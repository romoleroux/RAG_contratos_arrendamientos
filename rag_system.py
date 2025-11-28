# rag_system.py

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import EnsembleRetriever
import streamlit as st

from config import *
from prompts import *
import os
from dotenv import load_dotenv
load_dotenv()

@st.cache_resource
def initialize_rag_system():
    vectorstore = Chroma(
        embedding_function=OpenAIEmbeddings(model = EMBEDDING_MODEL),
        persist_directory= CHROMA_DB_PATH)

    llm_queries = ChatOpenAI(model = QUERY_MODEL, temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
    llm_generation = ChatOpenAI(model = GENERATION_MODEL, temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))

    # Retriever MMR (Maximal Marginal Relevance)
    base_retriever = vectorstore.as_retriever(search_type= SEARCH_TYPE, search_kwargs={"k":SEARCH_K, "fetch_k": MMR_FETCH_K,"lambda_mult": MMR_DIVERSITY_LAMBDA })

    # Retriever adicina con similarity para comparar
    similarity_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": SEARCH_K})

    # prompt personalizado para multi query retriever
    multi_query_prompt = PromptTemplate.from_template(MULTI_QUERY_PROMPT)

    # MultiQueryRetriever con prompt personalizado
    mmr_multi_retriever = MultiQueryRetriever.from_llm(
        llm=llm_queries,
        retriever=base_retriever,
        prompt=multi_query_prompt
    )

    # Ensemble retriever para combinar MMR y similarity
    if ENABLE_HYBRID_SEARCH:
        ensemble_retriever = EnsembleRetriever(
            retrievers=[mmr_multi_retriever, similarity_retriever],
            weights=[0.7, 0.3],
            similarity_threshold=SIMILARITY_THRESHOLD
        )
        final_retriever = ensemble_retriever
    else:
        final_retriever = mmr_multi_retriever

    prompt = PromptTemplate.from_template(RAG_TEMPLATE)

    # fucnipon para formatear y procesar los documents recuperados
    def format_docs(docs):
        formatted = []
        for i, doc in enumerate(docs):
            header = f'[fragmente {i}]'
            if doc.metadata:
                if 'source' in doc.metadata:
                    source = doc.metadata['source'].split("\\")[-1] if '\\' in doc.metadata['source'] else doc.metadata['source']
                    header += f' (source: {source})'
                if 'page' in doc.metadata:
                    header += f' (page: {doc.metadata["page"]})'
            content = doc.page_content.strip()
            formatted.append(f'{header}\n{content}\n')
        return '\n'.join(formatted)

    rag_chain = (
        {
            "context" : final_retriever | format_docs,
            "question": RunnablePassthrough()
        } 
        |prompt 
        |llm_generation 
        |StrOutputParser()
    )
    return rag_chain, mmr_multi_retriever

def query_rag(question):
    try:
        rag_chain, retriever = initialize_rag_system()
        response = rag_chain.invoke(question)

        # Obtener documents para mostrarlos
        docs = retriever.get_relevant_documents(question)

        # Formatear los documentos para mostrar
        docs_info = []
        for i, doc in enumerate(docs[:SEARCH_K], 1):
            doc_info = {
                "fragmento": i,
                "contenido": doc.page_content[:1000] + "..." if len(doc.page_content) > 1000 else doc.page_content,
                "fuente": doc.metadata.get("source", "Desconocida").split("\\")[-1],
                "pagina": doc.metadata.get("page", "Desconocida")
            ,
            }

            docs_info.append(doc_info)
        return response, docs_info
    except Exception as e:
        return f"Error al procesar la consulta: {str(e)}", []


def get_retriever_info():
    """Obtiene informacion sobre la configuracion del retriever"""
    return {
        "tipo": f"{SEARCH_TYPE.upper()}c+ MultiQuery" + (" + Hybrid" if ENABLE_HYBRID_SEARCH else ""), 
        "documento":SEARCH_K,
        "diversidad": MMR_DIVERSITY_LAMBDA,
        "candidatos" : MMR_FETCH_K,
        "umbral" : SIMILARITY_THRESHOLD if ENABLE_HYBRID_SEARCH else "N/A"
    }
