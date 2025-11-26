from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StructuredOutputParser
from langchain.retrievers.multi_query import MultiQueryRetriever
import streamlit as st

from config import *
import os
from dotenv import load_dotenv
load_dotenv()

def initialize_rag_system():
    vectorstore = Chroma(
        embedding_function=OpenAIEmbeddings(model = EMBEDDING_MODEL),
        persist_directory= CHROMA_DB_PATH)

    llm_queries = ChatOpenAI(model = QUERY_MODEL, temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
    llm_generation = ChatOpenAI(model = GENERATION_MODEL, temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))

    # Retriever MMR (Maximal Marginal Relevance)
    base_retriever = vectorstore.as_retriever(search_type= SEARCH_TYPE, search_kwargs={"k":SEARCH_K, "fetch_k": MMR_FETCH_K,"lambda_mult": MMR_DIVERSITY_LAMBDA })

