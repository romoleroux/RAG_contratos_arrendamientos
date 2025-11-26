# import libraries
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import *
import os
from dotenv import load_dotenv
load_dotenv()

loader =  PyPDFDirectoryLoader("C:\\Users\\romol\\Documents\\Software\\AI_AGENT\\santiago_ciber\\langhcain_langraph_course\\sistema_RAG\\contratos")
documentos = loader.load()

print(f'Se cargaron {len(documentos)} documentos desde el directorio.')

# Text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=5000,
    chunk_overlap=1000,
)

docs_split = text_splitter.split_documents(documentos)
print(f'Se crearon {len(docs_split)} chunks de texto.')



vectorstore = Chroma.from_documents(
    embedding=OpenAIEmbeddings(model = EMBEDDING_MODEL),
    documents=docs_split,
    persist_directory=CHROMA_DB_PATH)

print("¡Base de datos creada y guardada con éxito!")

