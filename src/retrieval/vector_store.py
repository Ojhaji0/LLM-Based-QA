"""
Vector Store — Chroma DB integration.
"""
import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from loguru import logger

def get_vector_store(persist_directory: str = None) -> Chroma:
    if persist_directory is None:
        persist_directory = os.path.join(os.getcwd(), "vectorstore")
    
    logger.info(f"Loading Chroma DB from {persist_directory}")
    hf_embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    return Chroma(persist_directory=persist_directory, embedding_function=hf_embeddings)
