"""
BM25 Retriever — sparse retrieval fallback.
"""
from __future__ import annotations
from rank_bm25 import BM25Okapi
from loguru import logger
from langchain_community.vectorstores import Chroma

class BM25Retriever:
    def __init__(self, vectorstore: Chroma):
        # We need to build the BM25 index from all documents in Chroma
        # For a production system, you'd want to persist this index or load from a separate corpus file.
        logger.info("Initializing BM25 index from Chroma vector store...")
        try:
            # We can get all docs from Chroma
            data = vectorstore.get()
            self.corpus = data['documents']
            self.metadatas = data['metadatas']
            self.ids = data['ids']
            
            tokenized_corpus = [doc.lower().split() for doc in self.corpus]
            self.bm25 = BM25Okapi(tokenized_corpus)
            logger.info(f"BM25 initialized with {len(self.corpus)} documents.")
        except Exception as e:
            logger.error(f"Failed to initialize BM25: {e}")
            self.corpus = []
            self.bm25 = None

    def retrieve(self, query: str, k: int = 20) -> list[tuple]:
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(query.lower().split())
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        
        # Return format: (doc_text, metadata, score, id)
        return [(self.corpus[i], self.metadatas[i], scores[i], self.ids[i]) for i in top_idx]
