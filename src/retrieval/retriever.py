"""
Retriever Orchestrator — combines Chroma, BM25, and bge-reranker.
"""
from __future__ import annotations
from loguru import logger
from src.retrieval.vector_store import get_vector_store
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import reciprocal_rank_fusion
from src.retrieval.reranker import rerank

class HybridRetrieverPipeline:
    def __init__(self, persist_directory: str = "vectorstore"):
        self.vectorstore = get_vector_store(persist_directory)
        self.bm25 = BM25Retriever(self.vectorstore)
        
    def retrieve(self, query: str, top_k: int = 5, top_k_dense: int = 25, top_k_sparse: int = 25, metadata_filter: dict | None = None):
        logger.info(f"Retrieving for query: '{query}' with filter: {metadata_filter}")
        
        # 1. Dense Retrieval (Chroma)
        # Pass the filter to vectorstore
        dense_docs = self.vectorstore.similarity_search_with_score(query, k=top_k_dense, filter=metadata_filter)
        dense_results = []
        for doc, score in dense_docs:
            doc_id = doc.metadata.get("id") or f"hash_{hash(doc.page_content)}"
            dense_results.append((doc.page_content, doc.metadata, score, doc_id))
            
        logger.debug(f"Dense retrieved {len(dense_results)} chunks")
        
        # 2. Sparse Retrieval (BM25)
        # Note: Current BM25Retriever doesn't support filters directly, but we can filter results after retrieval
        sparse_results = self.bm25.retrieve(query, k=top_k_sparse)
        if metadata_filter:
            sparse_results = [
                res for res in sparse_results 
                if all(res[1].get(k) == v for k, v in metadata_filter.items())
            ]
        logger.debug(f"Sparse retrieved {len(sparse_results)} chunks (after filtering)")
        
        # 3. Hybrid Fusion (RRF)
        fused_results = reciprocal_rank_fusion(dense_results, sparse_results)
        
        # Deduplication (pre-rerank)
        unique_candidates = []
        seen_texts = set()
        for res in fused_results:
            text_prefix = res[0][:200].lower() # res[0] is the document text
            if text_prefix not in seen_texts:
                seen_texts.add(text_prefix)
                unique_candidates.append(res)
                
        logger.debug(f"Fused and deduplicated to {len(unique_candidates)} unique candidates")
        
        # Take top 20 for reranking (Deeper pool for Cross-Encoder)
        candidates_to_rerank = unique_candidates[:20]
        
        # 4. Reranking (Cross-Encoder)
        final_results = rerank(query, candidates_to_rerank, top_k=top_k)
        logger.info(f"Final retrieved {len(final_results)} chunks after reranking top 20 pool")
        
        return final_results

if __name__ == "__main__":
    # Test script for Phase 3
    retriever = HybridRetrieverPipeline()
    test_query = "What is the attention mechanism?"
    print(f"\n--- Testing Query: {test_query} ---\n")
    res = retriever.retrieve(test_query)
    for i, (doc, meta, score, _) in enumerate(res):
        print(f"\n--- Rank {i+1} (Score: {score:.4f}) ---")
        print(f"Source: {meta.get('source_file')} | Page: {meta.get('page')}")
        print(doc[:250] + "...\n")
