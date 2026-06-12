"""
Reranker — bge-reranker-base cross-encoder.
"""
from __future__ import annotations
from FlagEmbedding import FlagReranker
from loguru import logger

_reranker = None

def get_reranker(model_name: str = "BAAI/bge-reranker-base", use_fp16: bool = False):
    global _reranker
    if _reranker is None:
        logger.info(f"Loading reranker model: {model_name} on CPU")
        # Ensure it runs on CPU since GPU is 6GB and used by LLM + Embedder
        _reranker = FlagReranker(model_name, use_fp16=use_fp16)
    return _reranker

def rerank(query: str, candidates: list[tuple], top_k: int = 5) -> list[tuple]:
    """
    candidates: list of (doc, metadata, score, id)
    Returns: list of (doc, metadata, rerank_score, id)
    """
    reranker = get_reranker()
    if not candidates:
        return []
        
    pairs = [[query, item[0]] for item in candidates]
    scores = reranker.compute_score(pairs)
    
    # If there's only 1 candidate, FlagReranker might return a float instead of a list
    if isinstance(scores, float):
        scores = [scores]
        
    ranked = []
    for i, score in enumerate(scores):
        doc, metadata, _, doc_id = candidates[i]
        ranked.append((doc, metadata, score, doc_id))
        
    # Sort by reranker score descending
    ranked = sorted(ranked, key=lambda x: x[2], reverse=True)
    return ranked[:top_k]
