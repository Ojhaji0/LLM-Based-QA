"""
Hybrid Retriever — Reciprocal Rank Fusion of dense and sparse results.
"""

def reciprocal_rank_fusion(dense_results, sparse_results, k=60):
    """
    RRF algorithm: score = 1 / (k + rank)
    dense_results: list of (doc, metadata, score, id) from Chroma
    sparse_results: list of (doc, metadata, score, id) from BM25
    Returns sorted list of (doc, metadata, fused_score, id)
    """
    scores = {}
    docs_map = {}
    
    # Process dense
    for rank, item in enumerate(dense_results):
        doc, metadata, score, doc_id = item
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
        docs_map[doc_id] = (doc, metadata)
        
    # Process sparse
    for rank, item in enumerate(sparse_results):
        doc, metadata, score, doc_id = item
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
        docs_map[doc_id] = (doc, metadata)
        
    fused = []
    for doc_id, fused_score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        doc, metadata = docs_map[doc_id]
        fused.append((doc, metadata, fused_score, doc_id))
        
    return fused
