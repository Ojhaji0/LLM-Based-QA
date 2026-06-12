import os
import sys
from src.retrieval.retriever import HybridRetrieverPipeline
from src.retrieval.vector_store import get_vector_store

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

retriever = HybridRetrieverPipeline()

# Define test query
query = sys.argv[1] if len(sys.argv) > 1 else "What is consistent hashing and how does it resolve hotspot keys?"

print(f"\n======================================")
print(f"QUERY: {query}")
print(f"======================================\n")

# Let's intercept dense, sparse, fused, and reranked results
metadata_filter = None
top_k_dense = 25
top_k_sparse = 25
top_k = 5

# 1. Dense (Chroma)
dense_docs = retriever.vectorstore.similarity_search_with_score(query, k=top_k_dense, filter=metadata_filter)
dense_results = []
for doc, score in dense_docs:
    doc_id = doc.metadata.get("id") or f"hash_{hash(doc.page_content)}"
    dense_results.append((doc.page_content, doc.metadata, score, doc_id))

print(f"--- DENSE RETRIEVAL (Top 3 Chunks) ---")
for i, item in enumerate(dense_results[:3]):
    print(f"Rank {i+1} (Chroma Distance Score: {item[2]:.4f})")
    print(f"Source: {os.path.basename(item[1].get('source_file', ''))} | Page: {item[1].get('page', '')}")
    print(f"Content: {item[0][:150].strip()}...")
    print("-" * 40)

# 2. Sparse (BM25)
sparse_results = retriever.bm25.retrieve(query, k=top_k_sparse)
print(f"\n--- SPARSE RETRIEVAL (Top 3 Chunks) ---")
for i, item in enumerate(sparse_results[:3]):
    print(f"Rank {i+1} (BM25 Score: {item[2]:.4f})")
    print(f"Source: {os.path.basename(item[1].get('source_file', ''))} | Page: {item[1].get('page', '')}")
    print(f"Content: {item[0][:150].strip()}...")
    print("-" * 40)

# 3. Hybrid Fusion (RRF)
from src.retrieval.hybrid_retriever import reciprocal_rank_fusion
fused_results = reciprocal_rank_fusion(dense_results, sparse_results)
unique_candidates = []
seen_texts = set()
for res in fused_results:
    text_prefix = res[0][:200].lower()
    if text_prefix not in seen_texts:
        seen_texts.add(text_prefix)
        unique_candidates.append(res)

print(f"\n--- FUSED CANDIDATES (RRF Top 3 Chunks) ---")
for i, item in enumerate(unique_candidates[:3]):
    print(f"Rank {i+1} (RRF Score: {item[2]:.4f})")
    print(f"Source: {os.path.basename(item[1].get('source_file', ''))} | Page: {item[1].get('page', '')}")
    print(f"Content: {item[0][:150].strip()}...")
    print("-" * 40)

# 4. Reranked (Cross-Encoder)
from src.retrieval.reranker import rerank
final_results = rerank(query, unique_candidates[:20], top_k=top_k)

print(f"\n--- FINAL RERANKED (Cross-Encoder Top 3 Chunks) ---")
for i, item in enumerate(final_results[:3]):
    print(f"Rank {i+1} (Rerank Score: {item[2]:.4f})")
    print(f"Source: {os.path.basename(item[1].get('source_file', ''))} | Page: {item[1].get('page', '')}")
    print(f"Content: {item[0][:150].strip()}...")
    print("-" * 40)
