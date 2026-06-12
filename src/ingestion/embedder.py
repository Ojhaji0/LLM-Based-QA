"""
Embedder — SBERT all-MiniLM-L6-v2 dense vector generation.
GPU-accelerated (CUDA), batch_size=32 for 16GB RAM constraint.
"""
from __future__ import annotations
from sentence_transformers import SentenceTransformer
from loguru import logger

_model = None  # lazy load


def get_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
              device: str = "cuda") -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {model_name} on {device}")
        _model = SentenceTransformer(model_name, device=device)
    return _model


def embed_texts(texts: list[str], batch_size: int = 32,
                device: str = "cuda") -> list[list[float]]:
    """
    Encode a list of texts into 384-dim dense vectors.
    Returns list of float lists (JSON-serializable).
    """
    model = get_model(device=device)
    logger.info(f"Embedding {len(texts)} texts (batch={batch_size})")
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # cosine similarity ready
    )
    return vectors.tolist()


def embed_query(query: str, device: str = "cuda") -> list[float]:
    """Encode a single query string."""
    model = get_model(device=device)
    vec = model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
    return vec.tolist()
