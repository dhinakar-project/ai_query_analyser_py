"""Embedder using ChromaDB's built-in DefaultEmbeddingFunction.

Uses chromadb.utils.embedding_functions.DefaultEmbeddingFunction which
runs all-MiniLM-L6-v2 via onnxruntime — no torch, no torchvision,
no transformers required. Works on Python 3.14+.
"""

import logging
from typing import List

logger = logging.getLogger(__name__)

_embedding_fn = None


def _get_embedding_fn():
    """Get or create the ChromaDB default embedding function singleton."""
    global _embedding_fn
    if _embedding_fn is None:
        logger.info("Loading ChromaDB DefaultEmbeddingFunction (onnxruntime)...")
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
        _embedding_fn = DefaultEmbeddingFunction()
        logger.info("Embedding function loaded successfully")
    return _embedding_fn


def encode_texts(texts: List[str]) -> List[List[float]]:
    """Encode a list of texts into embedding vectors.

    Args:
        texts: List of text strings to encode.

    Returns:
        List of embedding vectors (each a list of floats).
    """
    fn = _get_embedding_fn()
    embeddings = fn(texts)
    return [list(emb) for emb in embeddings]


def encode_query(query: str) -> List[float]:
    """Encode a single query string into an embedding vector.

    Args:
        query: Query text to encode.

    Returns:
        Embedding vector as a list of floats.
    """
    fn = _get_embedding_fn()
    embeddings = fn([query])
    return list(embeddings[0])


def get_embedder():
    """Return the embedding function (for compatibility with any code that calls get_embedder()).

    Returns:
        The ChromaDB DefaultEmbeddingFunction instance.
    """
    return _get_embedding_fn()