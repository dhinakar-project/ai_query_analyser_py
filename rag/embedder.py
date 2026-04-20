"""Embedder for semantic search using SentenceTransformer."""

import logging
from typing import List

from sentence_transformers import SentenceTransformer

from utils.llm import _get_api_key

logger = logging.getLogger(__name__)

_embedder: SentenceTransformer | None = None


def get_embedder() -> SentenceTransformer:
    """Get or create a singleton SentenceTransformer embedder.
    
    Returns:
        SentenceTransformer instance using all-MiniLM-L6-v2 model.
    """
    global _embedder
    
    if _embedder is None:
        logger.info("Loading SentenceTransformer embedder...")
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Embedder loaded successfully")
    
    return _embedder


def encode_texts(texts: List[str]) -> List[List[float]]:
    """Encode texts into embeddings.
    
    Args:
        texts: List of text strings to encode.
        
    Returns:
        List of embedding vectors.
    """
    embedder = get_embedder()
    embeddings = embedder.encode(texts, convert_to_numpy=True)
    return [emb.tolist() for emb in embeddings]


def encode_query(query: str) -> List[float]:
    """Encode a single query text.
    
    Args:
        query: Query text to encode.
        
    Returns:
        Embedding vector as a list of floats.
    """
    embedder = get_embedder()
    embedding = embedder.encode([query], convert_to_numpy=True)[0]
    return embedding.tolist()