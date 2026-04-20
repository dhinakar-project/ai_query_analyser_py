"""RAG (Retrieval-Augmented Generation) module."""

from rag.embedder import get_embedder
from rag.store import initialize_store, get_store
from rag.retriever import retrieve

__all__ = ["get_embedder", "initialize_store", "get_store", "retrieve"]