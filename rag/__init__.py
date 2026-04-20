"""RAG package — retrieval-augmented generation for support responses."""

from rag.store import initialize_store, get_store
from rag.retriever import retrieve

__all__ = ["initialize_store", "get_store", "retrieve"]