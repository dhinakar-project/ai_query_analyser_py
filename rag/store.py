"""ChromaDB vector store for RAG.

Uses chromadb.utils.embedding_functions.DefaultEmbeddingFunction (onnxruntime)
so no torch/torchvision/sentence-transformers is needed.
"""

import json
import logging
import os
from typing import List, Optional

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from models.analysis_result import Article

logger = logging.getLogger(__name__)

_client: Optional[chromadb.PersistentClient] = None
_collection = None
_embedding_fn = DefaultEmbeddingFunction()


def load_articles() -> List[Article]:
    """Load articles from the knowledge base JSONL file."""
    articles = []
    kb_path = os.path.join(
        os.path.dirname(__file__), "..", "knowledge_base", "support_articles.jsonl"
    )

    if not os.path.exists(kb_path):
        logger.error(f"Knowledge base not found: {kb_path}")
        return articles

    with open(kb_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                articles.append(Article(**data))

    logger.info(f"Loaded {len(articles)} articles from knowledge base")
    return articles


def get_store() -> chromadb.PersistentClient:
    """Get or create the ChromaDB persistent client."""
    global _client

    if _client is None:
        logger.info("Initializing ChromaDB client...")
        os.makedirs("chroma_db", exist_ok=True)
        _client = chromadb.PersistentClient(
            path="chroma_db",
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )
        logger.info("ChromaDB client initialized")

    return _client


def initialize_store() -> None:
    """Initialize the vector store with articles. Re-indexes if count changes."""
    global _collection

    client = get_store()

    _collection = client.get_or_create_collection(
        name="support_articles",
        embedding_function=_embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    articles = load_articles()
    article_count = len(articles)
    
    current_count = _collection.count()
    
    if current_count != article_count or current_count == 0:
        logger.info(f"Re-indexing articles: {current_count} -> {article_count}")
        
        try:
            client.delete_collection("support_articles")
        except Exception:
            pass
            
        _collection = client.get_or_create_collection(
            name="support_articles",
            embedding_function=_embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        
        if articles:
            _collection.add(
                ids=[a.id for a in articles],
                documents=[a.content for a in articles],
                metadatas=[{"category": a.category, "title": a.title} for a in articles],
            )
            logger.info(f"Indexed {len(articles)} articles into ChromaDB")
    else:
        logger.info(f"Collection already has {current_count} articles")


def get_collection():
    """Get the articles collection, initializing if needed."""
    if _collection is None:
        initialize_store()
    return _collection