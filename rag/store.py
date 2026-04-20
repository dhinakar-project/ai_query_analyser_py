"""ChromaDB vector store for RAG."""

import json
import logging
import os
from typing import List, Optional

import chromadb
from chromadb.config import Settings

from models.analysis_result import Article
from rag.embedder import encode_texts

logger = logging.getLogger(__name__)

_store: chromadb.PersistentClient | None = None
_collection = None


def load_articles() -> List[Article]:
    """Load articles from the knowledge base JSONL file.
    
    Returns:
        List of Article objects.
    """
    articles = []
    kb_path = os.path.join(os.path.dirname(__file__), "..", "knowledge_base", "support_articles.jsonl")
    
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
    """Get or create the ChromaDB persistent client.
    
    Returns:
        ChromaDB client instance.
    """
    global _store
    
    if _store is None:
        logger.info("Initializing ChromaDB client...")
        
        os.makedirs("chroma_db", exist_ok=True)
        
        _store = chromadb.PersistentClient(
            path="chroma_db",
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            )
        )
        logger.info("ChromaDB client initialized")
    
    return _store


def initialize_store() -> None:
    """Initialize the vector store with articles if collection is empty."""
    global _collection
    
    client = get_store()
    
    try:
        _collection = client.get_collection("support_articles")
        
        if _collection.count() == 0:
            logger.info("Collection is empty, upserting articles...")
            articles = load_articles()
            
            ids = [a.id for a in articles]
            documents = [a.content for a in articles]
            metadatas = [{"category": a.category, "title": a.title} for a in articles]
            
            embeddings = encode_texts(documents)
            
            _collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
            logger.info(f"Upserted {len(articles)} articles")
        
    except Exception as e:
        logger.error(f"Error initializing store: {e}")
        
        _collection = client.create_collection(
            "support_articles",
            metadata={"hnsw:space": "cosine"}
        )
        
        articles = load_articles()
        
        ids = [a.id for a in articles]
        documents = [a.content for a in articles]
        metadatas = [{"category": a.category, "title": a.title} for a in articles]
        
        embeddings = encode_texts(documents)
        
        _collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
        logger.info(f"Created collection and added {len(articles)} articles")


def get_collection():
    """Get the articles collection.
    
    Returns:
        ChromaDB collection.
    """
    if _collection is None:
        initialize_store()
    return _collection