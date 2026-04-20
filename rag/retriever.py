"""Retriever for RAG-based query responses."""

import logging
from typing import List

from models.analysis_result import Article
from rag.store import get_collection, initialize_store

logger = logging.getLogger(__name__)

_initialized = False


def _ensure_initialized() -> None:
    global _initialized
    if not _initialized:
        initialize_store()
        _initialized = True


async def retrieve(query: str, category: str, k: int = 3) -> List[Article]:
    """Retrieve relevant support articles for a query.

    Filters by category metadata then performs semantic search.

    Args:
        query: The customer query text.
        category: Category to filter by.
        k: Number of articles to retrieve.

    Returns:
        List of relevant Article objects.
    """
    _ensure_initialized()

    try:
        collection = get_collection()

        if collection is None or collection.count() == 0:
            logger.warning("Collection is empty — skipping RAG retrieval")
            return []

        results = collection.query(
            query_texts=[query],
            n_results=min(k, collection.count()),
            where={"category": category},
            include=["documents", "metadatas"],
        )

        articles = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                articles.append(
                    Article(
                        id=results["ids"][0][i] if results["ids"] else f"art-{i}",
                        title=meta.get("title", "Unknown"),
                        category=meta.get("category", category),
                        content=doc,
                        tags=[],
                    )
                )

        logger.debug(f"Retrieved {len(articles)} articles for category={category}")
        return articles

    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        return []