"""SQLite database for persistent query analytics."""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Optional, Dict, Any, List

import aiosqlite

from observability.logger import get_query_hash

logger = logging.getLogger(__name__)

DB_PATH = "data/analytics.db"


class Database:
    """Async SQLite database for query analytics."""
    
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._connection: Optional[aiosqlite.Connection] = None
        
    async def connect(self) -> None:
        """Connect to the database and create tables if needed."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        self._connection = await aiosqlite.connect(self.db_path)
        self._connection.row_factory = aiosqlite.Row
        
        await self._create_tables()
        logger.info(f"Connected to database: {self.db_path}")
        
    async def _create_tables(self) -> None:
        """Create tables if they don't exist."""
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                query_hash TEXT NOT NULL,
                category TEXT NOT NULL,
                sentiment TEXT NOT NULL,
                priority TEXT NOT NULL,
                escalated INTEGER NOT NULL,
                language TEXT NOT NULL,
                latency_ms INTEGER NOT NULL,
                cost_usd REAL NOT NULL,
                category_confidence INTEGER NOT NULL,
                sentiment_confidence INTEGER NOT NULL
            )
        """)
        
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT NOT NULL,
                query_count INTEGER NOT NULL DEFAULT 0,
                total_cost_usd REAL NOT NULL DEFAULT 0
            )
        """)

        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS voice_calls (
                call_id TEXT PRIMARY KEY,
                transcript TEXT,
                category TEXT,
                sentiment TEXT,
                priority TEXT,
                should_escalate INTEGER,
                ai_response TEXT,
                duration_seconds INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_hash TEXT NOT NULL,
                category TEXT,
                was_helpful INTEGER NOT NULL,
                timestamp TEXT NOT NULL
            )
        """)

        await self._connection.commit()
        
    async def close(self) -> None:
        """Close the database connection."""
        if self._connection:
            await self._connection.close()
            logger.info("Database connection closed")
            
    async def record_query(
        self,
        query: str,
        category: str,
        sentiment: str,
        priority: str,
        escalated: bool,
        language: str,
        latency_ms: int,
        cost_usd: float,
        category_confidence: int,
        sentiment_confidence: int
    ) -> None:
        """Record a query to the database.
        
        Args:
            query: The original query text.
            category: Category classification.
            sentiment: Sentiment analysis result.
            priority: Priority level.
            escalated: Whether query was escalated.
            language: Detected language.
            latency_ms: Processing time in ms.
            cost_usd: Cost of the LLM call.
            category_confidence: Category confidence score.
            sentiment_confidence: Sentiment confidence score.
        """
        query_hash = get_query_hash(query)
        timestamp = datetime.now().isoformat()
        
        await self._connection.execute(
            """
            INSERT INTO queries 
            (timestamp, query_hash, category, sentiment, priority, escalated, 
             language, latency_ms, cost_usd, category_confidence, sentiment_confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (timestamp, query_hash, category, sentiment, priority, 
             int(escalated), language, latency_ms, cost_usd, 
             category_confidence, sentiment_confidence)
        )
        
        await self._connection.commit()
        logger.debug(f"Recorded query: {category} / {sentiment} / {priority}")
        
    async def get_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get summary statistics for the specified number of days.
        
        Args:
            days: Number of days to look back.
            
        Returns:
            Dictionary with summary statistics.
        """
        cutoff = datetime.now()
        from datetime import timedelta
        cutoff = cutoff - timedelta(days=days)
        cutoff_str = cutoff.isoformat()
        
        async with self._connection.execute(
            """
            SELECT 
                COUNT(*) as total_queries,
                AVG(latency_ms) as avg_latency,
                SUM(CASE WHEN escalated = 1 THEN 1 ELSE 0 END) as escalations,
                SUM(cost_usd) as total_cost
            FROM queries
            WHERE timestamp >= ?
            """,
            (cutoff_str,)
        ) as cursor:
            row = await cursor.fetchone()
            
        async with self._connection.execute(
            """
            SELECT category, COUNT(*) as count
            FROM queries
            WHERE timestamp >= ?
            GROUP BY category
            """,
            (cutoff_str,)
        ) as cursor:
            category_rows = await cursor.fetchall()
            
        async with self._connection.execute(
            """
            SELECT sentiment, COUNT(*) as count
            FROM queries
            WHERE timestamp >= ?
            GROUP BY sentiment
            """,
            (cutoff_str,)
        ) as cursor:
            sentiment_rows = await cursor.fetchall()
            
        return {
            "total_queries": row["total_queries"] if row else 0,
            "avg_latency_ms": row["avg_latency"] if row and row["avg_latency"] else 0,
            "escalation_count": row["escalations"] if row else 0,
            "escalation_rate": (row["escalations"] / row["total_queries"] * 100) if row and row["total_queries"] else 0,
            "total_cost_usd": row["total_cost"] if row else 0,
            "category_breakdown": {r["category"]: r["count"] for r in category_rows},
            "sentiment_distribution": {r["sentiment"]: r["count"] for r in sentiment_rows}
        }
        
    async def get_trends(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get trends over time.
        
        Args:
            days: Number of days to look back.
            
        Returns:
            List of daily trend data.
        """
        cutoff = datetime.now()
        from datetime import timedelta
        cutoff = cutoff - timedelta(days=days)
        cutoff_str = cutoff.isoformat()
        
        async with self._connection.execute(
            """
            SELECT 
                DATE(timestamp) as date,
                COUNT(*) as query_count,
                AVG(latency_ms) as avg_latency
            FROM queries
            WHERE timestamp >= ?
            GROUP BY DATE(timestamp)
            ORDER BY date
            """,
            (cutoff_str,)
        ) as cursor:
            rows = await cursor.fetchall()
            
        return [
{
                "date": r["date"],
                "query_count": r["query_count"],
                "avg_latency_ms": r["avg_latency_ms"]
            }
            for r in rows
        ]

    async def save_voice_call(
        self,
        call_id: str,
        transcript: str,
        category: str,
        sentiment: str,
        priority: str,
        should_escalate: bool,
        ai_response: str,
        duration_seconds: int,
    ) -> None:
        """Save a voice call result.

        Args:
            call_id: Call ID
            transcript: Voice transcript
            category: Category classification
            sentiment: Sentiment analysis
            priority: Priority level
            should_escalate: Whether to escalate
            ai_response: AI response text
            duration_seconds: Call duration in seconds
        """
        timestamp = datetime.now().isoformat()

        await self._connection.execute(
            """
            INSERT OR REPLACE INTO voice_calls
            (call_id, transcript, category, sentiment, priority,
             should_escalate, ai_response, duration_seconds, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (call_id, transcript, category, sentiment, priority,
             int(should_escalate), ai_response, duration_seconds, timestamp)
        )

        await self._connection.commit()
        logger.debug(f"Saved voice call: {call_id}")

    async def get_voice_calls(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent voice calls.

        Args:
            limit: Maximum number of calls to return

        Returns:
            List of voice call dicts
        """
        async with self._connection.execute(
            """
            SELECT * FROM voice_calls
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,)
        ) as cursor:
            rows = await cursor.fetchall()

        return [
            {
                "call_id": r["call_id"],
                "transcript": r["transcript"],
                "category": r["category"],
                "sentiment": r["sentiment"],
                "priority": r["priority"],
                "should_escalate": bool(r["should_escalate"]),
                "ai_response": r["ai_response"],
                "duration_seconds": r["duration_seconds"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    async def get_voice_call(self, call_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific voice call.

        Args:
            call_id: Call ID

        Returns:
            Voice call dict or None
        """
        async with self._connection.execute(
            "SELECT * FROM voice_calls WHERE call_id = ?",
            (call_id,)
        ) as cursor:
            row = await cursor.fetchone()

        if not row:
            return None

        return {
            "call_id": row["call_id"],
            "transcript": row["transcript"],
            "category": row["category"],
            "sentiment": row["sentiment"],
            "priority": row["priority"],
            "should_escalate": bool(row["should_escalate"]),
            "ai_response": row["ai_response"],
            "duration_seconds": row["duration_seconds"],
            "created_at": row["created_at"],
        }
        
    async def record_feedback(self, query_hash: str, category: str, was_helpful: bool) -> None:
        await self._connection.execute(
            "INSERT INTO feedback (query_hash, category, was_helpful, timestamp) VALUES (?, ?, ?, ?)",
            (query_hash, category, int(was_helpful), datetime.now().isoformat())
        )
        await self._connection.commit()
  
    async def get_feedback_stats(self) -> dict:
        async with self._connection.execute(
            "SELECT COUNT(*) as total, SUM(was_helpful) as helpful FROM feedback"
        ) as cursor:
            row = await cursor.fetchone()
        total = row["total"] or 0
        helpful = row["helpful"] or 0
        return {
            "total_rated": total,
            "helpful_count": helpful,
            "helpful_rate": round((helpful / total * 100), 1) if total > 0 else 0
        }


_db_instance: Optional[Database] = None


async def get_database() -> Database:
    """Get or create the database singleton.
    
    Returns:
        Database instance.
    """
    global _db_instance
    
    if _db_instance is None:
        _db_instance = Database()
        await _db_instance.connect()
        
    return _db_instance


async def close_database() -> None:
    """Close the database connection."""
    global _db_instance
    
    if _db_instance:
        await _db_instance.close()
        _db_instance = None