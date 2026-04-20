"""Structured JSON logger for observability."""

import json
import logging
import os
import hashlib
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from typing import Optional


trace_id_counter = 0


class JSONFormatter(logging.Formatter):
    """Custom formatter that outputs JSON to log files."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        if hasattr(record, "node_name"):
            log_data["node_name"] = record.node_name
        if hasattr(record, "model"):
            log_data["model"] = record.model
        if hasattr(record, "input_tokens"):
            log_data["input_tokens"] = record.input_tokens
        if hasattr(record, "output_tokens"):
            log_data["output_tokens"] = record.output_tokens
        if hasattr(record, "latency_ms"):
            log_data["latency_ms"] = record.latency_ms
        if hasattr(record, "trace_id"):
            log_data["trace_id"] = record.trace_id
        if hasattr(record, "query_hash"):
            log_data["query_hash"] = record.query_hash
        if hasattr(record, "total_nodes"):
            log_data["total_nodes"] = record.total_nodes
        if hasattr(record, "total_latency_ms"):
            log_data["total_latency_ms"] = record.total_latency_ms
        if hasattr(record, "final_category"):
            log_data["final_category"] = record.final_category
        if hasattr(record, "final_sentiment"):
            log_data["final_sentiment"] = record.final_sentiment
        if hasattr(record, "final_priority"):
            log_data["final_priority"] = record.final_priority
        if hasattr(record, "escalated"):
            log_data["escalated"] = record.escalated
        if hasattr(record, "confidence_category"):
            log_data["confidence_category"] = record.confidence_category
        if hasattr(record, "confidence_sentiment"):
            log_data["confidence_sentiment"] = record.confidence_sentiment
            
        return json.dumps(log_data)


def setup_logger() -> logging.Logger:
    """Set up the observability logger with file handler."""
    os.makedirs("logs", exist_ok=True)
    
    logger = logging.getLogger("observability")
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = TimedRotatingFileHandler(
            "logs/queries.jsonl",
            when="midnight",
            interval=1,
            backupCount=7
        )
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        
        console = logging.StreamHandler()
        console.setLevel(logging.WARNING)
        logger.addHandler(console)
    
    return logger


def get_query_hash(query: str) -> str:
    """Generate a SHA256 hash of the query for privacy-safe logging.
    
    Args:
        query: The query string to hash.
        
    Returns:
        Hex string of the hash.
    """
    return hashlib.sha256(query.encode()).hexdigest()[:16]


def get_trace_id() -> str:
    """Generate a unique trace ID for a graph run."""
    global trace_id_counter
    trace_id_counter += 1
    return f"trace_{trace_id_counter}_{datetime.now().strftime('%Y%m%d%H%M%S')}"


logger = setup_logger()


def log_llm_call(
    node_name: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    latency_ms: int,
    query_hash: Optional[str] = None,
    trace_id: Optional[str] = None
) -> None:
    """Log an LLM call with tokens and latency.
    
    Args:
        node_name: Name of the graph node making the call.
        model: Model identifier used.
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens.
        latency_ms: Time taken in milliseconds.
        query_hash: Hash of the query for privacy-safe logging.
        trace_id: Trace identifier for the run.
    """
    extra = {
        "node_name": node_name,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_ms": latency_ms,
    }
    if query_hash:
        extra["query_hash"] = query_hash
    if trace_id:
        extra["trace_id"] = trace_id
        
    logger.info(
        f"LLM call: {node_name} ({model})",
        extra=extra
    )


def log_graph_run(
    trace_id: str,
    total_nodes: int,
    total_latency_ms: int,
    final_category: str,
    final_sentiment: str,
    final_priority: str,
    escalated: bool,
    confidence_category: int,
    confidence_sentiment: int,
    query_hash: Optional[str] = None
) -> None:
    """Log a complete graph run with all metrics.
    
    Args:
        trace_id: Unique trace identifier.
        total_nodes: Number of nodes visited.
        total_latency_ms: Total processing time in milliseconds.
        final_category: Final category classification.
        final_sentiment: Final sentiment analysis.
        final_priority: Final priority level.
        escalated: Whether the query was escalated.
        confidence_category: Category confidence score.
        confidence_sentiment: Sentiment confidence score.
        query_hash: Hash of the query for privacy-safe logging.
    """
    extra = {
        "trace_id": trace_id,
        "total_nodes": total_nodes,
        "total_latency_ms": total_latency_ms,
        "final_category": final_category,
        "final_sentiment": final_sentiment,
        "final_priority": final_priority,
        "escalated": escalated,
        "confidence_category": confidence_category,
        "confidence_sentiment": confidence_sentiment,
    }
    if query_hash:
        extra["query_hash"] = query_hash
        
    logger.info(
        f"Graph run complete: {final_category} ({confidence_category}%) / {final_sentiment} / {final_priority}",
        extra=extra
    )