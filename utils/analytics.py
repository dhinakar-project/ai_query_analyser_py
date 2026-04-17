"""Analytics module for tracking and analyzing query metrics."""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


class QueryAnalytics:
    """Analytics tracker for customer query analysis sessions.
    
    This class maintains in-memory analytics for the current Streamlit session,
    tracking metrics like categories, sentiments, response times, and escalations.
    
    Attributes:
        total_queries: Total number of queries analyzed.
        category_counts: Dictionary mapping categories to their occurrence counts.
        sentiment_counts: Dictionary mapping sentiments to their occurrence counts.
        response_times_ms: List of response times in milliseconds.
        escalation_count: Number of queries flagged for escalation.
        language_counts: Dictionary mapping languages to occurrence counts.
    """
    
    def __init__(self) -> None:
        """Initialize the analytics tracker with default values."""
        self.total_queries: int = 0
        self.category_counts: Dict[str, int] = defaultdict(int)
        self.sentiment_counts: Dict[str, int] = defaultdict(int)
        self.response_times_ms: List[int] = []
        self.escalation_count: int = 0
        self.language_counts: Dict[str, int] = defaultdict(int)
        self.start_time: datetime = datetime.now()
        self._priority_counts: Dict[str, int] = defaultdict(int)
        self._queries_history: List[Dict[str, Any]] = []
        logger.info("QueryAnalytics initialized")
    
    def record_query(
        self,
        category: str,
        sentiment: str,
        priority: str,
        response_time_ms: int,
        should_escalate: bool,
        language: str
    ) -> None:
        """Record metrics for an analyzed query.
        
        Args:
            category: The classified category of the query.
            sentiment: The detected sentiment of the query.
            priority: The assessed priority level.
            response_time_ms: Time taken to process the query in milliseconds.
            should_escalate: Whether the query was flagged for escalation.
            language: The detected language of the query.
        """
        self.total_queries += 1
        self.category_counts[category] += 1
        self.sentiment_counts[sentiment] += 1
        self._priority_counts[priority] += 1
        self.response_times_ms.append(response_time_ms)
        self.language_counts[language] += 1
        
        if should_escalate:
            self.escalation_count += 1
        
        query_record = {
            "timestamp": datetime.now().isoformat(),
            "category": category,
            "sentiment": sentiment,
            "priority": priority,
            "response_time_ms": response_time_ms,
            "should_escalate": should_escalate,
            "language": language
        }
        self._queries_history.append(query_record)
        
        logger.debug(f"Recorded query metrics: category={category}, sentiment={sentiment}, "
                    f"priority={priority}, escalate={should_escalate}")
    
    def get_average_response_time_ms(self) -> float:
        """Calculate the average response time in milliseconds.
        
        Returns:
            Average response time in ms, or 0.0 if no queries recorded.
        """
        if not self.response_times_ms:
            return 0.0
        return sum(self.response_times_ms) / len(self.response_times_ms)
    
    def get_escalation_rate(self) -> float:
        """Calculate the escalation rate as a percentage.
        
        Returns:
            Escalation rate as a percentage (0-100), or 0.0 if no queries.
        """
        if self.total_queries == 0:
            return 0.0
        return (self.escalation_count / self.total_queries) * 100
    
    def get_most_common_category(self) -> Optional[str]:
        """Get the most frequently occurring category.
        
        Returns:
            The category with the highest count, or None if no categories recorded.
        """
        if not self.category_counts:
            return None
        return max(self.category_counts.items(), key=lambda x: x[1])[0]
    
    def get_most_common_sentiment(self) -> Optional[str]:
        """Get the most frequently occurring sentiment.
        
        Returns:
            The sentiment with the highest count, or None if no sentiments recorded.
        """
        if not self.sentiment_counts:
            return None
        return max(self.sentiment_counts.items(), key=lambda x: x[1])[0]
    
    def get_category_breakdown(self) -> Dict[str, int]:
        """Get the complete category breakdown.
        
        Returns:
            Dictionary mapping categories to their counts.
        """
        return dict(self.category_counts)
    
    def get_sentiment_distribution(self) -> Dict[str, int]:
        """Get the complete sentiment distribution.
        
        Returns:
            Dictionary mapping sentiments to their counts.
        """
        return dict(self.sentiment_counts)
    
    def get_priority_distribution(self) -> Dict[str, int]:
        """Get the priority distribution.
        
        Returns:
            Dictionary mapping priorities to their counts.
        """
        return dict(self._priority_counts)
    
    def get_language_distribution(self) -> Dict[str, int]:
        """Get the language distribution.
        
        Returns:
            Dictionary mapping languages to their counts.
        """
        return dict(self.language_counts)
    
    def get_session_duration_seconds(self) -> float:
        """Get the session duration in seconds.
        
        Returns:
            Seconds elapsed since analytics initialization.
        """
        delta = datetime.now() - self.start_time
        return delta.total_seconds()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a complete summary of all analytics metrics.
        
        Returns:
            Dictionary containing all analytics metrics and statistics.
        """
        return {
            "total_queries": self.total_queries,
            "average_response_time_ms": round(self.get_average_response_time_ms(), 2),
            "escalation_rate": round(self.get_escalation_rate(), 2),
            "most_common_category": self.get_most_common_category(),
            "most_common_sentiment": self.get_most_common_sentiment(),
            "category_breakdown": self.get_category_breakdown(),
            "sentiment_distribution": self.get_sentiment_distribution(),
            "priority_distribution": self.get_priority_distribution(),
            "language_distribution": self.get_language_distribution(),
            "session_duration_seconds": round(self.get_session_duration_seconds(), 2)
        }
    
    def reset(self) -> None:
        """Reset all analytics to initial state."""
        self.__init__()
        logger.info("Analytics reset")
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get the history of all recorded queries.
        
        Returns:
            List of query record dictionaries.
        """
        return self._queries_history.copy()
    
    def export_to_csv_format(self) -> str:
        """Export analytics data in CSV format.
        
        Returns:
            CSV-formatted string containing all query records.
        """
        if not self._queries_history:
            return "timestamp,category,sentiment,priority,response_time_ms,should_escalate,language\n"
        
        lines = ["timestamp,category,sentiment,priority,response_time_ms,should_escalate,language"]
        for record in self._queries_history:
            line = f"{record['timestamp']},{record['category']},{record['sentiment']},"
            line += f"{record['priority']},{record['response_time_ms']},"
            line += f"{record['should_escalate']},{record['language']}"
            lines.append(line)
        
        return "\n".join(lines)


def get_default_analytics() -> QueryAnalytics:
    """Get a default analytics instance with default values.
    
    Returns:
        A new QueryAnalytics instance.
    """
    return QueryAnalytics()
