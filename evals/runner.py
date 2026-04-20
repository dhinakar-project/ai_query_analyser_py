"""Evaluation runner for query analysis."""

import json
import logging
import time
from typing import List, Optional, Literal
from datetime import datetime

from pydantic import BaseModel, Field

from evals.golden_dataset import QueryTestCase, get_golden_dataset

logger = logging.getLogger(__name__)


class CaseResult(BaseModel):
    """Result of evaluating a single test case."""
    
    query: str = Field(description="The query that was tested")
    expected_category: str = Field(description="Expected category")
    actual_category: str = Field(description="Actual category")
    category_match: bool = Field(description="Whether category matched")
    expected_sentiment: str = Field(description="Expected sentiment")
    actual_sentiment: str = Field(description="Actual sentiment")
    sentiment_match: bool = Field(description="Whether sentiment matched")
    expected_priority: str = Field(description="Expected priority")
    actual_priority: str = Field(description="Actual priority")
    priority_match: bool = Field(description="Whether priority matched")
    expected_escalate: bool = Field(description="Expected escalation")
    actual_escalate: bool = Field(description="Actual escalation")
    escalate_match: bool = Field(description="Whether escalation matched")
    latency_ms: int = Field(description="Processing time in ms")
    confidence_category: int = Field(description="Category confidence score")
    confidence_sentiment: int = Field(description="Sentiment confidence score")


class EvalReport(BaseModel):
    """Complete evaluation report."""
    
    total_cases: int = Field(description="Total number of test cases")
    category_accuracy: float = Field(description="Category classification accuracy")
    sentiment_accuracy: float = Field(description="Sentiment analysis accuracy")
    priority_accuracy: float = Field(description="Priority assessment accuracy")
    escalation_precision: float = Field(description="Escalation precision")
    escalation_recall: float = Field(description="Escalation recall")
    avg_confidence_category: float = Field(description="Average category confidence")
    avg_confidence_sentiment: float = Field(description="Average sentiment confidence")
    avg_latency_ms: float = Field(description="Average processing time")
    timestamp: str = Field(description="When the eval was run")
    case_results: List[CaseResult] = Field(description="Per-case results")


async def run_single_case(
    test_case: QueryTestCase,
    verbose: bool = False
) -> CaseResult:
    """Run a single test case through the graph.
    
    Args:
        test_case: The test case to evaluate.
        verbose: Whether to print progress.
        
    Returns:
        CaseResult with actual vs expected values.
    """
    from graph.runner import run_graph
    
    start_time = time.time()
    
    try:
        result = await run_graph(test_case.query)
        
        latency_ms = int((time.time() - start_time) * 1000)
        
        actual_category = result.get("category", "Unknown")
        actual_sentiment = result.get("sentiment", "Unknown")
        actual_priority = result.get("priority", "Medium")
        actual_escalate = result.get("should_escalate", False)
        confidence_category = result.get("confidence_category", 0)
        confidence_sentiment = result.get("confidence_sentiment", 0)
        
    except Exception as e:
        logger.error(f"Error running test case: {e}")
        latency_ms = int((time.time() - start_time) * 1000)
        
        actual_category = "Error"
        actual_sentiment = "Error"
        actual_priority = "Error"
        actual_escalate = False
        confidence_category = 0
        confidence_sentiment = 0
    
    if verbose:
        print(f"  {test_case.query[:50]}... => cat:{actual_category}, sent:{actual_sentiment}, pri:{actual_priority}")
    
    return CaseResult(
        query=test_case.query[:100],
        expected_category=test_case.expected_category,
        actual_category=actual_category,
        category_match=actual_category == test_case.expected_category,
        expected_sentiment=test_case.expected_sentiment,
        actual_sentiment=actual_sentiment,
        sentiment_match=actual_sentiment == test_case.expected_sentiment,
        expected_priority=test_case.expected_priority,
        actual_priority=actual_priority,
        priority_match=actual_priority == test_case.expected_priority,
        expected_escalate=test_case.expected_escalate,
        actual_escalate=actual_escalate,
        escalate_match=actual_escalate == test_case.expected_escalate,
        latency_ms=latency_ms,
        confidence_category=confidence_category,
        confidence_sentiment=confidence_sentiment
    )


async def run_evals(
    dataset: Optional[List[QueryTestCase]] = None,
    verbose: bool = False
) -> EvalReport:
    """Run evaluation on the golden dataset.
    
    Args:
        dataset: Optional custom dataset. Uses golden dataset if not provided.
        verbose: Whether to print progress.
        
    Returns:
        EvalReport with all metrics and per-case results.
    """
    if dataset is None:
        dataset = get_golden_dataset()
    
    if verbose:
        print(f"Running evals on {len(dataset)} test cases...")
    
    results: List[CaseResult] = []
    
    for i, test_case in enumerate(dataset):
        if verbose:
            print(f"[{i+1}/{len(dataset)}] ", end="")
        
        result = await run_single_case(test_case, verbose)
        results.append(result)
    
    category_matches = sum(1 for r in results if r.category_match)
    sentiment_matches = sum(1 for r in results if r.sentiment_match)
    priority_matches = sum(1 for r in results if r.priority_match)
    
    true_positives = sum(1 for r in results if r.expected_escalate and r.actual_escalate)
    false_positives = sum(1 for r in results if not r.expected_escalate and r.actual_escalate)
    false_negatives = sum(1 for r in results if r.expected_escalate and not r.actual_escalate)
    
    total = len(results)
    
    category_accuracy = category_matches / total if total > 0 else 0
    sentiment_accuracy = sentiment_matches / total if total > 0 else 0
    priority_accuracy = priority_matches / total if total > 0 else 0
    
    escalation_precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    escalation_recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    avg_confidence_cat = sum(r.confidence_category for r in results) / total if total > 0 else 0
    avg_confidence_sent = sum(r.confidence_sentiment for r in results) / total if total > 0 else 0
    avg_latency = sum(r.latency_ms for r in results) / total if total > 0 else 0
    
    report = EvalReport(
        total_cases=total,
        category_accuracy=round(category_accuracy, 3),
        sentiment_accuracy=round(sentiment_accuracy, 3),
        priority_accuracy=round(priority_accuracy, 3),
        escalation_precision=round(escalation_precision, 3),
        escalation_recall=round(escalation_recall, 3),
        avg_confidence_category=round(avg_confidence_cat, 1),
        avg_confidence_sentiment=round(avg_confidence_sent, 1),
        avg_latency_ms=round(avg_latency, 0),
        timestamp=datetime.now().isoformat(),
        case_results=results
    )
    
    return report


def save_report(report: EvalReport, path: str = "evals/last_report.json") -> None:
    """Save evaluation report to a JSON file.
    
    Args:
        report: The evaluation report to save.
        path: Path to save the report.
    """
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    with open(path, "w") as f:
        json.dump(report.model_dump(), f, indent=2)
    
    logger.info(f"Saved eval report to {path}")


def load_report(path: str = "evals/last_report.json") -> Optional[EvalReport]:
    """Load a saved evaluation report.
    
    Args:
        path: Path to the report file.
        
    Returns:
        EvalReport if file exists, None otherwise.
    """
    try:
        with open(path) as f:
            data = json.load(f)
        return EvalReport(**data)
    except FileNotFoundError:
        return None
    except Exception as e:
        logger.error(f"Error loading report: {e}")
        return None