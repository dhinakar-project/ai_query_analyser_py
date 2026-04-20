"""CLI for running evaluations."""

import asyncio
import sys
from typing import Optional, List

from rich.console import Console
from rich.table import Table

from evals.golden_dataset import QueryTestCase, get_golden_dataset
from evals.runner import run_evals, save_report, load_report


console = Console()


def filter_by_category(
    dataset: List[QueryTestCase],
    category: str
) -> List[QueryTestCase]:
    """Filter test cases by category.
    
    Args:
        dataset: Full dataset.
        category: Category to filter by.
        
    Returns:
        Filtered dataset.
    """
    return [tc for tc in dataset if tc.expected_category == category]


def print_report(report, verbose: bool = False) -> None:
    """Print the evaluation report using Rich.
    
    Args:
        report: The evaluation report.
        verbose: Whether to show per-case details.
    """
    console.print("\n[bold cyan]Evaluation Results[/bold cyan]\n")
    
    metrics_table = Table(show_header=False, box=None)
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Value", style="green")
    
    metrics_table.add_row("Total Cases", str(report.total_cases))
    metrics_table.add_row("Category Accuracy", f"{report.category_accuracy:.1%}")
    metrics_table.add_row("Sentiment Accuracy", f"{report.sentiment_accuracy:.1%}")
    metrics_table.add_row("Priority Accuracy", f"{report.priority_accuracy:.1%}")
    metrics_table.add_row("Escalation Precision", f"{report.escalation_precision:.1%}")
    metrics_table.add_row("Escalation Recall", f"{report.escalation_recall:.1%}")
    metrics_table.add_row("Avg Category Confidence", f"{report.avg_confidence_category:.1f}")
    metrics_table.add_row("Avg Sentiment Confidence", f"{report.avg_confidence_sentiment:.1f}")
    metrics_table.add_row("Avg Latency", f"{report.avg_latency_ms:.0f}ms")
    
    console.print(metrics_table)
    console.print(f"\n[dim]Evaluated at: {report.timestamp}[/dim]\n")
    
    if verbose:
        console.print("\n[bold cyan]Per-Case Results[/bold cyan]\n")
        
        case_table = Table()
        case_table.add_column("Query", style="cyan", width=30)
        case_table.add_column("Category", justify="center")
        case_table.add_column("Sentiment", justify="center")
        case_table.add_column("Priority", justify="center")
        case_table.add_column("Escalate", justify="center")
        
        for result in report.case_results:
            cat_icon = "[green]✓[/green]" if result.category_match else "[red]✗[/red]"
            sent_icon = "[green]✓[/green]" if result.sentiment_match else "[red]✗[/red]"
            prio_icon = "[green]✓[/green]" if result.priority_match else "[red]✗[/red]"
            esc_icon = "[green]✓[/green]" if result.escalate_match else "[red]✗[/red]"
            
            case_table.add_row(
                result.query[:30] + "...",
                f"{cat_icon} {result.actual_category}",
                f"{sent_icon} {result.actual_sentiment}",
                f"{prio_icon} {result.actual_priority}",
                f"{esc_icon} {result.actual_escalate}"
            )
        
        console.print(case_table)


async def main() -> None:
    """Main CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run evaluation on query analyzer")
    parser.add_argument(
        "--category",
        type=str,
        help="Filter by category (e.g., Billing, Technical Support)"
    )
    parser.add_argument(
        "--fail-below",
        type=float,
        default=0.0,
        help="Exit with code 1 if accuracy is below threshold (e.g., 0.8)"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show per-case results"
    )
    parser.add_argument(
        "--load",
        type=str,
        help="Load and display a saved report"
    )
    
    args = parser.parse_args()
    
    if args.load:
        report = load_report(args.load)
        if report:
            print_report(report, args.verbose)
        else:
            console.print(f"[red]Report not found: {args.load}[/red]")
            sys.exit(1)
        return
    
    dataset = get_golden_dataset()
    
    if args.category:
        dataset = filter_by_category(dataset, args.category)
        if not dataset:
            console.print(f"[red]No test cases found for category: {args.category}[/red]")
            sys.exit(1)
        console.print(f"[cyan]Running {len(dataset)} test cases for {args.category}...\n")
    
    report = await run_evals(dataset, verbose=args.verbose)
    
    save_report(report)
    
    print_report(report, args.verbose)
    
    if args.fail_below > 0:
        min_accuracy = min(
            report.category_accuracy,
            report.sentiment_accuracy,
            report.priority_accuracy
        )
        if min_accuracy < args.fail_below:
            console.print(f"\n[red]FAIL: Accuracy {min_accuracy:.1%} below threshold {args.fail_below:.1%}[/red]")
            sys.exit(1)
        else:
            console.print(f"\n[green]PASS: All metrics above threshold[/green]")


if __name__ == "__main__":
    asyncio.run(main())