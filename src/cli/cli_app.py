"""Command-line interface implementation for the two-agent pipeline.

This module contains the real CLI logic. The root-level ``cli_app.py`` file is
only a small launcher so users can run the project from the repository root.
"""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.analysis_agent import AnalysisAgentError, DataAnalysisAgent
from src.agents.orchestrator import AgentOrchestrator
from src.agents.visualization_agent import DataVisualizationAgent
from src.auth.access_control import AccessControlError
from src.db.connector import DatabaseQueryError
from src.db.registry import DatabaseRegistryError
from src.db.sql_guard import UnsafeSQLError
from src.visualization.chart_factory import ChartCreationError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Autonomous AI agents for SQLite data analysis and visualization."
    )
    parser.add_argument("--user", required=True, help="Configured username, for example alice")
    parser.add_argument("--db", required=True, help="Database name, for example chinook")
    parser.add_argument("--question", required=True, help="Natural-language analysis question")
    parser.add_argument("--limit", type=int, default=100, help="Maximum rows returned")
    parser.add_argument(
        "--chart",
        default="auto",
        choices=["auto", "bar", "line", "scatter", "table"],
        help="Chart type for the visualization agent",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use built-in sample-question SQL templates instead of the OpenAI API",
    )
    parser.add_argument(
        "--preview-rows",
        type=int,
        default=10,
        help="Number of result rows to print in the terminal",
    )
    return parser


def print_pipeline_result(result, preview_rows: int) -> None:
    analysis = result.analysis

    print(f"User: {result.user}")
    print(f"Database: {result.database}")
    print(f"Question: {analysis.question}")
    print(f"SQL source: {analysis.sql_source}")

    print("\nGenerated SQL:")
    print(analysis.sql)

    print("\nSummary:")
    print(analysis.summary)

    print(f"\nRows returned: {analysis.row_count}")
    if not analysis.dataframe.empty:
        print("\nPreview:")
        print(analysis.dataframe.head(preview_rows).to_string(index=False))

    if result.visualization is not None:
        print("\nVisualization:")
        print(f"Chart type: {result.visualization.chart_type}")
        print(f"Chart saved to: {result.visualization.chart_path}")
        print(f"Data saved to: {result.visualization.data_path}")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    analysis_agent = DataAnalysisAgent(
        row_limit=args.limit,
        use_openai=not args.offline,
    )
    visualization_agent = DataVisualizationAgent()
    orchestrator = AgentOrchestrator(
        analysis_agent=analysis_agent,
        visualization_agent=visualization_agent,
    )

    try:
        result = orchestrator.run_pipeline(
            user=args.user,
            database=args.db,
            question=args.question,
            chart_type=args.chart,
        )
    except (
        AccessControlError,
        AnalysisAgentError,
        DatabaseRegistryError,
        DatabaseQueryError,
        UnsafeSQLError,
        ChartCreationError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print_pipeline_result(result, preview_rows=args.preview_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

