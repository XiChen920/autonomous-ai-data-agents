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
from src.agents.analysis_agent import is_supported_fallback_question
from src.agents.orchestrator import AgentOrchestrator
from src.agents.visualization_agent import DataVisualizationAgent
from src.auth.access_control import AccessControl, AccessControlError
from src.db.connector import DatabaseQueryError
from src.db.registry import DatabaseRegistry, DatabaseRegistryError
from src.db.sql_guard import UnsafeSQLError
from src.observability.analysis_logger import AnalysisLogError, AnalysisLogger
from src.visualization.chart_factory import ChartCreationError


# Defines CLI arguments for running analysis and adding databases.
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Autonomous AI agents for SQLite data analysis and visualization."
    )
    parser.add_argument("--user", help="Configured username, for example alice")
    parser.add_argument("--db", help="Database name, for example chinook")
    parser.add_argument("--question", help="Natural-language analysis question")
    parser.add_argument("--limit", type=int, default=100, help="Maximum rows returned")
    parser.add_argument(
        "--chart",
        default="auto",
        choices=["auto", "bar", "line", "scatter", "table"],
        help="Chart type for the visualization agent",
    )
    parser.add_argument(
        "--mode",
        choices=["online", "offline"],
        help="Analysis mode. Online uses OpenAI; offline uses exact sample-question SQL templates.",
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
    parser.add_argument(
        "--add-database",
        action="store_true",
        help="Add or update a SQLite database entry in config/databases.yaml.",
    )
    parser.add_argument(
        "--db-name",
        help="Logical name for --add-database, for example custom_sales.",
    )
    parser.add_argument(
        "--db-path",
        help="SQLite file path for --add-database, for example data/custom_sales.sqlite.",
    )
    parser.add_argument(
        "--description",
        help="Human-readable database description for --add-database.",
    )
    parser.add_argument(
        "--grant-user",
        action="append",
        default=[],
        help="Grant the added database to a user. Repeat this option for multiple users.",
    )
    parser.add_argument(
        "--feedback-run-id",
        help="Save feedback for a previous analysis run id.",
    )
    parser.add_argument(
        "--feedback-rating",
        choices=["correct", "partially correct", "incorrect"],
        help="Feedback rating for --feedback-run-id.",
    )
    parser.add_argument(
        "--feedback-comment",
        default="",
        help="Optional comment for --feedback-run-id.",
    )
    return parser


# Validates required arguments for the standard analysis run.
def require_analysis_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    missing = [
        flag
        for flag, value in {
            "--user": args.user,
            "--db": args.db,
            "--question": args.question,
        }.items()
        if not value
    ]
    if missing:
        parser.error(f"analysis mode requires: {', '.join(missing)}")


# Validates required arguments for the database integration command.
def require_database_add_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    missing = [
        flag
        for flag, value in {
            "--db-name": args.db_name,
            "--db-path": args.db_path,
            "--description": args.description,
        }.items()
        if not value
    ]
    if missing:
        parser.error(f"--add-database requires: {', '.join(missing)}")


# Validates required arguments for feedback submission.
def require_feedback_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not args.feedback_rating:
        parser.error("--feedback-run-id requires --feedback-rating.")


# Adds or updates a database registry entry and optionally grants user access.
def add_database_from_cli(args: argparse.Namespace) -> int:
    registry = DatabaseRegistry()
    access_control = AccessControl()

    update_result = registry.add_or_update_database(
        database_name=args.db_name,
        database_path=args.db_path,
        description=args.description,
    )

    print(f"Database '{update_result.database_name}' status: {update_result.status}")
    print(f"Path: {update_result.current_database['path']}")
    print(f"Description: {update_result.current_database['description']}")

    if update_result.status == "updated":
        changed = ", ".join(update_result.changed_fields)
        print(f"Changed fields: {changed}")

    if not args.grant_user:
        print("No user permissions changed. Grant access in Streamlit or config/users.yaml.")
        print("New databases have no offline sample questions; use online mode and type a custom question.")
        return 0

    for username in args.grant_user:
        grant_result = access_control.grant_database_to_user(
            username,
            update_result.database_name,
        )
        permissions = ", ".join(grant_result.current_databases)
        print(
            f"User '{grant_result.username}' status: {grant_result.status}. "
            f"Permissions: {permissions}"
        )

    print("New databases have no offline sample questions; use online mode and type a custom question.")
    return 0


# Saves user feedback against a logged analysis run.
def add_feedback_from_cli(args: argparse.Namespace) -> int:
    logger = AnalysisLogger()
    updated_record = logger.record_feedback(
        run_id=args.feedback_run_id,
        rating=args.feedback_rating,
        comment=args.feedback_comment,
        user=args.user or "",
    )
    print(f"Feedback saved for run: {updated_record['run_id']}")
    print(f"Rating: {updated_record['user_feedback']['rating']}")
    return 0


# Chooses online or offline analysis from current and legacy CLI arguments.
def should_use_openai(parser: argparse.ArgumentParser, args: argparse.Namespace) -> bool:
    if args.offline and args.mode == "online":
        parser.error("Use either --offline or --mode online, not both.")

    return not (args.offline or args.mode == "offline")


# Prints the pipeline result in a readable terminal format.
def print_pipeline_result(result, preview_rows: int) -> None:
    analysis = result.analysis

    print(f"User: {result.user}")
    print(f"Database: {result.database}")
    print(f"Question: {analysis.question}")
    print(f"SQL source: {analysis.sql_source}")
    print(f"Run id: {result.run_id}")

    print("\nGenerated SQL:")
    print(analysis.sql)

    print("\nSummary:")
    print(analysis.summary)

    if analysis.retrieved_tables:
        print("\nRetrieved schema context:")
        print(f"Tables: {', '.join(analysis.retrieved_tables)}")
        if analysis.retrieved_columns:
            print(f"Columns: {', '.join(analysis.retrieved_columns[:12])}")

    print(f"\nRows returned: {analysis.row_count}")
    if not analysis.dataframe.empty:
        print("\nPreview:")
        print(analysis.dataframe.head(preview_rows).to_string(index=False))

    if result.visualization is not None:
        print("\nVisualization:")
        print(f"Chart type: {result.visualization.chart_type}")
        print(f"Chart saved to: {result.visualization.chart_path}")
        print(f"Data saved to: {result.visualization.data_path}")


# Runs the CLI workflow and returns a process exit code.
def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.add_database:
        require_database_add_args(parser, args)
        try:
            return add_database_from_cli(args)
        except (AccessControlError, DatabaseRegistryError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

    if args.feedback_run_id:
        require_feedback_args(parser, args)
        try:
            return add_feedback_from_cli(args)
        except AnalysisLogError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

    require_analysis_args(parser, args)
    use_openai = should_use_openai(parser, args)

    if not use_openai and not is_supported_fallback_question(args.db, args.question):
        print(
            "Error: Offline fallback only supports predefined sample questions. "
            "For newly added databases, use --mode online and type a custom question.",
            file=sys.stderr,
        )
        return 1

    analysis_agent = DataAnalysisAgent(
        row_limit=args.limit,
        use_openai=use_openai,
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
