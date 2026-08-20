"""Starter evaluation harness for SQL generation quality.

The normal test suite validates that the pipeline works, but it does not
measure OpenAI-generated SQL accuracy at scale. This module makes that
limitation explicit and provides a small, extendable benchmark runner.
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from src.agents.analysis_agent import (
    AnalysisAgentError,
    AnalysisResult,
    DataAnalysisAgent,
    InvalidAnalysisQuestionError,
    SQLGenerationError,
)
from src.db.connector import DatabaseQueryError, SQLiteConnector
from src.db.registry import DatabaseRegistry
from src.db.sql_guard import UnsafeSQLError, ensure_limit
from src.utils.config_loader import load_yaml


DEFAULT_CASES_PATH = Path(__file__).resolve().with_name("sql_generation_cases.yaml")
VALID_MODES = {"offline", "online"}
LLM_SQL_ACCURACY_LIMITATION = (
    "Current automated tests validate the pipeline and starter benchmark cases, "
    "but they are not a large-scale accuracy evaluation for OpenAI-generated SQL."
)
IDENTIFIER_SYNONYMS = {
    "avg": "average",
    "cnt": "count",
    "num": "number",
    "qty": "quantity",
}


class SQLGenerationEvaluationError(RuntimeError):
    """Raised when SQL-generation evaluation cases or options are invalid."""


@dataclass(frozen=True)
class SQLGenerationEvalCase:
    """One golden case for checking generated SQL result shape."""

    id: str
    database: str
    question: str
    expected_columns: tuple[str, ...]
    min_rows: int = 1
    expected_sql_contains: tuple[str, ...] = ()
    golden_sql: str = ""
    golden_result: tuple[dict[str, Any], ...] = ()
    numeric_tolerance: float = 0.01
    online_only: bool = False


@dataclass(frozen=True)
class SQLGenerationEvalResult:
    """Result for one SQL-generation evaluation case."""

    case_id: str
    database: str
    question: str
    status: str
    sql_source: str = ""
    rows: int = 0
    failures: tuple[str, ...] = ()
    error: str = ""
    generated_sql: str = ""
    execution_success: bool = False
    result_accuracy_success: bool = False
    invalid_sql: bool = False
    repair_attempted: bool = False
    repair_succeeded: bool = False

    @property
    def passed(self) -> bool:
        """Return whether the case passed all checks."""

        return self.status == "passed"

    @property
    def skipped(self) -> bool:
        """Return whether the case was intentionally skipped."""

        return self.status == "skipped"


# Builds the command-line parser for this evaluator.
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run starter SQL-generation evaluation cases."
    )
    parser.add_argument(
        "--cases",
        default=str(DEFAULT_CASES_PATH),
        help="Path to a YAML file containing SQL-generation eval cases.",
    )
    parser.add_argument(
        "--mode",
        choices=sorted(VALID_MODES),
        default="offline",
        help="offline uses fixed sample-question SQL; online calls OpenAI.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum rows returned by each generated SQL query.",
    )
    return parser


# Converts a YAML value into a tuple of strings.
def _string_tuple(value: object, field_name: str, case_id: str) -> tuple[str, ...]:
    if value is None:
        return ()

    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise SQLGenerationEvaluationError(
            f"Case '{case_id}' field '{field_name}' must be a list of strings."
        )

    return tuple(item.strip() for item in value if item.strip())


# Converts a YAML golden_result value into immutable row dictionaries.
def _golden_result_rows(value: object, case_id: str) -> tuple[dict[str, Any], ...]:
    if value is None:
        return ()

    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise SQLGenerationEvaluationError(
            f"Case '{case_id}' field 'golden_result' must be a list of row objects."
        )

    return tuple(dict(item) for item in value)


# Normalizes SQL/result identifiers so reasonable LLM aliases can still match.
def normalize_identifier(identifier: object) -> str:
    text = str(identifier).strip()
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    raw_tokens = re.findall(r"[A-Za-z0-9]+", text.lower())
    tokens = [IDENTIFIER_SYNONYMS.get(token, token) for token in raw_tokens]
    if len(tokens) > 1 and tokens[-1] in {"label", "name"}:
        tokens = tokens[:-1]
    return "".join(tokens)


# Returns whether two values should be compared as numbers.
def _is_number_like(value: object) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return not pd.isna(value)


# Compares two scalar values exactly or with numeric tolerance.
def _values_match(expected: object, actual: object, numeric_tolerance: float) -> bool:
    if pd.isna(expected) and pd.isna(actual):
        return True

    if _is_number_like(expected) and _is_number_like(actual):
        return abs(float(expected) - float(actual)) <= numeric_tolerance

    return str(expected).strip() == str(actual).strip()


# Aligns actual result columns to the golden result using normalized aliases.
def _map_expected_columns(
    expected_columns: list[str],
    actual_columns: list[str],
) -> dict[str, str]:
    actual_by_normalized = {
        normalize_identifier(column): column
        for column in actual_columns
    }
    return {
        expected_column: actual_by_normalized[normalize_identifier(expected_column)]
        for expected_column in expected_columns
        if normalize_identifier(expected_column) in actual_by_normalized
    }


# Compares an actual result DataFrame against a golden DataFrame.
def compare_result_dataframes(
    expected: pd.DataFrame,
    actual: pd.DataFrame,
    numeric_tolerance: float = 0.01,
) -> tuple[str, ...]:
    failures: list[str] = []
    expected_columns = [str(column) for column in expected.columns]
    actual_columns = [str(column) for column in actual.columns]
    column_map = _map_expected_columns(expected_columns, actual_columns)

    missing_columns = [
        column
        for column in expected_columns
        if column not in column_map
    ]
    if missing_columns:
        failures.append(
            f"Golden result columns missing from actual result: {missing_columns}."
        )
        return tuple(failures)

    if len(expected) != len(actual):
        failures.append(
            f"Golden result row count mismatch: expected {len(expected)}, got {len(actual)}."
        )

    rows_to_compare = min(len(expected), len(actual))
    for row_index in range(rows_to_compare):
        for expected_column, actual_column in column_map.items():
            expected_value = expected.iloc[row_index][expected_column]
            actual_value = actual.iloc[row_index][actual_column]
            if not _values_match(expected_value, actual_value, numeric_tolerance):
                failures.append(
                    "Golden result value mismatch at "
                    f"row {row_index + 1}, column '{expected_column}': "
                    f"expected {expected_value!r}, got {actual_value!r}."
                )
                if len(failures) >= 5:
                    failures.append("Additional golden result mismatches omitted.")
                    return tuple(failures)

    return tuple(failures)


# Builds a golden DataFrame from inline rows or golden SQL.
def build_golden_dataframe(
    case: SQLGenerationEvalCase,
    registry: DatabaseRegistry,
    row_limit: int,
) -> pd.DataFrame | None:
    if case.golden_result:
        return pd.DataFrame(list(case.golden_result))

    if not case.golden_sql:
        return None

    connector = SQLiteConnector()
    golden_sql = ensure_limit(case.golden_sql, limit=row_limit)
    return connector.run_query(registry.resolve_path(case.database), golden_sql)


# Loads golden SQL-generation cases from YAML.
def load_eval_cases(cases_path: str | Path = DEFAULT_CASES_PATH) -> list[SQLGenerationEvalCase]:
    data = load_yaml(cases_path)
    raw_cases = data.get("cases")

    if not isinstance(raw_cases, list) or not raw_cases:
        raise SQLGenerationEvaluationError("SQL-generation eval file must contain cases.")

    cases: list[SQLGenerationEvalCase] = []
    for index, raw_case in enumerate(raw_cases, start=1):
        if not isinstance(raw_case, dict):
            raise SQLGenerationEvaluationError(f"Case #{index} must be a YAML object.")

        case_id = str(raw_case.get("id", "")).strip()
        database = str(raw_case.get("database", "")).strip()
        question = str(raw_case.get("question", "")).strip()
        if not case_id or not database or not question:
            raise SQLGenerationEvaluationError(
                f"Case #{index} must include id, database, and question."
            )

        min_rows = int(raw_case.get("min_rows", 1))
        if min_rows < 0:
            raise SQLGenerationEvaluationError(f"Case '{case_id}' min_rows cannot be negative.")

        numeric_tolerance = float(raw_case.get("numeric_tolerance", 0.01))
        if numeric_tolerance < 0:
            raise SQLGenerationEvaluationError(
                f"Case '{case_id}' numeric_tolerance cannot be negative."
            )

        cases.append(
            SQLGenerationEvalCase(
                id=case_id,
                database=database,
                question=question,
                expected_columns=_string_tuple(
                    raw_case.get("expected_columns"),
                    "expected_columns",
                    case_id,
                ),
                min_rows=min_rows,
                expected_sql_contains=_string_tuple(
                    raw_case.get("expected_sql_contains"),
                    "expected_sql_contains",
                    case_id,
                ),
                golden_sql=str(raw_case.get("golden_sql", "")).strip(),
                golden_result=_golden_result_rows(raw_case.get("golden_result"), case_id),
                numeric_tolerance=numeric_tolerance,
                online_only=bool(raw_case.get("online_only", False)),
            )
        )

    return cases


# Checks one AnalysisResult against the golden expectations for a case.
def evaluate_analysis_result(
    case: SQLGenerationEvalCase,
    result: AnalysisResult,
    golden_dataframe: pd.DataFrame | None = None,
) -> tuple[str, ...]:
    failures: list[str] = []

    if result.row_count < case.min_rows:
        failures.append(
            f"Expected at least {case.min_rows} rows, got {result.row_count}."
        )

    actual_columns = {
        normalize_identifier(column): str(column)
        for column in result.dataframe.columns
    }
    for expected_column in case.expected_columns:
        if normalize_identifier(expected_column) not in actual_columns:
            failures.append(
                f"Expected column '{expected_column}' not found. "
                f"Actual columns: {list(result.dataframe.columns)}."
            )

    generated_sql = result.sql.lower()
    for fragment in case.expected_sql_contains:
        if fragment.lower() not in generated_sql:
            failures.append(f"Expected SQL fragment '{fragment}' not found.")

    if golden_dataframe is not None:
        failures.extend(
            compare_result_dataframes(
                expected=golden_dataframe,
                actual=result.dataframe,
                numeric_tolerance=case.numeric_tolerance,
            )
        )

    return tuple(failures)


# Runs one eval case through the Data Analysis Agent.
def run_eval_case(
    case: SQLGenerationEvalCase,
    mode: str,
    row_limit: int,
    registry: DatabaseRegistry | None = None,
) -> SQLGenerationEvalResult:
    if mode == "offline" and case.online_only:
        return SQLGenerationEvalResult(
            case_id=case.id,
            database=case.database,
            question=case.question,
            status="skipped",
            error="Case requires OpenAI SQL generation; offline mode skipped it.",
        )

    registry = registry or DatabaseRegistry()
    agent = DataAnalysisAgent(row_limit=row_limit, use_openai=(mode == "online"))
    database_info = registry.get_database(case.database)

    try:
        golden_dataframe = build_golden_dataframe(case, registry, row_limit)
        result = agent.analyze(
            database_path=database_info["path"],
            database_name=case.database,
            question=case.question,
            database_description=str(database_info.get("description", "")),
        )
    except (AnalysisAgentError, DatabaseQueryError, UnsafeSQLError) as error:
        metrics = agent._sql_generation_metrics
        return SQLGenerationEvalResult(
            case_id=case.id,
            database=case.database,
            question=case.question,
            status="failed",
            error=f"{type(error).__name__}: {error}",
            invalid_sql=isinstance(error, (SQLGenerationError, DatabaseQueryError, UnsafeSQLError))
            and not isinstance(error, InvalidAnalysisQuestionError),
            repair_attempted=metrics.sql_repair_count > 0,
            repair_succeeded=False,
        )
    except Exception as error:
        metrics = agent._sql_generation_metrics
        return SQLGenerationEvalResult(
            case_id=case.id,
            database=case.database,
            question=case.question,
            status="failed",
            error=f"Unexpected {type(error).__name__}: {error}",
            repair_attempted=metrics.sql_repair_count > 0,
            repair_succeeded=False,
        )

    failures = list(evaluate_analysis_result(case, result, golden_dataframe))
    if mode == "online" and result.sql_source != "openai":
        failures.append(
            f"Expected OpenAI-generated SQL in online eval, got source '{result.sql_source}'."
        )

    return SQLGenerationEvalResult(
        case_id=case.id,
        database=case.database,
        question=case.question,
        status="failed" if failures else "passed",
        sql_source=result.sql_source,
        rows=result.row_count,
        failures=tuple(failures),
        generated_sql=result.sql,
        execution_success=True,
        result_accuracy_success=not failures,
        invalid_sql=False,
        repair_attempted=result.sql_repair_count > 0,
        repair_succeeded=result.sql_repair_succeeded,
    )


# Runs all cases and returns structured per-case results.
def run_evaluation(
    cases_path: str | Path = DEFAULT_CASES_PATH,
    mode: str = "offline",
    row_limit: int = 20,
    registry: DatabaseRegistry | None = None,
) -> list[SQLGenerationEvalResult]:
    if mode not in VALID_MODES:
        raise SQLGenerationEvaluationError(f"Mode must be one of: {sorted(VALID_MODES)}")

    if row_limit <= 0:
        raise SQLGenerationEvaluationError("Row limit must be positive.")

    load_dotenv()
    if mode == "online" and not os.getenv("OPENAI_API_KEY"):
        raise SQLGenerationEvaluationError(
            "OPENAI_API_KEY is required for online SQL-generation evaluation."
        )

    cases = load_eval_cases(cases_path)
    return [
        run_eval_case(case, mode=mode, row_limit=row_limit, registry=registry)
        for case in cases
    ]


# Prints a compact report for command-line use.
def print_eval_report(results: list[SQLGenerationEvalResult], mode: str) -> None:
    passed = sum(result.passed for result in results)
    skipped = sum(result.skipped for result in results)
    failed = len(results) - passed - skipped
    evaluated = [result for result in results if not result.skipped]
    evaluated_count = len(evaluated)
    execution_successes = sum(result.execution_success for result in evaluated)
    result_accuracy_successes = sum(result.result_accuracy_success for result in evaluated)
    invalid_sql_count = sum(result.invalid_sql for result in evaluated)
    repair_attempts = [result for result in evaluated if result.repair_attempted]
    repair_successes = sum(result.repair_succeeded for result in repair_attempts)

    print(LLM_SQL_ACCURACY_LIMITATION)
    print()
    print(f"SQL generation eval mode: {mode}")
    print(f"Cases: {len(results)} | Passed: {passed} | Failed: {failed} | Skipped: {skipped}")
    if evaluated_count:
        print(f"Execution accuracy: {execution_successes}/{evaluated_count} ({execution_successes / evaluated_count:.0%})")
        print(f"Result accuracy: {result_accuracy_successes}/{evaluated_count} ({result_accuracy_successes / evaluated_count:.0%})")
        print(f"Invalid SQL rate: {invalid_sql_count}/{evaluated_count} ({invalid_sql_count / evaluated_count:.0%})")
    if repair_attempts:
        print(f"Repair success rate: {repair_successes}/{len(repair_attempts)} ({repair_successes / len(repair_attempts):.0%})")
    else:
        print("Repair success rate: N/A (no repair attempts observed)")
    print()

    for result in results:
        label = result.status.upper()
        source = f" source={result.sql_source}" if result.sql_source else ""
        rows = f" rows={result.rows}" if result.rows else ""
        print(f"- [{label}] {result.case_id} db={result.database}{source}{rows}")
        if result.repair_attempted:
            print(f"  repair: {'succeeded' if result.repair_succeeded else 'failed'}")

        for failure in result.failures:
            print(f"  issue: {failure}")

        if result.error:
            print(f"  note: {result.error}")

        if result.generated_sql and result.status == "failed":
            one_line_sql = " ".join(result.generated_sql.split())
            print(f"  sql: {one_line_sql[:240]}")


# CLI entry point for the SQL-generation evaluator.
def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        results = run_evaluation(
            cases_path=args.cases,
            mode=args.mode,
            row_limit=args.limit,
        )
    except SQLGenerationEvaluationError as error:
        print(f"SQL generation evaluation failed: {error}")
        return 1

    print_eval_report(results, mode=args.mode)
    return 0 if all(result.status != "failed" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
