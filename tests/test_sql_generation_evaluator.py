"""Tests for the starter SQL-generation evaluation harness."""

import pandas as pd

from src.agents.analysis_agent import AnalysisResult
from src.tools.sql_generation_evaluator import (
    DEFAULT_CASES_PATH,
    LLM_SQL_ACCURACY_LIMITATION,
    SQLGenerationEvalCase,
    evaluate_analysis_result,
    load_eval_cases,
    normalize_identifier,
    run_evaluation,
)


# Verifies the benchmark YAML loads and documents the current eval limitation.
def test_sql_generation_eval_cases_load_seed_benchmark() -> None:
    cases = load_eval_cases(DEFAULT_CASES_PATH)

    assert len(cases) >= 4
    assert any(case.online_only for case in cases)
    assert "not a large-scale accuracy evaluation" in LLM_SQL_ACCURACY_LIMITATION


# Verifies result-shape checks catch the expected columns, row count, and SQL fragments.
def test_sql_generation_eval_checks_expected_result_shape() -> None:
    case = SQLGenerationEvalCase(
        id="demo_case",
        database="chinook",
        question="Show total sales by country",
        expected_columns=("country", "total_sales"),
        min_rows=1,
        expected_sql_contains=("invoices", "SUM"),
    )
    result = AnalysisResult(
        database_name="chinook",
        question=case.question,
        sql="SELECT BillingCountry AS country, SUM(Total) AS total_sales FROM invoices",
        summary="Returned one row.",
        dataframe=pd.DataFrame([{"country": "USA", "total_sales": 523.06}]),
        sql_source="injected",
    )

    assert evaluate_analysis_result(case, result) == ()


# Verifies reasonable LLM aliases match the expected benchmark column names.
def test_sql_generation_eval_accepts_normalized_llm_column_aliases() -> None:
    case = SQLGenerationEvalCase(
        id="alias_case",
        database="chinook",
        question="Which countries have the highest average invoice value?",
        expected_columns=("country", "average_invoice_value", "total_sales"),
        min_rows=1,
        expected_sql_contains=("invoices", "AVG"),
    )
    result = AnalysisResult(
        database_name="chinook",
        question=case.question,
        sql=(
            "SELECT BillingCountry AS Country, AVG(Total) AS AvgInvoiceValue, "
            "SUM(Total) AS TotalSales FROM invoices"
        ),
        summary="Returned one row.",
        dataframe=pd.DataFrame(
            [{"Country": "USA", "AvgInvoiceValue": 5.65, "TotalSales": 523.06}]
        ),
        sql_source="openai",
    )

    assert normalize_identifier("TotalSales") == normalize_identifier("total_sales")
    assert normalize_identifier("AvgInvoiceValue") == normalize_identifier(
        "average_invoice_value"
    )
    assert evaluate_analysis_result(case, result) == ()


# Verifies offline evaluation runs deterministic sample cases without OpenAI.
def test_sql_generation_eval_offline_runs_supported_cases_and_skips_online_only() -> None:
    results = run_evaluation(DEFAULT_CASES_PATH, mode="offline", row_limit=5)
    failed_results = [result for result in results if result.status == "failed"]

    assert failed_results == []
    assert any(result.status == "passed" for result in results)
    assert any(result.status == "skipped" for result in results)
