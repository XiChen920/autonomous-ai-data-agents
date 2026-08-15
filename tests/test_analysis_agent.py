"""Tests for Data Analysis Agent behavior and orchestrator access failures."""

import pytest

from src.agents.analysis_agent import (
    DataAnalysisAgent,
    InvalidAnalysisQuestionError,
    SQLGenerationError,
)
from src.agents.orchestrator import AgentOrchestrator
from src.auth.access_control import AccessDeniedError
from src.db.registry import DatabaseRegistry


def test_analysis_agent_runs_injected_sql_against_chinook() -> None:
    registry = DatabaseRegistry()

    def fixed_sql(database_name: str, question: str, schema_text: str) -> str:
        return """
        SELECT BillingCountry AS country, ROUND(SUM(Total), 2) AS total_sales
        FROM invoices
        GROUP BY BillingCountry
        ORDER BY total_sales DESC
        """

    agent = DataAnalysisAgent(
        row_limit=5,
        use_openai=False,
        sql_generator=fixed_sql,
    )

    result = agent.analyze(
        database_path=registry.resolve_path("chinook"),
        database_name="chinook",
        question="Show total sales by country",
    )

    assert result.sql_source == "injected"
    assert result.row_count == 5
    assert list(result.dataframe.columns) == ["country", "total_sales"]
    assert "LIMIT 5" in result.sql


def test_analysis_agent_fallback_supports_common_chinook_question() -> None:
    registry = DatabaseRegistry()
    agent = DataAnalysisAgent(row_limit=3, use_openai=False)

    result = agent.analyze(
        database_path=registry.resolve_path("chinook"),
        database_name="chinook",
        question="Show total sales by country",
    )

    assert result.sql_source == "fallback"
    assert result.row_count == 3
    assert "total_sales" in result.dataframe.columns


def test_analysis_agent_rejects_irrelevant_question() -> None:
    registry = DatabaseRegistry()
    agent = DataAnalysisAgent(row_limit=3, use_openai=False)

    with pytest.raises(InvalidAnalysisQuestionError):
        agent.analyze(
            database_path=registry.resolve_path("chinook"),
            database_name="chinook",
            question="Tell me a joke about pizza",
        )


def test_analysis_agent_offline_rejects_custom_question_without_sample_template() -> None:
    registry = DatabaseRegistry()
    agent = DataAnalysisAgent(row_limit=3, use_openai=False)

    with pytest.raises(SQLGenerationError):
        agent.analyze(
            database_path=registry.resolve_path("chinook"),
            database_name="chinook",
            question="Show total sales by city",
        )


def test_orchestrator_blocks_unauthorized_database_access() -> None:
    orchestrator = AgentOrchestrator(
        analysis_agent=DataAnalysisAgent(row_limit=3, use_openai=False)
    )

    try:
        orchestrator.run_analysis(
            user="bob",
            database="chinook",
            question="Show total sales by country",
        )
    except AccessDeniedError as exc:
        assert "not allowed" in str(exc)
    else:
        raise AssertionError("Expected AccessDeniedError")
