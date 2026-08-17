"""Tests for Data Analysis Agent behavior and orchestrator access failures."""

import pytest

from src.agents.analysis_agent import (
    DataAnalysisAgent,
    InvalidAnalysisQuestionError,
    SQLGenerationError,
    is_supported_fallback_question,
)
from src.agents.orchestrator import AgentOrchestrator
from src.auth.access_control import AccessDeniedError
from src.db.registry import DatabaseRegistry


# Verifies injected SQL can be executed and limited by the agent.
def test_analysis_agent_runs_injected_sql_against_chinook() -> None:
    registry = DatabaseRegistry()

    # Supplies deterministic SQL so the test does not depend on OpenAI.
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


# Verifies a supported offline sample question returns expected data.
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


# Verifies non-analysis questions are rejected before SQL generation.
def test_analysis_agent_rejects_irrelevant_question() -> None:
    registry = DatabaseRegistry()
    agent = DataAnalysisAgent(row_limit=3, use_openai=False)

    with pytest.raises(InvalidAnalysisQuestionError):
        agent.analyze(
            database_path=registry.resolve_path("chinook"),
            database_name="chinook",
            question="Tell me a joke about pizza",
        )


# Verifies offline mode rejects questions without fixed SQL templates.
def test_analysis_agent_offline_rejects_custom_question_without_sample_template() -> None:
    registry = DatabaseRegistry()
    agent = DataAnalysisAgent(row_limit=3, use_openai=False)

    with pytest.raises(SQLGenerationError):
        agent.analyze(
            database_path=registry.resolve_path("chinook"),
            database_name="chinook",
            question="Show total sales by city",
        )


# Verifies offline mode requires exact sample-question wording.
def test_analysis_agent_offline_requires_exact_sample_question() -> None:
    registry = DatabaseRegistry()
    agent = DataAnalysisAgent(row_limit=3, use_openai=False)

    assert is_supported_fallback_question("chinook", "Show total sales by country")
    assert not is_supported_fallback_question("chinook", "Show country total sales")

    with pytest.raises(SQLGenerationError):
        agent.analyze(
            database_path=registry.resolve_path("chinook"),
            database_name="chinook",
            question="Show country total sales",
        )


# Verifies newly added databases do not get offline sample templates automatically.
def test_custom_database_has_no_offline_sample_question_by_default() -> None:
    assert not is_supported_fallback_question(
        "custom_sales",
        "Show total amount by region",
    )


# Verifies the orchestrator blocks unauthorized database access.
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
