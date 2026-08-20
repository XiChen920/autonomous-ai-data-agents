"""Tests for Data Analysis Agent behavior and orchestrator access failures."""

import json
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


class FakeFunctionCall:
    # Mimics a Responses API function_call item.
    def __init__(self, name: str, arguments: dict, call_id: str) -> None:
        self.type = "function_call"
        self.name = name
        self.arguments = json.dumps(arguments)
        self.call_id = call_id


class FakeResponse:
    # Mimics the response fields used by the analysis agent.
    def __init__(self, response_id: str, output=None, output_text: str = "") -> None:
        self.id = response_id
        self.output = output or []
        self.output_text = output_text


class FakeResponsesClient:
    # Returns scripted model responses and records tool outputs sent back by the agent.
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("No fake OpenAI responses left.")
        return self.responses.pop(0)


class FakeOpenAIClient:
    # Provides the .responses.create interface used by OpenAI().
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = FakeResponsesClient(responses)


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


# Verifies the online tool loop can observe a SQL error and repair it.
def test_analysis_agent_openai_tool_loop_repairs_sql_after_execution_error(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    registry = DatabaseRegistry()
    bad_sql = "SELECT MissingColumn FROM invoices"
    fixed_sql = """
        SELECT BillingCountry AS country, ROUND(SUM(Total), 2) AS total_sales
        FROM invoices
        GROUP BY BillingCountry
        ORDER BY total_sales DESC
    """
    fake_client = FakeOpenAIClient(
        [
            FakeResponse(
                "response-1",
                output=[FakeFunctionCall("get_schema", {}, "call-schema")],
            ),
            FakeResponse(
                "response-2",
                output=[FakeFunctionCall("run_sql", {"sql": bad_sql}, "call-bad-sql")],
            ),
            FakeResponse(
                "response-3",
                output=[FakeFunctionCall("run_sql", {"sql": fixed_sql}, "call-fixed-sql")],
            ),
            FakeResponse("response-4", output_text=fixed_sql),
        ]
    )
    agent = DataAnalysisAgent(
        row_limit=5,
        use_openai=True,
        openai_client=fake_client,
    )

    result = agent.analyze(
        database_path=registry.resolve_path("chinook"),
        database_name="chinook",
        question="Show total sales by country",
    )

    tool_outputs = [
        tool_output
        for call in fake_client.responses.calls
        if isinstance(call.get("input"), list)
        for tool_output in call["input"]
    ]
    tool_output_text = "\n".join(item["output"] for item in tool_outputs)

    assert result.sql_source == "openai"
    assert result.row_count == 5
    assert list(result.dataframe.columns) == ["country", "total_sales"]
    assert "LIMIT 5" in result.sql
    assert '"ok": false' in tool_output_text
    assert "MissingColumn" in tool_output_text
    assert '"ok": true' in tool_output_text


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
