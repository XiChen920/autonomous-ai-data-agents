"""Tests for analysis run logging and feedback capture."""

from pathlib import Path

import pytest

from src.agents.analysis_agent import DataAnalysisAgent
from src.agents.orchestrator import AgentOrchestrator
from src.agents.visualization_agent import DataVisualizationAgent
from src.auth.access_control import AccessDeniedError
from src.observability.analysis_logger import AnalysisLogger


# Verifies run logs can be created and later updated with user feedback.
def test_analysis_logger_records_run_and_feedback(tmp_path: Path) -> None:
    logger = AnalysisLogger(tmp_path / "analysis_runs.json")

    run_id = logger.record_run(
        user="alice",
        database="chinook",
        question="Show total sales by country",
        generated_sql="SELECT 1",
        sql_source="test",
        row_count=1,
        chart_type="bar",
        latency_seconds=0.1234,
    )
    updated_record = logger.record_feedback(
        run_id=run_id,
        rating="incorrect",
        comment="The result should use countries, not cities.",
        user="alice",
    )

    records = logger.list_runs()
    assert len(records) == 1
    assert records[0]["run_id"] == run_id
    assert records[0]["latency_seconds"] == 0.123
    assert updated_record["user_feedback"]["rating"] == "incorrect"
    assert "countries" in updated_record["user_feedback"]["comment"]


# Verifies successful pipeline runs save SQL, row count, chart type, and latency.
def test_orchestrator_logs_successful_pipeline_run(tmp_path: Path) -> None:
    logger = AnalysisLogger(tmp_path / "analysis_runs.json")
    orchestrator = AgentOrchestrator(
        analysis_agent=DataAnalysisAgent(row_limit=5, use_openai=False),
        visualization_agent=DataVisualizationAgent(output_dir=tmp_path / "charts"),
        analysis_logger=logger,
    )

    result = orchestrator.run_pipeline(
        user="alice",
        database="chinook",
        question="Show total sales by country",
        chart_type="bar",
    )

    records = logger.list_runs()
    assert result.run_id == records[0]["run_id"]
    assert records[0]["status"] == "success"
    assert records[0]["database"] == "chinook"
    assert records[0]["generated_sql"]
    assert records[0]["sql_source"] == "fallback"
    assert records[0]["row_count"] == 5
    assert records[0]["chart_type"] == "bar"
    assert records[0]["latency_seconds"] >= 0


# Verifies failed runs are logged with an error message for root-cause tracing.
def test_orchestrator_logs_failed_access_check(tmp_path: Path) -> None:
    logger = AnalysisLogger(tmp_path / "analysis_runs.json")
    orchestrator = AgentOrchestrator(
        analysis_agent=DataAnalysisAgent(row_limit=5, use_openai=False),
        analysis_logger=logger,
    )

    with pytest.raises(AccessDeniedError):
        orchestrator.run_analysis(
            user="bob",
            database="chinook",
            question="Show total sales by country",
        )

    records = logger.list_runs()
    assert records[0]["status"] == "failed"
    assert records[0]["user"] == "bob"
    assert records[0]["database"] == "chinook"
    assert "not allowed" in records[0]["error_message"]
