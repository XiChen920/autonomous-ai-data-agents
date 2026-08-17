"""Tests for Data Visualization Agent chart generation and pipeline integration."""

from pathlib import Path

import pandas as pd

from src.agents.analysis_agent import DataAnalysisAgent, AnalysisResult
from src.agents.orchestrator import AgentOrchestrator
from src.agents.visualization_agent import DataVisualizationAgent


# Verifies category-and-number data becomes a styled bar chart.
def test_visualization_agent_creates_company_style_bar_chart(tmp_path: Path) -> None:
    dataframe = pd.DataFrame(
        {
            "country": ["USA", "Canada", "France"],
            "total_sales": [523.06, 303.96, 195.10],
        }
    )
    analysis_result = AnalysisResult(
        database_name="chinook",
        question="Show total sales by country",
        sql="SELECT ...",
        summary="Example analysis",
        dataframe=dataframe,
        sql_source="test",
    )
    agent = DataVisualizationAgent(output_dir=tmp_path)

    result = agent.visualize(analysis_result, chart_type="auto")

    assert result.chart_type == "bar"
    assert result.chart_path.exists()
    assert result.chart_path.stat().st_size > 0
    assert result.data_path.exists()


# Verifies empty data still produces a safe table image.
def test_visualization_agent_creates_table_for_empty_dataframe(tmp_path: Path) -> None:
    analysis_result = AnalysisResult(
        database_name="chinook",
        question="Empty query",
        sql="SELECT ...",
        summary="No data",
        dataframe=pd.DataFrame(),
        sql_source="test",
    )
    agent = DataVisualizationAgent(output_dir=tmp_path)

    result = agent.visualize(analysis_result)

    assert result.chart_type == "table"
    assert result.chart_path.exists()
    assert result.chart_path.stat().st_size > 0


# Verifies forced line/scatter charts do not crash with one numeric column.
def test_visualization_agent_handles_forced_line_and_scatter_with_one_numeric_column(
    tmp_path: Path,
) -> None:
    dataframe = pd.DataFrame(
        {
            "country": ["USA", "Canada", "France"],
            "total_sales": [523.06, 303.96, 195.10],
        }
    )
    analysis_result = AnalysisResult(
        database_name="chinook",
        question="Show total sales by country",
        sql="SELECT ...",
        summary="Example analysis",
        dataframe=dataframe,
        sql_source="test",
    )
    agent = DataVisualizationAgent(output_dir=tmp_path)

    line_result = agent.visualize(analysis_result, chart_type="line")
    scatter_result = agent.visualize(analysis_result, chart_type="scatter")

    assert line_result.chart_type == "line"
    assert line_result.chart_path.exists()
    assert line_result.chart_path.stat().st_size > 0
    assert scatter_result.chart_type == "scatter"
    assert scatter_result.chart_path.exists()
    assert scatter_result.chart_path.stat().st_size > 0


# Verifies the full two-agent pipeline returns both data and a chart.
def test_orchestrator_runs_analysis_and_visualization_pipeline(tmp_path: Path) -> None:
    orchestrator = AgentOrchestrator(
        analysis_agent=DataAnalysisAgent(row_limit=5, use_openai=False),
        visualization_agent=DataVisualizationAgent(output_dir=tmp_path),
    )

    result = orchestrator.run_pipeline(
        user="alice",
        database="chinook",
        question="Show total sales by country",
    )

    assert result.analysis.row_count == 5
    assert result.visualization is not None
    assert result.visualization.chart_type == "bar"
    assert result.visualization.chart_path.exists()
