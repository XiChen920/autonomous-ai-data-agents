"""Data Visualization Agent.

This agent receives the analysis DataFrame, delegates chart creation to the
chart factory, and saves both the PNG chart and CSV data output.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from src.agents.analysis_agent import AnalysisResult
from src.utils.config_loader import PROJECT_ROOT
from src.visualization.chart_factory import ChartFactory


@dataclass
class VisualizationResult:
    chart_type: str
    chart_path: Path
    data_path: Path


class DataVisualizationAgent:
    # Configures chart creation and output storage.
    def __init__(
        self,
        chart_factory: ChartFactory | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        self.chart_factory = chart_factory or ChartFactory()
        self.output_dir = Path(output_dir) if output_dir else PROJECT_ROOT / "outputs" / "charts"

    # Turns an analysis DataFrame into a styled chart and CSV output.
    def visualize(
        self,
        analysis_result: AnalysisResult,
        chart_type: str = "auto",
    ) -> VisualizationResult:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        base_name = self._build_filename(
            analysis_result.database_name,
            analysis_result.question,
        )
        chart_path = self.output_dir / f"{base_name}.png"
        data_path = self.output_dir / f"{base_name}.csv"
        title = self._build_title(analysis_result.question)

        selected_chart_type, saved_chart_path = self.chart_factory.create_chart(
            dataframe=analysis_result.dataframe,
            output_path=chart_path,
            title=title,
            chart_type=chart_type,
        )
        analysis_result.dataframe.to_csv(data_path, index=False)

        return VisualizationResult(
            chart_type=selected_chart_type,
            chart_path=saved_chart_path,
            data_path=data_path,
        )

    # Builds a safe filename from database name and question text.
    def _build_filename(self, database_name: str, question: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", question.lower()).strip("-")
        slug = slug[:60].strip("-") or "analysis"
        return f"{database_name}-{slug}"

    # Shortens long chart titles so they fit the saved figure.
    def _build_title(self, question: str) -> str:
        question = question.strip()
        if len(question) <= 90:
            return question
        return question[:87].rstrip() + "..."
