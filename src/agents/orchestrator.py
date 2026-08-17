"""Coordinates access control, database lookup, analysis, and visualization agents."""

from dataclasses import dataclass

from src.agents.analysis_agent import AnalysisResult, DataAnalysisAgent
from src.agents.visualization_agent import DataVisualizationAgent, VisualizationResult
from src.auth.access_control import AccessControl
from src.db.registry import DatabaseRegistry


@dataclass
class PipelineResult:
    user: str
    database: str
    analysis: AnalysisResult
    visualization: VisualizationResult | None = None


class AgentOrchestrator:
    # Wires access control, database lookup, analysis, and visualization services.
    def __init__(
            self,
            access_control: AccessControl | None = None,
            database_registry: DatabaseRegistry | None = None,
            analysis_agent: DataAnalysisAgent | None = None,
            visualization_agent: DataVisualizationAgent | None = None,
    ) -> None:
        self.access_control = access_control or AccessControl()
        self.database_registry = database_registry or DatabaseRegistry()
        self.analysis_agent = analysis_agent or DataAnalysisAgent()
        self.visualization_agent = visualization_agent or DataVisualizationAgent()

    # Runs only the access-controlled data analysis stage.
    def run_analysis(self, user: str, database: str, question: str) -> PipelineResult:
        self.access_control.require_access(user, database)  # check user access
        database_info = self.database_registry.get_database(database)
        analysis_result = self.analysis_agent.analyze(
            database_path=database_info["path"],
            database_name=database,
            question=question,
        )

        return PipelineResult(
            user=user,
            database=database,
            analysis=analysis_result,
        )

    # Runs analysis and then passes the result into the visualization agent.
    def run_pipeline(
            self,
            user: str,
            database: str,
            question: str,
            chart_type: str = "auto",
    ) -> PipelineResult:
        pipeline_result = self.run_analysis(user, database, question)
        visualization_result = self.visualization_agent.visualize(
            pipeline_result.analysis,
            chart_type=chart_type,
        )

        return PipelineResult(
            user=pipeline_result.user,
            database=pipeline_result.database,
            analysis=pipeline_result.analysis,
            visualization=visualization_result,
        )
