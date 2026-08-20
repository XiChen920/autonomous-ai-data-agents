"""Coordinates access control, database lookup, analysis, and visualization agents."""

from dataclasses import dataclass
from time import perf_counter

from src.agents.analysis_agent import AnalysisResult, DataAnalysisAgent
from src.agents.visualization_agent import DataVisualizationAgent, VisualizationResult
from src.auth.access_control import AccessControl
from src.db.registry import DatabaseRegistry
from src.observability.analysis_logger import AnalysisLogger


@dataclass
class PipelineResult:
    user: str
    database: str
    analysis: AnalysisResult
    visualization: VisualizationResult | None = None
    run_id: str = ""


class AgentOrchestrator:
    # Wires access control, database lookup, analysis, and visualization services.
    def __init__(
            self,
            access_control: AccessControl | None = None,
            database_registry: DatabaseRegistry | None = None,
            analysis_agent: DataAnalysisAgent | None = None,
            visualization_agent: DataVisualizationAgent | None = None,
            analysis_logger: AnalysisLogger | None = None,
    ) -> None:
        self.access_control = access_control or AccessControl()
        self.database_registry = database_registry or DatabaseRegistry()
        self.analysis_agent = analysis_agent or DataAnalysisAgent()
        self.visualization_agent = visualization_agent or DataVisualizationAgent()
        self.analysis_logger = analysis_logger or AnalysisLogger()

    # Runs the analysis stage without writing a log record.
    def _run_analysis_core(self, user: str, database: str, question: str) -> AnalysisResult:
        self.access_control.require_access(user, database)  # check user access
        database_info = self.database_registry.get_database(database)
        return self.analysis_agent.analyze(
            database_path=database_info["path"],
            database_name=database,
            question=question,
            database_description=str(database_info.get("description", "")),
        )

    # Runs only the access-controlled data analysis stage.
    def run_analysis(self, user: str, database: str, question: str) -> PipelineResult:
        started_at = perf_counter()
        analysis_result: AnalysisResult | None = None

        try:
            analysis_result = self._run_analysis_core(user, database, question)
        except Exception as error:
            latency_seconds = perf_counter() - started_at
            self._record_run(
                user=user,
                database=database,
                question=question,
                analysis_result=analysis_result,
                visualization_result=None,
                chart_type="",
                latency_seconds=latency_seconds,
                error=error,
            )
            raise

        latency_seconds = perf_counter() - started_at
        run_id = self._record_run(
            user=user,
            database=database,
            question=question,
            analysis_result=analysis_result,
            visualization_result=None,
            chart_type="",
            latency_seconds=latency_seconds,
        )

        return PipelineResult(
            user=user,
            database=database,
            analysis=analysis_result,
            run_id=run_id,
        )

    # Runs analysis and then passes the result into the visualization agent.
    def run_pipeline(
            self,
            user: str,
            database: str,
            question: str,
            chart_type: str = "auto",
    ) -> PipelineResult:
        started_at = perf_counter()
        analysis_result: AnalysisResult | None = None
        visualization_result: VisualizationResult | None = None

        try:
            analysis_result = self._run_analysis_core(user, database, question)
            visualization_result = self.visualization_agent.visualize(
                analysis_result,
                chart_type=chart_type,
            )
        except Exception as error:
            latency_seconds = perf_counter() - started_at
            self._record_run(
                user=user,
                database=database,
                question=question,
                analysis_result=analysis_result,
                visualization_result=visualization_result,
                chart_type=chart_type,
                latency_seconds=latency_seconds,
                error=error,
            )
            raise

        latency_seconds = perf_counter() - started_at
        run_id = self._record_run(
            user=user,
            database=database,
            question=question,
            analysis_result=analysis_result,
            visualization_result=visualization_result,
            chart_type=chart_type,
            latency_seconds=latency_seconds,
        )

        return PipelineResult(
            user=user,
            database=database,
            analysis=analysis_result,
            visualization=visualization_result,
            run_id=run_id,
        )

    # Saves one success or failure run record for later root-cause tracing.
    def _record_run(
        self,
        user: str,
        database: str,
        question: str,
        analysis_result: AnalysisResult | None,
        visualization_result: VisualizationResult | None,
        chart_type: str,
        latency_seconds: float,
        error: Exception | None = None,
    ) -> str:
        selected_chart_type = ""
        if visualization_result is not None:
            selected_chart_type = visualization_result.chart_type
        elif chart_type:
            selected_chart_type = chart_type

        return self.analysis_logger.record_run(
            user=user,
            database=database,
            question=question,
            generated_sql=analysis_result.sql if analysis_result else "",
            sql_source=analysis_result.sql_source if analysis_result else "",
            error_message=f"{type(error).__name__}: {error}" if error else "",
            row_count=analysis_result.row_count if analysis_result else 0,
            chart_type=selected_chart_type,
            latency_seconds=latency_seconds,
            status="failed" if error else "success",
        )
