"""Analysis run logging and user-feedback storage.

The logger stores each run as a JSON record so failed or disputed results can
be traced back to the user question, selected database, generated SQL, chart
type, latency, and any later user feedback.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.utils.config_loader import PROJECT_ROOT


class AnalysisLogError(RuntimeError):
    """Raised when run logs or feedback cannot be saved."""


class AnalysisLogger:
    # Configures where analysis run records are stored.
    def __init__(self, log_path: str | Path | None = None) -> None:
        self.log_path = Path(log_path) if log_path else PROJECT_ROOT / "outputs" / "logs" / "analysis_runs.json"

    # Appends one analysis run record and returns its stable run id.
    def record_run(
        self,
        user: str,
        database: str,
        question: str,
        generated_sql: str = "",
        sql_source: str = "",
        error_message: str = "",
        row_count: int = 0,
        chart_type: str = "",
        latency_seconds: float = 0.0,
        status: str = "success",
    ) -> str:
        run_id = uuid4().hex
        records = self.list_runs()
        records.append(
            {
                "run_id": run_id,
                "timestamp_utc": self._utc_now(),
                "status": status,
                "user": user,
                "database": database,
                "question": question,
                "generated_sql": generated_sql,
                "sql_source": sql_source,
                "error_message": error_message,
                "row_count": int(row_count or 0),
                "chart_type": chart_type,
                "latency_seconds": round(float(latency_seconds or 0.0), 3),
                "user_feedback": None,
            }
        )
        self._write_records(records)
        return run_id

    # Updates one run with user feedback.
    def record_feedback(
        self,
        run_id: str,
        rating: str,
        comment: str = "",
        user: str = "",
    ) -> dict[str, Any]:
        records = self.list_runs()
        for record in records:
            if record.get("run_id") == run_id:
                record["user_feedback"] = {
                    "rating": rating,
                    "comment": comment,
                    "user": user,
                    "timestamp_utc": self._utc_now(),
                }
                self._write_records(records)
                return record

        raise AnalysisLogError(f"Run id not found: {run_id}")

    # Loads existing run records, newest last.
    def list_runs(self) -> list[dict[str, Any]]:
        if not self.log_path.exists():
            return []

        try:
            with self.log_path.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except json.JSONDecodeError as exc:
            raise AnalysisLogError(f"Analysis log is not valid JSON: {self.log_path}") from exc

        if not isinstance(data, list):
            raise AnalysisLogError(f"Analysis log must contain a JSON list: {self.log_path}")

        return [dict(record) for record in data if isinstance(record, dict)]

    # Writes all records back to disk using an atomic replace.
    def _write_records(self, records: list[dict[str, Any]]) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.log_path.with_suffix(".tmp")
        with temp_path.open("w", encoding="utf-8") as file:
            json.dump(records, file, indent=2, ensure_ascii=False)
        temp_path.replace(self.log_path)

    # Returns the current UTC timestamp as an ISO string.
    def _utc_now(self) -> str:
        return datetime.now(timezone.utc).isoformat()
