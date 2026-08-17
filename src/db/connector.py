"""SQLite read-only connector used by the analysis agent."""

import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd


class DatabaseQueryError(RuntimeError):
    """Raised when a database query cannot be executed."""


class SQLiteConnector:
    # Stores the SQLite connection timeout.
    def __init__(self, timeout: int = 10) -> None:
        self.timeout = timeout

    # Opens a SQLite database in read-only mode.
    def connect_readonly(self, database_path: str | Path) -> sqlite3.Connection:
        path = Path(database_path).resolve()
        # mode=ro ensures agent-generated queries cannot modify the source DB.
        uri = f"file:{path.as_posix()}?mode=ro"
        return sqlite3.connect(uri, uri=True, timeout=self.timeout)

    # Executes a read-only SQL query and returns a pandas DataFrame.
    def run_query(
        self,
        database_path: str | Path,
        sql: str,
        params: dict[str, Any] | tuple[Any, ...] | None = None,
    ) -> pd.DataFrame:
        try:
            with self.connect_readonly(database_path) as connection:
                return pd.read_sql_query(sql, connection, params=params)
        except Exception as exc:
            raise DatabaseQueryError(f"Failed to execute query: {exc}") from exc
