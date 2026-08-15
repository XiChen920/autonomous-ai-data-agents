"""SQLite schema inspection helpers for feeding table/column context to the agent."""

from pathlib import Path
from typing import Any

from src.db.connector import SQLiteConnector


def quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


class SchemaReader:
    def __init__(self, connector: SQLiteConnector | None = None) -> None:
        self.connector = connector or SQLiteConnector()

    def list_tables(self, database_path: str | Path) -> list[str]:
        sql = """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
          AND name NOT LIKE 'sqlite_%'
        ORDER BY name
        """
        dataframe = self.connector.run_query(database_path, sql)
        return dataframe["name"].tolist()

    def get_table_columns(self, database_path: str | Path, table_name: str) -> list[dict[str, Any]]:
        sql = f"PRAGMA table_info({quote_identifier(table_name)})"
        dataframe = self.connector.run_query(database_path, sql)

        columns = []
        for row in dataframe.to_dict(orient="records"):
            columns.append(
                {
                    "name": row["name"],
                    "type": row["type"],
                    "not_null": bool(row["notnull"]),
                    "primary_key": bool(row["pk"]),
                }
            )

        return columns

    def get_schema_text(self, database_path: str | Path) -> str:
        # The LLM prompt consumes this compact text representation of the schema.
        lines: list[str] = []

        for table_name in self.list_tables(database_path):
            columns = self.get_table_columns(database_path, table_name)
            column_text = ", ".join(
                f"{column['name']} {column['type']}".strip()
                for column in columns
            )
            lines.append(f"Table {table_name}: {column_text}")

        return "\n".join(lines)
