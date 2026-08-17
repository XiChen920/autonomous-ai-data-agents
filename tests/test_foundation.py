"""Foundation tests for configuration, permissions, SQLite access, and SQL safety."""

from pathlib import Path
import sqlite3

import pytest

from src.agents.analysis_agent import DataAnalysisAgent
from src.agents.orchestrator import AgentOrchestrator
from src.auth.access_control import AccessControl, AccessDeniedError
from src.db.connector import SQLiteConnector
from src.db.registry import DatabaseRegistry
from src.db.schema_reader import SchemaReader
from src.db.sql_guard import UnsafeSQLError, ensure_limit, validate_select_only


# Creates a tiny SQLite file used by database-integration tests.
def create_custom_sales_database(database_path: Path) -> None:
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "CREATE TABLE sales (region TEXT NOT NULL, amount REAL NOT NULL)"
        )
        connection.executemany(
            "INSERT INTO sales (region, amount) VALUES (?, ?)",
            [
                ("Europe", 120.0),
                ("Europe", 80.0),
                ("Asia", 90.0),
            ],
        )


# Verifies configured users are allowed or blocked correctly.
def test_access_control_allows_and_blocks_configured_users() -> None:
    access_control = AccessControl()

    access_control.require_access("alice", "chinook")

    with pytest.raises(AccessDeniedError):
        access_control.require_access("bob", "chinook")


# Verifies admin-style user permission updates are persisted.
def test_access_control_can_add_user_permissions(tmp_path: Path) -> None:
    users_config = tmp_path / "users.yaml"
    users_config.write_text(
        """
users:
  admin:
    allowed_databases:
      - chinook
""",
        encoding="utf-8",
    )
    access_control = AccessControl(users_config)

    create_result = access_control.add_or_update_user("charlie", ["sakila", "chinook", "sakila"])
    updated_access_control = AccessControl(users_config)

    assert create_result.status == "created"
    assert create_result.current_databases == ["chinook", "sakila"]
    assert updated_access_control.can_access("charlie", "chinook")
    assert updated_access_control.can_access("charlie", "sakila")

    unchanged_result = updated_access_control.add_or_update_user("charlie", ["chinook", "sakila"])
    assert unchanged_result.status == "unchanged"

    update_result = updated_access_control.add_or_update_user("charlie", ["northwind"])
    assert update_result.status == "updated"
    assert update_result.added_databases == ["northwind"]
    assert update_result.removed_databases == ["chinook", "sakila"]


# Verifies a new user can be created and granted one database in one step.
def test_access_control_can_create_new_user_with_new_database_permission(tmp_path: Path) -> None:
    users_config = tmp_path / "users.yaml"
    users_config.write_text("users: {}\n", encoding="utf-8")
    access_control = AccessControl(users_config)

    grant_result = access_control.grant_database_to_user("trainee", "custom_sales")
    updated_access_control = AccessControl(users_config)

    assert grant_result.status == "created"
    assert grant_result.current_databases == ["custom_sales"]
    assert updated_access_control.can_access("trainee", "custom_sales")


# Verifies logical database names resolve to real SQLite files.
def test_database_registry_resolves_existing_sqlite_file() -> None:
    registry = DatabaseRegistry()

    database_path = registry.resolve_path("chinook")

    assert isinstance(database_path, Path)
    assert database_path.exists()
    assert database_path.name == "chinook.db"


# Verifies a new database can be added through the registry config API.
def test_database_registry_can_add_new_database_integration(tmp_path: Path) -> None:
    database_file = tmp_path / "custom_sales.sqlite"
    databases_config = tmp_path / "databases.yaml"
    create_custom_sales_database(database_file)
    databases_config.write_text("databases: {}\n", encoding="utf-8")

    registry = DatabaseRegistry(databases_config)
    create_result = registry.add_or_update_database(
        "custom_sales",
        database_file,
        "Custom sales analytics database",
    )
    updated_registry = DatabaseRegistry(databases_config)

    assert create_result.status == "created"
    assert updated_registry.resolve_path("custom_sales") == database_file.resolve()
    assert updated_registry.get_database("custom_sales")["description"] == (
        "Custom sales analytics database"
    )

    unchanged_result = updated_registry.add_or_update_database(
        "custom_sales",
        database_file,
        "Custom sales analytics database",
    )
    assert unchanged_result.status == "unchanged"


# Verifies schema inspection can read tables and columns.
def test_schema_reader_can_read_sqlite_schema() -> None:
    registry = DatabaseRegistry()
    schema_reader = SchemaReader()

    tables = schema_reader.list_tables(registry.resolve_path("chinook"))
    schema_text = schema_reader.get_schema_text(registry.resolve_path("chinook"))

    assert tables
    assert "Table" in schema_text


# Verifies SQL safety checks add limits and reject mutations.
def test_sql_guard_adds_limit_and_blocks_mutation_queries() -> None:
    safe_sql = ensure_limit("SELECT * FROM customers")

    assert safe_sql == "SELECT * FROM customers LIMIT 100"
    assert validate_select_only("SELECT Name FROM artists LIMIT 5") == (
        "SELECT Name FROM artists LIMIT 5"
    )

    with pytest.raises(UnsafeSQLError):
        validate_select_only("DROP TABLE customers")


# Verifies the connector can execute a read-only SELECT query.
def test_sqlite_connector_runs_readonly_select_query() -> None:
    registry = DatabaseRegistry()
    connector = SQLiteConnector()

    dataframe = connector.run_query(
        registry.resolve_path("chinook"),
        "SELECT COUNT(*) AS row_count FROM customers",
    )

    assert dataframe.loc[0, "row_count"] > 0


# Verifies a newly added database and newly added user work through the orchestrator.
def test_new_user_can_query_newly_added_database_with_orchestrator(tmp_path: Path) -> None:
    database_file = tmp_path / "custom_sales.sqlite"
    databases_config = tmp_path / "databases.yaml"
    users_config = tmp_path / "users.yaml"
    create_custom_sales_database(database_file)
    databases_config.write_text("databases: {}\n", encoding="utf-8")
    users_config.write_text("users: {}\n", encoding="utf-8")

    registry = DatabaseRegistry(databases_config)
    registry.add_or_update_database(
        "custom_sales",
        database_file,
        "Custom sales analytics database",
    )
    access_control = AccessControl(users_config)
    access_control.grant_database_to_user("trainee", "custom_sales")

    # Supplies deterministic SQL so the test does not depend on OpenAI.
    def fixed_sql(database_name: str, question: str, schema_text: str) -> str:
        return """
        SELECT region, ROUND(SUM(amount), 2) AS total_amount
        FROM sales
        GROUP BY region
        ORDER BY total_amount DESC
        """

    orchestrator = AgentOrchestrator(
        access_control=AccessControl(users_config),
        database_registry=DatabaseRegistry(databases_config),
        analysis_agent=DataAnalysisAgent(
            row_limit=10,
            use_openai=False,
            sql_generator=fixed_sql,
        ),
    )

    result = orchestrator.run_analysis(
        user="trainee",
        database="custom_sales",
        question="Show total amount by region",
    )

    assert result.analysis.row_count == 2
    assert list(result.analysis.dataframe.columns) == ["region", "total_amount"]
    assert result.analysis.dataframe.loc[0, "region"] == "Europe"
