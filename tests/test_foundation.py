"""Foundation tests for configuration, permissions, SQLite access, and SQL safety."""

from pathlib import Path

import pytest

from src.auth.access_control import AccessControl, AccessDeniedError
from src.db.connector import SQLiteConnector
from src.db.registry import DatabaseRegistry
from src.db.schema_reader import SchemaReader
from src.db.sql_guard import UnsafeSQLError, ensure_limit, validate_select_only


def test_access_control_allows_and_blocks_configured_users() -> None:
    access_control = AccessControl()

    access_control.require_access("alice", "chinook")

    with pytest.raises(AccessDeniedError):
        access_control.require_access("bob", "chinook")


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


def test_database_registry_resolves_existing_sqlite_file() -> None:
    registry = DatabaseRegistry()

    database_path = registry.resolve_path("chinook")

    assert isinstance(database_path, Path)
    assert database_path.exists()
    assert database_path.name == "chinook.db"


def test_schema_reader_can_read_sqlite_schema() -> None:
    registry = DatabaseRegistry()
    schema_reader = SchemaReader()

    tables = schema_reader.list_tables(registry.resolve_path("chinook"))
    schema_text = schema_reader.get_schema_text(registry.resolve_path("chinook"))

    assert tables
    assert "Table" in schema_text


def test_sql_guard_adds_limit_and_blocks_mutation_queries() -> None:
    safe_sql = ensure_limit("SELECT * FROM customers")

    assert safe_sql == "SELECT * FROM customers LIMIT 100"
    assert validate_select_only("SELECT Name FROM artists LIMIT 5") == (
        "SELECT Name FROM artists LIMIT 5"
    )

    with pytest.raises(UnsafeSQLError):
        validate_select_only("DROP TABLE customers")


def test_sqlite_connector_runs_readonly_select_query() -> None:
    registry = DatabaseRegistry()
    connector = SQLiteConnector()

    dataframe = connector.run_query(
        registry.resolve_path("chinook"),
        "SELECT COUNT(*) AS row_count FROM customers",
    )

    assert dataframe.loc[0, "row_count"] > 0
