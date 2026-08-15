"""Database registry.

Maps logical database names from config/databases.yaml to real SQLite files.
This keeps adding a new database mostly configuration-driven.
"""

from pathlib import Path
from typing import Any

from src.utils.config_loader import PROJECT_ROOT, load_config, load_yaml


class DatabaseRegistryError(RuntimeError):
    """Base class for database registry failures."""


class UnknownDatabaseError(DatabaseRegistryError):
    """Raised when a database name is not configured."""


class DatabaseFileNotFoundError(DatabaseRegistryError):
    """Raised when a configured database file does not exist."""


class DatabaseRegistry:
    def __init__(self, databases_config_path: str | Path | None = None) -> None:
        if databases_config_path is None:
            config = load_config("databases.yaml")
        else:
            config = load_yaml(databases_config_path)

        self.databases: dict[str, dict[str, Any]] = config.get("databases", {})

    def list_databases(self) -> list[str]:
        return sorted(self.databases.keys())

    def get_database(self, database_name: str) -> dict[str, Any]:
        if database_name not in self.databases:
            raise UnknownDatabaseError(f"Unknown database: {database_name}")

        database = dict(self.databases[database_name])
        database["name"] = database_name
        database["path"] = self.resolve_path(database_name)
        return database

    def resolve_path(self, database_name: str) -> Path:
        if database_name not in self.databases:
            raise UnknownDatabaseError(f"Unknown database: {database_name}")

        raw_path = Path(self.databases[database_name]["path"])
        path = raw_path if raw_path.is_absolute() else PROJECT_ROOT / raw_path

        if not path.exists():
            raise DatabaseFileNotFoundError(
                f"Database file for '{database_name}' does not exist: {path}"
            )

        return path
