"""Database registry.

Maps logical database names from config/databases.yaml to real SQLite files.
This keeps adding a new database mostly configuration-driven.
"""

from pathlib import Path
import re
import sqlite3
from typing import Any
from dataclasses import dataclass

from src.utils.config_loader import CONFIG_DIR, PROJECT_ROOT, load_config, load_yaml, save_yaml


class DatabaseRegistryError(RuntimeError):
    """Base class for database registry failures."""


class UnknownDatabaseError(DatabaseRegistryError):
    """Raised when a database name is not configured."""


class DatabaseFileNotFoundError(DatabaseRegistryError):
    """Raised when a configured database file does not exist."""


class InvalidDatabaseConfigError(DatabaseRegistryError):
    """Raised when a database registry update is invalid."""


@dataclass(frozen=True)
class DatabaseConfigUpdate:
    database_name: str
    status: str
    previous_database: dict[str, Any]
    current_database: dict[str, Any]

    # Lists registry fields that changed during the update.
    @property
    def changed_fields(self) -> list[str]:
        fields = set(self.previous_database).union(self.current_database)
        return sorted(
            field
            for field in fields
            if self.previous_database.get(field) != self.current_database.get(field)
        )


class DatabaseRegistry:
    # Loads logical database definitions from the database config file.
    def __init__(self, databases_config_path: str | Path | None = None) -> None:
        self.databases_config_path = (
            Path(databases_config_path) if databases_config_path else CONFIG_DIR / "databases.yaml"
        )

        if databases_config_path is None:
            config = load_config("databases.yaml")
        else:
            config = load_yaml(databases_config_path)

        self.databases: dict[str, dict[str, Any]] = config.get("databases", {})

    # Returns the configured database names.
    def list_databases(self) -> list[str]:
        return sorted(self.databases.keys())

    # Returns database metadata plus its resolved filesystem path.
    def get_database(self, database_name: str) -> dict[str, Any]:
        if database_name not in self.databases:
            raise UnknownDatabaseError(f"Unknown database: {database_name}")

        database = dict(self.databases[database_name])
        database["name"] = database_name
        database["path"] = self.resolve_path(database_name)
        return database

    # Converts a logical database name into an existing SQLite file path.
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

    # Creates or updates a logical database entry in config/databases.yaml.
    def add_or_update_database(
        self,
        database_name: str,
        database_path: str | Path,
        description: str,
    ) -> DatabaseConfigUpdate:
        database_name = database_name.strip().lower()
        if not re.fullmatch(r"[a-z][a-z0-9_-]{1,31}", database_name):
            raise InvalidDatabaseConfigError(
                "Database name must start with a lowercase letter and contain 2-32 lowercase letters, numbers, underscores, or hyphens."
            )

        path_text = str(database_path).strip()
        if not path_text:
            raise InvalidDatabaseConfigError("Database path is required.")

        description = description.strip()
        if not description:
            raise InvalidDatabaseConfigError("Database description is required.")

        raw_path = Path(path_text)
        resolved_path = raw_path if raw_path.is_absolute() else PROJECT_ROOT / raw_path
        resolved_path = resolved_path.resolve()
        self._validate_sqlite_file(resolved_path)

        stored_path = raw_path.as_posix()
        current_database = {
            "path": stored_path,
            "description": description,
        }
        previous_database = dict(self.databases.get(database_name, {}))

        if database_name not in self.databases:
            status = "created"
        elif previous_database == current_database:
            status = "unchanged"
        else:
            status = "updated"

        if status != "unchanged":
            self.databases[database_name] = current_database
            save_yaml(self.databases_config_path, {"databases": self.databases})

        return DatabaseConfigUpdate(
            database_name=database_name,
            status=status,
            previous_database=previous_database,
            current_database=current_database,
        )

    # Checks that a configured path points to an existing readable SQLite file.
    def _validate_sqlite_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            raise DatabaseFileNotFoundError(f"SQLite database file does not exist: {path}")

        try:
            with sqlite3.connect(path) as connection:
                connection.execute("SELECT name FROM sqlite_master LIMIT 1").fetchall()
        except sqlite3.Error as exc:
            raise InvalidDatabaseConfigError(
                f"File is not a readable SQLite database: {path}"
            ) from exc
