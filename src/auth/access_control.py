"""User access-control layer.

Users and their allowed databases are loaded from config/users.yaml. The same
class is also used by the Streamlit admin panel to add or update user access.
"""

from pathlib import Path
import re
from dataclasses import dataclass

from src.utils.config_loader import CONFIG_DIR, load_config, load_yaml, save_yaml


class AccessControlError(RuntimeError):
    """Base class for access-control failures."""


class UnknownUserError(AccessControlError):
    """Raised when a username is not configured."""


class AccessDeniedError(AccessControlError):
    """Raised when a user is not allowed to access a database."""


class InvalidUserConfigError(AccessControlError):
    """Raised when a user configuration update is invalid."""


@dataclass(frozen=True)
class UserPermissionUpdate:
    username: str
    status: str
    previous_databases: list[str]
    current_databases: list[str]

    # Lists databases newly granted by an update.
    @property
    def added_databases(self) -> list[str]:
        return sorted(set(self.current_databases) - set(self.previous_databases))

    # Lists databases removed by an update.
    @property
    def removed_databases(self) -> list[str]:
        return sorted(set(self.previous_databases) - set(self.current_databases))


class AccessControl:
    # Loads configured users and their database permissions.
    def __init__(self, users_config_path: str | Path | None = None) -> None:
        self.users_config_path = Path(users_config_path) if users_config_path else CONFIG_DIR / "users.yaml"

        if users_config_path is None:
            config = load_config("users.yaml")
        else:
            config = load_yaml(users_config_path)

        self.users: dict = config.get("users", {})

    # Returns all configured usernames for login hints and admin display.
    def list_users(self) -> list[str]:
        return sorted(self.users.keys())

    # Gets the database names a user is allowed to access.
    def allowed_databases(self, username: str) -> list[str]:
        if username not in self.users:
            raise UnknownUserError(f"Unknown user: {username}")

        allowed = self.users[username].get("allowed_databases", [])
        return list(allowed)

    # Checks access without raising when the database is allowed.
    def can_access(self, username: str, database_name: str) -> bool:
        return database_name in self.allowed_databases(username)

    # Enforces access and raises a readable error when blocked.
    def require_access(self, username: str, database_name: str) -> None:
        if not self.can_access(username, database_name):
            allowed = ", ".join(self.allowed_databases(username)) or "none"
            raise AccessDeniedError(
                f"User '{username}' is not allowed to access database "
                f"'{database_name}'. Allowed databases: {allowed}."
            )

    # Creates a new user or updates an existing user's database permissions.
    def add_or_update_user(
        self,
        username: str,
        allowed_databases: list[str],
    ) -> UserPermissionUpdate:
        username = username.strip()
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{1,31}", username):
            raise InvalidUserConfigError(
                "Username must start with a letter and contain 2-32 letters, numbers, underscores, or hyphens."
            )

        if not allowed_databases:
            raise InvalidUserConfigError("Select at least one database for the user.")

        current_databases = sorted(set(allowed_databases))
        previous_databases = sorted(self.users.get(username, {}).get("allowed_databases", []))

        if username not in self.users:
            status = "created"
        elif previous_databases == current_databases:
            status = "unchanged"
        else:
            status = "updated"

        if status != "unchanged":
            self.users[username] = {"allowed_databases": current_databases}
            save_yaml(self.users_config_path, {"users": self.users})

        return UserPermissionUpdate(
            username=username,
            status=status,
            previous_databases=previous_databases,
            current_databases=current_databases,
        )

    # Grants one database to a new or existing user.
    def grant_database_to_user(self, username: str, database_name: str) -> UserPermissionUpdate:
        username = username.strip()
        database_name = database_name.strip()
        previous_databases = list(self.users.get(username, {}).get("allowed_databases", []))
        return self.add_or_update_user(username, previous_databases + [database_name])
