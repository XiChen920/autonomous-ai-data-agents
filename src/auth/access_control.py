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

    @property
    def added_databases(self) -> list[str]:
        return sorted(set(self.current_databases) - set(self.previous_databases))

    @property
    def removed_databases(self) -> list[str]:
        return sorted(set(self.previous_databases) - set(self.current_databases))


class AccessControl:
    def __init__(self, users_config_path: str | Path | None = None) -> None:
        self.users_config_path = Path(users_config_path) if users_config_path else CONFIG_DIR / "users.yaml"

        if users_config_path is None:
            config = load_config("users.yaml")
        else:
            config = load_yaml(users_config_path)

        self.users: dict = config.get("users", {})

    def list_users(self) -> list[str]:
        return sorted(self.users.keys())

    def allowed_databases(self, username: str) -> list[str]:
        if username not in self.users:
            raise UnknownUserError(f"Unknown user: {username}")

        allowed = self.users[username].get("allowed_databases", [])
        return list(allowed)

    def can_access(self, username: str, database_name: str) -> bool:
        return database_name in self.allowed_databases(username)

    def require_access(self, username: str, database_name: str) -> None:
        if not self.can_access(username, database_name):
            allowed = ", ".join(self.allowed_databases(username)) or "none"
            raise AccessDeniedError(
                f"User '{username}' is not allowed to access database "
                f"'{database_name}'. Allowed databases: {allowed}."
            )

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
        save_yaml(self.users_config_path, {"users": self.users})
