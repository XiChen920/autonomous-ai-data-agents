"""YAML configuration loading and saving helpers."""

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"


class ConfigError(RuntimeError):
    """Raised when a configuration file is missing or invalid."""


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    if not config_path.exists():
        raise ConfigError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise ConfigError(f"Configuration file must contain a YAML object: {config_path}")

    return data


def load_config(filename: str) -> dict[str, Any]:
    return load_yaml(CONFIG_DIR / filename)


def save_yaml(path: str | Path, data: dict[str, Any]) -> None:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False)
