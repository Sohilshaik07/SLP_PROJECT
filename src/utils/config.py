"""
Utility: Configuration Loader
==============================
Loads and validates configs/config.yaml.
Provides a simple Config object accessible project-wide.
"""

import os
import yaml
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "configs" / "config.yaml"


def load_config(config_path: str | Path = None) -> dict:
    """
    Load the YAML configuration file.

    Parameters
    ----------
    config_path : str or Path, optional
        Path to the config file. Defaults to configs/config.yaml
        in the project root.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    config_path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            f"Expected location: {_DEFAULT_CONFIG_PATH}"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def get_project_root() -> Path:
    """Return the absolute path to the project root."""
    return _PROJECT_ROOT


def resolve_path(relative_path: str, config: dict = None) -> Path:
    """
    Resolve a path relative to the project root.

    Parameters
    ----------
    relative_path : str
        Path relative to the project root.
    config : dict, optional
        Config dict (unused here, kept for API consistency).

    Returns
    -------
    Path
        Absolute path.
    """
    return _PROJECT_ROOT / relative_path
