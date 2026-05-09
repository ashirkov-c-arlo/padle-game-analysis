from __future__ import annotations

from pathlib import Path

import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict:
    """Load YAML config, merging with defaults.

    If path is None, loads configs/default.yaml.
    If path is provided, loads defaults first then deep-merges the override file on top.
    """
    defaults = _load_yaml(_DEFAULT_CONFIG_PATH)

    if path is None:
        return defaults

    overrides = _load_yaml(Path(path))
    return _deep_merge(defaults, overrides)


def _load_yaml(path: Path) -> dict:
    """Load a YAML file and return its contents as a dict."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return data if data is not None else {}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning a new dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
