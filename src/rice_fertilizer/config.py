from __future__ import annotations
from pathlib import Path
from typing import Any
import copy
import yaml
from .exceptions import ConfigurationError

def project_root() -> Path:
    return Path(__file__).resolve().parents[2]

def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise ConfigurationError(f"Configuration file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ConfigurationError(f"Expected a mapping in {path}")
    return data

def load_all_configs(config_dir: str | Path | None = None) -> dict[str, Any]:
    config_dir = Path(config_dir) if config_dir else project_root() / "configs"
    return {path.stem: load_yaml(path) for path in sorted(config_dir.glob("*.yaml"))}

def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged

def require_value(config: dict[str, Any], dotted_key: str) -> Any:
    current: Any = config
    for key in dotted_key.split("."):
        if not isinstance(current, dict) or key not in current:
            raise ConfigurationError(f"Missing configuration key: {dotted_key}")
        current = current[key]
    if current is None or current == "":
        raise ConfigurationError(
            f"Configuration value '{dotted_key}' is not decided. "
            "Fill configs/decisions.yaml and read README_HIZKIA.md."
        )
    return current

def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root() / path

def ensure_runtime_directories(configs: dict[str, Any]) -> dict[str, Path]:
    paths = configs["paths"]
    directories = {
        "data_root": resolve_path(paths["data_root"]),
        "input_dir": resolve_path(paths["input_csv"]).parent,
        "interim_dir": resolve_path(paths["interim_dir"]),
        "processed_dir": resolve_path(paths["processed_dir"]),
        "artifacts_root": resolve_path(paths["artifacts_root"]),
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories
