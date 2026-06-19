from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml


class ConfigError(RuntimeError):
    pass


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        def replace_var(match: re.Match[str]) -> str:
            name = match.group(1) or match.group(2)
            return os.environ.get(name, match.group(0))

        expanded = re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)", replace_var, value)
        return os.path.expanduser(os.path.expandvars(expanded))
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def load_config(path: str | None = None, validate_paths: bool = False) -> dict[str, Any]:
    config_path = Path(path or "configs/default_paths.yaml")
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg = _expand(cfg)
    cfg.setdefault("project_root", ".")
    cfg.setdefault("data_root", "data")
    cfg.setdefault("output_root", "outputs")
    cfg.setdefault("models", {})
    cfg.setdefault("datasets", {})
    if validate_paths:
        validate_config_paths(cfg)
    return cfg


def require_path(cfg: dict[str, Any], section: str, key: str) -> str:
    value = cfg.get(section, {}).get(key)
    if not value:
        raise ConfigError(f"Missing required config path: {section}.{key}")
    path = Path(value)
    if not path.exists():
        raise ConfigError(f"Configured path does not exist for {section}.{key}: {path}")
    return str(path)


def validate_config_paths(cfg: dict[str, Any]) -> None:
    hf_home = cfg.get("hf_home")
    if hf_home in (None, "", "$HF_HOME"):
        raise ConfigError("HF_HOME is not set; set HF_HOME to the local Hugging Face cache root.")
    for section in ("models", "datasets"):
        for key, value in cfg.get(section, {}).items():
            if not value:
                continue
            if "$" in str(value):
                raise ConfigError(f"Unexpanded environment variable in {section}.{key}: {value}")
