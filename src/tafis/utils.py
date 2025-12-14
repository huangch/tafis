"""
tafis.utils

Small shared utility helpers used across the TAFIS codebase.

Design principles:
  - Keep this file minimal and dependency-free.
  - No business logic here; only generic helpers.
  - Avoid circular imports (utils should not import model/train/etc.).

Current responsibilities:
  - Safe JSON read/write helpers
  - YAML config loading (optional convenience)
  - Path normalization helpers
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def ensure_dir(path: str | Path) -> Path:
    """
    Ensure a directory exists and return it as a Path.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: str | Path) -> Dict[str, Any]:
    """
    Read a JSON file into a Python dict.
    """
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(
    obj: Dict[str, Any],
    path: str | Path,
    indent: int = 2,
    sort_keys: bool = True,
) -> None:
    """
    Write a Python dict to JSON with sane defaults.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, sort_keys=sort_keys)


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    This is optional sugar for users who prefer YAML over CLI flags.
    """
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def maybe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    """
    Try to cast a value to float, otherwise return default.
    """
    try:
        return float(x)
    except Exception:
        return default
