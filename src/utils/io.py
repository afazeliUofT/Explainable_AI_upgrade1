"""Lightweight I/O utilities."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict


def timestamp() -> str:
    return time.strftime("%Y-%m-%d_%H-%M-%S")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_json(obj: Dict[str, Any], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False)


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
