"""Configuration utilities.

We keep config in JSON to avoid extra dependencies on clusters.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False)


def deep_get(d: Dict[str, Any], key: str, default: Any = None) -> Any:
    """Get nested key with dot notation.

    Example:
      deep_get(cfg, "ours.reservoir.M_f")
    """
    cur: Any = d
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def deep_set(d: Dict[str, Any], key: str, value: Any) -> None:
    """Set nested key with dot notation."""
    parts = key.split(".")
    cur: Any = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


@dataclass
class RunPaths:
    out_dir: str
    results_json: str
    plot_png: str


def make_run_paths(out_dir: str) -> RunPaths:
    out_dir = os.path.abspath(out_dir)
    return RunPaths(
        out_dir=out_dir,
        results_json=os.path.join(out_dir, "results.json"),
        plot_png=os.path.join(out_dir, "bler_vs_ebno_db.png"),
    )
