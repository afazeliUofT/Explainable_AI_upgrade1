"""Merge partial Eb/N0 JSON files from SLURM array jobs and write summary files.

Usage:
    python -m src.experiments.merge_pole_transport_results --indir results/pole_transport_eval
"""
from __future__ import annotations

import argparse
import glob
import os
from typing import Any, Dict, List

from src.experiments.report_utils import write_summary_files
from src.utils.config import load_json, save_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True)
    args = ap.parse_args()

    pattern = os.path.join(args.indir, "result_ebno_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No partial result files found under {pattern}")

    merged: Dict[str, Any] = {"config": None, "meta": None, "results": []}
    rows: List[Dict[str, Any]] = []
    for fp in files:
        obj = load_json(fp)
        if merged["config"] is None:
            merged["config"] = obj.get("config")
        if merged["meta"] is None:
            merged["meta"] = obj.get("meta")
        for row in obj.get("results", []):
            rows.append(row)

    rows = sorted(rows, key=lambda x: float(x.get("ebno_db", 0.0)))
    merged["results"] = rows
    out_path = os.path.join(args.indir, "results.json")
    save_json(merged, out_path)
    write_summary_files(merged, args.indir)
    print("Merged results written to", out_path)


if __name__ == "__main__":
    main()
