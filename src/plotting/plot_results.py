"""Plot BLER curves from results.json."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt


def load(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", required=True, help="Path to results.json")
    ap.add_argument("--outfile", default=None, help="Path to output PNG (default: alongside infile)")
    args = ap.parse_args()

    obj = load(args.infile)
    results = obj.get("results", [])
    if not results:
        raise SystemExit("No 'results' found")

    # Collect receiver names
    all_rx = set()
    for r in results:
        all_rx.update(r.get("receivers", {}).keys())
    all_rx = sorted(all_rx)

    ebnos = [float(r["ebno_db"]) for r in results]

    plt.figure()
    for rx in all_rx:
        blers = []
        for r in results:
            rec = r.get("receivers", {}).get(rx, None)
            blers.append(float(rec.get("bler", float("nan"))) if rec else float("nan"))
        plt.semilogy(ebnos, blers, marker="o", label=rx)

    plt.xlabel("Eb/N0 [dB]")
    plt.ylabel("BLER")
    plt.grid(True, which="both")
    plt.legend()

    outfile = args.outfile
    if outfile is None:
        outfile = os.path.join(os.path.dirname(args.infile), "bler_vs_ebno_db.png")
    plt.savefig(outfile, dpi=200, bbox_inches="tight")
    print("Saved:", outfile)


if __name__ == "__main__":
    main()
