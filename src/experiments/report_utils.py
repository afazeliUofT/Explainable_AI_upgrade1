from __future__ import annotations

import csv
import json
import math
import os
from typing import Any, Dict, List, Sequence, Tuple


def _collect_receiver_names(results: Sequence[Dict[str, Any]]) -> List[str]:
    names = set()
    for row in results:
        names.update(row.get("receivers", {}).keys())
    return sorted(names)


def _interpolate_target(results: Sequence[Dict[str, Any]], rx_name: str, target_bler: float) -> float | None:
    pts = []
    for row in results:
        rec = row.get("receivers", {}).get(rx_name)
        if rec is None:
            continue
        bler = rec.get("bler", None)
        ebno = row.get("ebno_db", None)
        if bler is None or ebno is None:
            continue
        if bler <= 0.0:
            bler = 1e-12
        pts.append((float(ebno), float(bler)))
    pts = sorted(pts)
    if len(pts) < 2:
        return None
    target_log = math.log10(target_bler)
    for (e0, b0), (e1, b1) in zip(pts[:-1], pts[1:]):
        lb0 = math.log10(max(b0, 1e-12))
        lb1 = math.log10(max(b1, 1e-12))
        if (lb0 - target_log) * (lb1 - target_log) <= 0 and abs(lb1 - lb0) > 1e-12:
            alpha = (target_log - lb0) / (lb1 - lb0)
            return e0 + alpha * (e1 - e0)
    return None


def _format_gain(reference: str, contender: str, results: Sequence[Dict[str, Any]], target_bler: float) -> str:
    e_ref = _interpolate_target(results, reference, target_bler)
    e_new = _interpolate_target(results, contender, target_bler)
    if e_ref is None or e_new is None:
        return f"{contender} vs {reference} at BLER={target_bler:g}: not enough crossing points"
    gain = e_ref - e_new
    direction = "better" if gain > 0 else "worse"
    return (
        f"{contender} vs {reference} at BLER={target_bler:g}: "
        f"{gain:+.3f} dB ({direction} for {contender})"
    )


def write_summary_files(results_out: Dict[str, Any], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    results = results_out.get("results", [])
    rx_names = _collect_receiver_names(results)

    csv_path = os.path.join(out_dir, "comparison_table.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["ebno_db"]
        for name in rx_names:
            header.extend([f"{name}_bler", f"{name}_ber", f"{name}_n_blocks", f"{name}_n_block_errors"])
        writer.writerow(header)
        for row in sorted(results, key=lambda x: float(x.get("ebno_db", 0.0))):
            out_row = [float(row.get("ebno_db", 0.0))]
            recs = row.get("receivers", {})
            for name in rx_names:
                rec = recs.get(name, {})
                out_row.extend(
                    [
                        rec.get("bler", ""),
                        rec.get("ber", ""),
                        rec.get("n_blocks", ""),
                        rec.get("n_block_errors", ""),
                    ]
                )
            writer.writerow(out_row)

    report_lines = []
    report_lines.append("Pole-transport comparison report")
    report_lines.append("=" * 36)
    report_lines.append("")
    report_lines.append("Receivers:")
    for name in rx_names:
        report_lines.append(f"  - {name}")
    report_lines.append("")
    for row in sorted(results, key=lambda x: float(x.get("ebno_db", 0.0))):
        report_lines.append(f"Eb/N0 = {float(row['ebno_db']):+.2f} dB")
        recs = row.get("receivers", {})
        ranked = sorted(
            [(name, recs[name].get("bler", float("inf"))) for name in recs.keys()],
            key=lambda x: float(x[1]),
        )
        if ranked:
            report_lines.append(f"  best BLER: {ranked[0][0]} -> {float(ranked[0][1]):.4e}")
        for name in rx_names:
            rec = recs.get(name)
            if rec is None:
                continue
            report_lines.append(
                f"  {name:28s} BLER={float(rec['bler']):.4e} BER={float(rec['ber']):.4e} "
                f"block_errors={int(rec['n_block_errors'])}/{int(rec['n_blocks'])}"
            )
        report_lines.append("")

    if "lmmse_ls" in rx_names:
        for contender in ["pole_static", "pole_transport_heuristic", "pole_transport_learned"]:
            if contender in rx_names:
                report_lines.append(_format_gain("lmmse_ls", contender, results, target_bler=1e-1))
                report_lines.append(_format_gain("lmmse_ls", contender, results, target_bler=1e-2))
    report_lines.append("")

    report_path = os.path.join(out_dir, "comparison_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    # Optional plot.
    try:
        import matplotlib.pyplot as plt

        fig = plt.figure()
        for name in rx_names:
            xs = []
            ys = []
            for row in sorted(results, key=lambda x: float(x.get("ebno_db", 0.0))):
                rec = row.get("receivers", {}).get(name)
                if rec is None:
                    continue
                xs.append(float(row["ebno_db"]))
                ys.append(max(float(rec["bler"]), 1e-5))
            if xs:
                plt.semilogy(xs, ys, marker="o", label=name)
        plt.xlabel("Eb/N0 [dB]")
        plt.ylabel("BLER")
        plt.grid(True, which="both")
        plt.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "bler_comparison.png"), dpi=160)
        plt.close(fig)
    except Exception:
        pass

    with open(os.path.join(out_dir, "results_compact.json"), "w", encoding="utf-8") as f:
        json.dump(results_out, f, indent=2)
