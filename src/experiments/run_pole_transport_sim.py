"""Evaluate the pole-transport PUSCH receiver against baselines.

Usage:
    python -m src.experiments.run_pole_transport_sim \
        --config configs/pole_transport_umino_cfo05_singlelayer.json \
        --out results/pole_transport_eval

For SLURM array jobs, use:
    python -m src.experiments.run_pole_transport_sim ... --only_ebno_index 3
"""
from __future__ import annotations

import argparse
import copy
import os
import socket
from typing import Any, Callable, Dict, Optional

import tensorflow as tf

from src.data.pusch_link import PuschLink
from src.experiments.report_utils import write_summary_files
from src.receivers.baselines import build_baselines
from src.receivers.pole_transport_receiver import PoleTransportPuschReceiver
from src.utils.config import load_json, save_json
from src.utils.io import ensure_dir, timestamp
from src.utils.metrics import ErrorCounts, count_errors
from src.utils.seed import seed_everything


def _make_receiver_cfg(cfg: Dict[str, Any], *, policy_mode: str, selection: str) -> Dict[str, Any]:
    c = copy.deepcopy(cfg)
    c.setdefault("pole_transport", {})
    c["pole_transport"].setdefault("policy", {})
    c["pole_transport"].setdefault("receiver", {})
    c["pole_transport"]["policy"]["mode"] = str(policy_mode)
    c["pole_transport"]["receiver"]["selection"] = str(selection)
    return c


def _build_receivers(link: PuschLink, cfg: Dict[str, Any]):
    rx_cfg = cfg.get("receivers", {}) or {}
    receivers: Dict[str, Callable[[tf.Tensor, tf.Tensor, Optional[tf.Tensor]], tf.Tensor]] = {}

    # Baselines from the existing repo.
    base = build_baselines(link.tx, cfg)
    if bool(rx_cfg.get("run_lmmse_ls", True)) and "lmmse_ls" in base:
        fn = base["lmmse_ls"]
        receivers["lmmse_ls"] = lambda y, no, h, fn=fn: fn(y, no, None)
    if bool(rx_cfg.get("run_lmmse_perfect_csi", False)) and "lmmse_perfect_csi" in base:
        fn = base["lmmse_perfect_csi"]
        receivers["lmmse_perfect_csi"] = lambda y, no, h, fn=fn: fn(y, no, h)

    if bool(rx_cfg.get("run_pole_static", True)):
        cfg_static = _make_receiver_cfg(cfg, policy_mode="heuristic", selection="static")
        r = PoleTransportPuschReceiver(link.tx, cfg_static, selection_mode="static")
        receivers["pole_static"] = lambda y, no, h, r=r: r(y, no)

    if bool(rx_cfg.get("run_pole_transport_heuristic", True)):
        cfg_h = _make_receiver_cfg(cfg, policy_mode="heuristic", selection="best")
        r = PoleTransportPuschReceiver(link.tx, cfg_h, selection_mode="best")
        receivers["pole_transport_heuristic"] = lambda y, no, h, r=r: r(y, no)

    if bool(rx_cfg.get("run_pole_transport_learned", False)):
        cfg_l = _make_receiver_cfg(cfg, policy_mode="learned", selection="best")
        r = PoleTransportPuschReceiver(link.tx, cfg_l, selection_mode="best")
        weights_path = cfg_l.get("pole_transport", {}).get("policy", {}).get("weights_path", None)
        loaded = r.maybe_load_policy_weights(weights_path)
        if loaded:
            receivers["pole_transport_learned"] = lambda y, no, h, r=r: r(y, no)
        else:
            print(
                "[run_pole_transport_sim] learned pole-transport weights were not found; "
                "skipping pole_transport_learned"
            )

    return receivers


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only_ebno_index", type=int, default=None)
    args = ap.parse_args()

    cfg = load_json(args.config)
    ensure_dir(args.out)
    seed_everything(int(cfg.get("seed", 1)))

    link = PuschLink(cfg)
    receivers = _build_receivers(link, cfg)

    sim_cfg = cfg.get("sim", {})
    ebno_list = list(sim_cfg.get("ebno_db_list", [0]))
    batch_size = int(sim_cfg.get("batch_size", 32))
    min_block_errors = int(sim_cfg.get("min_block_errors", 120))
    max_batches = int(sim_cfg.get("max_batches", 400))
    stop_mode = str(sim_cfg.get("stop_mode", "ref")).lower()
    stop_ref = str(sim_cfg.get("stop_ref", "lmmse_ls"))
    ignore_min_err = set(sim_cfg.get("ignore_min_error_receivers", []))
    progress_every = int(sim_cfg.get("progress_every", 25))

    if args.only_ebno_index is not None:
        i = int(args.only_ebno_index)
        if i < 0 or i >= len(ebno_list):
            raise ValueError(f"only_ebno_index={i} is out of range")
        ebno_list = [ebno_list[i]]

    use_tf_function = bool(cfg.get("tf_function", False))
    jit = bool(cfg.get("jit_compile", False))
    tf_function_baselines = bool(cfg.get("tf_function_baselines", False))

    if use_tf_function or jit:
        print(
            "[run_pole_transport_sim] graph mode enabled: "
            f"tf_function={use_tf_function}, jit_compile={jit}, tf_function_baselines={tf_function_baselines}"
        )
        try:
            warm_batch = link.generate(batch_size=max(1, min(2, batch_size)), ebno_db=float(ebno_list[0]))
            for name, fn in receivers.items():
                try:
                    _ = fn(warm_batch.y, warm_batch.no, warm_batch.h)
                except Exception as e:
                    print(f"[warmup] receiver={name} failed during warmup: {repr(e)}")
        except Exception as e:
            print("[warmup] skipped:", repr(e))
        for k, fn in list(receivers.items()):
            is_baseline = k.startswith("lmmse_")
            do_wrap = (not is_baseline) or tf_function_baselines
            if do_wrap:
                receivers[k] = tf.function(fn, jit_compile=jit)

    print("=== pole-transport simulation ===")
    print("out_dir:", os.path.abspath(args.out))
    print("host:", socket.gethostname())
    print("receivers:", list(receivers.keys()))
    print("ebno_db_list:", ebno_list)

    results_out: Dict[str, Any] = {
        "config": cfg,
        "meta": {
            "timestamp": timestamp(),
            "host": socket.gethostname(),
            "tf_function": use_tf_function,
            "jit_compile": jit,
        },
        "results": [],
    }

    for ebno_db in ebno_list:
        acc: Dict[str, ErrorCounts] = {
            name: ErrorCounts(n_bits=0, n_bit_errors=0, n_blocks=0, n_block_errors=0) for name in receivers.keys()
        }
        n_batches = 0
        try:
            no_val = float(tf.get_static_value(link.ebnodb_to_no(float(ebno_db))) or 0.0)
        except Exception:
            no_val = None

        while True:
            n_batches += 1
            batch = link.generate(batch_size=batch_size, ebno_db=float(ebno_db))
            for name, rx_fn in receivers.items():
                b_hat = rx_fn(batch.y, batch.no, batch.h)
                errs = count_errors(batch.b, b_hat)
                prev = acc[name]
                acc[name] = ErrorCounts(
                    n_bits=prev.n_bits + errs.n_bits,
                    n_bit_errors=prev.n_bit_errors + errs.n_bit_errors,
                    n_blocks=prev.n_blocks + errs.n_blocks,
                    n_block_errors=prev.n_block_errors + errs.n_block_errors,
                )

            if (progress_every > 0) and (n_batches % progress_every == 0):
                ref_be = acc.get(stop_ref, None)
                ref_num = ref_be.n_block_errors if ref_be is not None else -1
                print(
                    f"[progress] Eb/N0={ebno_db:+.2f} dB | batch={n_batches}/{max_batches} | "
                    f"{stop_ref} block_err={ref_num}",
                    flush=True,
                )

            if n_batches >= max_batches:
                break
            if stop_mode == "ref":
                if stop_ref not in acc:
                    raise ValueError(f"stop_ref={stop_ref} not present in receivers")
                if acc[stop_ref].n_block_errors >= min_block_errors:
                    break
            else:
                if all(
                    (a.n_block_errors >= min_block_errors) or (name in ignore_min_err)
                    for name, a in acc.items()
                ):
                    break

        rec_summary: Dict[str, Any] = {}
        print(f"Eb/N0={ebno_db:.2f} dB | batches={n_batches}")
        for name, a in acc.items():
            rec_summary[name] = {
                "ber": a.ber(),
                "bler": a.bler(),
                "n_bits": a.n_bits,
                "n_bit_errors": a.n_bit_errors,
                "n_blocks": a.n_blocks,
                "n_block_errors": a.n_block_errors,
            }
            print(
                f"  {name:28s} BLER={rec_summary[name]['bler']:.4e} "
                f"BER={rec_summary[name]['ber']:.4e}"
            )

        results_out["results"].append(
            {"ebno_db": float(ebno_db), "no": no_val, "receivers": rec_summary}
        )

        if args.only_ebno_index is not None:
            fname = f"result_ebno_{float(ebno_db):+.2f}.json".replace("+", "p").replace("-", "m")
            out_path = os.path.join(args.out, fname)
            save_json(
                {"config": cfg, "meta": results_out["meta"], "results": results_out["results"][-1:]},
                out_path,
            )
            print("Saved partial:", out_path)
        else:
            out_path = os.path.join(args.out, "results.json")
            save_json(results_out, out_path)
            write_summary_files(results_out, args.out)
            print("Saved:", out_path)

    if args.only_ebno_index is None:
        out_path = os.path.join(args.out, "results.json")
        save_json(results_out, out_path)
        write_summary_files(results_out, args.out)
        print("Final results written to:", out_path)


if __name__ == "__main__":
    main()
