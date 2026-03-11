"""Main simulation script.

Usage:
  python -m src.experiments.run_sim --config configs/example_pusch_reservoir.json --out results/run1

Optional:
  --only_ebno_index i   # run only ebno_db_list[i] (useful for SLURM arrays)
"""

from __future__ import annotations

import argparse
import copy
import os
import socket
from typing import Any, Callable, Dict, Optional

import tensorflow as tf

from src.data.pusch_link import PuschLink
from src.receivers.baselines import build_baselines
from src.receivers.reservoir_receiver import ReservoirPuschReceiver
from src.utils.config import load_json, save_json
from src.utils.io import ensure_dir, timestamp
from src.utils.metrics import ErrorCounts, count_errors
from src.utils.seed import seed_everything


def _make_variant_cfg(cfg: Dict[str, Any], dd_enabled: bool, lowrank_enabled: bool) -> Dict[str, Any]:
    c = copy.deepcopy(cfg)
    c.setdefault("ours", {})
    c["ours"].setdefault("dd", {})
    c["ours"].setdefault("lowrank", {})
    c["ours"]["dd"]["enabled"] = bool(dd_enabled)
    c["ours"]["lowrank"]["enabled"] = bool(lowrank_enabled)
    return c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only_ebno_index", type=int, default=None)
    args = ap.parse_args()

    cfg = load_json(args.config)
    ensure_dir(args.out)

    seed = int(cfg.get("seed", 1))
    seed_everything(seed)

    # Build link
    link = PuschLink(cfg)

    # Build receivers
    rx_cfg = cfg.get("receivers", {})

    # Accept list-of-receiver-names configs (probe convenience / legacy)
    if isinstance(rx_cfg, (list, tuple)):
        names = set(rx_cfg)
        rx_cfg = {
            "run_lmmse_ls": ("lmmse_ls" in names),
            "run_lmmse_perfect_csi": ("lmmse_perfect_csi" in names),
            "run_kbest_ls": ("kbest_ls" in names),
            "run_ours_pilot_only_full": ("ours_pilot_only_full" in names),
            "run_ours_dd_full": ("ours_dd_full" in names),
            "run_ours_pilot_only_lowrank": ("ours_pilot_only_lowrank" in names),
            "run_ours_dd_lowrank": ("ours_dd_lowrank" in names),
        }
    elif rx_cfg is None:
        rx_cfg = {}
    elif not isinstance(rx_cfg, dict):
        raise TypeError(
            f"cfg['receivers'] must be a dict or list, got {type(rx_cfg)}"
        )

    receivers = {}
    # Receiver callables operate on tensors only.
    #
    # IMPORTANT (TensorFlow/XLA): We intentionally avoid passing the custom
    # `LinkBatch` object into `tf.function` because plain dataclasses are not
    # traceable by default. Passing tensors directly keeps compilation
    # compatible with `tf.function` and `jit_compile=True`.
    receivers: Dict[str, Callable[[tf.Tensor, tf.Tensor, Optional[tf.Tensor]], tf.Tensor]] = {}

    # Baselines
    base = build_baselines(link.tx, cfg)
    for name, fn in base.items():
        # Important: only pass perfect CSI to receivers that are meant to use it.
        if "perfect" in name:
            receivers[name] = lambda y, no, h, fn=fn: fn(y, no, h)
        else:
            receivers[name] = lambda y, no, h, fn=fn: fn(y, no, None)

    # Proposed variants
    if bool(rx_cfg.get("run_ours_pilot_only_full", True)):
        cfg_po = _make_variant_cfg(cfg, dd_enabled=False, lowrank_enabled=False)
        r = ReservoirPuschReceiver(link.tx, cfg_po)
        receivers["ours_pilot_only_full"] = lambda y, no, h, r=r: r(y, no)

    if bool(rx_cfg.get("run_ours_dd_full", True)):
        cfg_dd = _make_variant_cfg(cfg, dd_enabled=True, lowrank_enabled=False)
        r = ReservoirPuschReceiver(link.tx, cfg_dd)
        receivers["ours_dd_full"] = lambda y, no, h, r=r: r(y, no)


        # Proposed variants (LOWRANK pilot-only)
    if bool(rx_cfg.get("run_ours_pilot_only_lowrank", True)):
        cfg_lr_po = _make_variant_cfg(cfg, dd_enabled=False, lowrank_enabled=True)
        r = ReservoirPuschReceiver(link.tx, cfg_lr_po)
        receivers["ours_pilot_only_lowrank"] = lambda y, no, h, r=r: r(y, no)

    if bool(rx_cfg.get("run_ours_dd_lowrank", True)):
        cfg_lr = _make_variant_cfg(cfg, dd_enabled=True, lowrank_enabled=True)
        r = ReservoirPuschReceiver(link.tx, cfg_lr)
        receivers["ours_dd_lowrank"] = lambda y, no, h, r=r: r(y, no)

    # Simulation settings
    sim_cfg = cfg.get("sim", {})
    ebno_list = list(sim_cfg.get("ebno_db_list", [0]))
    batch_size = int(sim_cfg.get("batch_size", 32))
    min_block_errors = int(sim_cfg.get("min_block_errors", 200))
    max_batches = int(sim_cfg.get("max_batches", 400))

    # Stopping rule
    # - "all": stop when (all receivers, except ignored) have >= min_block_errors
    # - "ref": stop when a designated reference receiver has >= min_block_errors
    stop_mode = str(sim_cfg.get("stop_mode", "all")).lower()
    stop_ref = str(sim_cfg.get("stop_ref", "lmmse_ls"))
    ignore_min_err = set(sim_cfg.get("ignore_min_error_receivers", ["lmmse_perfect_csi"]))
    progress_every = int(sim_cfg.get("progress_every", 50))

    if args.only_ebno_index is not None:
        i = int(args.only_ebno_index)
        if i < 0 or i >= len(ebno_list):
            raise ValueError(f"only_ebno_index={i} out of range for ebno_db_list (len={len(ebno_list)})")
        ebno_list = [ebno_list[i]]

    # Optionally compile receivers
    #
    # `tf_function` enables graph tracing (good speed-up on GPU) without
    # requiring XLA. `jit_compile` enables XLA, which may require `ptxas`.
    # If your cluster lacks the CUDA toolchain in PATH, set `jit_compile=false`.
    #
    # IMPORTANT (Sionna/Keras): Some Sionna blocks create constant tensors in
    # `build()` and store them as layer attributes (e.g., CRC encoder matrices).
    # If the first call happens *inside* a traced `tf.function`, those constants
    # may become graph-tensors tied to that FuncGraph and later lead to
    # "out of scope" errors. We therefore run a one-shot warmup call in eager
    # mode before wrapping.
    use_tf_function = bool(cfg.get("tf_function", False))
    jit = bool(cfg.get("jit_compile", False))
    tf_function_baselines = bool(cfg.get("tf_function_baselines", False))
    if use_tf_function or jit:
        print(
            "[run_sim] Receiver graph mode enabled: tf_function=%s, jit_compile=%s, tf_function_baselines=%s"
            % (use_tf_function, jit, tf_function_baselines)
        )

        # Eager warmup to force Keras/Sionna layers to build outside tf.function
        try:
            warm_bs = max(1, min(2, int(batch_size)))
            warm_batch = link.generate(batch_size=warm_bs, ebno_db=float(ebno_list[0]))
            for name, fn in receivers.items():
                try:
                    _ = fn(warm_batch.y, warm_batch.no, warm_batch.h)
                except Exception as e:
                    print(f"[run_sim] Warmup failed for receiver '{name}' (continuing): {repr(e)}")
        except Exception as e:
            print("[run_sim] Warmup step failed (continuing without warmup):", repr(e))

        # Wrap selected receivers
        for k, fn in list(receivers.items()):
            is_baseline = (k.startswith("lmmse_") or k.startswith("kbest_"))
            do_wrap = (not is_baseline) or tf_function_baselines
            if do_wrap:
                receivers[k] = tf.function(fn, jit_compile=jit)

    print("=== Simulation config ===")
    print("out_dir           :", os.path.abspath(args.out))
    print("num_tx            :", link.num_tx)
    print("num_layers        :", link.num_layers)
    print("num_tx_ant        :", link.num_tx_ant)
    print("num_rx_ant        :", link.num_rx_ant)
    print("ebno_db_list      :", ebno_list)
    print("batch_size        :", batch_size)
    print("min_block_errors  :", min_block_errors)
    print("max_batches       :", max_batches)
    print("stop_mode         :", stop_mode)
    print("stop_ref          :", stop_ref)
    print("ignore_min_error  :", sorted(list(ignore_min_err)))
    print("receivers         :", list(receivers.keys()))

    results_out: Dict[str, Any] = {
        "config": cfg,
        "meta": {
            "timestamp": timestamp(),
            "host": socket.gethostname(),
            "tf_function": bool(cfg.get("tf_function", False)),
            "jit_compile": bool(cfg.get("jit_compile", False)),
        },
        "results": [],
    }

    for ebno_db in ebno_list:
        # Accumulators per receiver
        acc: Dict[str, ErrorCounts] = {}
        # Initialize counts
        for name in receivers.keys():
            acc[name] = ErrorCounts(n_bits=0, n_bit_errors=0, n_blocks=0, n_block_errors=0)

        n_batches = 0
        # Log the actual noise variance used at this Eb/N0 (when convertible to a python scalar)
        try:
            no_val = float(tf.get_static_value(link.ebnodb_to_no(float(ebno_db))) or 0.0)
        except Exception:
            no_val = None
        while True:
            n_batches += 1
            batch = link.generate(batch_size=batch_size, ebno_db=float(ebno_db))

            # Run all receivers on the same realization
            for name, rx_fn in receivers.items():
                b_hat = rx_fn(batch.y, batch.no, batch.h)
                errs = count_errors(batch.b, b_hat)
                a = acc[name]
                acc[name] = ErrorCounts(
                    n_bits=a.n_bits + errs.n_bits,
                    n_bit_errors=a.n_bit_errors + errs.n_bit_errors,
                    n_blocks=a.n_blocks + errs.n_blocks,
                    n_block_errors=a.n_block_errors + errs.n_block_errors,
                )

            # Periodic progress (helps on clusters with buffered stdout)
            if (progress_every > 0) and (n_batches % progress_every == 0):
                a0 = acc.get(stop_ref, None)
                ref_be = (a0.n_block_errors if a0 is not None else -1)
                print(
                    f"[progress] Eb/N0={ebno_db:+.2f} dB | batch={n_batches}/{max_batches} | {stop_ref} block_err={ref_be}",
                    flush=True,
                )

            # Stopping rule
            if n_batches >= max_batches:
                break

            if stop_mode == "ref":
                if stop_ref not in acc:
                    raise ValueError(
                        f"stop_ref='{stop_ref}' not found among receivers {list(acc.keys())}."
                    )
                if acc[stop_ref].n_block_errors >= min_block_errors:
                    break
            else:
                # stop_mode == "all"
                if all(
                    (a.n_block_errors >= min_block_errors) or (name in ignore_min_err)
                    for name, a in acc.items()
                ):
                    break

        # Summarize
        rec_summary: Dict[str, Any] = {}
        for name, a in acc.items():
            rec_summary[name] = {
                "ber": a.ber(),
                "bler": a.bler(),
                "n_bits": a.n_bits,
                "n_bit_errors": a.n_bit_errors,
                "n_blocks": a.n_blocks,
                "n_block_errors": a.n_block_errors,
            }

        print(f"Eb/N0={ebno_db:.2f} dB | batches={n_batches}")
        for name in rec_summary:
            print(f"  {name:20s}  BLER={rec_summary[name]['bler']:.4e}  BER={rec_summary[name]['ber']:.4e}")

        results_out["results"].append({
            "ebno_db": float(ebno_db),
            "no": no_val,
            "receivers": rec_summary,
        })

        # Write intermediate output
        if args.only_ebno_index is not None:
            # Avoid collisions for array jobs
            fname = f"result_ebno_{float(ebno_db):+.2f}.json".replace("+", "p").replace("-", "m")
            out_path = os.path.join(args.out, fname)
            save_json({"config": cfg, "meta": results_out["meta"], "results": results_out["results"][-1:]}, out_path)
            print("Saved partial:", out_path)
        else:
            out_path = os.path.join(args.out, "results.json")
            save_json(results_out, out_path)
            print("Saved:", out_path)

    # Final write
    if args.only_ebno_index is None:
        out_path = os.path.join(args.out, "results.json")
        save_json(results_out, out_path)

        # Plot
        try:
            from src.plotting.plot_results import main as plot_main

            # Fake argv for plotter
            import sys

            argv0 = sys.argv
            sys.argv = ["plot_results", "--infile", out_path]
            plot_main()
            sys.argv = argv0
        except Exception as e:
            print("Plotting failed (non-fatal):", repr(e))


if __name__ == "__main__":
    main()
