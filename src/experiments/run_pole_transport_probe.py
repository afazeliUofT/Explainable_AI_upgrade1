
"""Probe the pole-transport detector before decoding.

This script is meant to diagnose whether the detector is failing at:
1) pilot fitting / candidate selection,
2) symbol estimation before LLR generation, or
3) the decoding/LLR path.

It measures per-candidate pilot MSE, symbol MSE/EVM on data REs, and hard symbol
error rate before the NR decoder.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List

import numpy as np
import tensorflow as tf

from src.data.pusch_link import PuschLink
from src.receivers.pole_transport_receiver import PoleTransportPuschReceiver
from src.utils.config import load_json, save_json
from src.utils.io import ensure_dir
from src.utils.seed import seed_everything


def _build_points(detector) -> tf.Tensor:
    pts = getattr(detector, "_sionna_points", None)
    if pts is not None:
        return tf.cast(tf.reshape(pts, [-1]), detector.cdtype)

    # Internal square-QAM fallback from PAM levels
    levels = tf.cast(detector._pam_levels, detector.rdtype)
    re, im = tf.meshgrid(levels, levels, indexing="ij")
    pts = tf.cast(re, detector.cdtype) + 1j * tf.cast(im, detector.cdtype)
    return tf.reshape(pts, [-1])


def _hard_slice(points: tf.Tensor, x: tf.Tensor) -> tf.Tensor:
    # x: [B,S,N]
    dist = tf.abs(x[..., None] - points[None, None, None, :]) ** 2
    idx = tf.argmin(dist, axis=-1, output_type=tf.int32)
    return tf.gather(points, idx)


def _eval_candidate(detector, y_tilde, L_sym, K_eff, poles, x_d_true):
    cand = detector._candidate(y_tilde, L_sym, K_eff, poles)
    x_hat_d = cand["x_hat_d"]
    mse = tf.reduce_mean(tf.square(tf.abs(x_hat_d - x_d_true)), axis=[1, 2])
    xpow = tf.reduce_mean(tf.square(tf.abs(x_d_true)), axis=[1, 2])
    evm = mse / tf.maximum(xpow, tf.cast(1e-9, detector.rdtype))

    pts = _build_points(detector)
    x_hard = _hard_slice(pts, x_hat_d)
    ser = tf.reduce_mean(tf.cast(tf.not_equal(x_hard, x_d_true), detector.rdtype), axis=[1, 2])

    return {
        "pilot_mse": cand["pilot_mse"],
        "sigma2_eff": cand["sigma2_eff"],
        "data_mse": mse,
        "data_evm": evm,
        "hard_ser": ser,
        "x_hat_d": x_hat_d,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ebno_db", type=float, required=True)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_batches", type=int, default=10)
    ap.add_argument("--weights_path", default=None)
    args = ap.parse_args()

    cfg = load_json(args.config)
    ensure_dir(args.out)
    seed_everything(int(cfg.get("seed", 1)))

    link = PuschLink(cfg)
    rx = PoleTransportPuschReceiver(link.tx, cfg, selection_mode="best")
    if args.weights_path:
        _ = rx.maybe_load_policy_weights(args.weights_path)
    else:
        _ = rx.maybe_load_policy_weights()

    det = rx.detector
    records: List[Dict[str, Any]] = []

    for batch_idx in range(args.num_batches):
        batch = link.generate(batch_size=args.batch_size, ebno_db=float(args.ebno_db))

        y_eff = det._remove_nulled_subcarriers(batch.y)
        L_sym = tf.shape(y_eff)[2]
        K_eff = tf.shape(y_eff)[3]
        y_flat = det._flatten_grid(y_eff)
        y_tilde, _ = det._whiten(y_flat)
        summary = det._pilot_summary(y_tilde, batch.no)
        cond_poles = det.policy(summary)

        x_d_true = det.extract_data_symbols_from_x(batch.x)

        static = _eval_candidate(det, y_tilde, L_sym, K_eff, det.static_poles, x_d_true)
        cond = _eval_candidate(det, y_tilde, L_sym, K_eff, cond_poles, x_d_true)

        use_cond, x_hat_sel, sigma2_sel, pilot_mse_sel = det._select_candidate(static, cond, "best")
        pts = _build_points(det)
        x_hard_sel = _hard_slice(pts, x_hat_sel)
        data_mse_sel = tf.reduce_mean(tf.square(tf.abs(x_hat_sel - x_d_true)), axis=[1, 2])
        data_evm_sel = data_mse_sel / tf.maximum(
            tf.reduce_mean(tf.square(tf.abs(x_d_true)), axis=[1, 2]),
            tf.cast(1e-9, det.rdtype),
        )
        hard_ser_sel = tf.reduce_mean(tf.cast(tf.not_equal(x_hard_sel, x_d_true), det.rdtype), axis=[1, 2])

        rec = {
            "batch_idx": batch_idx,
            "summary_mean": tf.reduce_mean(summary, axis=0).numpy().astype(float).tolist(),
            "use_conditional_rate": float(tf.reduce_mean(tf.cast(use_cond, det.rdtype)).numpy()),
            "pilot_mse_static_mean": float(tf.reduce_mean(static["pilot_mse"]).numpy()),
            "pilot_mse_cond_mean": float(tf.reduce_mean(cond["pilot_mse"]).numpy()),
            "pilot_mse_selected_mean": float(tf.reduce_mean(pilot_mse_sel).numpy()),
            "sigma2_selected_mean": float(tf.reduce_mean(sigma2_sel).numpy()),
            "data_mse_static_mean": float(tf.reduce_mean(static["data_mse"]).numpy()),
            "data_mse_cond_mean": float(tf.reduce_mean(cond["data_mse"]).numpy()),
            "data_mse_selected_mean": float(tf.reduce_mean(data_mse_sel).numpy()),
            "data_evm_static_mean": float(tf.reduce_mean(static["data_evm"]).numpy()),
            "data_evm_cond_mean": float(tf.reduce_mean(cond["data_evm"]).numpy()),
            "data_evm_selected_mean": float(tf.reduce_mean(data_evm_sel).numpy()),
            "hard_ser_static_mean": float(tf.reduce_mean(static["hard_ser"]).numpy()),
            "hard_ser_cond_mean": float(tf.reduce_mean(cond["hard_ser"]).numpy()),
            "hard_ser_selected_mean": float(tf.reduce_mean(hard_ser_sel).numpy()),
        }
        records.append(rec)
        print(
            f"[probe] batch={batch_idx} "
            f"use_cond={rec['use_conditional_rate']:.3f} "
            f"pilot_mse_sel={rec['pilot_mse_selected_mean']:.4e} "
            f"evm_sel={rec['data_evm_selected_mean']:.4e} "
            f"ser_sel={rec['hard_ser_selected_mean']:.4e}",
            flush=True,
        )

    # Aggregate
    keys = [
        "use_conditional_rate",
        "pilot_mse_static_mean",
        "pilot_mse_cond_mean",
        "pilot_mse_selected_mean",
        "sigma2_selected_mean",
        "data_mse_static_mean",
        "data_mse_cond_mean",
        "data_mse_selected_mean",
        "data_evm_static_mean",
        "data_evm_cond_mean",
        "data_evm_selected_mean",
        "hard_ser_static_mean",
        "hard_ser_cond_mean",
        "hard_ser_selected_mean",
    ]
    agg = {k: float(np.mean([r[k] for r in records])) for k in keys}
    out = {
        "config_path": os.path.abspath(args.config),
        "ebno_db": float(args.ebno_db),
        "batch_size": int(args.batch_size),
        "num_batches": int(args.num_batches),
        "records": records,
        "aggregate": agg,
    }
    out_path = os.path.join(args.out, f"probe_ebno_{args.ebno_db:+.2f}.json".replace("+", "p").replace("-", "m"))
    save_json(out, out_path)
    print("Saved:", out_path, flush=True)


if __name__ == "__main__":
    main()
