"""Train the tiny pole-policy network for the pole-transport receiver.

This script does *not* train a full neural receiver. It trains only the tiny
slot-conditioned pole policy, while the per-TTI ridge readout remains learned
online from current pilots.

Usage:
    python -m src.experiments.train_pole_transport_policy \
        --config configs/pole_transport_umino_cfo05_singlelayer.json \
        --out checkpoints/pole_transport
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from typing import Any, Dict, List

import tensorflow as tf

from src.data.pusch_link import PuschLink
from src.receivers.pole_transport_receiver import PoleTransportPuschReceiver
from src.utils.config import load_json
from src.utils.io import ensure_dir, timestamp
from src.utils.seed import seed_everything


def _force_learned_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    c = copy.deepcopy(cfg)
    c.setdefault("pole_transport", {})
    c["pole_transport"].setdefault("policy", {})
    c["pole_transport"].setdefault("receiver", {})
    c["pole_transport"]["policy"]["mode"] = "learned"
    c["pole_transport"]["receiver"]["selection"] = "conditional"
    return c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = load_json(args.config)
    cfg = _force_learned_cfg(cfg)
    ensure_dir(args.out)
    seed_everything(int(cfg.get("seed", 1)))

    train_cfg = cfg.get("train", {})
    steps = int(train_cfg.get("steps", 1500))
    batch_size = int(train_cfg.get("batch_size", 16))
    ebno_list = list(train_cfg.get("ebno_db_list", cfg.get("sim", {}).get("ebno_db_list", [0, 2, 4, 6, 8])))
    learning_rate = float(train_cfg.get("learning_rate", 3e-3))
    weight_decay = float(train_cfg.get("weight_decay", 1e-5))
    save_every = int(train_cfg.get("save_every", 100))
    log_every = int(train_cfg.get("log_every", 25))
    alpha_pilot = float(train_cfg.get("alpha_pilot", 0.05))

    weights_path = cfg.get("pole_transport", {}).get("policy", {}).get("weights_path", None)
    if not weights_path:
        weights_path = os.path.join(args.out, "pole_policy.weights.h5")
        cfg["pole_transport"]["policy"]["weights_path"] = weights_path
    elif not os.path.isabs(weights_path):
        weights_path = os.path.join(os.getcwd(), weights_path)
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)

    link = PuschLink(cfg)
    receiver = PoleTransportPuschReceiver(link.tx, cfg, selection_mode="conditional")

    # Build variables once.
    warm_batch = link.generate(batch_size=max(1, min(2, batch_size)), ebno_db=float(ebno_list[0]))
    _ = receiver.detector.forward_data(warm_batch.y, warm_batch.no, selection="conditional")

    if args.resume:
        _ = receiver.maybe_load_policy_weights(weights_path)

    trainables = list(receiver.detector.policy.trainable_variables)
    if len(trainables) == 0:
        raise RuntimeError(
            "No trainable variables found. Check that pole_transport.policy.mode is 'learned'."
        )

    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    history: List[Dict[str, Any]] = []

    @tf.function(jit_compile=False)
    def train_step(batch_y, batch_no, batch_x):
        with tf.GradientTape() as tape:
            x_hat_d, aux = receiver.detector.forward_data(batch_y, batch_no, selection="conditional", return_aux=True)
            x_true_d = receiver.detector.extract_data_symbols_from_x(batch_x)
            loss_sym = tf.reduce_mean(tf.square(tf.abs(x_hat_d - x_true_d)))
            loss_pilot = tf.reduce_mean(aux["pilot_mse_cond"])
            l2 = tf.add_n([tf.reduce_sum(tf.square(v)) for v in trainables])
            loss = loss_sym + tf.cast(alpha_pilot, receiver.detector.rdtype) * tf.cast(loss_pilot, receiver.detector.rdtype)
            loss = loss + tf.cast(weight_decay, receiver.detector.rdtype) * tf.cast(l2, receiver.detector.rdtype)
        grads = tape.gradient(loss, trainables)
        opt.apply_gradients(zip(grads, trainables))
        return loss, loss_sym, loss_pilot

    print("=== training pole transport policy ===")
    print("out_dir:", os.path.abspath(args.out))
    print("weights_path:", weights_path)
    print("steps:", steps)
    print("batch_size:", batch_size)
    print("ebno_db_list:", ebno_list)

    for step in range(1, steps + 1):
        ebno_db = float(ebno_list[(step - 1) % len(ebno_list)])
        batch = link.generate(batch_size=batch_size, ebno_db=ebno_db)
        loss, loss_sym, loss_pilot = train_step(batch.y, batch.no, batch.x)
        record = {
            "step": step,
            "ebno_db": ebno_db,
            "loss": float(loss.numpy()),
            "loss_sym": float(loss_sym.numpy()),
            "loss_pilot": float(loss_pilot.numpy()),
        }
        history.append(record)
        if (log_every > 0) and (step % log_every == 0):
            print(
                f"[train] step={step:05d}/{steps} ebno={ebno_db:+.2f}dB "
                f"loss={record['loss']:.4e} sym={record['loss_sym']:.4e} pilot={record['loss_pilot']:.4e}",
                flush=True,
            )
        if (save_every > 0) and (step % save_every == 0):
            receiver.detector.policy.net.save_weights(weights_path)
            with open(os.path.join(args.out, "train_log.json"), "w", encoding="utf-8") as f:
                json.dump({"meta": {"timestamp": timestamp()}, "history": history}, f, indent=2)
            print("[train] checkpoint saved to", weights_path)

    receiver.detector.policy.net.save_weights(weights_path)
    with open(os.path.join(args.out, "train_log.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": {"timestamp": timestamp()}, "history": history}, f, indent=2)
    with open(os.path.join(args.out, "train_config_resolved.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print("Training finished. Final weights written to", weights_path)


if __name__ == "__main__":
    main()
