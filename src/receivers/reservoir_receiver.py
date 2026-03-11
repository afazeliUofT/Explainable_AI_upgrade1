"""Full PUSCH receiver wrapper for the reservoir detector.

Pipeline:
  y -> ReservoirDetector -> llr(streams)
    -> LayerDemapper -> llr(layers)
    -> TBDecoder -> b_hat

This mirrors the relevant parts of Sionna's PUSCHReceiver, but skips explicit
channel estimation and classical MIMO equalization.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import tensorflow as tf

from .reservoir_detector import (
    DDParams,
    LowRankParams,
    PoleAdaptParams,
    ReservoirDetector,
    ReservoirParams,
    SubbandParams,
)


def _import_nr_blocks():
    try:
        from sionna.phy.nr import LayerDemapper, TBDecoder
        return LayerDemapper, TBDecoder
    except Exception:
        from sionna.nr import LayerDemapper, TBDecoder
        return LayerDemapper, TBDecoder


def _get_num_bits_per_symbol(pusch_transmitter: Any) -> int:
    # Prefer transmitter cache
    n = getattr(pusch_transmitter, "_num_bits_per_symbol", None)
    if n is not None:
        return int(n)
    # Try config (attribute names differ across Sionna versions)
    for attr in ["_pusch_config", "pusch_config"]:
        cfg = getattr(pusch_transmitter, attr, None)
        if cfg is not None:
            try:
                return int(cfg.tb.num_bits_per_symbol)
            except Exception:
                pass

    cfgs = getattr(pusch_transmitter, "_pusch_configs", None)
    if cfgs is not None:
        try:
            return int(cfgs[0].tb.num_bits_per_symbol)
        except Exception:
            pass

    return 2


class ReservoirPuschReceiver(tf.keras.layers.Layer):
    def __init__(self, pusch_transmitter: Any, cfg: Dict[str, Any]):
        super().__init__(name="reservoir_pusch_receiver")

        LayerDemapper, TBDecoder = _import_nr_blocks()

        self.tx = pusch_transmitter
        self.rg = self.tx.resource_grid

        # Extract config sections
        ours = cfg.get("ours", {})
        self._llr_scale = float(ours.get("llr_scale", 1.0))
        res_cfg = ours.get("reservoir", {})
        dd_cfg = ours.get("dd", {})
        lr_cfg = ours.get("lowrank", {})
        whit_cfg = ours.get("whitening", {})
        ridge_cfg = ours.get("ridge", {})
        sb_cfg = ours.get("subband", {})

        precision = str(cfg.get("precision", "single")).lower()
        seed = int(cfg.get("seed", 1))

        num_rx_ant = int(cfg.get("channel", {}).get("num_rx_ant", 4))
        q_m = _get_num_bits_per_symbol(self.tx)

        reservoir = ReservoirParams(
            M_f=int(res_cfg.get("M_f", 8)),
            M_t=int(res_cfg.get("M_t", 8)),
            d_f=int(res_cfg.get("d_f", 1)),
            d_t=int(res_cfg.get("d_t", 1)),
            pole_policy=str(res_cfg.get("pole_policy", "logspace")),
            pole_warp=float(res_cfg.get("pole_warp", 4.0)),
            rho_f_min=float(res_cfg.get("rho_f_min", 0.3)),
            rho_f_max=float(res_cfg.get("rho_f_max", 0.98)),
            rho_t_min=float(res_cfg.get("rho_t_min", 0.3)),
            rho_t_max=float(res_cfg.get("rho_t_max", 0.98)),
            disable_time_memory_single_dmrs=bool(res_cfg.get("disable_time_memory_single_dmrs", True)),
        )

        dd = DDParams(
            enabled=bool(dd_cfg.get("enabled", True)),
            alpha_min=float(dd_cfg.get("alpha_min", 0.9)),
            Q_max=int(dd_cfg.get("Q_max", 64)),
            temperature=float(dd_cfg.get("temperature", 1.0)),
            mu=float(dd_cfg.get("mu", 1.0)),
            accept_tol=float(dd_cfg.get("accept_tol", 0.0)),
        )

        lowrank = LowRankParams(
            enabled=bool(lr_cfg.get("enabled", False)),
            R=int(lr_cfg.get("R", 16)),
        )

        # Subbanded readout (frequency selectivity fix)
        size_sc = int(sb_cfg.get("size_sc", 0))
        size_prbs = int(sb_cfg.get("size_prbs", 0))
        if (size_sc <= 0) and (size_prbs > 0):
            size_sc = 12 * size_prbs

        subband = SubbandParams(
            enabled=bool(sb_cfg.get("enabled", False)),
            size_sc=size_sc,
            mode=str(sb_cfg.get("mode", "auto")),
        )

        # Optional: pilot-driven pole adaptation (covariance -> lattice -> poles)
        pa_cfg = ours.get("pole_adapt", {})
        pole_adapt = PoleAdaptParams(
    enabled=bool(pa_cfg.get("enabled", False)),
    adapt_dim=str(pa_cfg.get("adapt_dim", "f")),
    order=int(pa_cfg.get("order", 4)),
    order_f=None if pa_cfg.get("order_f", None) is None else int(pa_cfg.get("order_f")),
    order_t=None if pa_cfg.get("order_t", None) is None else int(pa_cfg.get("order_t")),
    pilot_stream=int(pa_cfg.get("pilot_stream", 0)),
    blend_mode=str(pa_cfg.get("blend_mode", "fixed")),
    blend=float(pa_cfg.get("blend", 0.0)),
    prior_var=float(pa_cfg.get("prior_var", 0.02)),
    min_blend=float(pa_cfg.get("min_blend", 0.0)),
    max_blend=float(pa_cfg.get("max_blend", 1.0)),
    eps=float(pa_cfg.get("eps", 1e-6)),
    max_abs_pole=float(pa_cfg.get("max_abs_pole", 0.999)),
)

        whitening_enabled = bool(whit_cfg.get("enabled", False))
        whitening_epsilon = float(whit_cfg.get("epsilon", 1e-3))

                # Ridge regularization:
        # - `lambda_feat`: regularize only the reservoir/memory branch (paper Eq. (21)/(34))
        # - `lambda_y`: optional regularization of the skip connection D_t
        #              (None => use the same value as lambda_feat)
        ridge_lambda_feat = float(ridge_cfg.get("lambda_feat", ridge_cfg.get("lambda", 1e-3)))
        ridge_lambda_y = ridge_cfg.get("lambda_y", None)
        ridge_lambda_y = None if ridge_lambda_y is None else float(ridge_lambda_y)

        self.detector = ReservoirDetector(
            resource_grid=self.rg,
            num_rx_ant=num_rx_ant,
            num_bits_per_symbol=q_m,
            reservoir=reservoir,
            ridge_lambda=ridge_lambda_feat,
            ridge_lambda_y=ridge_lambda_y,
            dd=dd,
            lowrank=lowrank,
            whitening_enabled=whitening_enabled,
            whitening_epsilon=whitening_epsilon,
            subband=subband,
            pole_adapt=pole_adapt,
            precision=precision,
            seed=seed,
        )

        # These are constructed by the transmitter, but attribute names can differ.
        layer_mapper = getattr(self.tx, "_layer_mapper", None)
        if layer_mapper is None:
            layer_mapper = getattr(self.tx, "layer_mapper", None)
        if layer_mapper is None:
            raise RuntimeError("Could not access layer mapper from PUSCHTransmitter")

        # LayerDemapper signature is stable in recent Sionna versions.
        self._layer_demapper = LayerDemapper(layer_mapper, q_m)

        # TBDecoder signatures vary slightly across versions; be defensive.
        tb_encoder = getattr(self.tx, "_tb_encoder", None)
        if tb_encoder is None:
            tb_encoder = getattr(self.tx, "tb_encoder", None)
        if tb_encoder is None:
            raise RuntimeError("Could not access TB encoder from PUSCHTransmitter")

        try:
            self._tb_decoder = TBDecoder(tb_encoder)
        except TypeError:
            self._tb_decoder = TBDecoder(tb_encoder, output_dtype=tf.int32)

    def call(self, y: tf.Tensor, no: tf.Tensor) -> tf.Tensor:
        llr_streams = self.detector(y, no)
        if self._llr_scale != 1.0:
            llr_streams = llr_streams * tf.cast(self._llr_scale, llr_streams.dtype)
        llr_layers = self._layer_demapper(llr_streams)
        out = self._tb_decoder(llr_layers)
        b_hat = out[0] if isinstance(out, (tuple, list)) else out

        # Ensure shape matches transmitter output for error counting:
        # expected: [B, num_tx, num_bits]
        if b_hat.shape.rank == 2:
            b_hat = b_hat[:, None, :]
        elif b_hat.shape.rank == 1:
            b_hat = b_hat[None, None, :]

        return b_hat
