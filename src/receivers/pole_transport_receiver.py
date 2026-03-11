from __future__ import annotations

from typing import Any, Dict, Optional

import tensorflow as tf

from .pole_transport_detector import PoleTransportDetector


def _import_nr_blocks():
    try:
        from sionna.phy.nr import LayerDemapper, TBDecoder

        return LayerDemapper, TBDecoder
    except Exception:
        from sionna.nr import LayerDemapper, TBDecoder

        return LayerDemapper, TBDecoder


def _get_num_bits_per_symbol(pusch_transmitter: Any) -> int:
    n = getattr(pusch_transmitter, "_num_bits_per_symbol", None)
    if n is not None:
        return int(n)
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


class PoleTransportPuschReceiver(tf.keras.layers.Layer):
    """Full PUSCH receiver wrapper for the pole-transport detector."""

    def __init__(self, pusch_transmitter: Any, cfg: Dict[str, Any], selection_mode: str = "best"):
        super().__init__(name=f"pole_transport_pusch_receiver_{selection_mode}")
        LayerDemapper, TBDecoder = _import_nr_blocks()
        self.tx = pusch_transmitter
        self.rg = self.tx.resource_grid
        self.selection_mode = str(selection_mode).lower().strip()

        precision = str(cfg.get("precision", "single")).lower()
        seed = int(cfg.get("seed", 1))
        num_rx_ant = int(cfg.get("channel", {}).get("num_rx_ant", 4))
        q_m = _get_num_bits_per_symbol(self.tx)
        pt_cfg = cfg.get("pole_transport", {})
        self.detector = PoleTransportDetector(
            resource_grid=self.rg,
            num_rx_ant=num_rx_ant,
            num_bits_per_symbol=q_m,
            cfg=pt_cfg,
            precision=precision,
            seed=seed,
        )

        layer_mapper = getattr(self.tx, "_layer_mapper", None)
        if layer_mapper is None:
            layer_mapper = getattr(self.tx, "layer_mapper", None)
        if layer_mapper is None:
            raise RuntimeError("Could not access layer mapper from PUSCHTransmitter")
        self._layer_demapper = LayerDemapper(layer_mapper, q_m)

        tb_encoder = getattr(self.tx, "_tb_encoder", None)
        if tb_encoder is None:
            tb_encoder = getattr(self.tx, "tb_encoder", None)
        if tb_encoder is None:
            raise RuntimeError("Could not access TB encoder from PUSCHTransmitter")
        try:
            self._tb_decoder = TBDecoder(tb_encoder)
        except TypeError:
            self._tb_decoder = TBDecoder(tb_encoder, output_dtype=tf.int32)

    def maybe_load_policy_weights(self, path: Optional[str] = None) -> bool:
        return self.detector.policy.load_external_weights_if_available(path)

    def call(self, y: tf.Tensor, no: tf.Tensor) -> tf.Tensor:
        llr_streams = self.detector(y, no, selection=self.selection_mode)
        llr_layers = self._layer_demapper(llr_streams)
        out = self._tb_decoder(llr_layers)
        b_hat = out[0] if isinstance(out, (tuple, list)) else out
        if b_hat.shape.rank == 2:
            b_hat = b_hat[:, None, :]
        elif b_hat.shape.rank == 1:
            b_hat = b_hat[None, None, :]
        return b_hat
