"""Baselines based on Sionna's built-in :class:`~sionna.nr.PUSCHReceiver`.

Notes for Sionna v0.18:

* The default receiver expects inputs ``[y, no]``.
* A perfect-CSI baseline is enabled by constructing
  ``PUSCHReceiver(..., channel_estimator="perfect")`` and then calling it with
  inputs ``[y, h, no]``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import tensorflow as tf


def _import_nr_receiver():
    try:
        from sionna.phy.nr import PUSCHReceiver
        return PUSCHReceiver
    except Exception:
        from sionna.nr import PUSCHReceiver
        return PUSCHReceiver


def _import_kbest():
    """Try to import KBestDetector. Return None if not available."""
    try:
        from sionna.phy.ofdm import KBestDetector
        return KBestDetector
    except Exception:
        try:
            from sionna.ofdm import KBestDetector
            return KBestDetector
        except Exception:
            return None


def build_baselines(
    pusch_transmitter: Any,
    cfg: Dict[str, Any],
) -> Dict[str, Callable[[tf.Tensor, tf.Tensor, Optional[tf.Tensor]], tf.Tensor]]:
    """Create baseline receiver callables.

    Returns a dict mapping receiver name -> function(y, no, h_optional) -> b_hat
    """
    PUSCHReceiver = _import_nr_receiver()

    recv_cfg = cfg.get("receivers", {})
        # Accept both:
    #   (A) dict-style receiver config (recommended)
    #   (B) legacy/probe list-of-receiver-names
    if isinstance(recv_cfg, (list, tuple)):
        names = set(recv_cfg)
        recv_cfg = {
            "run_lmmse_ls": ("lmmse_ls" in names),
            "run_lmmse_perfect_csi": ("lmmse_perfect_csi" in names),
            "run_kbest_ls": ("kbest_ls" in names),
        }
    elif recv_cfg is None:
        recv_cfg = {}
    elif not isinstance(recv_cfg, dict):
        raise TypeError(
            f"cfg['receivers'] must be a dict or list, got {type(recv_cfg)}"
        )
    pusch_cfg = cfg.get("pusch", {})

    baselines: Dict[str, Callable[[tf.Tensor, tf.Tensor, Optional[tf.Tensor]], tf.Tensor]] = {}

    def _call_rx(
        rx: Any,
        y: tf.Tensor,
        no: tf.Tensor,
        h: Optional[tf.Tensor],
        *,
        perfect_csi: bool,
    ) -> tf.Tensor:
        """Call a Sionna receiver with best-effort compatibility.

        We primarily support the Sionna v0.18 API:

        * default: ``rx([y, no])``
        * perfect CSI: ``rx([y, h, no])`` when constructed with
          ``channel_estimator="perfect"``.

        For robustness across versions, we try both list- and positional-style
        calls. We avoid silently swapping argument order.
        """
        if perfect_csi:
            if h is None:
                raise ValueError("Perfect-CSI baseline requires channel h")
            inputs = [y, h, no]
            try:
                out = rx(inputs)
            except TypeError:
                out = rx(y, h, no)
        else:
            inputs = [y, no]
            try:
                out = rx(inputs)
            except TypeError:
                out = rx(y, no)

        # Some blocks may return a tuple/list; the decoded bits are conventionally first.
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out

    # Default receiver: LS channel estimation + linear detector (LMMSE)
    if bool(recv_cfg.get("run_lmmse_ls", True)):
        # IMPORTANT: bind the receiver object into the closure.
        #
        # Python closures capture variables by reference. If we reuse the same
        # variable name (e.g., `rx`) for different baselines below, then the
        # earlier function would accidentally reference the *last* assigned
        # receiver (e.g., the perfect-CSI receiver). We therefore bind `rx` as
        # a default argument.
        rx = PUSCHReceiver(pusch_transmitter)
        rx_ls = rx

        def fn(
            y: tf.Tensor,
            no: tf.Tensor,
            h: Optional[tf.Tensor] = None,
            rx: Any = rx_ls,
        ) -> tf.Tensor:
            # LS/estimated CSI (uses DM-RS internally)
            return _call_rx(rx, y, no, h, perfect_csi=False)

        baselines["lmmse_ls"] = fn

    # Perfect CSI baseline
    # Sionna v0.18 supports this through channel_estimator="perfect" and the
    # input tuple (y, h, no). We keep it optional because older versions might
    # not support it.
    if bool(recv_cfg.get("run_lmmse_perfect_csi", True)):
        try:
            rx = PUSCHReceiver(pusch_transmitter, channel_estimator="perfect")
        except TypeError:
            print(
                "[baselines] This Sionna install does not support channel_estimator=\"perfect\". "
                "Skipping lmmse_perfect_csi."
            )
            rx = None

        if rx is not None:
            rx_p = rx

            def fn(
                y: tf.Tensor,
                no: tf.Tensor,
                h: Optional[tf.Tensor] = None,
                rx: Any = rx_p,
            ) -> tf.Tensor:
                return _call_rx(rx, y, no, h, perfect_csi=True)

            baselines["lmmse_perfect_csi"] = fn

    # Optional KBest baseline
    if bool(recv_cfg.get("run_kbest_ls", False)):
        KBestDetector = _import_kbest()
        if KBestDetector is None:
            print("[baselines] KBestDetector not available in this Sionna install. Skipping kbest_ls.")
        else:
            # Best-effort: infer number of streams and bits per symbol from transmitter
            rg = pusch_transmitter.resource_grid
            num_streams = rg.num_streams_per_tx

            # MCS-derived bits per symbol is stored in pusch_configs[0].tb.num_bits_per_symbol in recent versions,
            # but transmitter also caches _num_bits_per_symbol.
            num_bits_per_symbol = getattr(pusch_transmitter, "_num_bits_per_symbol", None)
            if num_bits_per_symbol is None:
                try:
                    # In Sionna >= 1.0, pusch_transmitter has _pusch_configs
                    cfg0 = pusch_transmitter._pusch_configs[0]
                    num_bits_per_symbol = int(cfg0.tb.num_bits_per_symbol)
                except Exception:
                    num_bits_per_symbol = 2

            # k parameter controls complexity
            k = int(pusch_cfg.get("kbest_k", 8))

            # We need stream management; easiest is to let PUSCHReceiver create it, then reuse.
            tmp_rx = PUSCHReceiver(pusch_transmitter)
            stream_management = getattr(tmp_rx, "_stream_management", None)
            if stream_management is None:
                # Some versions may expose it publicly
                stream_management = getattr(tmp_rx, "stream_management", None)

            if stream_management is None:
                raise RuntimeError("Could not access stream management from PUSCHReceiver; cannot build KBest baseline")

            kbest = KBestDetector(
                "bit",
                num_streams,
                k,
                rg,
                stream_management,
                "qam",
                num_bits_per_symbol,
            )

            rx = PUSCHReceiver(pusch_transmitter, mimo_detector=kbest)
            rx_k = rx

            def fn(
                y: tf.Tensor,
                no: tf.Tensor,
                h: Optional[tf.Tensor] = None,
                rx: Any = rx_k,
            ) -> tf.Tensor:
                return _call_rx(rx, y, no, h, perfect_csi=False)

            baselines["kbest_ls"] = fn

    return baselines
