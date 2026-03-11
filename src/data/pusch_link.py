"""PUSCH link generation using Sionna.

This module provides a thin wrapper around:
- PUSCHTransmitter
- OFDMChannel / TimeChannel with 3GPP 38.901 scenarios

It is intentionally conservative and uses try/except imports to support
both older and newer Sionna releases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf


def _import_nr():
    try:
        from sionna.phy.nr import PUSCHConfig, PUSCHTransmitter
        return PUSCHConfig, PUSCHTransmitter
    except Exception:
        from sionna.nr import PUSCHConfig, PUSCHTransmitter
        return PUSCHConfig, PUSCHTransmitter


def _import_channel():
    """Return channel-related classes with version fallback."""
    try:
        from sionna.phy.channel import (
            OFDMChannel,
            TimeChannel,
            RayleighBlockFading,
            gen_single_sector_topology as gen_topology,
        )
        from sionna.phy.channel.tr38901 import AntennaArray, UMi, UMa, RMa
        return OFDMChannel, TimeChannel, RayleighBlockFading, gen_topology, AntennaArray, UMi, UMa, RMa
    except Exception:
        from sionna.channel import (
            OFDMChannel,
            TimeChannel,
            RayleighBlockFading,
            gen_single_sector_topology as gen_topology,
        )
        from sionna.channel.tr38901 import AntennaArray, UMi, UMa, RMa
        return OFDMChannel, TimeChannel, RayleighBlockFading, gen_topology, AntennaArray, UMi, UMa, RMa


def _import_utils():
    try:
        from sionna.phy.utils import ebnodb2no
        return ebnodb2no
    except Exception:
        from sionna.utils import ebnodb2no
        return ebnodb2no


def _try_set_attr(obj: Any, name: str, value: Any) -> bool:
    """Best-effort attribute setter.

    Some Sionna config fields are exposed as read-only properties. In this case,
    ``setattr`` raises an exception. We silently ignore such fields.

    Returns True if the attribute existed and was set successfully.
    """
    if not hasattr(obj, name):
        return False
    try:
        setattr(obj, name, value)
        return True
    except Exception:
        return False


def _try_set_nested(obj: Any, path: str, value: Any) -> bool:
    """Set nested attribute if all prefixes exist.

    Returns True if set, False if any attribute missing.
    """
    parts = path.split(".")
    cur = obj
    for p in parts[:-1]:
        if not hasattr(cur, p):
            return False
        cur = getattr(cur, p)
    if not hasattr(cur, parts[-1]):
        return False
    try:
        setattr(cur, parts[-1], value)
        return True
    except Exception:
        return False


def _normalize_subcarrier_spacing(value: Any) -> int:
    """Normalize subcarrier spacing for Sionna NR.

    Sionna validates the subcarrier spacing against the discrete set
    {15, 30, 60, 120, 240, 480, 960} and expects the unit to be kHz.

    In practice, configuration files are often written in Hz (e.g., 30000).
    This helper accepts both conventions:
      - if ``value >= 1000`` we interpret it as Hz and convert to kHz;
      - otherwise we interpret it as kHz.
    """
    scs = float(value)
    if scs >= 1000.0:
        scs = scs / 1000.0

    scs_int = int(round(scs))
    if abs(scs - scs_int) > 1e-6:
        raise ValueError(
            "Invalid subcarrier spacing. Provide an integer in kHz (e.g., 30) "
            "or in Hz (e.g., 30000). Got: " + str(value)
        )

    allowed = [15, 30, 60, 120, 240, 480, 960]
    if scs_int not in allowed:
        raise ValueError(
            f"Invalid subcarrier spacing {scs_int} kHz. Sionna allows {allowed}. "
            "Provide e.g. 30 (kHz) or 30000 (Hz)."
        )
    return scs_int


def _make_ut_array(AntennaArray: Any, num_tx_ant: int, carrier_frequency_hz: float) -> Any:
    """Create a UT antenna array compatible with ``num_tx_ant``.

    For uplink PUSCH, a single-polarized UT array is often sufficient.
    If ``num_tx_ant`` is even, we use a compact dual-polarized array.
    """
    if num_tx_ant <= 1:
        return AntennaArray(
            num_rows=1,
            num_cols=1,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=carrier_frequency_hz,
        )

    if (num_tx_ant % 2) == 0:
        return AntennaArray(
            num_rows=1,
            num_cols=int(num_tx_ant // 2),
            polarization="dual",
            polarization_type="cross",
            antenna_pattern="omni",
            carrier_frequency=carrier_frequency_hz,
        )

    # Fallback for odd antenna counts > 1
    return AntennaArray(
        num_rows=1,
        num_cols=int(num_tx_ant),
        polarization="single",
        polarization_type="V",
        antenna_pattern="omni",
        carrier_frequency=carrier_frequency_hz,
    )


def _make_bs_array(AntennaArray: Any, num_rx_ant: int, carrier_frequency_hz: float) -> Any:
    """Create a BS antenna array compatible with ``num_rx_ant``."""
    if (num_rx_ant % 2) == 0:
        return AntennaArray(
            num_rows=1,
            num_cols=int(num_rx_ant // 2),
            polarization="dual",
            polarization_type="cross",
            antenna_pattern="38.901",
            carrier_frequency=carrier_frequency_hz,
        )

    # Fallback for odd antenna counts
    return AntennaArray(
        num_rows=1,
        num_cols=int(num_rx_ant),
        polarization="single",
        polarization_type="V",
        antenna_pattern="38.901",
        carrier_frequency=carrier_frequency_hz,
    )


@dataclass
class LinkBatch:
    x: tf.Tensor  # transmitted resource grid (frequency-domain)
    b: tf.Tensor  # information bits
    y: tf.Tensor  # received grid
    h: Optional[tf.Tensor]  # channel (if available)
    no: tf.Tensor  # noise variance


class PuschLink:
    """End-to-end PUSCH link: transmitter + channel."""

    def __init__(self, cfg: Dict[str, Any]):
        PUSCHConfig, PUSCHTransmitter = _import_nr()
        OFDMChannel, TimeChannel, RayleighBlockFading, gen_topology, AntennaArray, UMi, UMa, RMa = _import_channel()
        self._ebnodb2no = _import_utils()

        self.cfg = cfg
        pusch_cfg = cfg.get("pusch", {})
        chan_cfg = cfg.get("channel", {})

        self.num_tx = int(pusch_cfg.get("num_tx", 1))
        self.num_layers = int(pusch_cfg.get("num_layers", 1))
        # In Sionna NR (e.g., v0.18.0), `num_layers` must be equal to `num_antenna_ports`.
        # We default to `num_layers` if `num_antenna_ports` is not provided.
        self.num_tx_ant = int(pusch_cfg.get("num_antenna_ports", self.num_layers))
        if self.num_tx_ant != self.num_layers:
            raise ValueError(
                "Invalid PUSCH configuration: 'num_antenna_ports' must equal 'num_layers' "
                f"for Sionna NR. Got num_layers={self.num_layers}, num_antenna_ports={self.num_tx_ant}."
            )
        self.num_rx_ant = int(chan_cfg.get("num_rx_ant", 4))

        # Build one PUSCHConfig per transmitter
        self.pusch_configs: List[Any] = []
        for tx in range(self.num_tx):
            c = PUSCHConfig()

            # Basic MU settings
            _try_set_attr(c, "num_layers", self.num_layers)
            _try_set_attr(c, "num_antenna_ports", self.num_tx_ant)

            # If the config exposes scrambling IDs, make them different across users
            # (best-effort; does nothing if attributes do not exist)
            _try_set_attr(c, "n_rnti", 1000 + tx)
            _try_set_attr(c, "n_id", 1 + tx)

            # TB settings (best-effort)
            tb_cfg = pusch_cfg.get("tb", {})
            if isinstance(tb_cfg, dict):
                if "mcs_index" in tb_cfg:
                    _try_set_nested(c, "tb.mcs_index", int(tb_cfg["mcs_index"]))

                        # ------------------------------------------------------------
            # DMRS settings (best-effort)
            # ------------------------------------------------------------
            # Sionna exposes PUSCH DMRS configuration through `c.dmrs.*`,
            # e.g., length (1/2) and additional_position (0..3). :contentReference[oaicite:4]{index=4}
            dmrs_cfg = pusch_cfg.get("dmrs", {})
            if isinstance(dmrs_cfg, dict) and dmrs_cfg:
                any_set = False

                if "config_type" in dmrs_cfg:
                    any_set |= _try_set_nested(c, "dmrs.config_type", int(dmrs_cfg["config_type"]))
                if "length" in dmrs_cfg:
                    any_set |= _try_set_nested(c, "dmrs.length", int(dmrs_cfg["length"]))
                if "additional_position" in dmrs_cfg:
                    any_set |= _try_set_nested(c, "dmrs.additional_position", int(dmrs_cfg["additional_position"]))
                if "num_cdm_groups_without_data" in dmrs_cfg:
                    any_set |= _try_set_nested(
                        c, "dmrs.num_cdm_groups_without_data", int(dmrs_cfg["num_cdm_groups_without_data"])
                    )
                if "type_a_position" in dmrs_cfg:
                    any_set |= _try_set_nested(c, "dmrs.type_a_position", int(dmrs_cfg["type_a_position"]))
                if "dmrs_port_set" in dmrs_cfg:
                    any_set |= _try_set_nested(c, "dmrs.dmrs_port_set", [int(x) for x in dmrs_cfg["dmrs_port_set"]])

                if not any_set:
                    raise ValueError(
                        "You provided pusch.dmrs in the JSON, but none of the DMRS fields could be set on "
                        "PUSCHConfig. Check your Sionna version/API."
                    )

            # Keep track of requested resource parameters to avoid running with
            # unintended defaults if a field is read-only in a given Sionna version.
            target_scs_khz = None
            target_n_prbs = None

            # Resource settings (best-effort; attribute names depend on Sionna version)
            res_cfg = pusch_cfg.get("resource", {})
            if isinstance(res_cfg, dict):
                if "subcarrier_spacing" in res_cfg:
                    # Common in Sionna: carrier.subcarrier_spacing (unit: kHz)
                    scs_khz = _normalize_subcarrier_spacing(res_cfg["subcarrier_spacing"])
                    _try_set_nested(c, "carrier.subcarrier_spacing", scs_khz)
                    target_scs_khz = float(scs_khz)
                if "num_prbs" in res_cfg:
                    # Try a few likely locations
                    n_prbs = int(res_cfg["num_prbs"])
                    target_n_prbs = int(n_prbs)
                    # IMPORTANT (Sionna 0.18): `num_resource_blocks` / `num_prbs` can be a
                    # derived (read-only) property. The stable, settable knobs are the
                    # bandwidth part (BWP) and carrier grid sizes.

                    # Top-level (PUSCHConfig)
                    _try_set_attr(c, "n_size_bwp", n_prbs)
                    _try_set_attr(c, "n_start_bwp", 0)

                    # CarrierConfig (often settable)
                    _try_set_nested(c, "carrier.n_size_grid", n_prbs)
                    _try_set_nested(c, "carrier.n_start_grid", 0)

                    # Alternative BWP nesting (older/other variants)
                    _try_set_nested(c, "bwp.n_size_bwp", n_prbs)
                    _try_set_nested(c, "bwp.n_start_bwp", 0)

            # Let the config validate and update derived quantities if available
            try:
                c.check_config()
            except Exception:
                pass

            # Detect silent configuration mismatches early.
            if target_scs_khz is not None:
                try:
                    cur_scs = float(getattr(c.carrier, "subcarrier_spacing"))
                except Exception:
                    cur_scs = None
                if (cur_scs is not None) and (cur_scs != float(target_scs_khz)):
                    raise ValueError(
                        f"Requested subcarrier spacing {target_scs_khz} kHz, but got {cur_scs} kHz. "
                        "Please check your Sionna version and config keys."
                    )

            if target_n_prbs is not None:
                # `n_size_bwp` is the most stable knob across Sionna NR releases.
                cur_n = getattr(c, "n_size_bwp", None)
                if cur_n is not None:
                    try:
                        cur_n_int = int(cur_n)
                    except Exception:
                        cur_n_int = None
                    if (cur_n_int is not None) and (cur_n_int != int(target_n_prbs)):
                        raise ValueError(
                            f"Requested num_prbs={target_n_prbs}, but PUSCHConfig.n_size_bwp={cur_n_int}. "
                            "Please check that `n_size_bwp` is settable in your Sionna version."
                        )

            self.pusch_configs.append(c)

        # Build transmitter (single-user or multi-user)
        if len(self.pusch_configs) == 1:
            try:
                self.tx = PUSCHTransmitter(self.pusch_configs[0])
            except ValueError as e:
                msg = str(e)
                if ("coderate" in msg.lower()) or ("repetition" in msg.lower()):
                    raise ValueError(
                        "Failed to build PUSCHTransmitter due to an unsupported LDPC code rate. "
                        "This can happen for low MCS indices (e.g., MCS 10 in table 1 has target "
                        "coderate 340/1024 \u2248 0.332 < 1/3). "
                        "Fix: increase `pusch.tb.mcs_index` (try 11 or higher), or adjust the "
                        "transport block configuration to avoid very low code rates. "
                        f"Original error: {msg}"
                    ) from e
                raise
        else:
            try:
                self.tx = PUSCHTransmitter(self.pusch_configs)
            except Exception as e:
                raise RuntimeError(
                    "This Sionna version does not accept PUSCHTransmitter(list_of_configs). "
                    "Set pusch.num_tx=1 and use pusch.num_layers>1 to emulate MU/MIMO streams."
                ) from e
        self.rg = self.tx.resource_grid

        # Channel settings
        self.domain = str(chan_cfg.get("domain", "freq")).lower()
        self.scenario = str(chan_cfg.get("scenario", "Rayleigh"))
        self.fc = float(chan_cfg.get("carrier_frequency_hz", 3.5e9))
        self.speed = float(chan_cfg.get("speed_mps", 0.0))
        self.enable_pathloss = bool(chan_cfg.get("enable_pathloss", False))
        self.enable_shadow_fading = bool(chan_cfg.get("enable_shadow_fading", False))

        # Optional OFDM impairments (research)
        #
        # These are meant to *intentionally* break RE orthogonality (create ICI/ISI),
        # so the convolutional model in Explanable_AI.pdf is exercised.
        #
        # Config example:
        #   "channel": {
        #     ...,
        #     "impairments": {"cfo_normalized": 0.05}
        #   }
        #
        # `cfo_normalized` is normalized to subcarrier spacing (i.e., 1.0 means one
        # full subcarrier offset). Use small values like 0.01...0.2.
        imp_cfg = chan_cfg.get("impairments", {})
        self.cfo_normalized = float(chan_cfg.get("cfo_normalized", imp_cfg.get("cfo_normalized", 0.0)))

        self._gen_topology = gen_topology

        def _ofdm_channel(model: Any, rg: Any, add_awgn: bool, normalize_channel: bool, return_channel: bool) -> Any:
            """Construct OFDMChannel with version-safe keyword arguments."""
            try:
                return OFDMChannel(
                    model,
                    rg,
                    add_awgn=add_awgn,
                    normalize_channel=normalize_channel,
                    return_channel=return_channel,
                )
            except TypeError:
                # Older versions may not expose `add_awgn`
                return OFDMChannel(
                    model,
                    rg,
                    normalize_channel=normalize_channel,
                    return_channel=return_channel,
                )

        def _time_channel(model: Any, bandwidth: Any, num_time_samples: Any, add_awgn: bool, normalize_channel: bool, return_channel: bool) -> Any:
            """Construct TimeChannel with version-safe keyword arguments."""
            try:
                return TimeChannel(
                    model,
                    bandwidth,
                    num_time_samples,
                    add_awgn=add_awgn,
                    normalize_channel=normalize_channel,
                    return_channel=return_channel,
                )
            except TypeError:
                return TimeChannel(
                    model,
                    bandwidth,
                    num_time_samples,
                    normalize_channel=normalize_channel,
                    return_channel=return_channel,
                )

        # Build channel model
        if self.scenario.lower() == "rayleigh":
            rayleigh = RayleighBlockFading(
                num_rx=1,
                num_rx_ant=self.num_rx_ant,
                num_tx=self.rg.num_tx,
                num_tx_ant=self.num_tx_ant,
            )
            # Always return channel for perfect-CSI baselines
            self.channel = _ofdm_channel(
                rayleigh,
                self.rg,
                add_awgn=True,
                normalize_channel=True,
                return_channel=True,
            )
            self._channel_model = rayleigh
            self._needs_topology = False
        else:
            # Configure antenna arrays for 3GPP TR 38.901 system-level channels.
            #
            # Sionna's TR 38.901 scenarios require choosing an outdoor-to-indoor (O2I)
            # penetration loss model for indoor UTs. We expose it as a lightweight
            # config knob, defaulting to the low-loss model.
            o2i_model = str(chan_cfg.get("o2i_model", "low")).lower()
            if o2i_model not in ["low", "high"]:
                raise ValueError(
                    "Invalid channel.o2i_model. Use 'low' or 'high'. Got: " + str(o2i_model)
                )

            ut_array = _make_ut_array(AntennaArray, self.num_tx_ant, self.fc)
            bs_array = _make_bs_array(AntennaArray, self.num_rx_ant, self.fc)

            def _make_tr38901(ModelCls: Any) -> Any:
                """Instantiate a TR38.901 scenario with backward compatibility."""
                try:
                    return ModelCls(
                        carrier_frequency=self.fc,
                        o2i_model=o2i_model,
                        ut_array=ut_array,
                        bs_array=bs_array,
                        direction="uplink",
                        enable_pathloss=self.enable_pathloss,
                        enable_shadow_fading=self.enable_shadow_fading,
                    )
                except TypeError:
                    # Older Sionna versions did not require o2i_model
                    return ModelCls(
                        carrier_frequency=self.fc,
                        ut_array=ut_array,
                        bs_array=bs_array,
                        direction="uplink",
                        enable_pathloss=self.enable_pathloss,
                        enable_shadow_fading=self.enable_shadow_fading,
                    )

            scen = self.scenario.lower()
            if scen == "umi":
                channel_model = _make_tr38901(UMi)
            elif scen == "uma":
                channel_model = _make_tr38901(UMa)
            elif scen == "rma":
                channel_model = _make_tr38901(RMa)
            else:
                raise ValueError(f"Unknown scenario: {self.scenario}. Use Rayleigh/UMi/UMa/RMa")

            self._channel_model = channel_model
            if self.domain == "freq":
                self.channel = _ofdm_channel(
                    channel_model,
                    self.rg,
                    add_awgn=True,
                    normalize_channel=True,
                    return_channel=True,
                )
            else:
                # Time-domain simulation (optional)
                # NOTE: the TimeChannel needs bandwidth and num_time_samples
                self.channel = _time_channel(
                    channel_model,
                    self.rg.bandwidth,
                    self.rg.num_time_samples,
                    add_awgn=True,
                    normalize_channel=True,
                    return_channel=True,
                )
            self._needs_topology = True

        # Cache these for noise conversion
        self._num_bits_per_symbol = getattr(self.tx, "_num_bits_per_symbol", None)
        self._target_coderate = getattr(self.tx, "_target_coderate", None)

    def new_topology(self, batch_size: int) -> None:
        if not self._needs_topology:
            return
        topology = self._gen_topology(
            batch_size,
            self.num_tx,
            self.scenario.lower(),
            min_ut_velocity=self.speed,
            max_ut_velocity=self.speed,
        )
        self._channel_model.set_topology(*topology)

    def ebnodb_to_no(self, ebno_db: float) -> tf.Tensor:
        # Prefer transmitter-provided values
        num_bits_per_symbol = self._num_bits_per_symbol
        target_coderate = self._target_coderate

        # Fallbacks (best effort)
        if num_bits_per_symbol is None:
            # Try PUSCHConfig.tb.num_bits_per_symbol
            try:
                num_bits_per_symbol = int(self.pusch_configs[0].tb.num_bits_per_symbol)
            except Exception:
                num_bits_per_symbol = 2

        if target_coderate is None:
            try:
                target_coderate = float(self.pusch_configs[0].tb.target_coderate)
            except Exception:
                target_coderate = 0.5

        return self._ebnodb2no(
            tf.constant(ebno_db, tf.float32),
            num_bits_per_symbol,
            target_coderate,
            self.rg,
        )
    




    def _apply_cfo(self, y: tf.Tensor) -> tf.Tensor:
        """Apply a synthetic carrier-frequency offset (CFO) impairment.

        Implemented by:
          y_freq --IFFT--> y_time --phase ramp--> y_time' --FFT--> y_freq'

        This creates structured inter-carrier interference (ICI), i.e., RE leakage
        along the subcarrier axis.
        """
        eps = float(getattr(self, "cfo_normalized", 0.0))
        if abs(eps) < 1e-12:
            return y

        cdtype = y.dtype
        K = tf.shape(y)[-1]
        n = tf.cast(tf.range(K), tf.float32)

        # Phase ramp exp(j*2*pi*eps*n/K)
                # IMPORTANT: Avoid Python complex scalars (1j) here. If you do
        # (complex_scalar * float_tensor), TF tries to cast the complex scalar
        # to float32 and throws:
        #   TypeError: Cannot convert ...j to EagerTensor of dtype float
        theta = (2.0 * np.pi * tf.cast(eps, tf.float32)) * n / tf.cast(K, tf.float32)  # [K], float
        phase = tf.exp(tf.complex(tf.zeros_like(theta), theta))  # exp(j*theta), [K] complex
        phase = tf.cast(phase, cdtype)

        y_time = tf.signal.ifft(tf.cast(y, cdtype))
        y_time = y_time * phase[None, None, None, None, :]
        y_freq = tf.signal.fft(y_time)
        return y_freq



    def generate(self, batch_size: int, ebno_db: float) -> LinkBatch:
        """Generate one batch of slots."""
        self.new_topology(batch_size)
        x, b = self.tx(batch_size)
        no = self.ebnodb_to_no(float(ebno_db))
        # Channel call style depends on Sionna version.
        # Many releases implement `call(self, inputs)` where inputs is a list/tuple.
        try:
            out_ch = self.channel([x, no])
        except Exception:
            out_ch = self.channel(x, no)
        
        if isinstance(out_ch, (tuple, list)) and len(out_ch) >= 2:
            y, h = out_ch[0], out_ch[1]
        else:
            y, h = out_ch, None

        # Apply optional impairments (e.g., CFO) to create RE coupling
        y = self._apply_cfo(y)

        return LinkBatch(x=x, b=b, y=y, h=h, no=no)
