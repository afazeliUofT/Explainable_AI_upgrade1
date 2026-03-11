"""Reservoir-based semi-blind detector for NR PUSCH.

Implements the core receiver-side learning from `Explanable_AI_modified.tex`:
- frequency/time IIR state banks
- outer-product feature phi or low-rank feature psi
- ridge regression from DM-RS (and optional reliability-selected DATA)

The detector outputs **bit LLRs** shaped like an OFDM detector:
  [batch_size, num_tx, num_streams_per_tx, num_data_symbols*num_bits_per_symbol]

So it can be fed into Sionna's LayerDemapper and TBDecoder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import tensorflow as tf


def _finite_mask(x: tf.Tensor) -> tf.Tensor:
    """
    tf.math.is_finite does NOT support complex tensors.
    For complex, finite iff both real and imag are finite.
    """
    if x.dtype.is_complex:
        return tf.math.is_finite(tf.math.real(x)) & tf.math.is_finite(tf.math.imag(x))
    return tf.math.is_finite(x)


def _try_import_mapping():
    """Try importing Sionna's mapping blocks.

    When available, we use these blocks to ensure that the LLR convention and
    bit ordering are consistent with Sionna's NR transmitter/receiver chain.
    """
    try:
        from sionna.mapping import Constellation, Demapper

        return Constellation, Demapper
    except Exception:
        try:
            from sionna.phy.mapping import Constellation, Demapper

            return Constellation, Demapper
        except Exception:
            return None, None


def _try_import_remove_nulled():
    try:
        from sionna.phy.ofdm import RemoveNulledSubcarriers
        return RemoveNulledSubcarriers
    except Exception:
        try:
            from sionna.ofdm import RemoveNulledSubcarriers
            return RemoveNulledSubcarriers
        except Exception:
            return None


@dataclass
class ReservoirParams:
    M_f: int
    M_t: int
    d_f: int
    d_t: int
    pole_policy: str
    rho_f_min: float
    rho_f_max: float
    rho_t_min: float
    rho_t_max: float

    # If True, we disable the time-memory recursion when there is only one DMRS OFDM symbol.
    # Default keeps legacy behavior. Set False to force time-memory ON in single-DMRS scenarios.
    pole_warp: float = 4.0
    disable_time_memory_single_dmrs: bool = True


@dataclass
class PoleAdaptParams:
    """Pilot-driven adaptive pole selection + fusion with a static prior.

    Supports:
      - adapt_dim: "f", "t", or "ft"
      - blend_mode: "fixed" or "map" (Bayesian/MAP shrinkage)
      - nonzero blending clamp via min_blend/max_blend
    """
    enabled: bool = False

    # Dimension(s) to adapt: "f", "t", or "ft".
    adapt_dim: str = "f"

    # AR order(s); order_f/order_t override order if provided.
    order: int = 4
    order_f: Optional[int] = None
    order_t: Optional[int] = None

    # Which pilot stream to use for de-rotation
    pilot_stream: int = 0

    # Fusion mode: "fixed" convex blend, or "map" Bayesian shrinkage
    blend_mode: str = "fixed"

    # Fixed blend factor (used for blend_mode="fixed")
    blend: float = 0.0

    # MAP/Bayes fusion hyperparameters (used for blend_mode="map")
    prior_var: float = 0.02
    min_blend: float = 0.0
    max_blend: float = 1.0

    # Numerical safety / clipping
    eps: float = 1e-6
    max_abs_pole: float = 0.999


@dataclass
class DDParams:
    enabled: bool
    alpha_min: float
    Q_max: int
    temperature: float
    # µ in Eq. (34) of the paper: scales influence of pseudo-labels vs pilots
    mu: float = 1.0

    # Allow DD update even if pilot MSE is slightly worse (fractional tolerance).
    # Accept if pilot_mse1 <= pilot_mse0*(1+accept_tol).
    accept_tol: float = 0.0


@dataclass
class LowRankParams:
    enabled: bool
    R: int

@dataclass
class SubbandParams:
    """
    Piecewise-constant frequency adaptation.

    If enabled, the detector can adapt the readout across contiguous blocks
    of subcarriers (subbands).

    IMPORTANT (confirmed by PROBE V4b):
      With only ONE DMRS symbol and small subbands (e.g., 5 PRBs => 60 pilots/group),
      gating BOTH feature+skip per subband ("all") can be underdetermined for FULL
      features (F0 = F_feat + N_rx). This causes severe overfit and poor data NMSE.

    We therefore support multiple gating modes and a safe "auto" mode.
    """
    enabled: bool
    # Number of *effective* subcarriers per subband (K_block).
    # Example: size_sc=60 corresponds to 5 PRBs (5*12).
    size_sc: int
    # How to apply subband gating:
    # - "all"      : gate BOTH reservoir features and skip y (block-diagonal per subband)
    # - "skip_only": gate ONLY the skip y (global reservoir features; subband-specific linear equalizer)
    # - "feat_only": gate ONLY reservoir features (global skip y)
    # - "auto"     : choose a safe mode based on pilot budget (recommended)
    mode: str = "auto"


def _make_poles(
    M: int,
    rho_min: float,
    rho_max: float,
    policy: str,
    *,
    warp: float = 4.0,
) -> np.ndarray:
    """Generate real-valued pole radii.

    Supported policies:
      - "linspace": uniform radii in [rho_min, rho_max]
      - "logspace": log-uniform radii in [rho_min, rho_max] (denser near rho_min)
      - "dense_unit" (aka "paper"/"unit"): denser near rho_max (|p| ~ 1)
    """
    if M <= 0:
        return np.zeros([0], dtype=np.float32)
    if M == 1:
        return np.array([rho_max], dtype=np.float32)

    policy = str(policy).lower().strip()

    # Legacy policies
    if policy == "logspace":
        rho_min_c = max(float(rho_min), 1e-12)
        rho_max_c = max(float(rho_max), 1e-12)
        return np.exp(np.linspace(np.log(rho_min_c), np.log(rho_max_c), M)).astype(np.float32)

    if policy == "linspace":
        return np.linspace(float(rho_min), float(rho_max), M, dtype=np.float32)

    # Paper-style prior: dense near |p| ~= rho_max (close to unit circle)
    if policy in ["dense_unit", "dense_unit_log", "paper", "unit"]:
        gamma = float(warp)
        if (not np.isfinite(gamma)) or (gamma <= 1.0):
            gamma = 4.0

        u = np.linspace(0.0, 1.0, M, dtype=np.float32)
        # warping that concentrates samples near 1
        u_w = 1.0 - np.power(1.0 - u, gamma)

        rho_min_c = max(float(rho_min), 1e-12)
        rho_max_c = max(float(rho_max), 1e-12)
        log_min = np.log(rho_min_c)
        log_max = np.log(rho_max_c)
        return np.exp(log_min + u_w * (log_max - log_min)).astype(np.float32)

    raise ValueError(
        f"Unknown pole_policy={policy}. Use logspace/linspace/dense_unit."
    )


def _levinson_durbin(
    r: tf.Tensor,
    order: int,
    eps: float = 1e-6,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Levinson–Durbin recursion for complex Toeplitz autocorrelation.

    Args:
        r: Complex autocorrelation sequence of shape [B, order+1] where r[:,0] is real >= 0.
        order: AR order.
        eps: Numerical epsilon for stability.

    Returns:
        a: Complex AR coefficients a_1..a_order of shape [B, order]
           for the model x[n] = sum_i a_i x[n-i] + e[n].
        kappa: Complex reflection (lattice) coefficients of shape [B, order].
    """
    r = tf.convert_to_tensor(r)
    dtype = r.dtype
    B = tf.shape(r)[0]
    order = int(order)

    # Prediction error power (real)
    e = tf.math.real(r[:, 0])
    e = tf.maximum(e, tf.cast(eps, e.dtype))

    a = tf.zeros([B, 0], dtype=dtype)
    kappas = []

    for i in range(1, order + 1):
        # acc = r[i] - sum_{j=1}^{i-1} a_j r[i-j]
        if i == 1:
            acc = r[:, 1]
        else:
            r_vec = r[:, 1:i]                 # [B, i-1] -> r[1..i-1]
            r_rev = tf.reverse(r_vec, axis=[1])  # [B, i-1] -> r[i-1..1]
            acc = r[:, i] - tf.reduce_sum(a * r_rev, axis=1)

        k = acc / tf.cast(e, dtype)
        kappas.append(k)

        # a <- [a - k*conj(reverse(a)), k]
        if i == 1:
            a = tf.expand_dims(k, axis=1)
        else:
            a = a - tf.expand_dims(k, axis=1) * tf.math.conj(tf.reverse(a, axis=[1]))
            a = tf.concat([a, tf.expand_dims(k, axis=1)], axis=1)

        # e <- e * (1 - |k|^2)
        e = e * tf.maximum(1.0 - tf.abs(k) ** 2, tf.cast(eps, e.dtype))

    kappa = tf.stack(kappas, axis=1) if kappas else tf.zeros([B, 0], dtype=dtype)
    err_var = e  # [B], real prediction error power proxy
    return a, kappa, err_var


def _ar_poles_from_coeffs(a: tf.Tensor) -> tf.Tensor:
    """Convert AR coefficients to poles via companion-matrix eigenvalues.

    a is shape [B,p] for x[n] = sum_i a_i x[n-i] + e[n].
    Poles are eigenvalues of the companion matrix with first row a.

    IMPORTANT:
      tf.linalg.eigvals will hard-fail if the input contains NaN/Inf.
      Under pilot-starved / high-stress conditions, the AR fit can produce
      non-finite coefficients. We sanitize to avoid crashing and let the
      caller's fusion-with-prior handle degraded estimates gracefully.
    """
    a = tf.convert_to_tensor(a)
    dtype = a.dtype
    B = tf.shape(a)[0]
    p = tf.shape(a)[1]

    # Sanitize coefficients (complex-safe)
    a = tf.where(_finite_mask(a), a, tf.zeros_like(a))

    def _empty():
        return tf.zeros([B, 0], dtype=dtype)

    def _nonempty():
        eye = tf.eye(p - 1, dtype=dtype)
        bottom = tf.pad(eye, paddings=[[0, 0], [0, 1]])  # (p-1) x p
        bottom = tf.broadcast_to(bottom[None, :, :], [B, p - 1, p])
        top = tf.expand_dims(a, axis=1)  # B x 1 x p
        C = tf.concat([top, bottom], axis=1)  # B x p x p

        # Extra safety: sanitize companion matrix too
        C = tf.where(_finite_mask(C), C, tf.zeros_like(C))

                # tf.linalg.eigvals can occasionally fail to converge for ill-conditioned
        # companion matrices (seen in pilot-starved Doppler stress tests). Use a numpy
        # fallback with exception handling to avoid hard crashes.
        def _eigvals_safe_np(C_np):
            import numpy as _np
            C_np = _np.asarray(C_np)

            # Expect [B, p, p]. If something ever comes in as [p, p], batch it.
            if C_np.ndim == 2:
                C_np = C_np[None, :, :]

            try:
                w = _np.linalg.eigvals(C_np.astype(_np.complex128)).astype(_np.complex64)
            except Exception:
                B_ = C_np.shape[0]
                p_ = C_np.shape[-1]
                w = _np.zeros((B_, p_), dtype=_np.complex64)

            w[~_np.isfinite(w)] = _np.complex64(0.0)
            return w

        poles = tf.numpy_function(_eigvals_safe_np, [C], Tout=tf.complex64)
        poles = tf.reshape(poles, [B, p])
        poles = tf.where(_finite_mask(poles), poles, tf.zeros_like(poles))
        return poles

    return tf.cond(p > 0, _nonempty, _empty)


def _unit_norm_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    nrm = np.sqrt(np.sum(np.abs(x) ** 2, axis=-1, keepdims=True))
    return x / np.maximum(nrm, eps)


def _make_projection(M: int, d: int, n_in: int, seed: int) -> np.ndarray:
    """Create fixed complex projection matrices B_m of shape [M, d, n_in]."""
    rng = np.random.default_rng(seed)
    # Complex Gaussian, row-normalized
    re = rng.standard_normal(size=(M, d, n_in)).astype(np.float32)
    im = rng.standard_normal(size=(M, d, n_in)).astype(np.float32)
    B = re + 1j * im
    B = _unit_norm_rows(B.reshape(M * d, n_in)).reshape(M, d, n_in)
    return B.astype(np.complex64)


def _complex_dtype(precision: str) -> tf.dtypes.DType:
    return tf.complex128 if precision == "double" else tf.complex64


def _real_dtype(precision: str) -> tf.dtypes.DType:
    return tf.float64 if precision == "double" else tf.float32


class ReservoirDetector(tf.keras.layers.Layer):
    def __init__(
        self,
        resource_grid: Any,
        num_rx_ant: int,
        num_bits_per_symbol: int,
        reservoir: ReservoirParams,
        ridge_lambda: float,
        dd: DDParams,
        lowrank: LowRankParams,
        whitening_enabled: bool,
        whitening_epsilon: float,
        ridge_lambda_y: Optional[float] = None,
        subband: Optional[SubbandParams] = None,
        pole_adapt: Optional[PoleAdaptParams] = None,
        precision: str = "single",
        seed: int = 1,
        name: str = "reservoir_detector",
    ):
        super().__init__(name=name)

        self.rg = resource_grid
        self.num_rx_ant = int(num_rx_ant)
        self.num_bits_per_symbol = int(num_bits_per_symbol)

        self.res = reservoir

        # Ridge regularization:
        # - ridge_lambda   : applied to the reservoir/memory features (paper Eq. (21)/(34))
        # - ridge_lambda_y : optional regularization for the skip-connection block D_t
        #                   (None => use the same value as ridge_lambda)
        self.ridge_lambda = float(ridge_lambda)
        self.ridge_lambda_y = None if ridge_lambda_y is None else float(ridge_lambda_y)

        self.dd = dd
        self.lowrank = lowrank

        # Subbanded readout (frequency selectivity fix)
        self.subband = subband if subband is not None else SubbandParams(enabled=False, size_sc=0)
        self._subband_enabled = bool(self.subband.enabled) and (int(self.subband.size_sc) > 0)
        self._subband_num_groups = 1
        self._pilot_group = None  # tf.int32 [P]
        self._data_group = None   # tf.int32 [Nd]
        self._subband_mode = "none"  # resolved mode used at call()
        self._subband_min_pilots: Optional[int] = None

        # Optional pilot-driven pole adaptation (covariance -> lattice/reflection coeffs -> poles)
        self.pole_adapt = pole_adapt if pole_adapt is not None else PoleAdaptParams(enabled=False)

        self.whitening_enabled = bool(whitening_enabled)
        self.whitening_epsilon = float(whitening_epsilon)

        self.precision = precision
        self.cdtype = _complex_dtype(precision)
        self.rdtype = _real_dtype(precision)

        # Dimensions
        self.num_tx = int(self.rg.num_tx)
        self.num_streams_per_tx = int(self.rg.num_streams_per_tx)
        self.num_streams_total = self.num_tx * self.num_streams_per_tx

        # Pilot pattern (mask/pilots)
        pp = self.rg.pilot_pattern
        mask = pp.mask  # [num_tx, num_streams, L, K_eff]
        pilots = pp.pilots  # [num_tx, num_streams, P]

        # Convert to numpy once for indexing
        mask_np = np.array(mask)
        # Use stream (0,0) as reference (PUSCH DMRS mask is typically identical across streams)
        mask00 = mask_np[0, 0].astype(bool)  # [L, K_eff]
        mask00_flat = mask00.reshape(-1)  # [L*K_eff]

        self.pilot_ind = tf.constant(np.flatnonzero(mask00_flat), dtype=tf.int32)
        self.data_ind = tf.constant(np.flatnonzero(~mask00_flat), dtype=tf.int32)

        # Pilot OFDM symbol locations (diagnostics / feature gating)
        # If pilots occupy only a single OFDM symbol, time-recursive states can create a
        # large train/test distribution shift across symbols (pilots at one l, data at others).
        pilot_re_per_sym = mask00.sum(axis=1)  # [L]
        pilot_symbols = np.flatnonzero(pilot_re_per_sym > 0)
        self.pilot_ofdm_symbols = [int(i) for i in pilot_symbols.tolist()]
        self._single_dmrs_symbol = (len(self.pilot_ofdm_symbols) == 1)

        # Pilot ordering helpers (used e.g. for pilot-driven pole adaptation).
        # We keep a selection of pilots belonging to the *first* pilot OFDM symbol,
        # ordered by frequency index k.
        K_eff_static_all = int(mask00.shape[1])
        pilot_ind_np0 = np.flatnonzero(mask00_flat).astype(np.int32)
        pilot_l_np0 = (pilot_ind_np0 // K_eff_static_all).astype(np.int32)
        pilot_k_np0 = (pilot_ind_np0 % K_eff_static_all).astype(np.int32)
        if len(self.pilot_ofdm_symbols) > 0:
            l0 = int(self.pilot_ofdm_symbols[0])
            sel0 = np.flatnonzero(pilot_l_np0 == l0).astype(np.int32)
            sel0 = sel0[np.argsort(pilot_k_np0[sel0])]
        else:
            sel0 = np.arange(pilot_ind_np0.shape[0], dtype=np.int32)
        self._pilot_sel_for_ar = tf.constant(sel0, dtype=tf.int32)  # indices into the P-pilot list
        self._pilot_ind_for_ar = tf.constant(pilot_ind_np0[sel0], dtype=tf.int32)  # indices into [L*K]

        self.num_pilots = int(self.pilot_ind.shape[0])
        # Number of data REs (flattened over the effective subcarriers)
        # Note: In Sionna 0.18, `pp.num_data_symbols` may be a scalar tf.Tensor.
        # We avoid `int(tf.Tensor)` and derive the count from the pilot mask.
        self.num_data_symbols = int(self.data_ind.shape[0])

        # If Sionna exposes a static python integer (or a statically known scalar tensor),
        # we clip to that value as a conservative guard.
        try:
            n_pp = tf.get_static_value(pp.num_data_symbols)
            if n_pp is not None:
                n_pp = int(n_pp)
                if n_pp < self.num_data_symbols:
                    self.num_data_symbols = n_pp
                    self.data_ind = self.data_ind[: self.num_data_symbols]
        except Exception:
            pass
        
        # ============================================================
        # Subband grouping (frequency-only, shared across OFDM symbols)
        # ============================================================
        if self._subband_enabled:
            # Static effective grid dimensions from the pilot mask
            K_eff_static = int(mask00.shape[1])
            L_static = int(mask00.shape[0])

                        # Effective subband width (K_block)
            K_block = int(self.subband.size_sc)

                        # In "auto" mode with LOWRANK readout, smaller subbands are usually better.
            # Empirically (PROBE 32/35):
            #   - skip_only is the only stable gating choice for lowrank
            #   - size_sc in the 6–8 range gives much better BLER than 12 or larger
            mode0 = str(getattr(self.subband, "mode", "auto")).lower().strip()
            if mode0 in ["", "none", "off"]:
                mode0 = "all"  # backward compatible

            if (mode0 == "auto") and self.lowrank.enabled:
                # If user specified an explicit size_sc, respect it (but cap extremely large values).
                # If not specified / invalid, default to 6.
                if K_block <= 0:
                    K_block = 6
                else:
                    K_block = min(K_block, 8)

            # Keep around for debugging / probes (cfg vs effective after auto-capping)
            self._subband_size_sc_cfg = int(getattr(self.subband, "size_sc", 0) or 0)
            self._subband_size_sc_eff = int(K_block)

            # Degenerate settings: disable for efficiency / safety
            if (K_block <= 0) or (K_block >= K_eff_static):
                self._subband_enabled = False
                self._subband_num_groups = 1
                self._pilot_group = None
                self._data_group = None
            else:
                # ------------------------------------------------------------
                # Subband grouping: avoid a tiny remainder group.
                #
                # With ONE DMRS symbol, pilots/group == (# subcarriers in group).
                # In skip_only mode we estimate at least N_rx coefficients per group.
                # If the last group is a tiny remainder (e.g., 12 subcarriers),
                # then pilots/group < N_rx => underdetermined => catastrophic BLER.
                #
                # Fix: use floor(K_eff/K_block) groups and clamp the remainder
                # subcarriers into the last group (so the last group is larger,
                # never tiny).
                # ------------------------------------------------------------
                self._subband_num_groups = int(K_eff_static // K_block)
                if self._subband_num_groups < 1:
                    self._subband_num_groups = 1

                # Flatten order is l-major then k (same as mask00.reshape(-1))
                group_k = (np.arange(K_eff_static) // K_block).astype(np.int32)  # [K_eff]
                group_k = np.minimum(group_k, self._subband_num_groups - 1).astype(np.int32)
                group_re = np.tile(group_k, L_static).astype(np.int32)           # [L*K_eff]

                # Convert indices to numpy safely
                pilot_ind_np = tf.get_static_value(self.pilot_ind)
                if pilot_ind_np is None:
                    pilot_ind_np = self.pilot_ind.numpy()
                data_ind_np = tf.get_static_value(self.data_ind)
                if data_ind_np is None:
                    data_ind_np = self.data_ind.numpy()

                self._pilot_group = tf.constant(group_re[pilot_ind_np], dtype=tf.int32)
                self._data_group = tf.constant(group_re[data_ind_np], dtype=tf.int32)

                # ------------------------------------------------------------
        # Resolve subband gating mode.
        #
        # PROBE V4b confirmed that for FULL features:
        #   F0 = F_feat + N_rx = 64 + 16 = 80
        # With size_prbs=5 and a single DMRS symbol:
        #   pilots/group = 60  => underdetermined if we use "all" gating.
        #
        # "auto" chooses:
        #   - "all"      when per-group pilot budget is sufficient
        #   - "skip_only" otherwise (always safe; only N_rx coeffs/group)
        # ------------------------------------------------------------
        if self._subband_enabled and (self._pilot_group is not None):
            try:
                counts = np.bincount(self._pilot_group.numpy(), minlength=self._subband_num_groups)
                self._subband_min_pilots = int(counts.min())
            except Exception:
                self._subband_min_pilots = None

            mode = str(getattr(self.subband, "mode", "auto")).lower().strip()
            if mode in ["", "none", "off"]:
                mode = "all"  # backward compatible
            if mode not in ["auto", "all", "skip_only", "feat_only"]:
                raise ValueError(f"Unknown subband.mode={mode}. Use auto/all/skip_only/feat_only")

            if mode == "auto":
                # Safe default:
                #  - lowrank: gate ONLY the skip connection per subband (better generalization)
                #  - full   : keep DOF check for "all" vs "skip_only"
                if self.lowrank.enabled:
                    self._subband_mode = "skip_only"
                else:
                    F_feat_base = int(self.res.M_f * self.res.d_f * self.res.M_t * self.res.d_t)
                    F0 = int(F_feat_base + self.num_rx_ant)

                    # DOF safety check (critical for FULL)
                    if (self._subband_min_pilots is not None) and (self._subband_min_pilots < F0):
                        self._subband_mode = "skip_only"
                    else:
                        self._subband_mode = "all"
            else:
                # Respect explicit config request: all / skip_only / feat_only.
                # NOTE: For LOWRANK readout, "all" and "feat_only" are numerically unstable
                # in this codebase and tend to produce near-random BLER (see PROBE 32).
                # To avoid accidental foot-shooting, force them to skip_only.
                if self.lowrank.enabled and (mode in ["all", "feat_only"]):
                    print(
                        f"[ReservoirDetector] WARNING: lowrank + subband.mode={mode} is unstable; "
                        f"forcing subband.mode=skip_only"
                    )
                    self._subband_mode = "skip_only"
                else:
                    self._subband_mode = mode
        else:
            self._subband_mode = "none"    

        # Pilot symbols for all streams, stacked as [N_s, P]
        pilots_tf = tf.cast(pilots, self.cdtype)
        pilots_flat = tf.reshape(pilots_tf, [self.num_streams_total, -1])
        self.pilots_all = pilots_flat  # [N_s, P]

        # Poles
        warp = float(getattr(self.res, "pole_warp", 4.0))
        p_f = _make_poles(
            self.res.M_f,
            self.res.rho_f_min,
            self.res.rho_f_max,
            self.res.pole_policy,
            warp=warp,
        )
        p_t = _make_poles(
            self.res.M_t,
            self.res.rho_t_min,
            self.res.rho_t_max,
            self.res.pole_policy,
            warp=warp,
        )
        self.p_f = tf.constant(p_f, dtype=self.rdtype)
        self.p_t = tf.constant(p_t, dtype=self.rdtype)

        # Projection matrices
        B_f = _make_projection(self.res.M_f, self.res.d_f, self.num_rx_ant, seed=seed + 10)
        B_t = _make_projection(self.res.M_t, self.res.d_t, self.num_rx_ant, seed=seed + 20)
        self.B_f = tf.constant(B_f, dtype=self.cdtype)
        self.B_t = tf.constant(B_t, dtype=self.cdtype)

        # Low-rank factors (fixed)
        if self.lowrank.enabled:
            Mfdf = self.res.M_f * self.res.d_f
            Mtdt = self.res.M_t * self.res.d_t
            rng = np.random.default_rng(seed + 30)
            b = rng.standard_normal((self.lowrank.R, Mfdf)).astype(np.float32) + 1j * rng.standard_normal(
                (self.lowrank.R, Mfdf)
            ).astype(np.float32)
            c = rng.standard_normal((self.lowrank.R, Mtdt)).astype(np.float32) + 1j * rng.standard_normal(
                (self.lowrank.R, Mtdt)
            ).astype(np.float32)
            b = _unit_norm_rows(b)
            c = _unit_norm_rows(c)
            self.b_lr = tf.constant(b.astype(np.complex64), dtype=self.cdtype)  # [R, Mfdf]
            self.c_lr = tf.constant(c.astype(np.complex64), dtype=self.cdtype)  # [R, Mtdt]
        else:
            self.b_lr = None
            self.c_lr = None

        # Optional Sionna pre-processing
        RemoveNulledSubcarriers = _try_import_remove_nulled()
        self._remove_nulled = RemoveNulledSubcarriers(self.rg) if RemoveNulledSubcarriers is not None else None

        # Precompute PAM/QAM metadata for demapping and slicing
        if self.num_bits_per_symbol % 2 != 0:
            raise ValueError("This implementation assumes square QAM (even bits per symbol)")
        self._pam_m = 2 ** (self.num_bits_per_symbol // 2)
        self._pam_bits = self.num_bits_per_symbol // 2
        self._pam_levels = self._qam_levels(self._pam_m, self.rdtype)  # [m]
        self._pam_bit_labels = self._gray_bits(self._pam_m, self._pam_bits)  # [m, pam_bits]

        # Preferred: use Sionna's mapping blocks (if available) to keep bit
        # order and LLR conventions consistent with the NR chain.
        Constellation, Demapper = _try_import_mapping()
        self._sionna_constellation = None
        self._sionna_demapper = None
        self._sionna_points = None

        if (Constellation is not None) and (Demapper is not None):
            try:
                self._sionna_constellation = Constellation(
                    "qam", num_bits_per_symbol=self.num_bits_per_symbol, normalize=True
                )
            except TypeError:
                # Older API
                try:
                    self._sionna_constellation = Constellation(
                        "qam", self.num_bits_per_symbol, normalize=True
                    )
                except Exception:
                    self._sionna_constellation = Constellation("qam", self.num_bits_per_symbol)

            # Cache constellation points for slicing (best-effort attribute names)
            pts = getattr(self._sionna_constellation, "points", None)
            if pts is None:
                pts = getattr(self._sionna_constellation, "_points", None)
            if pts is not None:
                self._sionna_points = tf.cast(pts, self.cdtype)

            # Create demapper (max-log preferred for speed)
            #
            # NOTE (Sionna v0.18.x): Demapper signature is
            #   Demapper(demapping_method, constellation_type=None,
            #            num_bits_per_symbol=None, constellation=None, ...)
            # hence passing the Constellation object positionally would be
            # interpreted as `constellation_type` and triggers
            # "Wrong constellation type".
            dm = None
            for method in ["maxlog", "app"]:
                # 1) Preferred: pass the Constellation object by keyword
                try:
                    dm = Demapper(method,
                                 constellation=self._sionna_constellation,
                                 hard_out=False)
                    break
                except Exception:
                    dm = None

                # 2) Alternative: request internal QAM constellation by type
                try:
                    dm = Demapper(method,
                                 constellation_type="qam",
                                 num_bits_per_symbol=self.num_bits_per_symbol,
                                 hard_out=False)
                    break
                except Exception:
                    dm = None

                # 3) Fallback: older positional API (constellation_type, bps)
                try:
                    dm = Demapper(method, "qam", self.num_bits_per_symbol, hard_out=False)
                    break
                except Exception:
                    dm = None

            self._sionna_demapper = dm

    @staticmethod
    def _gray_bits(m: int, nbits: int) -> tf.Tensor:
        """Return Gray code labels for integers 0..m-1 as shape [m, nbits]."""
        x = tf.range(m, dtype=tf.int32)
        g = tf.bitwise.bitwise_xor(x, tf.bitwise.right_shift(x, 1))
        bits = []
        for i in reversed(range(nbits)):
            bits.append(tf.bitwise.bitwise_and(tf.bitwise.right_shift(g, i), 1))
        return tf.stack(bits, axis=1)  # [m, nbits]

    @staticmethod
    def _qam_levels(m: int, rdtype: tf.dtypes.DType) -> tf.Tensor:
        """Normalized PAM levels for square QAM (unit average energy)."""
        # Unnormalized levels: -(m-1), -(m-3), ..., (m-1)
        levels = tf.cast(tf.range(-(m - 1), m, delta=2), rdtype)  # [m]
        # QAM normalization: average energy per dimension = (m^2 - 1)/3
        # total average energy = 2*(m^2 - 1)/3
        scale = tf.sqrt(tf.cast(2.0 * (m**2 - 1) / 3.0, rdtype))
        return levels / scale

    def _remove_nulled_subcarriers(self, y: tf.Tensor) -> tf.Tensor:
        """Return y_eff of shape [B, N_r, L, K_eff]."""
        if self._remove_nulled is not None:
            y_eff = self._remove_nulled(y)  # [B, num_rx, N_r, L, K_eff]
        else:
            y_eff = y
        # Assume num_rx=1
        y_eff = y_eff[:, 0]  # [B, N_r, L, K_eff]
        return tf.cast(y_eff, self.cdtype)

    def _flatten_grid(self, y_eff: tf.Tensor) -> tf.Tensor:
        """Flatten [B, N_r, L, K] -> [B, N_re, N_r] with RE order matching pilot mask flatten."""
        y_perm = tf.transpose(y_eff, [0, 2, 3, 1])  # [B, L, K, N_r]
        B = tf.shape(y_perm)[0]
        L = tf.shape(y_perm)[1]
        K = tf.shape(y_perm)[2]
        y_flat = tf.reshape(y_perm, [B, L * K, self.num_rx_ant])
        return y_flat

    def _whiten(self, y_flat: tf.Tensor) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
        """Whiten y using sample covariance estimated from pilots.

        Inputs:
          y_flat: [B, N_re, N_r]

        Returns:
          y_tilde_flat: [B, N_re, N_r]
          L: Cholesky factor of covariance [B, N_r, N_r] (or None if disabled)
        """
        if not self.whitening_enabled:
            return y_flat, None

        # Gather pilots: [B, P, N_r]
        y_p = tf.gather(y_flat, self.pilot_ind, axis=1)
        y_p = tf.cast(y_p, self.cdtype)

        # Covariance R = (1/P) Y^H Y, where Y is [P, N_r]
        P = tf.cast(tf.shape(y_p)[1], self.rdtype)
        R = tf.matmul(y_p, y_p, adjoint_a=True) / tf.cast(P, self.cdtype)  # [B, N_r, N_r]

        eps = tf.cast(self.whitening_epsilon, self.rdtype)
        R = R + tf.cast(eps, self.cdtype) * tf.eye(self.num_rx_ant, dtype=self.cdtype)[None, :, :]

        L = tf.linalg.cholesky(R)  # [B, N_r, N_r]

        # Whiten all REs: y_tilde = L^{-1} y
        rhs = tf.transpose(y_flat, [0, 2, 1])  # [B, N_r, N_re]
        y_tilde = tf.linalg.triangular_solve(L, rhs, lower=True)  # [B, N_r, N_re]
        y_tilde = tf.transpose(y_tilde, [0, 2, 1])  # [B, N_re, N_r]
        return y_tilde, L
    

    def _resolve_ar_order(self, dim: str, override: Optional[int], fallback: int) -> int:
        """Resolve AR order for pole adaptation with backward compatibility."""
        if override is not None:
            return max(int(override), 1)

        pa = self.pole_adapt
        dim = str(dim).lower().strip()

        # Optional per-dimension overrides (new config)
        if dim == "f":
            v = getattr(pa, "order_f", None)
            if v is not None:
                return max(int(v), 1)
        if dim == "t":
            v = getattr(pa, "order_t", None)
            if v is not None:
                return max(int(v), 1)

        # Legacy single order
        v = getattr(pa, "order", None)
        if v is not None:
            return max(int(v), 1)

        return max(int(fallback), 1)

    def _fuse_poles_with_prior(
        self,
        p_adapt: tf.Tensor,
        p_prior: tf.Tensor,
        meas_var: Optional[tf.Tensor],
    ) -> tf.Tensor:
        """Fuse adaptive poles with a static prior (fixed blend or MAP/Bayesian shrinkage)."""
        pa = self.pole_adapt
        mode = str(getattr(pa, "blend_mode", "fixed")).lower().strip()

        # Fixed convex blend: p = blend*prior + (1-blend)*adapt
        if mode in ["", "fixed", "manual"]:
            blend = float(getattr(pa, "blend", 0.0))
            if blend <= 0.0:
                return p_adapt
            blend = min(max(blend, 0.0), 1.0)
            b = tf.cast(blend, self.cdtype)
            return b * p_prior + tf.cast(1.0 - blend, self.cdtype) * p_adapt

        # MAP/Bayesian shrinkage:
        # weight on prior increases when measurement uncertainty is high
        if mode in ["map", "bayes", "bayesian"]:
            # Fallback if meas_var is unavailable
            if meas_var is None:
                blend = float(getattr(pa, "blend", 0.0))
                if blend <= 0.0:
                    return p_adapt
                blend = min(max(blend, 0.0), 1.0)
                b = tf.cast(blend, self.cdtype)
                return b * p_prior + tf.cast(1.0 - blend, self.cdtype) * p_adapt

            eps = tf.cast(getattr(pa, "eps", 1e-6), self.rdtype)
            prior_var = tf.cast(getattr(pa, "prior_var", 0.02), self.rdtype)
            prior_var = tf.maximum(prior_var, eps)

            mv = tf.cast(meas_var, self.rdtype)
            mv = tf.maximum(mv, eps)

            # Posterior mean fusion (Gaussian): weight on prior = mv/(mv+prior_var)
            w = mv / (mv + prior_var)  # [B]

            # Enforce nonzero blending if requested
            min_b = tf.cast(getattr(pa, "min_blend", 0.0), self.rdtype)
            max_b = tf.cast(getattr(pa, "max_blend", 1.0), self.rdtype)
            w = tf.clip_by_value(w, min_b, max_b)

            w = tf.cast(w, self.cdtype)[:, None]  # [B,1]
            return w * p_prior + (tf.cast(1.0, self.cdtype) - w) * p_adapt

        raise ValueError(f"Unknown pole_adapt.blend_mode={mode}. Use fixed/map.")

    def _adaptive_poles_f_from_pilots(
        self,
        y_tilde_flat: tf.Tensor,
        pilots: tf.Tensor,
        *,
        order: Optional[int] = None,
    ) -> Optional[tf.Tensor]:
        """Estimate frequency poles p_f from pilots using a lattice/AR policy.

        Returns:
            p_f_adapt: [B, M_f] complex poles, or None if disabled.
        """
        if (self.pole_adapt is None) or (not bool(self.pole_adapt.enabled)):
            return None
        adapt_dim = str(self.pole_adapt.adapt_dim).lower().replace(" ", "")
        if "f" not in adapt_dim:
            return None

        # AR order
        p = self._resolve_ar_order("f", order, fallback=int(self.res.M_f))
        p = max(p, 1)
        P_ar_static = int(self._pilot_sel_for_ar.shape[0] or 0)
        if P_ar_static <= 1:
            return None
        p = min(p, P_ar_static - 1)
        p = max(p, 1)
        Mf = int(self.res.M_f)

        # Use only the first pilot OFDM symbol's pilots (ordered by frequency)
        y_p = tf.gather(y_tilde_flat, self._pilot_ind_for_ar, axis=1)  # [B, P_ar, N_r]
        pilots = tf.cast(pilots, self.cdtype)
        pilots_full = tf.reshape(pilots, [self.num_tx * self.num_streams_per_tx, -1])  # [S,P]
        pilot_ref = pilots_full[int(self.pole_adapt.pilot_stream), :]  # [P]
        pilot_ref = tf.gather(pilot_ref, self._pilot_sel_for_ar, axis=0)  # [P_ar]

        # De-rotate by the (unit-modulus) pilot so that the frequency correlation reflects the channel.
        y_de = y_p * tf.math.conj(pilot_ref)[None, :, None]

        # Collapse RX dimension to a single complex process (still coherent across frequency)
        s = tf.reduce_mean(y_de, axis=-1)  # [B, P_ar]

        # Autocorrelation r[0..p]
        eps_r = tf.cast(self.pole_adapt.eps, self.rdtype)
        eps_c = tf.cast(eps_r, self.cdtype)
        P_ar = tf.shape(s)[1]
        r0 = tf.reduce_mean(s * tf.math.conj(s), axis=1)  # [B]
        r_list = [r0]
        for lag in range(1, p + 1):
            rk = tf.cond(
                P_ar > lag,
                lambda lag=lag: tf.reduce_mean(s[:, lag:] * tf.math.conj(s[:, :-lag]), axis=1),
                lambda: tf.zeros_like(r0),
            )
            r_list.append(rk)
        r = tf.stack(r_list, axis=1)  # [B, p+1]

        # Normalize so r[0]=1 for numerical stability / scale invariance
        denom = tf.where(tf.abs(r[:, 0:1]) > eps_r, r[:, 0:1], eps_c * tf.ones_like(r[:, 0:1]))
        r = r / denom

        # Levinson–Durbin -> lattice/reflection coeffs and AR coeffs
        a, _kappa, err_var = _levinson_durbin(r, p, eps=float(self.pole_adapt.eps))
        a = tf.where(_finite_mask(a), a, tf.zeros_like(a))
        err_var = tf.where(tf.math.is_finite(err_var), err_var, tf.ones_like(err_var))
        poles = _ar_poles_from_coeffs(a)  # [B, p]

        # Clip to stability radius
        max_abs = tf.cast(self.pole_adapt.max_abs_pole, self.rdtype)
        abs_p = tf.abs(poles)
        scale = tf.minimum(1.0, max_abs / (abs_p + tf.cast(self.pole_adapt.eps, self.rdtype)))
        poles = poles * tf.cast(scale, self.cdtype)

        # Select top-|pole| poles (pad if p < M_f)
        if p >= Mf:
            abs_p = tf.abs(poles)
            idx = tf.argsort(abs_p, axis=1, direction="DESCENDING")[:, :Mf]
            p_sel = tf.gather(poles, idx, batch_dims=1)  # [B, M_f]
        else:
            p_sel = poles
            B = tf.shape(p_sel)[0]
            pad = tf.cast(self.p_f, self.cdtype)[None, : (Mf - p)]
            pad = tf.broadcast_to(pad, [B, Mf - p])
            p_sel = tf.concat([p_sel, pad], axis=1)

        # Optional blend with the static pole set
                # Fuse with static prior (fixed blend or MAP shrinkage)
        B = tf.shape(p_sel)[0]
        p_prior = tf.cast(self.p_f, self.cdtype)[None, :]
        p_prior = tf.broadcast_to(p_prior, [B, Mf])
        p_sel = self._fuse_poles_with_prior(p_sel, p_prior, meas_var=err_var)

        # Final stability clip
        max_abs = tf.cast(self.pole_adapt.max_abs_pole, self.rdtype)
        abs_p = tf.abs(p_sel)
        scale = tf.minimum(1.0, max_abs / (abs_p + tf.cast(self.pole_adapt.eps, self.rdtype)))
        p_sel = p_sel * tf.cast(scale, self.cdtype)

        return p_sel
    

    def _adaptive_poles_t_from_pilots(
        self,
        y_tilde_flat: tf.Tensor,
        pilots: tf.Tensor,
        *,
        order: Optional[int] = None,
        L_sym: Optional[int] = None,
    ) -> Optional[tf.Tensor]:
        """Estimate time poles p_t from pilots.

        Only meaningful when pilots occupy >=2 OFDM symbols in the slot.
        If pilots occupy a single symbol, returns None (keep static prior).

        Uses a short AR(p) fit on per-OFDM-symbol averaged pilot observations.
        """
        if (self.pole_adapt is None) or (not bool(self.pole_adapt.enabled)):
            return None

        adapt_dim = str(self.pole_adapt.adapt_dim).lower().replace(" ", "")
        if "t" not in adapt_dim:
            return None

        if (self.pilot_ofdm_symbols is None) or (len(self.pilot_ofdm_symbols) < 2):
            return None
        if L_sym is None:
            return None

        # AR order (clipped by available pilot symbols)
        p = self._resolve_ar_order("t", order, fallback=int(self.res.M_t))
        p = min(p, len(self.pilot_ofdm_symbols) - 1)
        p = max(p, 1)
        Mt = int(self.res.M_t)

        # Pilot samples: y_p is [B,P,N_r]
        y_p = tf.gather(y_tilde_flat, self.pilot_ind, axis=1)

        pilots = tf.cast(pilots, self.cdtype)
        pilots_full = tf.reshape(pilots, [self.num_tx * self.num_streams_per_tx, -1])  # [S,P]
        pilot_ref = pilots_full[int(self.pole_adapt.pilot_stream), :]  # [P]

        # De-rotate pilots -> approximate channel samples at pilot REs
        y_de = y_p * tf.math.conj(pilot_ref)[None, :, None]  # [B,P,N_r]
        s = tf.reduce_mean(y_de, axis=-1)  # [B,P]

        # Compute l-index for each pilot from pilot_ind and (L_sym, K_eff)
        # N_re = L_sym*K_eff = tf.shape(y_tilde_flat)[1]
        N_re = tf.shape(y_tilde_flat)[1]
        K_eff = tf.cast(N_re // tf.cast(L_sym, tf.int32), tf.int32)
        l_ids = tf.math.floordiv(self.pilot_ind, K_eff)  # [P]

        # Average pilots per OFDM symbol => short time series seq[b, l]
        onehot = tf.one_hot(l_ids, depth=tf.cast(L_sym, tf.int32), dtype=self.rdtype)  # [P,L]
        sum_l = tf.matmul(s, tf.cast(onehot, self.cdtype))  # [B,L]
        cnt_l = tf.reduce_sum(onehot, axis=0)  # [L]

        eps_r = tf.cast(self.pole_adapt.eps, self.rdtype)
        cnt_l = tf.maximum(cnt_l, eps_r)
        mean_l = sum_l / tf.cast(cnt_l[None, :], self.cdtype)  # [B,L]

        # Restrict to pilot-bearing OFDM symbols only
        pilot_syms_tf = tf.constant(self.pilot_ofdm_symbols, dtype=tf.int32)
        seq = tf.gather(mean_l, pilot_syms_tf, axis=1)  # [B, L_pilot]

        # Autocorrelation r[0..p] along time
        r0 = tf.reduce_mean(seq * tf.math.conj(seq), axis=1)  # [B]
        r_list = [r0]
        for lag in range(1, p + 1):
            rk = tf.reduce_mean(seq[:, lag:] * tf.math.conj(seq[:, :-lag]), axis=1)
            r_list.append(rk)
        r = tf.stack(r_list, axis=1)  # [B,p+1]

        # Normalize so r[0]=1
        eps_c = tf.cast(eps_r, self.cdtype)
        denom = tf.where(tf.abs(r[:, 0:1]) > eps_r, r[:, 0:1], eps_c * tf.ones_like(r[:, 0:1]))
        r = r / denom

        # Levinson–Durbin -> poles
        a, _kappa, err_var = _levinson_durbin(r, p, eps=float(self.pole_adapt.eps))
        poles = _ar_poles_from_coeffs(a)  # [B,p]

        # Clip to stability radius
        max_abs = tf.cast(self.pole_adapt.max_abs_pole, self.rdtype)
        abs_p = tf.abs(poles)
        scale = tf.minimum(1.0, max_abs / (abs_p + tf.cast(self.pole_adapt.eps, self.rdtype)))
        poles = poles * tf.cast(scale, self.cdtype)

        # Pad/select to Mt
        if p >= Mt:
            abs_p = tf.abs(poles)
            idx = tf.argsort(abs_p, axis=1, direction="DESCENDING")[:, :Mt]
            p_sel = tf.gather(poles, idx, batch_dims=1)  # [B,Mt]
        else:
            p_sel = poles
            B = tf.shape(p_sel)[0]
            pad = tf.cast(self.p_t, self.cdtype)[None, : (Mt - p)]
            pad = tf.broadcast_to(pad, [B, Mt - p])
            p_sel = tf.concat([p_sel, pad], axis=1)

        # Fuse with static prior
        B = tf.shape(p_sel)[0]
        p_prior = tf.cast(self.p_t, self.cdtype)[None, :]
        p_prior = tf.broadcast_to(p_prior, [B, Mt])
        p_sel = self._fuse_poles_with_prior(p_sel, p_prior, meas_var=err_var)

        # Final safety clip
        max_abs = tf.cast(self.pole_adapt.max_abs_pole, self.rdtype)
        abs_p = tf.abs(p_sel)
        scale = tf.minimum(1.0, max_abs / (abs_p + tf.cast(self.pole_adapt.eps, self.rdtype)))
        p_sel = p_sel * tf.cast(scale, self.cdtype)
        return p_sel

    def _compute_states(
        self,
        y_tilde_flat: tf.Tensor,
        L_sym: int,
        K_eff: int,
        *,
        disable_time_memory: bool = False,
        p_f_override: Optional[tf.Tensor] = None,
        p_t_override: Optional[tf.Tensor] = None,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Compute frequency/time state stacks.

        Inputs:
          y_tilde_flat: [B, N_re, N_r]
          L_sym, K_eff: grid dims

        Returns:
          s_f_flat: [B, N_re, M_f*d_f]
          s_t_flat: [B, N_re, M_t*d_t]
        """
        B = tf.shape(y_tilde_flat)[0]

        # Reshape to [B, N_r, L, K]
        y_grid = tf.reshape(y_tilde_flat, [B, L_sym, K_eff, self.num_rx_ant])
        y_grid = tf.transpose(y_grid, [0, 3, 1, 2])  # [B, N_r, L, K]

        # === Frequency recursion: process each OFDM symbol independently ===
        y_f = tf.transpose(y_grid, [0, 2, 3, 1])  # [B, L, K, N_r]
        y_f = tf.reshape(y_f, [B * L_sym, K_eff, self.num_rx_ant])  # [B*L, K, N_r]

        # u_f[k] = B_f y
        u_f = tf.einsum("mdr,bkr->bkmd", self.B_f, y_f)  # [B*L, K, M_f, d_f]
        u_f = tf.transpose(u_f, [1, 0, 2, 3])  # [K, B*L, M_f, d_f]

        # Frequency poles. Default: static poles from config.
        if p_f_override is None:
            p_f = tf.cast(self.p_f, self.cdtype)[None, :, None]  # [1, M_f, 1]
        else:
            p_f_override = tf.cast(p_f_override, self.cdtype)
            # Accept either [M_f] or [B, M_f]
            if p_f_override.shape.rank == 1:
                p_f_override = p_f_override[None, :]
            # Repeat across OFDM symbols because we flattened B*L
            p_f_bl = tf.repeat(p_f_override, repeats=L_sym, axis=0)  # [B*L, M_f]
            p_f = p_f_bl[:, :, None]  # [B*L, M_f, 1]

        def step_f(prev, u_k):
            # prev,u_k: [B*L, M_f, d_f]
            return p_f * prev + u_k

        init_f = tf.zeros([B * L_sym, self.res.M_f, self.res.d_f], dtype=self.cdtype)
        s_f_seq = tf.scan(step_f, u_f, initializer=init_f)  # [K, B*L, M_f, d_f]
        s_f_seq = tf.transpose(s_f_seq, [1, 0, 2, 3])  # [B*L, K, M_f, d_f]
        s_f_seq = tf.reshape(s_f_seq, [B, L_sym, K_eff, self.res.M_f, self.res.d_f])
        s_f_vec = tf.reshape(s_f_seq, [B, L_sym, K_eff, self.res.M_f * self.res.d_f])

        # Flatten to [B, N_re, Mfdf]
        s_f_flat = tf.reshape(s_f_vec, [B, L_sym * K_eff, self.res.M_f * self.res.d_f])

        # === Time recursion: process each subcarrier independently ===
        y_t = tf.transpose(y_grid, [0, 3, 2, 1])  # [B, K, L, N_r]
        y_t = tf.reshape(y_t, [B * K_eff, L_sym, self.num_rx_ant])  # [B*K, L, N_r]

        u_t = tf.einsum("ndr,blr->blnd", self.B_t, y_t)  # [B*K, L, M_t, d_t]
        u_t = tf.transpose(u_t, [1, 0, 2, 3])  # [L, B*K, M_t, d_t]

        # Time poles. Default: static poles from config.
        if p_t_override is None:
            p_t = tf.cast(self.p_t, self.cdtype)[None, :, None]  # [1, M_t, 1]
        else:
            p_t_override = tf.cast(p_t_override, self.cdtype)
            if p_t_override.shape.rank == 1:
                p_t_override = p_t_override[None, :]
            # Repeat across subcarriers because we flattened B*K
            p_t_bk = tf.repeat(p_t_override, repeats=K_eff, axis=0)  # [B*K, M_t]
            p_t = p_t_bk[:, :, None]  # [B*K, M_t, 1]
        if disable_time_memory:
            # With single-symbol DM-RS, time-recursive features are weakly supervised
            # and can cause a large train/test distribution shift across OFDM symbols.
            # Using p_t=0 makes the time state depend only on the current symbol (no memory).
            p_t = tf.zeros_like(p_t)

        def step_t(prev, u_l):
            return p_t * prev + u_l

        init_t = tf.zeros([B * K_eff, self.res.M_t, self.res.d_t], dtype=self.cdtype)
        s_t_seq = tf.scan(step_t, u_t, initializer=init_t)  # [L, B*K, M_t, d_t]
        s_t_seq = tf.transpose(s_t_seq, [1, 0, 2, 3])  # [B*K, L, M_t, d_t]
        s_t_seq = tf.reshape(s_t_seq, [B, K_eff, L_sym, self.res.M_t, self.res.d_t])
        s_t_seq = tf.transpose(s_t_seq, [0, 2, 1, 3, 4])  # [B, L, K, M_t, d_t]
        s_t_vec = tf.reshape(s_t_seq, [B, L_sym, K_eff, self.res.M_t * self.res.d_t])
        s_t_flat = tf.reshape(s_t_vec, [B, L_sym * K_eff, self.res.M_t * self.res.d_t])

        return s_f_flat, s_t_flat

    def _phi_features(self, s_f: tf.Tensor, s_t: tf.Tensor) -> tf.Tensor:
        """Compute full outer-product feature phi for selected REs.

        Inputs:
          s_f: [B, N_sel, Mfdf]
          s_t: [B, N_sel, Mtdt]
        Returns:
          phi: [B, N_sel, Mfdf*Mtdt]
        """
        # Outer product over the feature dimensions for each selected RE.
        # s_f: [B, N_sel, Mfdf] with indices (b,n,a)
        # s_t: [B, N_sel, Mtdt] with indices (b,n,c)
        # -> outer: [B, N_sel, Mfdf, Mtdt] with indices (b,n,a,c)
        outer = tf.einsum("bna,bnc->bnac", s_f, s_t)  # [B, N_sel, Mfdf, Mtdt]
        B = tf.shape(outer)[0]
        N = tf.shape(outer)[1]
        return tf.reshape(outer, [B, N, self.res.M_f * self.res.d_f * self.res.M_t * self.res.d_t])

    def _psi_features(self, s_f: tf.Tensor, s_t: tf.Tensor) -> tf.Tensor:
        """Compute low-rank feature psi for selected REs.

        psi_r = (b_r^H s_f) * (c_r^H s_t)

        Inputs:
          s_f: [B, N_sel, Mfdf]
          s_t: [B, N_sel, Mtdt]
        Returns:
          psi: [B, N_sel, R]
        """
        assert self.b_lr is not None and self.c_lr is not None
        # b_lr: [R, Mfdf]
        # inner products: [B, N_sel, R]
        bf = tf.einsum("ra,bna->bnr", tf.math.conj(self.b_lr), s_f)
        ct = tf.einsum("ra,bna->bnr", tf.math.conj(self.c_lr), s_t)
        return bf * ct

    def _ridge_solve(
        self,
        Z: tf.Tensor,
        X: tf.Tensor,
        w: Optional[tf.Tensor] = None,
        reg_diag: Optional[tf.Tensor] = None,
    ) -> tf.Tensor:
        """Solve multi-output (weighted) ridge regression.

        We solve (per batch item):
            argmin_G  || X - G Z ||_F^2 + tr(G Λ G^H)

        where:
          - Z: [B, F, N]  (columns are feature vectors)
          - X: [B, S, N]  (columns are target symbols)
          - Λ: [F,F] diagonal with non-negative entries (Tikhonov regularization)

        When `reg_diag` is None, we use Λ = λ I with λ = self.ridge_lambda and
        keep the original primal/dual switching for speed.

        Optional sample weights `w` (one per column/sample):
          - w: [B, N], w>=0
          We implement weights via Z_w = Z*sqrt(w), X_w = X*sqrt(w).

        Returns:
          G: [B, S, F]
        """
        F = tf.shape(Z)[1]
        N = tf.shape(Z)[2]

        if w is not None:
            sqrt_w = tf.sqrt(tf.cast(w, self.rdtype))  # [B,N]
            sqrt_w = tf.cast(sqrt_w, self.cdtype)
            Zs = Z * sqrt_w[:, None, :]
            Xs = X * sqrt_w[:, None, :]
        else:
            Zs = Z
            Xs = X

        # XZ = X Z^H -> [B,S,F]
        XZ = tf.matmul(Xs, Zs, adjoint_b=True)

        if reg_diag is not None:
            # General diagonal Tikhonov: A = Z Z^H + diag(reg_diag)
            reg = tf.cast(reg_diag, self.cdtype)  # [F]
            A = tf.matmul(Zs, Zs, adjoint_b=True)  # [B,F,F]
            A = A + tf.linalg.diag(reg)[None, :, :]

            # Solve A * G^H = (XZ)^H  => G = (A^{-1} (XZ)^H)^H
            G_h = tf.linalg.solve(A, tf.linalg.adjoint(XZ))  # [B,F,S]
            return tf.linalg.adjoint(G_h)  # [B,S,F]

        # Scalar ridge (legacy path): Λ = λ I
        lam = tf.cast(self.ridge_lambda, self.cdtype)

        def _solve_primal() -> tf.Tensor:
            A = tf.matmul(Zs, Zs, adjoint_b=True)  # [B,F,F]
            A = A + lam * tf.eye(F, dtype=self.cdtype)[None, :, :]

            G_h = tf.linalg.solve(A, tf.linalg.adjoint(XZ))  # [B,F,S]
            return tf.linalg.adjoint(G_h)  # [B,S,F]

        def _solve_dual() -> tf.Tensor:
            Gram = tf.matmul(Zs, Zs, adjoint_a=True)  # [B,N,N]
            Gram = Gram + lam * tf.eye(N, dtype=self.cdtype)[None, :, :]

            # Beta^H = inv(Gram) X^H
            Xh = tf.linalg.adjoint(Xs)  # [B,N,S]
            Beta_h = tf.linalg.solve(Gram, Xh)  # [B,N,S]
            Beta = tf.linalg.adjoint(Beta_h)  # [B,S,N]
            return tf.matmul(Beta, Zs, adjoint_b=True)  # [B,S,F]

        use_primal = tf.less_equal(F, N)
        return tf.cond(use_primal, _solve_primal, _solve_dual)

    def _qam_slice_and_reliability(
        self, x_hat: tf.Tensor, sigma2_eff: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Nearest-neighbor slicer for square QAM + reliability.

        Inputs:
          x_hat: [B, S, N] complex
          sigma2_eff: [B, S] real (effective complex variance)

        Returns:
          x_bar: [B, S, N] complex (hard decisions)
          alpha: [B, N] real in [0,1] (RE-level reliability, min over streams)
        """
        # Preferred: slice using Sionna's constellation points, which guarantees
        # consistent normalization with the transmitter.
        if self._sionna_points is not None:
            pts = self._sionna_points  # [M]
            # Squared distance to all constellation points
            # d2: [B,S,N,M]
            d2 = tf.square(tf.abs(x_hat[..., None] - pts[None, None, None, :]))
            idx = tf.argmin(d2, axis=-1, output_type=tf.int32)  # [B,S,N]
            x_bar = tf.gather(pts, idx)  # [B,S,N]
            dmin = tf.reduce_min(d2, axis=-1)  # [B,S,N]

            eps = tf.cast(1e-12, self.rdtype)
            sigma2 = tf.maximum(tf.cast(sigma2_eff, self.rdtype), eps)  # [B,S]
            sigma2 = sigma2[:, :, None]  # [B,S,1]

            temp = tf.cast(self.dd.temperature, self.rdtype)
            alpha_stream = tf.exp(-dmin / (temp * sigma2))  # [B,S,N]
            alpha = tf.reduce_min(alpha_stream, axis=1)  # [B,N]
            return tf.cast(x_bar, self.cdtype), alpha

        # Split real/imag
        y_re = tf.cast(tf.math.real(x_hat), self.rdtype)
        y_im = tf.cast(tf.math.imag(x_hat), self.rdtype)

        levels = self._pam_levels  # [m]

        # Distances to 1D levels
        # Shapes: [B,S,N, m]
        dist_re = tf.square(y_re[..., None] - levels[None, None, None, :])
        dist_im = tf.square(y_im[..., None] - levels[None, None, None, :])

        # Find nearest indices
        idx_re = tf.argmin(dist_re, axis=-1, output_type=tf.int32)  # [B,S,N]
        idx_im = tf.argmin(dist_im, axis=-1, output_type=tf.int32)  # [B,S,N]

        # Hard decided levels
        x_re = tf.gather(levels, idx_re)
        x_im = tf.gather(levels, idx_im)
        x_bar = tf.cast(x_re, self.cdtype) + 1j * tf.cast(x_im, self.cdtype)

        # Min squared distance per symbol
        d_re = tf.reduce_min(dist_re, axis=-1)  # [B,S,N]
        d_im = tf.reduce_min(dist_im, axis=-1)
        d = d_re + d_im  # [B,S,N]

        # Reliability alpha in [0,1]
        # Use per-stream scaling, then take min over streams
        # Avoid division by zero
        eps = tf.cast(1e-12, self.rdtype)
        sigma2 = tf.maximum(tf.cast(sigma2_eff, self.rdtype), eps)  # [B,S]
        sigma2 = sigma2[:, :, None]  # [B,S,1]

        temp = tf.cast(self.dd.temperature, self.rdtype)
        alpha_stream = tf.exp(-d / (temp * sigma2))  # [B,S,N]
        alpha = tf.reduce_min(alpha_stream, axis=1)  # [B,N]
        return x_bar, alpha

    def _qam_maxlog_llr(self, x_hat: tf.Tensor, sigma2_eff: tf.Tensor) -> tf.Tensor:
        """Efficient max-log LLR for square QAM using separable PAM distances.

        Inputs:
          x_hat: [B, S, N] complex
          sigma2_eff: [B, S] real (effective complex variance)

        Returns:
          llr: [B, S, N*Qm] real
        """
        y_re = tf.cast(tf.math.real(x_hat), self.rdtype)
        y_im = tf.cast(tf.math.imag(x_hat), self.rdtype)

        levels = self._pam_levels  # [m]
        bits = self._pam_bit_labels  # [m, pam_bits]

        # Real-dimension noise variance
        eps = tf.cast(1e-12, self.rdtype)
        sigma2_c = tf.maximum(tf.cast(sigma2_eff, self.rdtype), eps)  # complex variance
        sigma2_r = sigma2_c / 2.0  # per real dim
        sigma2_r = sigma2_r[:, :, None]  # [B,S,1]

        # Distances: [B,S,N,m]
        dist_re = tf.square(y_re[..., None] - levels[None, None, None, :])
        dist_im = tf.square(y_im[..., None] - levels[None, None, None, :])

        big = tf.cast(1e9, self.rdtype)

        def pam_llr(dist: tf.Tensor) -> tf.Tensor:
            # dist: [B,S,N,m]
            llrs = []
            for b in range(self._pam_bits):
                # masks for bit=0/1
                mask0 = tf.equal(bits[:, b], 0)  # [m]
                mask1 = tf.logical_not(mask0)

                d0 = tf.reduce_min(tf.where(mask0[None, None, None, :], dist, big), axis=-1)
                d1 = tf.reduce_min(tf.where(mask1[None, None, None, :], dist, big), axis=-1)
                llr_b = (d1 - d0) / sigma2_r  # [B,S,N]
                llrs.append(llr_b)
            return tf.stack(llrs, axis=-1)  # [B,S,N,pam_bits]

        llr_re = pam_llr(dist_re)
        llr_im = pam_llr(dist_im)

        llr = tf.concat([llr_re, llr_im], axis=-1)  # [B,S,N,Qm]
        B = tf.shape(llr)[0]
        S = tf.shape(llr)[1]
        N = tf.shape(llr)[2]
        llr = tf.reshape(llr, [B, S, N * self.num_bits_per_symbol])
        return llr

    def call(self, y: tf.Tensor, no: tf.Tensor) -> tf.Tensor:
        """Compute LLRs.

        Inputs:
          y: [B, num_rx(=1), N_r, L, fft_size]
          no: scalar (or [B]) noise variance

        Output:
          llr: [B, num_tx, num_streams_per_tx, num_data_symbols*num_bits_per_symbol]
        """
        # 1) Remove nulled subcarriers
        y_eff = self._remove_nulled_subcarriers(y)  # [B, N_r, L, K_eff]

        B = tf.shape(y_eff)[0]
        L_sym = tf.shape(y_eff)[2]
        K_eff = tf.shape(y_eff)[3]

        # 2) Flatten
        y_flat = self._flatten_grid(y_eff)  # [B, N_re, N_r]

        # 3) Whitening
        y_tilde, _ = self._whiten(y_flat)

        # Retrieve pilots at call-time (DM-RS may change with slot/frame index).
        pilots_full = tf.cast(self.rg.pilot_pattern.pilots, self.cdtype)  # [num_tx, num_streams, P]

        # Optional: pilot-driven adaptive pole selection (frequency poles)
                # Optional: pilot-driven adaptive pole selection
        #
        # IMPORTANT: This must never crash the simulation. In pilot-starved or extreme
        # channels (high Doppler), some linear algebra steps may become numerically
        # unstable. If anything fails, we fall back to static poles.
        p_f_override = None
        p_t_override = None
        try:
            if (self.pole_adapt is not None) and bool(self.pole_adapt.enabled):
                adapt_dim = str(self.pole_adapt.adapt_dim).lower().replace(" ", "")
                if "f" in adapt_dim:
                    p_f_override = self._adaptive_poles_f_from_pilots(y_tilde, pilots_full)
                if "t" in adapt_dim:
                    p_t_override = self._adaptive_poles_t_from_pilots(
                        y_tilde, pilots_full, L_sym=L_sym
                    )
        except Exception as e:
            # Safe fallback: do not crash; just revert to static poles.
            tf.print("[pole_adapt] failed; falling back to static poles. Error:", str(e))
            p_f_override = None
            p_t_override = None

        # 4) States
        s_f_flat, s_t_flat = self._compute_states(
            y_tilde,
            L_sym,
            K_eff,
            disable_time_memory=(self._single_dmrs_symbol and self.res.disable_time_memory_single_dmrs),
            p_f_override=p_f_override,
            p_t_override=p_t_override,
        )

        # 5) Gather pilots and data
        y_p = tf.gather(y_tilde, self.pilot_ind, axis=1)  # [B,P,N_r]
        y_d = tf.gather(y_tilde, self.data_ind, axis=1)  # [B,Nd,N_r]

        s_f_p = tf.gather(s_f_flat, self.pilot_ind, axis=1)  # [B,P,Mfdf]
        s_t_p = tf.gather(s_t_flat, self.pilot_ind, axis=1)  # [B,P,Mtdt]

        s_f_d = tf.gather(s_f_flat, self.data_ind, axis=1)  # [B,Nd,Mfdf]
        s_t_d = tf.gather(s_t_flat, self.data_ind, axis=1)  # [B,Nd,Mtdt]

        # 6) Feature choice
        if self.lowrank.enabled:
            feat_p = self._psi_features(s_f_p, s_t_p)  # [B,P,R]
            feat_d = self._psi_features(s_f_d, s_t_d)  # [B,Nd,R]
            F_feat = self.lowrank.R
        else:
            feat_p = self._phi_features(s_f_p, s_t_p)  # [B,P,F_phi]
            feat_d = self._phi_features(s_f_d, s_t_d)  # [B,Nd,F_phi]
            F_feat = int(self.res.M_f * self.res.d_f * self.res.M_t * self.res.d_t)

        # 7) Build Z matrices (concat feature + skip y)
        # 7) Build Z matrices (feature + skip y), with optional subband gating.
        subband_groups = 1
        subband_mode = "none"

        if self._subband_enabled and (self._pilot_group is not None) and (self._data_group is not None):
            subband_groups = self._subband_num_groups
            subband_mode = getattr(self, "_subband_mode", "all")

            onehot_p = tf.one_hot(self._pilot_group, depth=subband_groups, dtype=self.rdtype)  # [P,G]
            onehot_d = tf.one_hot(self._data_group, depth=subband_groups, dtype=self.rdtype)   # [Nd,G]
            onehot_p = tf.cast(onehot_p, self.cdtype)
            onehot_d = tf.cast(onehot_d, self.cdtype)

            if subband_mode == "all":
                z_p0 = tf.concat([feat_p, y_p], axis=-1)  # [B,P,F0]
                z_d0 = tf.concat([feat_d, y_d], axis=-1)  # [B,Nd,F0]
                F0 = tf.shape(z_p0)[2]

                z_p_g = tf.einsum("pg,bpf->bpgf", onehot_p, z_p0)  # [B,P,G,F0]
                z_d_g = tf.einsum("ng,bnf->bngf", onehot_d, z_d0)  # [B,Nd,G,F0]

                z_p = tf.reshape(z_p_g, [B, tf.shape(z_p_g)[1], subband_groups * F0])
                z_d = tf.reshape(z_d_g, [B, tf.shape(z_d_g)[1], subband_groups * F0])

            elif subband_mode == "skip_only":
                # Gate ONLY skip-connection y per subband; keep reservoir features global.
                y_p_g = tf.einsum("pg,bpa->bpga", onehot_p, y_p)  # [B,P,G,N_rx]
                y_d_g = tf.einsum("ng,bna->bnga", onehot_d, y_d)  # [B,Nd,G,N_rx]
                y_p_g = tf.reshape(y_p_g, [B, tf.shape(y_p_g)[1], subband_groups * self.num_rx_ant])
                y_d_g = tf.reshape(y_d_g, [B, tf.shape(y_d_g)[1], subband_groups * self.num_rx_ant])
                z_p = tf.concat([feat_p, y_p_g], axis=-1)
                z_d = tf.concat([feat_d, y_d_g], axis=-1)

            elif subband_mode == "feat_only":
                # Gate ONLY reservoir features per subband; keep skip-connection y global.
                feat_p_g = tf.einsum("pg,bpf->bpgf", onehot_p, feat_p)  # [B,P,G,F_feat]
                feat_d_g = tf.einsum("ng,bnf->bngf", onehot_d, feat_d)  # [B,Nd,G,F_feat]
                feat_p_g = tf.reshape(feat_p_g, [B, tf.shape(feat_p_g)[1], subband_groups * F_feat])
                feat_d_g = tf.reshape(feat_d_g, [B, tf.shape(feat_d_g)[1], subband_groups * F_feat])
                z_p = tf.concat([feat_p_g, y_p], axis=-1)
                z_d = tf.concat([feat_d_g, y_d], axis=-1)

            else:
                raise ValueError(f"Unknown resolved subband mode: {subband_mode}")
        else:
            z_p = tf.concat([feat_p, y_p], axis=-1)
            z_d = tf.concat([feat_d, y_d], axis=-1)

        # Targets X_p (DM-RS): [B,S,P]
        pilots = tf.reshape(pilots_full, [self.num_tx * self.num_streams_per_tx, -1])  # [S,P]
        X_p = tf.tile(pilots[None, :, :], [B, 1, 1])

        # 8) Pilot-only ridge
        reg_diag = None
        if self.ridge_lambda_y is not None:
                        # Allow noise-scaled ridge by setting ridge lambdas negative.
            # Example:
            #   ridge.lambda_y = -1.0   -> lam_y = 1.0 * no
            #   ridge.lambda_y = -0.25  -> lam_y = 0.25 * no
            if self.ridge_lambda < 0:
                lam_feat_r = tf.cast(-self.ridge_lambda, self.rdtype) * tf.cast(no, self.rdtype)
            else:
                lam_feat_r = tf.cast(self.ridge_lambda, self.rdtype)

            if self.ridge_lambda_y < 0:
                lam_y_r = tf.cast(-self.ridge_lambda_y, self.rdtype) * tf.cast(no, self.rdtype)
            else:
                lam_y_r = tf.cast(self.ridge_lambda_y, self.rdtype)

            lam_feat = tf.cast(lam_feat_r, self.cdtype)
            lam_y = tf.cast(lam_y_r, self.cdtype)

            if (subband_groups > 1) and (subband_mode == "all"):
                reg_base = tf.concat(
                    [tf.fill([F_feat], lam_feat), tf.fill([self.num_rx_ant], lam_y)],
                    axis=0,
                )
                reg_diag = tf.tile(reg_base, [subband_groups])

            elif (subband_groups > 1) and (subband_mode == "skip_only"):
                reg_diag = tf.concat(
                    [tf.fill([F_feat], lam_feat), tf.fill([subband_groups * self.num_rx_ant], lam_y)],
                    axis=0,
                )

            elif (subband_groups > 1) and (subband_mode == "feat_only"):
                reg_diag = tf.concat(
                    [tf.fill([subband_groups * F_feat], lam_feat), tf.fill([self.num_rx_ant], lam_y)],
                    axis=0,
                )

            else:
                reg_diag = tf.concat(
                    [tf.fill([F_feat], lam_feat), tf.fill([self.num_rx_ant], lam_y)],
                    axis=0,
                )
        # Ensure design matrices exist for all control-flow paths (tf.function-safe)
        Z_p = tf.transpose(z_p, [0, 2, 1])  # [B, F, P]
        Z_d = tf.transpose(z_d, [0, 2, 1])  # [B, F, Nd]

        G_p = self._ridge_solve(Z_p, X_p, w=None, reg_diag=reg_diag) 

        # Initial estimates for DD selection
        x_hat_d0 = tf.matmul(G_p, Z_d)  # [B,S,Nd]

        # Estimate effective variance from pilots (pilot-only)
        x_hat_p0 = tf.matmul(G_p, Z_p)  # [B,S,P]
        err_p0 = X_p - x_hat_p0
        sigma2_eff0 = tf.reduce_mean(tf.square(tf.abs(err_p0)), axis=-1)  # [B,S]
        sigma2_eff0 = tf.maximum(tf.cast(sigma2_eff0, self.rdtype), tf.cast(1e-6, self.rdtype))

        # Pilot MSE (scalar per batch) for DD accept/reject sanity
        pilot_mse0 = tf.reduce_mean(tf.square(tf.abs(err_p0)), axis=[1, 2])  # [B]

        G = G_p
        sigma2_eff = sigma2_eff0

        # 9) Reliability-gated semi-blind refinement (optional)
        if self.dd.enabled and self.dd.Q_max > 0:
            # Hard decisions + reliability for all data
            x_bar_d0, alpha = self._qam_slice_and_reliability(x_hat_d0, sigma2_eff0)  # x_bar: [B,S,Nd], alpha:[B,Nd]

            # Select top-Q_max indices by alpha (fixed-size selection)
            Nd = int(self.data_ind.shape[0])
            Q = int(min(self.dd.Q_max, Nd))
            alpha_top, idx_top = tf.math.top_k(alpha, k=Q, sorted=True)

            # Apply threshold gating by setting weights to zero
            alpha_min = tf.cast(self.dd.alpha_min, self.rdtype)
            mu = tf.cast(self.dd.mu, self.rdtype)
            w_q = tf.where(alpha_top >= alpha_min, mu * alpha_top, tf.zeros_like(alpha_top))  # [B,Q]

            # Gather selected features and pseudo labels
            z_q = tf.gather(z_d, idx_top, axis=1, batch_dims=1)  # [B,Q,F]
            x_q = tf.gather(x_bar_d0, idx_top, axis=2, batch_dims=1)  # [B,S,Q]

            # Augmented training set
            z_a = tf.concat([z_p, z_q], axis=1)  # [B,P+Q,F]
            Z_a = tf.transpose(z_a, [0, 2, 1])  # [B,F,P+Q]

            X_a = tf.concat([X_p, x_q], axis=2)  # [B,S,P+Q]

            w_a = tf.concat([tf.ones([B, tf.shape(z_p)[1]], dtype=self.rdtype), w_q], axis=1)  # [B,P+Q]

            # Weighted ridge
                        # Weighted ridge (pilot + pseudo-labels)
            G_dd = self._ridge_solve(Z_a, X_a, w=w_a, reg_diag=reg_diag)

            # --- DD accept/reject safeguard (pilot-only validation) ---
            # If adding pseudo-labels does NOT improve the pilot fit, we revert to pilot-only.
            # This prevents harmful DD updates when pseudo-label correctness is low.
            x_hat_p1 = tf.matmul(G_dd, Z_p)  # [B,S,P]
            err_p1 = X_p - x_hat_p1
            pilot_mse1 = tf.reduce_mean(tf.square(tf.abs(err_p1)), axis=[1, 2])  # [B]

            # DD accept/reject:
            # Default (accept_tol=0.0): accept DD only if pilot MSE does not worsen.
            # If accept_tol>0, allow a small fractional pilot-MSE increase.
            tol = tf.cast(getattr(self.dd, "accept_tol", 0.0), self.rdtype)
            tol = tf.maximum(tol, tf.cast(0.0, self.rdtype))
            accept_dd = pilot_mse1 <= pilot_mse0 * (tf.cast(1.0, self.rdtype) + tol)  # [B] bool

            # Choose G (per batch item)
            G = tf.where(accept_dd[:, None, None], G_dd, G_p)

            # Update effective variance using pilots (after chosen solution)
            x_hat_p = tf.matmul(G, Z_p)
            err_p = X_p - x_hat_p
            sigma2_eff = tf.reduce_mean(tf.square(tf.abs(err_p)), axis=-1)  # [B,S]
            sigma2_eff = tf.maximum(tf.cast(sigma2_eff, self.rdtype), tf.cast(1e-6, self.rdtype))

        # 10) Final symbol estimates
        x_hat_d = tf.matmul(G, Z_d)  # [B,S,Nd]

                # 11) Demap to LLR
        if self._sionna_demapper is not None:
            # Sionna demapper expects a real-valued noise variance.
            # We provide an effective error variance estimated from DM-RS.
            no_eff = tf.cast(sigma2_eff, tf.float32)  # [B,S]
            no_eff = no_eff[:, :, None]  # [B,S,1]
            # Ensure broadcastability with y=[B,S,Nd]
            no_eff = tf.tile(no_eff, [1, 1, tf.shape(x_hat_d)[2]])  # [B,S,Nd]

            llr_out = None
            # Sionna Demapper call signature differs across versions:
            #   - Some expect positional: dm(x, no)
            #   - Some expect list/tuple: dm([x, no])
            try:
                llr_out = self._sionna_demapper(x_hat_d, no_eff)
            except TypeError:
                try:
                    llr_out = self._sionna_demapper([x_hat_d, no_eff])
                except Exception:
                    llr_out = None
            except Exception:
                try:
                    llr_out = self._sionna_demapper([x_hat_d, no_eff])
                except Exception:
                    llr_out = None

            if llr_out is None:
                # Fallback to internal max-log LLR if the Sionna demapper call fails
                llr = self._qam_maxlog_llr(x_hat_d, sigma2_eff)  # [B,S,Nd*Qm]
            else:
                # Sionna v0.18 returns [..., n*num_bits_per_symbol].
                # Keep a small compatibility hook in case another version returns
                # [..., n, num_bits_per_symbol].
                if (llr_out.shape.rank is not None) and (llr_out.shape.rank == 4):
                    llr = tf.reshape(
                        llr_out,
                        [B, tf.shape(llr_out)[1], tf.shape(llr_out)[2] * tf.shape(llr_out)[3]],
                    )
                else:
                    llr = llr_out  # [B,S,Nd*Qm]
        else:
            llr = self._qam_maxlog_llr(x_hat_d, sigma2_eff)  # [B,S,Nd*Qm]

        # IMPORTANT: Match Sionna OFDM-detector convention expected by LayerDemapper:
        #   [B, num_tx, num_streams_per_tx, Nd*Qm]
        Nd = tf.shape(x_hat_d)[2]  # dynamic Nd (should equal self.num_data_symbols)
        llr = tf.reshape(
            llr,
            [tf.shape(llr)[0], self.num_tx, self.num_streams_per_tx, Nd * self.num_bits_per_symbol],
        )

        return tf.cast(llr, self.rdtype)
