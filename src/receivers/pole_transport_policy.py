from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf


@dataclass
class PolePolicyConfig:
    mode: str = "heuristic"  # heuristic | learned
    M: int = 12
    hidden: int = 16
    rho_min: float = 0.85
    rho_max: float = 0.999
    omega_max: float = float(np.pi / 3.0)
    sigma_max: float = float(np.pi / 10.0)
    smooth_arc: float = float(0.05 * np.pi)
    pilot_stream: int = 0
    weights_path: Optional[str] = None


class PoleTransportPolicy(tf.keras.layers.Layer):
    """Tiny slot-conditioned pole policy.

    The policy reads a 7-dimensional pilot summary vector and returns
    a bank of stable complex poles of size ``M``. It is intentionally tiny.

    Summary vector order:
        [Re(c1), Im(c1), Re(c2), Im(c2), residual_u, anisotropy_kappa, snr_proxy_nu]
    """

    def __init__(self, cfg: PolePolicyConfig, precision: str = "single", name: str = "pole_transport_policy"):
        super().__init__(name=name)
        self.cfg = cfg
        self.precision = precision
        self.rdtype = tf.float64 if str(precision).lower() == "double" else tf.float32
        self.cdtype = tf.complex128 if str(precision).lower() == "double" else tf.complex64

        self.mode = str(cfg.mode).lower().strip()
        self.M = int(cfg.M)
        self.M_smooth = max(2, self.M // 2)
        self.M_osc_pairs = max(1, (self.M - self.M_smooth) // 2)
        self.smooth_grid = tf.constant(
            np.linspace(-float(cfg.smooth_arc), float(cfg.smooth_arc), self.M_smooth).astype(
                np.float64 if self.rdtype == tf.float64 else np.float32
            ),
            dtype=self.rdtype,
        )
        self.osc_grid = tf.constant(
            np.linspace(-1.0, 1.0, self.M_osc_pairs).astype(
                np.float64 if self.rdtype == tf.float64 else np.float32
            ),
            dtype=self.rdtype,
        )

        if self.mode == "learned":
            self.net = tf.keras.Sequential(
                [
                    tf.keras.layers.Input(shape=(7,), dtype=self.rdtype),
                    tf.keras.layers.Dense(int(cfg.hidden), activation="tanh", dtype=self.rdtype),
                    tf.keras.layers.Dense(int(cfg.hidden), activation="tanh", dtype=self.rdtype),
                    tf.keras.layers.Dense(5, activation=None, dtype=self.rdtype),
                ],
                name="pole_transport_mlp",
            )
        else:
            self.net = None

    def build(self, input_shape):
        if self.net is not None:
            self.net.build(input_shape)
        super().build(input_shape)

    def _heuristic_base(self, s: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Return heuristic parameters (r_s, r_o, omega, sigma, lambda) before learned corrections."""
        s = tf.cast(s, self.rdtype)
        c1_re, c1_im, c2_re, c2_im, u, kappa, nu = [s[:, i] for i in range(7)]
        eps = tf.cast(1e-6, self.rdtype)

        c1_abs = tf.sqrt(tf.maximum(c1_re * c1_re + c1_im * c1_im, eps))
        omega_seed = tf.atan2(c1_im, c1_re)
        c2_ang = tf.atan2(c2_im, c2_re)
        phase_disp = tf.abs(c2_ang - 2.0 * omega_seed)

        rho_lo = tf.cast(self.cfg.rho_min, self.rdtype)
        rho_hi = tf.cast(self.cfg.rho_max, self.rdtype)
        omega_max = tf.cast(self.cfg.omega_max, self.rdtype)
        sigma_max = tf.cast(self.cfg.sigma_max, self.rdtype)

        # Keep the seed close to the measured pilot correlation but robust to noise.
        r_seed = tf.clip_by_value(c1_abs * tf.exp(-0.10 * tf.maximum(u, 0.0)), rho_lo, rho_hi)
        r_s = tf.clip_by_value(r_seed + 0.005 * tf.tanh(nu), rho_lo, rho_hi)
        r_o = tf.clip_by_value(r_seed - 0.010 * tf.maximum(u, 0.0), rho_lo, rho_hi)

        omega = tf.clip_by_value(omega_seed, -omega_max, omega_max)
        sigma = tf.clip_by_value(0.30 * phase_disp + 0.10 * tf.maximum(u, 0.0), 0.0, sigma_max)

        # lambda controls how much of the oscillatory template opens up.
        lam = tf.clip_by_value(
            0.65 * tf.abs(omega) / tf.maximum(omega_max, eps)
            + 0.20 * tf.maximum(u, 0.0)
            + 0.15 * tf.maximum(kappa, 0.0),
            0.0,
            1.0,
        )
        return r_s, r_o, omega, sigma, lam

    def _apply_learned_corrections(
        self,
        s: tf.Tensor,
        r_s: tf.Tensor,
        r_o: tf.Tensor,
        omega: tf.Tensor,
        sigma: tf.Tensor,
        lam: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        if self.net is None:
            return r_s, r_o, omega, sigma, lam

        raw = tf.cast(self.net(tf.cast(s, self.rdtype)), self.rdtype)
        dr_s = 0.03 * tf.tanh(raw[:, 0])
        dr_o = 0.03 * tf.tanh(raw[:, 1])
        domega = 0.15 * tf.tanh(raw[:, 2])
        dsigma = 0.08 * tf.tanh(raw[:, 3])
        dlam = 0.25 * tf.tanh(raw[:, 4])

        rho_lo = tf.cast(self.cfg.rho_min, self.rdtype)
        rho_hi = tf.cast(self.cfg.rho_max, self.rdtype)
        omega_max = tf.cast(self.cfg.omega_max, self.rdtype)
        sigma_max = tf.cast(self.cfg.sigma_max, self.rdtype)

        r_s = tf.clip_by_value(r_s + dr_s, rho_lo, rho_hi)
        r_o = tf.clip_by_value(r_o + dr_o, rho_lo, rho_hi)
        omega = tf.clip_by_value(omega + domega, -omega_max, omega_max)
        sigma = tf.clip_by_value(sigma + dsigma, 0.0, sigma_max)
        lam = tf.clip_by_value(lam + dlam, 0.0, 1.0)
        return r_s, r_o, omega, sigma, lam

    def poles_from_summary(self, s: tf.Tensor) -> tf.Tensor:
        """Return conditional pole bank of shape [B, M]."""
        s = tf.cast(s, self.rdtype)
        r_s, r_o, omega, sigma, lam = self._heuristic_base(s)
        r_s, r_o, omega, sigma, lam = self._apply_learned_corrections(s, r_s, r_o, omega, sigma, lam)

        # Smooth poles stay close to angle 0 but cover a small arc.
        smooth_angles = self.smooth_grid[None, :] * (1.0 - 0.5 * lam[:, None])
        smooth = tf.cast(r_s[:, None], self.cdtype) * tf.exp(
            tf.complex(tf.zeros_like(smooth_angles), tf.cast(smooth_angles, self.rdtype))
        )

        # Oscillatory template opens proportionally to lam.
        osc_angles = lam[:, None] * (omega[:, None] + sigma[:, None] * self.osc_grid[None, :])
        osc_pos = tf.cast(r_o[:, None], self.cdtype) * tf.exp(
            tf.complex(tf.zeros_like(osc_angles), tf.cast(osc_angles, self.rdtype))
        )
        osc_neg = tf.cast(r_o[:, None], self.cdtype) * tf.exp(
            tf.complex(tf.zeros_like(osc_angles), tf.cast(-osc_angles, self.rdtype))
        )

        poles = tf.concat([smooth, osc_pos, osc_neg], axis=1)
        # Guard against small rounding errors outside the unit disk.
        abs_p = tf.maximum(tf.abs(poles), tf.cast(1e-12, self.rdtype))
        max_abs = tf.cast(self.cfg.rho_max, self.rdtype)
        scale = tf.minimum(tf.ones_like(abs_p), max_abs / abs_p)
        poles = poles * tf.cast(scale, self.cdtype)
        return poles

    def call(self, s: tf.Tensor) -> tf.Tensor:
        return self.poles_from_summary(s)

    def load_external_weights_if_available(self, path: Optional[str] = None) -> bool:
        """Best-effort weight loading.

        Returns True if weights were loaded, else False.
        """
        if self.net is None:
            return False
        weights_path = path or self.cfg.weights_path
        if not weights_path:
            return False
        try:
            # Make sure variables exist before loading.
            dummy = tf.zeros([1, 7], dtype=self.rdtype)
            _ = self(dummy)
            self.net.load_weights(weights_path)
            tf.print("[PoleTransportPolicy] Loaded weights from", weights_path)
            return True
        except Exception as e:
            tf.print("[PoleTransportPolicy] Could not load weights from", weights_path, ":", str(e))
            return False
