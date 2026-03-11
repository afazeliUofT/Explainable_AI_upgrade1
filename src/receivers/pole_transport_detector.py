from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import tensorflow as tf

from .pole_transport_policy import PolePolicyConfig, PoleTransportPolicy


def _try_import_mapping():
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


def _unit_norm_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    nrm = np.sqrt(np.sum(np.abs(x) ** 2, axis=-1, keepdims=True))
    return x / np.maximum(nrm, eps)


def _make_projection(M: int, n_in: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    re = rng.standard_normal(size=(M, n_in)).astype(np.float32)
    im = rng.standard_normal(size=(M, n_in)).astype(np.float32)
    B = _unit_norm_rows(re + 1j * im)
    return B.astype(np.complex64)


def _make_dense_unit_poles(M: int, rho_min: float, rho_max: float, warp: float = 4.0) -> np.ndarray:
    if M <= 0:
        return np.zeros([0], dtype=np.complex64)
    if M == 1:
        return np.array([np.complex64(rho_max)], dtype=np.complex64)
    u = np.linspace(0.0, 1.0, M, dtype=np.float32)
    gamma = float(warp) if np.isfinite(warp) and warp > 1.0 else 4.0
    u_w = 1.0 - np.power(1.0 - u, gamma)
    log_min = np.log(max(float(rho_min), 1e-6))
    log_max = np.log(max(float(rho_max), 1e-6))
    radii = np.exp(log_min + u_w * (log_max - log_min)).astype(np.float32)
    return radii.astype(np.complex64)


class PoleTransportDetector(tf.keras.layers.Layer):
    """TTI-conditioned pole-transport reservoir detector.

    This detector is frequency-dominant by design. It uses:
      1) pilot-covariance whitening,
      2) a tiny pilot-conditioned pole policy,
      3) bidirectional frequency reservoirs,
      4) a short FIR-like skip window,
      5) a closed-form ridge readout learned from current pilots.

    The detector can evaluate two candidates inside each TTI:
      * a static dense-near-unit pole bank (paper-style prior), and
      * a conditional pole bank predicted from the current DM-RS.

    The better one is chosen based on pilot reconstruction loss.
    """

    def __init__(
        self,
        resource_grid: Any,
        num_rx_ant: int,
        num_bits_per_symbol: int,
        cfg: Dict[str, Any],
        precision: str = "single",
        seed: int = 1,
        name: str = "pole_transport_detector",
    ):
        super().__init__(name=name)
        self.rg = resource_grid
        self.num_rx_ant = int(num_rx_ant)
        self.num_bits_per_symbol = int(num_bits_per_symbol)
        self.precision = str(precision).lower()
        self.rdtype = tf.float64 if self.precision == "double" else tf.float32
        self.cdtype = tf.complex128 if self.precision == "double" else tf.complex64
        self.seed = int(seed)

        self.num_tx = int(self.rg.num_tx)
        self.num_streams_per_tx = int(self.rg.num_streams_per_tx)
        self.num_streams_total = self.num_tx * self.num_streams_per_tx

        static_cfg = cfg.get("static", {}) if isinstance(cfg, dict) else {}
        policy_sec = cfg.get("policy", {}) if isinstance(cfg, dict) else {}
        recv_cfg = cfg.get("receiver", {}) if isinstance(cfg, dict) else {}

        self.M = int(policy_sec.get("M", static_cfg.get("M", 12)))
        self.static_rho_min = float(static_cfg.get("rho_min", 0.35))
        self.static_rho_max = float(static_cfg.get("rho_max", 0.995))
        self.static_warp = float(static_cfg.get("warp", 4.0))
        self.ridge_lambda = float(recv_cfg.get("ridge_lambda", 1e-3))
        self.whitening_enabled = bool(recv_cfg.get("whitening_enabled", True))
        self.whitening_epsilon = float(recv_cfg.get("whitening_epsilon", 1e-3))
        self.local_skip = max(0, int(recv_cfg.get("local_skip", 1)))
        self.lowrank_R = max(0, int(recv_cfg.get("lowrank_R", 4)))
        self.fallback_enabled = bool(recv_cfg.get("fallback_enabled", True))
        self.default_selection = str(recv_cfg.get("selection", "best")).lower().strip()

        policy_cfg = PolePolicyConfig(
            mode=str(policy_sec.get("mode", "heuristic")),
            M=self.M,
            hidden=int(policy_sec.get("hidden", 16)),
            rho_min=float(policy_sec.get("rho_min", 0.85)),
            rho_max=float(policy_sec.get("rho_max", 0.999)),
            omega_max=float(policy_sec.get("omega_max", float(np.pi / 3.0))),
            sigma_max=float(policy_sec.get("sigma_max", float(np.pi / 10.0))),
            smooth_arc=float(policy_sec.get("smooth_arc", float(0.05 * np.pi))),
            pilot_stream=int(policy_sec.get("pilot_stream", 0)),
            weights_path=policy_sec.get("weights_path", None),
        )
        self.policy_cfg = policy_cfg
        self.policy = PoleTransportPolicy(policy_cfg, precision=precision)

        # Pilot pattern metadata.
        pp = self.rg.pilot_pattern
        mask = np.array(pp.mask)
        pilots = np.array(pp.pilots)
        mask00 = mask[0, 0].astype(bool)  # [L, K_eff]
        self._mask00 = mask00
        mask00_flat = mask00.reshape(-1)
        self.pilot_ind = tf.constant(np.flatnonzero(mask00_flat).astype(np.int32), dtype=tf.int32)
        self.data_ind = tf.constant(np.flatnonzero(~mask00_flat).astype(np.int32), dtype=tf.int32)
        self.num_pilots = int(self.pilot_ind.shape[0])
        self.num_data_symbols = int(self.data_ind.shape[0])

        pilot_ind_np = np.flatnonzero(mask00_flat).astype(np.int32)
        K_eff_static = int(mask00.shape[1])
        pilot_l_np = (pilot_ind_np // K_eff_static).astype(np.int32)
        pilot_k_np = (pilot_ind_np % K_eff_static).astype(np.int32)
        self.pilot_ofdm_symbols = np.unique(pilot_l_np).astype(np.int32).tolist()
        if len(self.pilot_ofdm_symbols) > 0:
            l0 = int(self.pilot_ofdm_symbols[0])
            sel0 = np.flatnonzero(pilot_l_np == l0).astype(np.int32)
            sel0 = sel0[np.argsort(pilot_k_np[sel0])]
        else:
            sel0 = np.arange(pilot_ind_np.shape[0], dtype=np.int32)
        self._pilot_sel_for_summary = tf.constant(sel0, dtype=tf.int32)
        self._pilot_ind_for_summary = tf.constant(pilot_ind_np[sel0], dtype=tf.int32)
        self._summary_pilot_count = int(sel0.shape[0])

        pilots_flat = pilots.reshape(self.num_streams_total, -1)
        self.pilots_all = tf.constant(pilots_flat, dtype=self.cdtype)  # [S, P]

        # Static bank + fixed projections.
        static_poles = _make_dense_unit_poles(
            self.M,
            self.static_rho_min,
            self.static_rho_max,
            warp=self.static_warp,
        )
        self.static_poles = tf.constant(static_poles, dtype=self.cdtype)
        self.proj_fwd = tf.constant(_make_projection(self.M, self.num_rx_ant, seed=self.seed + 11), dtype=self.cdtype)
        self.proj_bwd = tf.constant(_make_projection(self.M, self.num_rx_ant, seed=self.seed + 17), dtype=self.cdtype)

        if self.lowrank_R > 0:
            rng = np.random.default_rng(self.seed + 29)
            u = rng.standard_normal((self.lowrank_R, self.M)).astype(np.float32) + 1j * rng.standard_normal(
                (self.lowrank_R, self.M)
            ).astype(np.float32)
            v = rng.standard_normal((self.lowrank_R, self.M)).astype(np.float32) + 1j * rng.standard_normal(
                (self.lowrank_R, self.M)
            ).astype(np.float32)
            self.mix_u = tf.constant(_unit_norm_rows(u).astype(np.complex64), dtype=self.cdtype)
            self.mix_v = tf.constant(_unit_norm_rows(v).astype(np.complex64), dtype=self.cdtype)
        else:
            self.mix_u = None
            self.mix_v = None

        RemoveNulledSubcarriers = _try_import_remove_nulled()
        self._remove_nulled = RemoveNulledSubcarriers(self.rg) if RemoveNulledSubcarriers is not None else None

        # Demapper compatibility hooks.
        if self.num_bits_per_symbol % 2 != 0:
            raise ValueError("PoleTransportDetector assumes square QAM (even bits per symbol)")
        self._pam_m = 2 ** (self.num_bits_per_symbol // 2)
        self._pam_bits = self.num_bits_per_symbol // 2
        self._pam_levels = self._qam_levels(self._pam_m, self.rdtype)
        self._pam_bit_labels = self._gray_bits(self._pam_m, self._pam_bits)

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
                try:
                    self._sionna_constellation = Constellation("qam", self.num_bits_per_symbol, normalize=True)
                except Exception:
                    self._sionna_constellation = None
            if self._sionna_constellation is not None:
                pts = getattr(self._sionna_constellation, "points", None)
                if pts is None:
                    pts = getattr(self._sionna_constellation, "_points", None)
                if pts is not None:
                    self._sionna_points = tf.cast(pts, self.cdtype)
                try:
                    self._sionna_demapper = Demapper(
                        "maxlog",
                        constellation=self._sionna_constellation,
                        hard_out=False,
                        dtype=self.rdtype,
                    )
                except TypeError:
                    try:
                        self._sionna_demapper = Demapper(
                            "app",
                            constellation=self._sionna_constellation,
                            hard_out=False,
                            dtype=self.rdtype,
                        )
                    except Exception:
                        self._sionna_demapper = None

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _gray_bits(M: int, num_bits: int) -> tf.Tensor:
        vals = np.arange(M, dtype=np.int32)
        gray = vals ^ (vals >> 1)
        bits = ((gray[:, None] >> np.arange(num_bits - 1, -1, -1)) & 1).astype(np.int32)
        return tf.constant(bits, dtype=tf.int32)

    @staticmethod
    def _qam_levels(M: int, dtype: tf.dtypes.DType) -> tf.Tensor:
        vals = np.arange(M, dtype=np.float32)
        gray_to_level = 2.0 * vals - (M - 1)
        norm = np.sqrt((2.0 / 3.0) * (M * M - 1))
        return tf.constant(gray_to_level / norm, dtype=dtype)

    def _remove_nulled_subcarriers(self, y: tf.Tensor) -> tf.Tensor:
        out = y
        if self._remove_nulled is not None:
            try:
                out = self._remove_nulled(y)
            except Exception:
                out = y
        if out.shape.rank == 5:
            try:
                out = tf.squeeze(out, axis=1)
            except Exception:
                pass
        if out.shape.rank != 4:
            raise ValueError(f"Expected y_eff rank 4 after nulled-subcarrier removal, got rank {out.shape.rank}")
        return tf.cast(out, self.cdtype)

    def _remove_nulled_x(self, x: tf.Tensor) -> tf.Tensor:
        out = x
        if self._remove_nulled is not None:
            try:
                out = self._remove_nulled(x)
            except Exception:
                out = x
        return tf.cast(out, self.cdtype)

    def _flatten_grid(self, y_eff: tf.Tensor) -> tf.Tensor:
        # [B, N_r, L, K] -> [B, L*K, N_r]
        B = tf.shape(y_eff)[0]
        L_sym = tf.shape(y_eff)[2]
        K_eff = tf.shape(y_eff)[3]
        y_flat = tf.transpose(y_eff, [0, 2, 3, 1])
        y_flat = tf.reshape(y_flat, [B, L_sym * K_eff, self.num_rx_ant])
        return y_flat

    def extract_data_symbols_from_x(self, x: tf.Tensor) -> tf.Tensor:
        x_eff = self._remove_nulled_x(x)
        if x_eff.shape.rank != 5:
            raise ValueError(f"Expected x rank 5 after nulled-subcarrier removal, got rank {x_eff.shape.rank}")
        B = tf.shape(x_eff)[0]
        L_sym = tf.shape(x_eff)[3]
        K_eff = tf.shape(x_eff)[4]
        x_flat = tf.reshape(x_eff, [B, self.num_streams_total, L_sym * K_eff])
        x_d = tf.gather(x_flat, self.data_ind, axis=2)
        return tf.cast(x_d, self.cdtype)

    def _whiten(self, y_flat: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        B = tf.shape(y_flat)[0]
        eye = tf.eye(self.num_rx_ant, dtype=self.cdtype, batch_shape=[B])
        if not self.whitening_enabled:
            return y_flat, eye
        y_p = tf.gather(y_flat, self.pilot_ind, axis=1)
        P = tf.cast(tf.maximum(tf.shape(y_p)[1], 1), self.cdtype)
        R = tf.einsum("bpr,bps->brs", y_p, tf.math.conj(y_p)) / P
        eps = tf.cast(self.whitening_epsilon, self.cdtype) * eye
        try:
            L = tf.linalg.cholesky(R + eps)
            W = tf.linalg.inv(L)
            y_tilde = tf.einsum("bij,bnj->bni", W, y_flat)
            return y_tilde, W
        except Exception:
            return y_flat, eye

    def _pilot_summary(self, y_tilde: tf.Tensor, no: tf.Tensor) -> tf.Tensor:
        B = tf.shape(y_tilde)[0]
        if self._summary_pilot_count <= 0:
            return tf.zeros([B, 7], dtype=self.rdtype)

        y_p = tf.gather(y_tilde, self._pilot_ind_for_summary, axis=1)  # [B, P0, N_r]
        pilot_ref = tf.gather(
            self.pilots_all[int(self.policy_cfg.pilot_stream)], self._pilot_sel_for_summary, axis=0
        )  # [P0]
        h = y_p * tf.math.conj(pilot_ref[None, :, None])
        h = tf.reduce_mean(h, axis=-1)  # [B, P0]
        eps = tf.cast(1e-6, self.rdtype)

        def _norm_corr(x: tf.Tensor, lag: int) -> tf.Tensor:
            if self._summary_pilot_count <= lag:
                return tf.zeros([B], dtype=self.cdtype)
            x0 = x[:, :-lag]
            x1 = x[:, lag:]
            den = tf.reduce_sum(tf.square(tf.abs(x0)), axis=1) + tf.cast(eps, self.rdtype)
            num = tf.reduce_sum(x1 * tf.math.conj(x0), axis=1)
            return num / tf.cast(den, self.cdtype)

        c1 = _norm_corr(h, 1)
        c2 = _norm_corr(h, 2)

        if self._summary_pilot_count > 1:
            pred = c1[:, None] * h[:, :-1]
            u = tf.reduce_mean(tf.square(tf.abs(h[:, 1:] - pred)), axis=1) / (
                tf.reduce_mean(tf.square(tf.abs(h[:, 1:])), axis=1) + tf.cast(eps, self.rdtype)
            )
        else:
            u = tf.zeros([B], dtype=self.rdtype)

        P0 = tf.cast(tf.maximum(tf.shape(y_p)[1], 1), self.cdtype)
        R = tf.einsum("bpr,bps->brs", y_p, tf.math.conj(y_p)) / P0
        eigvals = tf.cast(tf.linalg.eigvalsh(R), self.rdtype)
        tr = tf.reduce_sum(eigvals, axis=-1) + tf.cast(eps, self.rdtype)
        kappa = tf.reduce_max(eigvals, axis=-1) / tr

        if no.shape.rank == 0:
            no_r = tf.fill([B], tf.cast(no, self.rdtype))
        elif no.shape.rank == 1:
            no_r = tf.cast(no, self.rdtype)
        else:
            no_r = tf.reshape(tf.cast(no, self.rdtype), [B])
        sig_pow = tf.reduce_mean(tf.square(tf.abs(h)), axis=1)
        nu = tf.math.log(sig_pow / (no_r + tf.cast(eps, self.rdtype)) + tf.cast(eps, self.rdtype))

        s = tf.stack(
            [
                tf.math.real(c1),
                tf.math.imag(c1),
                tf.math.real(c2),
                tf.math.imag(c2),
                tf.cast(u, self.rdtype),
                tf.cast(kappa, self.rdtype),
                tf.cast(nu, self.rdtype),
            ],
            axis=1,
        )
        return tf.cast(s, self.rdtype)

    def _scan_frequency_states(self, y_tilde: tf.Tensor, L_sym: tf.Tensor, K_eff: tf.Tensor, poles: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        B = tf.shape(y_tilde)[0]
        y_grid = tf.reshape(y_tilde, [B, L_sym, K_eff, self.num_rx_ant])
        y_seq = tf.reshape(y_grid, [B * L_sym, K_eff, self.num_rx_ant])

        if poles.shape.rank == 1:
            p = tf.tile(tf.cast(poles[None, :], self.cdtype), tf.stack([B * L_sym, 1]))
        else:
            p = tf.cast(poles, self.cdtype)
            p = tf.repeat(p, repeats=L_sym, axis=0)

        u_fwd = tf.einsum("mr,bkr->bkm", self.proj_fwd, y_seq)
        u_fwd = tf.transpose(u_fwd, [1, 0, 2])  # [K, BL, M]
        p_b = p

        def step_f(prev, u_k):
            return p_b * prev + u_k

        init = tf.zeros([B * L_sym, self.M], dtype=self.cdtype)
        s_fwd = tf.scan(step_f, u_fwd, initializer=init)
        s_fwd = tf.transpose(s_fwd, [1, 0, 2])  # [BL, K, M]

        y_rev = tf.reverse(y_seq, axis=[1])
        u_bwd = tf.einsum("mr,bkr->bkm", self.proj_bwd, y_rev)
        u_bwd = tf.transpose(u_bwd, [1, 0, 2])

        def step_b(prev, u_k):
            return p_b * prev + u_k

        s_bwd = tf.scan(step_b, u_bwd, initializer=init)
        s_bwd = tf.transpose(s_bwd, [1, 0, 2])
        s_bwd = tf.reverse(s_bwd, axis=[1])

        s_fwd = tf.reshape(s_fwd, [B, L_sym * K_eff, self.M])
        s_bwd = tf.reshape(s_bwd, [B, L_sym * K_eff, self.M])
        return s_fwd, s_bwd

    def _local_skip_features(self, y_tilde: tf.Tensor, L_sym: tf.Tensor, K_eff: tf.Tensor) -> tf.Tensor:
        B = tf.shape(y_tilde)[0]
        y_grid = tf.reshape(y_tilde, [B, L_sym, K_eff, self.num_rx_ant])
        if self.local_skip <= 0:
            return tf.reshape(y_grid, [B, L_sym * K_eff, self.num_rx_ant])
        y_pad = tf.pad(y_grid, [[0, 0], [0, 0], [self.local_skip, self.local_skip], [0, 0]])
        chunks = []
        for shift in range(-self.local_skip, self.local_skip + 1):
            start = self.local_skip + shift
            chunks.append(y_pad[:, :, start : start + K_eff, :])
        skip = tf.concat(chunks, axis=-1)
        return tf.reshape(skip, [B, L_sym * K_eff, (2 * self.local_skip + 1) * self.num_rx_ant])

    def _feature_bank(self, y_tilde: tf.Tensor, L_sym: tf.Tensor, K_eff: tf.Tensor, poles: tf.Tensor) -> tf.Tensor:
        s_fwd, s_bwd = self._scan_frequency_states(y_tilde, L_sym, K_eff, poles)
        skip = self._local_skip_features(y_tilde, L_sym, K_eff)
        feat = [skip, s_fwd, s_bwd]
        if (self.lowrank_R > 0) and (self.mix_u is not None) and (self.mix_v is not None):
            a = tf.einsum("rm,bnm->bnr", self.mix_u, s_fwd)
            b = tf.einsum("rm,bnm->bnr", self.mix_v, s_bwd)
            feat.append(a * b)
        return tf.concat(feat, axis=-1)

    def _solve_ridge(self, Z_p: tf.Tensor, X_p: tf.Tensor) -> tf.Tensor:
        B = tf.shape(Z_p)[0]
        F = tf.shape(Z_p)[1]
        eye = tf.eye(F, dtype=self.cdtype, batch_shape=[B])
        lam = tf.cast(self.ridge_lambda, self.cdtype)
        A = tf.matmul(Z_p, Z_p, adjoint_b=True) + lam * eye
        rhs = tf.transpose(tf.matmul(X_p, Z_p, adjoint_b=True), [0, 2, 1])  # [B,F,S]
        Gt = tf.linalg.solve(A, rhs)
        G = tf.transpose(Gt, [0, 2, 1])
        return G

    def _candidate(self, y_tilde: tf.Tensor, L_sym: tf.Tensor, K_eff: tf.Tensor, poles: tf.Tensor) -> Dict[str, tf.Tensor]:
        B = tf.shape(y_tilde)[0]
        feat = self._feature_bank(y_tilde, L_sym, K_eff, poles)  # [B,Nre,F]
        feat_p = tf.gather(feat, self.pilot_ind, axis=1)
        feat_d = tf.gather(feat, self.data_ind, axis=1)
        Z_p = tf.transpose(feat_p, [0, 2, 1])  # [B,F,P]
        Z_d = tf.transpose(feat_d, [0, 2, 1])  # [B,F,Nd]
        X_p = tf.broadcast_to(self.pilots_all[None, :, :], [B, self.num_streams_total, self.num_pilots])
        G = self._solve_ridge(Z_p, X_p)
        x_hat_p = tf.matmul(G, Z_p)
        x_hat_d = tf.matmul(G, Z_d)
        err_p = X_p - x_hat_p
        pilot_mse = tf.reduce_mean(tf.square(tf.abs(err_p)), axis=[1, 2])
        sigma2_eff = tf.reduce_mean(tf.square(tf.abs(err_p)), axis=-1)
        sigma2_eff = tf.maximum(tf.cast(sigma2_eff, self.rdtype), tf.cast(1e-6, self.rdtype))
        return {
            "features_p": feat_p,
            "features_d": feat_d,
            "Z_p": Z_p,
            "Z_d": Z_d,
            "G": G,
            "x_hat_p": x_hat_p,
            "x_hat_d": x_hat_d,
            "pilot_mse": pilot_mse,
            "sigma2_eff": sigma2_eff,
        }

    def _qam_maxlog_llr(self, x_hat: tf.Tensor, sigma2_eff: tf.Tensor) -> tf.Tensor:
        y_re = tf.cast(tf.math.real(x_hat), self.rdtype)
        y_im = tf.cast(tf.math.imag(x_hat), self.rdtype)
        levels = self._pam_levels
        bits = self._pam_bit_labels
        eps = tf.cast(1e-12, self.rdtype)
        sigma2_c = tf.maximum(tf.cast(sigma2_eff, self.rdtype), eps)
        sigma2_r = sigma2_c / 2.0
        sigma2_r = sigma2_r[:, :, None]
        dist_re = tf.square(y_re[..., None] - levels[None, None, None, :])
        dist_im = tf.square(y_im[..., None] - levels[None, None, None, :])
        big = tf.cast(1e9, self.rdtype)

        def pam_llr(dist: tf.Tensor) -> tf.Tensor:
            llrs = []
            for bit_idx in range(self._pam_bits):
                mask0 = tf.equal(bits[:, bit_idx], 0)
                mask1 = tf.logical_not(mask0)
                d0 = tf.reduce_min(tf.where(mask0[None, None, None, :], dist, big), axis=-1)
                d1 = tf.reduce_min(tf.where(mask1[None, None, None, :], dist, big), axis=-1)
                llrs.append((d1 - d0) / sigma2_r)
            return tf.stack(llrs, axis=-1)

        llr_re = pam_llr(dist_re)
        llr_im = pam_llr(dist_im)
        llr = tf.concat([llr_re, llr_im], axis=-1)
        B = tf.shape(llr)[0]
        S = tf.shape(llr)[1]
        N = tf.shape(llr)[2]
        return tf.reshape(llr, [B, S, N * self.num_bits_per_symbol])

    def _llrs_from_symbols(self, x_hat_d: tf.Tensor, sigma2_eff: tf.Tensor) -> tf.Tensor:
        B = tf.shape(x_hat_d)[0]
        Nd = tf.shape(x_hat_d)[2]
        if self._sionna_demapper is not None:
            no_eff = tf.cast(sigma2_eff, tf.float32)[:, :, None]
            no_eff = tf.tile(no_eff, [1, 1, Nd])
            llr_out = None
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
            if llr_out is not None:
                if (llr_out.shape.rank is not None) and (llr_out.shape.rank == 4):
                    llr = tf.reshape(
                        llr_out,
                        [B, tf.shape(llr_out)[1], tf.shape(llr_out)[2] * tf.shape(llr_out)[3]],
                    )
                else:
                    llr = llr_out
            else:
                llr = self._qam_maxlog_llr(x_hat_d, sigma2_eff)
        else:
            llr = self._qam_maxlog_llr(x_hat_d, sigma2_eff)

        llr = tf.reshape(
            llr,
            [B, self.num_tx, self.num_streams_per_tx, Nd * self.num_bits_per_symbol],
        )
        return tf.cast(llr, self.rdtype)

    def _select_candidate(
        self,
        static: Dict[str, tf.Tensor],
        cond: Dict[str, tf.Tensor],
        selection: str,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        sel = str(selection).lower().strip()
        if sel == "static":
            use_cond = tf.zeros_like(static["pilot_mse"], dtype=tf.bool)
        elif sel == "conditional":
            use_cond = tf.ones_like(static["pilot_mse"], dtype=tf.bool)
        else:
            if self.fallback_enabled:
                use_cond = cond["pilot_mse"] < static["pilot_mse"]
            else:
                use_cond = tf.ones_like(static["pilot_mse"], dtype=tf.bool)
        x_hat_d = tf.where(use_cond[:, None, None], cond["x_hat_d"], static["x_hat_d"])
        sigma2_eff = tf.where(use_cond[:, None], cond["sigma2_eff"], static["sigma2_eff"])
        pilot_mse = tf.where(use_cond, cond["pilot_mse"], static["pilot_mse"])
        return use_cond, x_hat_d, sigma2_eff, pilot_mse

    def forward_data(
        self,
        y: tf.Tensor,
        no: tf.Tensor,
        selection: Optional[str] = None,
        return_aux: bool = True,
    ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        sel = selection or self.default_selection
        y_eff = self._remove_nulled_subcarriers(y)
        L_sym = tf.shape(y_eff)[2]
        K_eff = tf.shape(y_eff)[3]
        y_flat = self._flatten_grid(y_eff)
        y_tilde, _ = self._whiten(y_flat)
        summary = self._pilot_summary(y_tilde, no)
        cond_poles = self.policy(summary)
        static = self._candidate(y_tilde, L_sym, K_eff, self.static_poles)
        cond = self._candidate(y_tilde, L_sym, K_eff, cond_poles)
        use_cond, x_hat_d, sigma2_eff, pilot_mse = self._select_candidate(static, cond, sel)

        aux = {
            "summary": summary,
            "use_conditional": use_cond,
            "pilot_mse_static": static["pilot_mse"],
            "pilot_mse_cond": cond["pilot_mse"],
            "pilot_mse_selected": pilot_mse,
            "sigma2_eff": sigma2_eff,
            "cond_poles": cond_poles,
            "static_poles": tf.broadcast_to(self.static_poles[None, :], tf.shape(cond_poles)),
            "x_hat_d_static": static["x_hat_d"],
            "x_hat_d_cond": cond["x_hat_d"],
        }
        if return_aux:
            return x_hat_d, aux
        return x_hat_d, {}

    def call(self, y: tf.Tensor, no: tf.Tensor, selection: Optional[str] = None) -> tf.Tensor:
        x_hat_d, aux = self.forward_data(y, no, selection=selection, return_aux=True)
        llr = self._llrs_from_symbols(x_hat_d, aux["sigma2_eff"])
        return llr
