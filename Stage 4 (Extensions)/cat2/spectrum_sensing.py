"""
cat2/spectrum_sensing.py
Extension 2b: Cognitive Spectrum Sensing via Energy Detection
Implements the Neyman-Pearson energy detector (Section 4.2 of Extension Plan).
"""
import numpy as np
from scipy.special import gammaincc, gammaincinv
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class SensingConfig:
    n_samples:   int   = 256    # N: number of complex samples per sensing slot
    pfa_target:  float = 0.01  # desired false-alarm probability P_fa
    noise_power: float = 1.0   # σ_w² — estimated noise power (linear watts)


class EnergyDetector:
    """
    Neyman-Pearson energy detector for a single candidate frequency channel.

    Statistical model (Section 4.2.1):
    - H0 (free):  T ~ Gamma(N, σ_w²/N)   → P_fa = Γ(N, N·λ/σ_w²)/Γ(N)
    - H1 (jammed): T ~ Gamma(N, (σ_w²+Ps)/N) → P_d = Γ(N, N·λ/(σ_w²+Ps))/Γ(N)
    - Threshold λ derived from P_fa target via inverse incomplete gamma.
    """

    def __init__(self, config: SensingConfig):
        self.cfg = config
        self._recompute_threshold()

    def _recompute_threshold(self):
        N, sig2 = self.cfg.n_samples, self.cfg.noise_power
        # gammaincinv(N, 1 - P_fa) gives the argument x s.t. gammaincc(N,x) = P_fa
        inv_arg = gammaincinv(N, 1.0 - self.cfg.pfa_target)
        self.threshold = sig2 * inv_arg / N

    def set_noise_power(self, noise_power: float):
        """Update noise floor estimate and recompute threshold."""
        self.cfg.noise_power = noise_power
        self._recompute_threshold()

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------
    def detect(self, y: np.ndarray) -> bool:
        """
        Run energy detection on N complex samples.
        Returns True if channel is declared occupied (H1).

        Parameters
        ----------
        y : ndarray (N,) complex — received baseband samples
        """
        T = float(np.mean(np.abs(y) ** 2))
        return T > self.threshold

    # ------------------------------------------------------------------
    # Analytical performance metrics
    # ------------------------------------------------------------------
    def compute_pd(self, snr_linear: float) -> float:
        """
        Theoretical detection probability P_d for a given linear SNR Ps/σ_w².
        """
        N, sig2 = self.cfg.n_samples, self.cfg.noise_power
        total_power = sig2 * (1.0 + snr_linear)
        arg = N * self.threshold / total_power
        return float(gammaincc(N, arg))

    def compute_roc(self, snr_linear: float,
                    n_points: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate parametric ROC curve (P_fa, P_d) by sweeping threshold λ.

        Returns
        -------
        pfa_arr, pd_arr : ndarray (n_points,)
        """
        N, sig2 = self.cfg.n_samples, self.cfg.noise_power
        lambdas = np.linspace(0.01 * sig2, 5.0 * sig2, n_points)
        pfa = np.array([gammaincc(N, N * lam / sig2) for lam in lambdas])
        pd  = np.array([gammaincc(N, N * lam / (sig2 * (1.0 + snr_linear)))
                        for lam in lambdas])
        return pfa, pd

    # ------------------------------------------------------------------
    # Multi-channel scan
    # ------------------------------------------------------------------
    def scan_channels(self, channel_samples: Dict[float, np.ndarray]) -> Dict[float, bool]:
        """
        Detect jammer occupancy across multiple candidate channels.

        Parameters
        ----------
        channel_samples : dict {freq_hz: np.ndarray(N,) complex}

        Returns
        -------
        occupancy : dict {freq_hz: bool}  — True = channel occupied/jammed
        """
        return {f: self.detect(y) for f, y in channel_samples.items()}

    def sensing_overhead_fraction(self) -> float:
        """
        Fraction of hop dwell time consumed by sensing (T_sense / T_hop).
        Useful for computing throughput penalty of cognitive sensing.
        """
        # Assumes sample_rate = 10 MHz (typical SDR), T_sense = N/f_s
        sample_rate_hz = 10e6
        T_sense = self.cfg.n_samples / sample_rate_hz
        T_hop   = 1 / 1600   # default FHSS hop rate
        return min(1.0, T_sense / T_hop)
