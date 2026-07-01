"""
cat2/fhss_jammer.py
Extension 2a: FHSS + Partial-Band Jammer & Follower Jammer
Implements the full mathematical model from the Extension Plan Section 4.1.
"""
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class JammerType(Enum):
    PARTIAL_BAND = "partial_band"
    FOLLOWER     = "follower"
    FULL_BAND    = "full_band"


@dataclass
class FHSSConfig:
    n_channels:     int   = 79           # Bluetooth/ISM: 79 channels
    bandwidth_hz:   float = 83.5e6      # Total spread bandwidth (2.4 GHz ISM)
    hop_period_s:   float = 1 / 1600    # dwell time per hop (seconds)
    seed:           int   = 42           # PRNG seed for reproducibility


@dataclass
class JammerConfig:
    jammer_type:       JammerType = JammerType.PARTIAL_BAND
    total_power_w:     float = 1.0       # total jammer transmit power (Watts)
    rho:               float = 0.3       # partial-band fraction ρ ∈ (0, 1]
    detection_delay_s: float = 2e-4     # follower jammer reaction latency (s)


class FHSSSystem:
    """
    Simulate FHSS communication link under partial-band or follower jamming.

    Mathematical basis (Section 4.1.1 of Extension Plan):
    - BER under PBJ (BFSK, non-coherent):
        BER = ρ * 0.5*exp(-Eb/N0 / (2*(1 + JSR/ρ))) + (1-ρ) * 0.5*exp(-Eb/N0 / 2)
    - Worst-case ρ*: minimise coverage, maximise damage.
    - Follower jammer: α_follow = max(0, 1 - τ_d / T_hop)
    """

    def __init__(self, config: FHSSConfig, jammer: JammerConfig):
        self.cfg = config
        self.jam = jammer
        self.rng = np.random.default_rng(config.seed)
        # Fixed frequency channel grid
        self._freqs = np.linspace(2.4e9,
                                  2.4e9 + config.bandwidth_hz,
                                  config.n_channels)

    # ------------------------------------------------------------------
    # Core hop SINR computation
    # ------------------------------------------------------------------
    def compute_per_hop_sinr_db(self,
                                 received_power_w: float,
                                 noise_floor_w:    float,
                                 n_hops:           int = 1000) -> np.ndarray:
        """
        Compute per-hop SINR (dB) over n_hops frequency hops.

        Returns
        -------
        sinr_db : ndarray (n_hops,)
        """
        ch_bw = self.cfg.bandwidth_hz / self.cfg.n_channels

        if self.jam.jammer_type == JammerType.PARTIAL_BAND:
            n_jammed = max(1, int(self.jam.rho * self.cfg.n_channels))
            jammed_set = set(
                self.rng.choice(self.cfg.n_channels, n_jammed, replace=False)
            )
            J_per_ch = self.jam.total_power_w / n_jammed

        elif self.jam.jammer_type == JammerType.FOLLOWER:
            alpha = max(0.0,
                        1.0 - self.jam.detection_delay_s / self.cfg.hop_period_s)
            J_per_ch   = self.jam.total_power_w * alpha
            jammed_set = set(range(self.cfg.n_channels))   # all channels, reduced power

        else:  # FULL_BAND
            jammed_set = set(range(self.cfg.n_channels))
            J_per_ch   = self.jam.total_power_w / self.cfg.n_channels

        hops = self.rng.integers(0, self.cfg.n_channels, size=n_hops)
        sinr_db = np.zeros(n_hops)
        for i, ch in enumerate(hops):
            interference = (J_per_ch if ch in jammed_set else 0.0) \
                           + noise_floor_w * ch_bw
            sinr_db[i] = 10 * np.log10(received_power_w / (interference + 1e-30))
        return sinr_db

    # ------------------------------------------------------------------
    # Analytical BER models
    # ------------------------------------------------------------------
    @staticmethod
    def bfsk_ber_pbj(ebno: float, jsr: float, rho: float) -> float:
        """
        Closed-form BER for non-coherent BFSK under Partial-Band Jammer.

        Parameters
        ----------
        ebno : float  Eb/N0 (linear, not dB)
        jsr  : float  Jammer-to-Signal Ratio (linear)
        rho  : float  Partial-band fraction ∈ (0, 1]
        """
        if not (0 < rho <= 1):
            raise ValueError(f"rho must be in (0, 1], got {rho}")
        ber_jammed = 0.5 * np.exp(-ebno / (2 * (1.0 + jsr / rho)))
        ber_clean  = 0.5 * np.exp(-ebno / 2.0)
        return rho * ber_jammed + (1.0 - rho) * ber_clean

    @staticmethod
    def worst_case_rho(ebno: float, jsr: float) -> float:
        """Optimal jammer fraction ρ* that maximises BER for PBJ."""
        return min(1.0, 0.709 / (ebno * jsr + 1e-30))

    # ------------------------------------------------------------------
    # Follower jammer efficiency
    # ------------------------------------------------------------------
    def follower_jam_fraction(self) -> float:
        """α_follow = fraction of hop dwell time the follower can jam."""
        return max(0.0,
                   1.0 - self.jam.detection_delay_s / self.cfg.hop_period_s)

    # ------------------------------------------------------------------
    # Summary stats
    # ------------------------------------------------------------------
    def sweep_rho_ber(self,
                      ebno_db:   float,
                      jsr_db:    float,
                      n_points:  int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sweep ρ ∈ [0.01, 1] and return (rho_arr, ber_arr) for plotting.
        Marks worst-case ρ* analytically.
        """
        ebno = 10 ** (ebno_db / 10)
        jsr  = 10 ** (jsr_db  / 10)
        rho_arr = np.linspace(0.01, 1.0, n_points)
        ber_arr = np.array([self.bfsk_ber_pbj(ebno, jsr, r) for r in rho_arr])
        return rho_arr, ber_arr
