"""
cat2/adaptive_mcs.py
Extension 2c: Adaptive Modulation and Coding (AMC)
Implements LTE-inspired MCS table with hysteresis switching (Section 4.3).
"""
import numpy as np
from scipy.special import erfc
from dataclasses import dataclass, field
from typing import List


@dataclass
class MCSEntry:
    index:          int
    label:          str    # e.g. "64QAM-5/6"
    bits_per_symbol: float
    code_rate:      float
    snr_thresh_db:  float  # switch-UP threshold (dB)
    hysteresis_db:  float  # switch-DOWN = snr_thresh_db - hysteresis_db

    @property
    def spectral_efficiency(self) -> float:
        """bits/symbol * code_rate = effective bits/channel use."""
        return self.bits_per_symbol * self.code_rate


# MCS table: 8 modes, MCS-0 is the fallback (always usable)
MCS_TABLE: List[MCSEntry] = [
    MCSEntry(0, "BPSK-1/2",    0.5,  0.5,  -np.inf, 0.0),
    MCSEntry(1, "BPSK-1",      1.0,  1.0,  -1.5,    1.5),
    MCSEntry(2, "QPSK-1/2",    1.0,  0.5,   1.0,    1.5),
    MCSEntry(3, "QPSK-3/4",    1.5,  0.75,  3.5,    1.5),
    MCSEntry(4, "16QAM-1/2",   2.0,  0.5,   7.0,    2.0),
    MCSEntry(5, "16QAM-3/4",   3.0,  0.75, 10.5,    2.0),
    MCSEntry(6, "64QAM-2/3",   4.0,  2/3,  15.0,    2.5),
    MCSEntry(7, "64QAM-5/6",   5.0,  5/6,  18.5,    2.5),
]


class AMCController:
    """
    Adaptive MCS selection with hysteresis to prevent oscillation.

    Switching rules (Section 4.3.1):
    - Switch UP   to MCS k+1  if SNR ≥ snr_thresh_db[k+1]
    - Switch DOWN to MCS k-1  if SNR <  snr_thresh_db[k] - hysteresis_db[k]
    """

    def __init__(self, channel_bw_hz: float = 5e6, ber_target: float = 1e-3):
        self.bw = channel_bw_hz
        self.ber_target = ber_target
        self.current_mcs_idx = 0
        self.history: List[int] = []

    def select_mcs(self, snr_db: float) -> MCSEntry:
        """
        Select MCS based on instantaneous SNR with hysteresis.

        Parameters
        ----------
        snr_db : float — instantaneous channel SNR in dB.

        Returns
        -------
        MCSEntry — the chosen modulation-coding scheme.
        """
        curr = self.current_mcs_idx

        # Try switching UP
        if curr < len(MCS_TABLE) - 1:
            if snr_db >= MCS_TABLE[curr + 1].snr_thresh_db:
                self.current_mcs_idx = curr + 1
                self.history.append(self.current_mcs_idx)
                return MCS_TABLE[self.current_mcs_idx]

        # Try switching DOWN
        if curr > 0:
            down_thresh = (MCS_TABLE[curr].snr_thresh_db
                           - MCS_TABLE[curr].hysteresis_db)
            if snr_db < down_thresh:
                self.current_mcs_idx = curr - 1
                self.history.append(self.current_mcs_idx)
                return MCS_TABLE[self.current_mcs_idx]

        self.history.append(curr)
        return MCS_TABLE[curr]

    def throughput_bps(self, snr_db: float) -> float:
        """
        Instantaneous throughput [bps] for the MCS selected at this SNR.
        throughput = spectral_efficiency [bits/sym] × channel_bandwidth [Hz]
        """
        mcs = self.select_mcs(snr_db)
        return mcs.spectral_efficiency * self.bw

    def mode_occupancy(self, sinr_trace: np.ndarray) -> dict:
        """
        Run the AMC controller over a SINR time-series.
        Returns mode occupancy as a fraction of time in each MCS.

        Parameters
        ----------
        sinr_trace : ndarray (T,) — SINR samples in dB over time.

        Returns
        -------
        dict {mcs_label: fraction_of_time}
        """
        self.current_mcs_idx = 0   # reset state
        self.history = []
        for s in sinr_trace:
            self.select_mcs(s)
        total = len(self.history)
        counts = {m.label: 0 for m in MCS_TABLE}
        for idx in self.history:
            counts[MCS_TABLE[idx].label] += 1
        return {k: v / total for k, v in counts.items()}

    # ------------------------------------------------------------------
    # Analytical BER models
    # ------------------------------------------------------------------
    @staticmethod
    def ber_mqam(M: int, snr_b_linear: float) -> float:
        """
        Approximate BER for Gray-coded square M-QAM (Section 4.3.1).

        Parameters
        ----------
        M            : int — constellation size (2, 4, 16, 64, …)
        snr_b_linear : float — Eb/N0 (linear, not dB)
        """
        if M == 2:   # BPSK
            return 0.5 * erfc(np.sqrt(snr_b_linear))
        bits = np.log2(M)
        k    = (4 / bits) * (1 - 1 / np.sqrt(M))
        arg  = np.sqrt(3 * snr_b_linear * bits / (M - 1))
        return k * 0.5 * erfc(arg / np.sqrt(2))

    @staticmethod
    def shannon_capacity_bps(snr_linear: float, bandwidth_hz: float) -> float:
        """Shannon capacity upper bound [bps]."""
        return bandwidth_hz * np.log2(1.0 + snr_linear)
