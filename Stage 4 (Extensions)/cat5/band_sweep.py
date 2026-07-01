"""
cat5/band_sweep.py
Extension 5a: Frequency Band Sweep (900 MHz / 2.4 GHz / 5.8 GHz)

Mathematical basis (Section 7.1.1 of Extension Plan):
  - FSPL(d, f) = 20*log10(d) + 20*log10(f) - 147.55
  - K(f) = K0 * (f/f0)^alpha_K,  alpha_K = -0.3  (ITU-R P.1411)
  - sigma_sf(f) = 4.0 + 1.5*log10(f/f0)  (shadow fading std dev)
  - Outage via rician_mrc_outage() from Stage 2 (parametrized by K and L_MRC)

Integration:
  - fspl_db()          from Stage 3 sinr_models.py (real validated formula)
  - rician_mrc_outage() from Stage 2 simulator_core_remediated.py
  - RICIAN_K, SINR_THRESH_DB, L_MRC, NOISE_DBM from channel_bridge constants
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))

import numpy as np
from dataclasses import dataclass, field
from typing import Dict

from channel_bridge import (
    fspl_db, rician_mrc_outage,
    RICIAN_K, SINR_THRESH_DB, NOISE_DBM, L_MRC
)

# ─────────────────────────────────────────────────────────────────────────────
# Band definitions
# ─────────────────────────────────────────────────────────────────────────────
BANDS_HZ: Dict[str, float] = {
    "900 MHz":  900e6,
    "2.4 GHz": 2.4e9,
    "5.8 GHz": 5.8e9,
}

# K-factor frequency scaling: K(f) = K0 * (f/f0)^alpha_K
K0_REF    = RICIAN_K     # K at 2.4 GHz (from Stage 2, = 12.0)
F0_REF    = 2.4e9        # reference frequency
ALPHA_K   = -0.3         # empirical exponent (ITU-R P.1411)

# Fixed-gain antenna assumed (EIRP constant across bands)
TX_EIRP_DBM = 40.0       # dBm  (10 dBW = consistent with Phase A GCS)


@dataclass
class BandSweepResult:
    band_name:    str
    freq_hz:      float
    distances_m:  np.ndarray
    fspl_db_arr:  np.ndarray
    k_factor:     float
    sigma_sf_db:  float       # shadow fading std dev
    snr_db_arr:   np.ndarray  # SNR before LCMV (signal only, no jammer)
    p_out_arr:    np.ndarray  # Rician MRC outage probability at each distance


def run_band_sweep(distances_m: np.ndarray = None,
                   tx_eirp_dbm: float = TX_EIRP_DBM,
                   noise_dbm:   float = NOISE_DBM,
                   l_mrc:       int   = L_MRC) -> Dict[str, BandSweepResult]:
    """
    Sweep over 900 MHz, 2.4 GHz, 5.8 GHz for a range of GCS–UAV distances.

    Returns
    -------
    dict {band_name: BandSweepResult}
    """
    if distances_m is None:
        distances_m = np.linspace(50, 3000, 300)

    results = {}
    for name, freq in BANDS_HZ.items():
        # FSPL at this frequency (Stage 3 validated formula)
        fspl = np.array([fspl_db(d, freq) for d in distances_m])

        # Received SNR (no jammer, no beamforming gain)
        snr_db = tx_eirp_dbm - fspl - noise_dbm

        # K-factor frequency scaling (ITU-R P.1411)
        K = K0_REF * (freq / F0_REF) ** ALPHA_K

        # Shadow fading std dev (larger at higher frequency in urban/semi-urban)
        sigma_sf = 4.0 + 1.5 * np.log10(freq / F0_REF)

        # Outage probability using Stage 2's rician_mrc_outage() at each distance
        p_out = np.array([
            rician_mrc_outage(SINR_THRESH_DB, s, L_fhss=l_mrc, K=K)
            for s in snr_db
        ])

        results[name] = BandSweepResult(
            band_name=name, freq_hz=freq,
            distances_m=distances_m,
            fspl_db_arr=fspl,
            k_factor=K,
            sigma_sf_db=sigma_sf,
            snr_db_arr=snr_db,
            p_out_arr=p_out,
        )

    return results


def coverage_range_m(result: BandSweepResult,
                     p_out_threshold: float = 0.01) -> float:
    """
    Return the max range (m) where outage probability < p_out_threshold.
    """
    mask = result.p_out_arr < p_out_threshold
    if not np.any(mask):
        return 0.0
    return float(result.distances_m[mask][-1])


def validate_fspl_deltas() -> bool:
    """
    Validate that FSPL deltas match the spec table to within 0.05 dB.
    """
    d = 1000.0
    delta_58_24 = fspl_db(d, 5.8e9) - fspl_db(d, 2.4e9)
    delta_09_24 = fspl_db(d, 0.9e9) - fspl_db(d, 2.4e9)
    ok_58 = abs(delta_58_24 - 7.67) < 0.05
    ok_09 = abs(delta_09_24 - (-8.55)) < 0.05
    return ok_58 and ok_09
