from scipy.constants import c, pi
from scipy.stats import ncx2
import numpy as np

RICIAN_K       = 12.0
SINR_THRESH_DB = 3.0          # outage threshold
NULL_DEPTH_DB  = 45.0         # Phase B LCMV null (conservative, accounts for pointing error)
RNCO_REDUCTION = 0.15         # Phase C: residual p_out fraction (85% reduction)
ARQ_K          = 3            # Phase D: retransmit attempts; effective PER = PER^(K+1)

def fspl_db(range_m: float, fc_hz: float) -> float:
    return 20 * np.log10(4 * pi * range_m * fc_hz / c)

def _raw_sinr_and_pout(p_signal_dbm, p_jammer_dbm, noise_dbm) -> tuple[float, float]:
    sig  = 10 ** (p_signal_dbm / 10)
    jam  = 10 ** (p_jammer_dbm / 10)
    nse  = 10 ** (noise_dbm    / 10)
    sinr_lin   = sig / (jam + nse)
    sinr_db    = 10 * np.log10(sinr_lin)
    snr_avg    = sig / nse
    gamma_th   = 10 ** (SINR_THRESH_DB / 10)
    p_out = float(np.clip(
        ncx2.cdf(2 * gamma_th / snr_avg, df=2, nc=2 * RICIAN_K), 0.0, 1.0
    ))
    return sinr_db, p_out

def baseline_channel(eirp_dbw: float) -> tuple[float, float]:
    P_signal_dbm = (10.0 + 30) - fspl_db(1000.0, 2.4e9)
    P_jammer_dbm = (eirp_dbw + 30) - fspl_db(500.0, 2.4e9)
    return _raw_sinr_and_pout(P_signal_dbm, P_jammer_dbm, -100.0)

def phase_b_channel(eirp_dbw: float) -> tuple[float, float]:
    P_signal_dbm = (10.0 + 30) - fspl_db(1000.0, 2.4e9)
    P_jammer_dbm = (eirp_dbw + 30) - fspl_db(500.0, 2.4e9)
    P_jammer_eff_dbm = P_jammer_dbm - NULL_DEPTH_DB
    return _raw_sinr_and_pout(P_signal_dbm, P_jammer_eff_dbm, -100.0)

def phase_bc_channel(eirp_dbw: float) -> tuple[float, float]:
    sinr_db, p_out = phase_b_channel(eirp_dbw)
    return sinr_db, p_out * RNCO_REDUCTION
