"""
cat5/alpha_sweep.py
Extension 5c: RNCO Aggressiveness (Alpha) Sensitivity Sweep

Mathematical basis (Section 7.3 of Extension Plan):
  Link score: S_k = SINR_k/gamma_thresh - 1 + alpha * dSINR/dt / gamma_thresh
  Link dropped when S_k < 0.

Wiring to Stage 2:
  - Uses baseline_channel() / phase_b_channel() from Stage 3 sinr_models.py
    to generate a realistic SINR time-series
  - Uses SINR_THRESH_DB from Stage 3 constants
  - Does NOT modify the RNCO class in Stage 2 — wraps it via the link-score
    formula applied to a synthetic SINR trace driven by the real channel model
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Callable

from channel_bridge import (
    baseline_channel, phase_b_channel, phase_bc_channel,
    SINR_THRESH_DB, NULL_DEPTH_DB
)


@dataclass
class AlphaSweepConfig:
    alpha_values: List[float] = field(default_factory=lambda: list(np.linspace(0.05, 3.0, 30)))
    n_monte_carlo: int = 300           # MC trials per alpha
    sim_duration_s: float = 120.0     # seconds (matches MTTK_SEC from Stage 2)
    dt_s: float = 1.0                 # RNCO epoch duration (1 s)
    gamma_thresh_db: float = SINR_THRESH_DB   # 3.0 dB from Stage 3 constants
    eirp_sweep_dbw: np.ndarray = field(
        default_factory=lambda: np.linspace(-5, 50, 100))


def _sinr_trajectory(duration_s: float, dt_s: float,
                     eirp_profile: np.ndarray,
                     rng: np.random.Generator) -> np.ndarray:
    """
    Build a SINR time-series using the Phase D channel model plus realistic
    UAV maneuvering-induced SINR variation.

    Physics: When the UAV manoeuvres, its angle to the jammer changes.
    The LCMV null is not perfectly tracking (finite update rate), so there are
    moments of increased jammer leakage. This is modelled as an oscillating
    component of ±A_manoeuvre dB superimposed on the Phase B base SINR.

    Also adds Rician K=12 fading variance: sigma_fading ≈ sqrt(1/(K+1)) * 4.34 dB
    (log-normal approximation for Rician power fluctuation at high K).
    """
    T = int(duration_s / dt_s)
    t = np.arange(T) * dt_s

    # Phase B base SINR from real Stage 3 channel (9 dB until null breaks)
    eirp_idx    = np.linspace(0, len(eirp_profile) - 1, T).astype(int)
    sinr_base   = np.array([phase_b_channel(eirp_profile[i])[0]
                             for i in eirp_idx])

    # UAV maneuvering oscillation: ±3 dB peak at T_man = 15 s period
    # Models null pointing error as UAV turns away from nominal heading
    A_man  = 3.0    # dB peak-to-peak amplitude
    T_man  = 15.0   # seconds per manoeuvre cycle
    manoeuvre = A_man * np.sin(2 * np.pi * t / T_man)

    # Rician K=12 amplitude variance in dB
    # sigma_dB = 4.34 / sqrt(K+1) = 4.34 / sqrt(13) ≈ 1.20 dB
    sigma_rician = 4.34 / np.sqrt(12.0 + 1)
    fading = rng.normal(0, sigma_rician, size=T)

    return sinr_base + manoeuvre + fading


def _link_score(sinr_db: float, sinr_prev_db: float,
                alpha: float, gamma_thresh_db: float,
                dt_s: float) -> float:
    """
    RNCO link score S_k (Section 7.3.1):
    S_k = SINR_k/gamma - 1 + alpha * dSINR/dt / gamma
    """
    gamma = 10 ** (gamma_thresh_db / 10)
    sinr_lin = 10 ** (sinr_db / 10)
    d_sinr_dt = (sinr_db - sinr_prev_db) / dt_s
    return sinr_lin / gamma - 1.0 + alpha * d_sinr_dt / gamma


def simulate_one_trial(alpha: float, seed: int,
                       config: AlphaSweepConfig) -> Dict:
    """
    Simulate one mission at a given alpha value.

    Returns
    -------
    dict with keys: n_drops, n_false_drops, n_total_links,
                    throughput_bps, mission_success
    """
    rng = np.random.default_rng(seed)
    sinr_trace = _sinr_trajectory(
        config.sim_duration_s, config.dt_s,
        config.eirp_sweep_dbw, rng
    )
    T = len(sinr_trace)
    gamma_thresh_db = config.gamma_thresh_db

    n_drops       = 0
    n_false_drops = 0
    throughput_samples = []

    for k in range(1, T):
        sinr_k    = sinr_trace[k]
        sinr_prev = sinr_trace[k - 1]
        score     = _link_score(sinr_k, sinr_prev, alpha,
                                 gamma_thresh_db, config.dt_s)
        actual_ok = sinr_k >= gamma_thresh_db

        if score < 0:          # link dropped by RNCO
            n_drops += 1
            if actual_ok:      # false drop: SINR was fine but alpha triggered drop
                n_false_drops += 1
            throughput_samples.append(0.0)
        else:
            # AMC throughput (capped at 8-mode MCS table max, 5 MHz channel)
            # MCS table (snr_thresh_db, spectral_efficiency bpcu)
            _MCS = [(-99,0.25),(-1.5,1.0),(1.0,0.5),(3.5,1.125),
                    (7.0,1.0),(10.5,2.25),(15.0,8/3),(18.5,25/6)]
            se = _MCS[0][1]
            for _thresh, _se in _MCS[1:]:
                if sinr_k >= _thresh: se = _se
            tput = se * 5e6
            throughput_samples.append(tput)

    # Mission success: at least 90% of epochs maintained link
    success_rate  = 1.0 - (n_drops / max(T - 1, 1))
    mission_success = success_rate >= 0.90

    return {
        'n_drops':        n_drops,
        'n_false_drops':  n_false_drops,
        'n_total_links':  T - 1,
        'throughput_bps': float(np.mean(throughput_samples)),
        'mission_success': float(mission_success),
    }


def run_alpha_sweep(config: AlphaSweepConfig) -> Dict[float, Dict]:
    """
    Full alpha sweep with Monte Carlo averaging.

    Returns
    -------
    results : dict {alpha: {mean_fdr, mean_throughput_mbps,
                             success_rate, mean_drops_per_min}}
    """
    results = {}
    for alpha in config.alpha_values:
        trials = [simulate_one_trial(alpha, seed, config)
                  for seed in range(config.n_monte_carlo)]

        n_drops    = np.mean([t['n_drops']        for t in trials])
        n_false    = np.mean([t['n_false_drops']   for t in trials])
        throughput = np.mean([t['throughput_bps']  for t in trials])
        success    = np.mean([t['mission_success'] for t in trials])
        n_total    = np.mean([t['n_total_links']   for t in trials])

        results[alpha] = {
            'mean_fdr':          n_false / (n_total + 1e-9),
            'mean_throughput_mbps': throughput / 1e6,
            'success_rate':      success,
            'mean_drops_per_min': n_drops / (config.sim_duration_s / 60),
        }
    return results


def find_critical_alpha(sweep_results: Dict[float, Dict],
                        fdr_threshold: float = 0.05) -> float:
    """
    Find smallest alpha where mean_fdr first exceeds fdr_threshold.
    Returns inf if FDR never reaches the threshold.
    """
    for alpha in sorted(sweep_results.keys()):
        if sweep_results[alpha]['mean_fdr'] >= fdr_threshold:
            return alpha
    return float('inf')
