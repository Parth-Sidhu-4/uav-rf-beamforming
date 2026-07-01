import numpy as np
from scipy.stats import binom
from mission_profile import MissionProfile
from sinr_models import ARQ_K

def compute_pmcs(
    per_timeseries: np.ndarray,
    critical_mask: np.ndarray,
    success_threshold: float = 0.90,
) -> float:
    """
    Mission success := >= 90% of critical telemetry windows decoded.
    Uses binomial CDF. Robust to finite-sample PER variance.
    """
    critical_pers = per_timeseries[critical_mask]
    p_pkt = 1.0 - float(np.mean(critical_pers))
    n     = int(np.sum(critical_mask))
    k_min = int(np.ceil(success_threshold * n))
    return float(1.0 - binom.cdf(k_min - 1, n, p_pkt))

def run_configuration(
    channel_fn,
    eirp_dbw:      float,
    mission:       MissionProfile,
    apply_arq:     bool  = False,
    pkts_per_step: int   = 200,
    seed:          int   = 42,
) -> float:
    """
    1. Call channel_fn(eirp_dbw) → (sinr_db, p_out)
    2. Build constant-valued trajectories of length mission.n_steps
    3. Call phase_d_runner.run_sweep() to get per-step app_per
    4. If apply_arq: per_step = per_step^(ARQ_K + 1)  [independent retransmission model]
    5. Return compute_pmcs(per_timeseries, mission.critical_mask)
    """
    from phase_d_runner import run_sweep

    sinr_db, p_out = channel_fn(eirp_dbw)
    sigma_theta    = 0.05  # fixed: Phase C assumed converged

    _, log = run_sweep(
        sinr_trajectory    = [sinr_db]    * mission.n_steps,
        p_out_trajectory   = [p_out]      * mission.n_steps,
        sigma_theta_trajectory = [sigma_theta] * mission.n_steps,
        pkts_per_step      = pkts_per_step,
        seed               = seed,
    )

    per_ts = np.array([s['app_per'] for s in log])

    if apply_arq:
        per_ts = per_ts ** (ARQ_K + 1)

    return compute_pmcs(per_ts, mission.critical_mask)
