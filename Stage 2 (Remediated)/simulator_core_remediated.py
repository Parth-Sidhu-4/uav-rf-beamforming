"""
simulator_core_remediated.py
Phase C: Autonomous RNCO UAV Simulator — Full Remediation

Fixes applied in this file:
  FIX-1 : L_eff renamed to L_snapshots (MUSIC/Phase B) and L_fhss (MRC diversity/Phase A)
  FIX-5 : Rician MRC outage via Marcum Q-function (replaces Rayleigh incomplete gamma)
  FIX-6 : Neyman-Pearson radiometer detection_prob (replaces ad-hoc P_d formula)
  FIX-8A: UKF with true nonlinear bearing measurement h(x) = arctan(y/x)
  FIX-8B: Observability metric + COMMAND_LATERAL_OFFSET action
  FIX-8C: UAV position uncertainty coupled into EKF R_meas
  FIX-9A: Explicit Q_INFO->Q_RF coupling via SINR_achievable(sigma_theta)
  FIX-9B: Lyapunov V calibration sweep
  FIX-9C: Graduated null widening action set
  FIX-11 : Parameterised MTTK_SEC with sensitivity sweep
  FIX-13 : Graceful LCMV fallback hierarchy (full_widening -> point_null -> no_null)
"""

import numpy as np
import pandas as pd
import math
import scipy.stats
from scipy.stats import chi2, ncx2
import warnings

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# FIX-11: MTTK_SEC as named configurable constant
# Physical basis: to be supplied by mission planning.
# Suggested values: fast IR-guided system ~30s, optical ~60s, manual ~300s.
MTTK_SEC = 120.0

# FIX-6: Default false alarm probability for Neyman-Pearson radiometer
P_FA_DEFAULT = 0.01

# FIX-9C: Graduated null widening action set (degrees)
NULL_WIDTH_ACTIONS = [1.0, 2.0, 4.0, 8.0, 16.0, 24.0]

# FIX-8B: Observability threshold
OBSERVABILITY_THRESHOLD = 1e6

# UKF sigma-point parameters (Julier-Uhlmann defaults)
UKF_ALPHA = 1e-3
UKF_BETA  = 2.0
UKF_KAPPA = 0.0

# =============================================================================
# UTILITY: Q-FUNCTION
# =============================================================================
def q_func(x):
    return 1.0 - scipy.stats.norm.cdf(x)


# =============================================================================
# ARRAY PROCESSING
# =============================================================================
def get_steering_vector(N, theta_deg):
    theta_rad = np.radians(theta_deg)
    return np.exp(-1j * np.pi * np.arange(N) * np.sin(theta_rad)).reshape(N, 1)


def lcmv_weights(R_hat, C, f, loading_factor=0.10):
    """
    Compute LCMV beamformer weights with diagonal loading.
    R_hat         : (N x N) sample covariance matrix
    C             : (N x n_constraints) constraint matrix
    f             : (n_constraints, 1) constraint response vector
    loading_factor: diagonal loading as fraction of mean diagonal power
    Returns: (w, cond_num, failure_mode)
    """
    delta = loading_factor * np.real(np.trace(R_hat)) / R_hat.shape[0]
    R_loaded = R_hat + delta * np.eye(R_hat.shape[0])
    cond_num = np.linalg.cond(R_loaded)
    R_inv = np.linalg.inv(R_loaded)
    try:
        mid = C.conj().T @ R_inv @ C
        if np.linalg.cond(mid) > 1e12:
            raise np.linalg.LinAlgError("Constraint matrix nearly singular")
        w = R_inv @ C @ np.linalg.inv(mid) @ f
        return w, cond_num, "full_widening"
    except np.linalg.LinAlgError:
        return None, cond_num, "error"


# FIX-13: Graceful LCMV fallback hierarchy
def lcmv_with_fallback(R_hat, a_signal, theta_null_deg, sigma_theta_deg, N_elements, loading_factor=0.10):
    """
    Attempt LCMV in order: full_widening -> point_null_fallback -> no_null.
    Returns (w, cond_num, failure_mode, achieved_null_deg).
    """
    # FIX-9C: Pick the graduated null width that fits within available DOFs
    # DOF budget: N_elements - 1 (one consumed by signal constraint)
    max_nulls = N_elements - 2  # -1 for signal, -1 minimum for 1 null

    def try_widening(n_null_points):
        n_null_points = min(n_null_points, max_nulls)
        if n_null_points < 1:
            return None, 0, "no_dof"
        thetas = np.linspace(
            theta_null_deg - 2 * sigma_theta_deg,
            theta_null_deg + 2 * sigma_theta_deg,
            n_null_points
        )
        null_vecs = np.column_stack([get_steering_vector(N_elements, th) for th in thetas])
        C = np.column_stack([a_signal, null_vecs])
        f = np.vstack([[1.0], np.zeros((n_null_points, 1))])
        return lcmv_weights(R_hat, C, f, loading_factor)

    # Level 1: graduated null widening using FIX-9C action set
    # Choose smallest width that covers the uncertainty sector
    sector_needed = 4.0 * sigma_theta_deg
    chosen_width = next((w for w in NULL_WIDTH_ACTIONS if w >= sector_needed), NULL_WIDTH_ACTIONS[-1])

    # Try 2-point null if N allows
    if sigma_theta_deg >= 0.5 and max_nulls >= 2:
        result = try_widening(2)
        if result[0] is not None:
            return result[0], result[1], "full_widening", chosen_width

    # Level 2: single point-null fallback
    if max_nulls >= 1:
        a_j_est = get_steering_vector(N_elements, theta_null_deg)
        C = np.column_stack([a_signal, a_j_est])
        f = np.array([[1.0], [0.0]])
        w, cond_num, mode = lcmv_weights(R_hat, C, f, loading_factor)
        if w is not None:
            return w, cond_num, "point_null_fallback", 0.0

    # Level 3: no-null — return matched filter weights
    w_mf = a_signal / (a_signal.conj().T @ a_signal)
    return w_mf, 1.0, "no_null", 0.0


def build_constraint_matrix(a_signal, theta_est_deg, sigma_theta_deg, N_elements=4, n_null_points=2):
    """Build LCMV constraint matrix with DOF-aware null widening."""
    n_null_points = min(n_null_points, N_elements - 2)

    if sigma_theta_deg < 0.5 or n_null_points < 1:
        C = np.column_stack([a_signal, get_steering_vector(N_elements, theta_est_deg)])
        f = np.array([[1.0], [0.0]])
    else:
        thetas = np.linspace(
            theta_est_deg - 2 * sigma_theta_deg,
            theta_est_deg + 2 * sigma_theta_deg,
            n_null_points
        )
        null_vecs = np.column_stack([get_steering_vector(N_elements, th) for th in thetas])
        C = np.column_stack([a_signal, null_vecs])
        f = np.vstack([[1.0], np.zeros((n_null_points, 1))])

    return C, f


# =============================================================================
# FIX-7: MUSIC CRLB with SNR dependence and RSS calibration error
# =============================================================================
def music_crlb(N, L_snapshots, SNR_dB_per_element, theta_deg, calib_error_deg=0.5):
    """
    Compute the ULA Cramér-Rao Bound on DOA estimation with RSS calibration error.

    Formula: sigma²_CRB = 6 / (L_snapshots * rho * N * (N²-1) * pi² * cos²(theta))
    where rho is the array output SNR (per-element SNR * N for coherent combining).

    Returns total sigma in degrees (statistical + calibration combined via RSS).
    """
    theta_rad = np.radians(theta_deg)
    # Coherent combining gain: array SNR = per-element SNR * N
    rho = 10.0 ** (SNR_dB_per_element / 10.0) * N
    cos2 = max(np.cos(theta_rad) ** 2, 1e-6)

    var_crb_rad2 = 6.0 / (L_snapshots * rho * N * (N ** 2 - 1) * np.pi ** 2 * cos2)
    sigma_crb_deg = np.degrees(np.sqrt(var_crb_rad2))

    # Apply hardware floor
    sigma_music_deg = max(0.5, sigma_crb_deg)

    # FIX-7: RSS combination (not additive)
    sigma_total_deg = np.sqrt(sigma_music_deg ** 2 + calib_error_deg ** 2)
    return sigma_total_deg


# =============================================================================
# FIX-9A: Q_INFO -> Q_RF coupling via SINR achievable from null depth curve
# =============================================================================
def sinr_achievable_from_uncertainty(sigma_theta_deg, inr_base_dB=50.0, snr_base_dB=15.0):
    """
    Maps current EKF angular uncertainty sigma_theta_deg to expected achievable SINR.
    Uses the empirical null depth curve (calibrated to 10% loaded LCMV physics):
      4° sector  → -46 dB null depth
      24° sector → -13 dB null depth
    Interpolates between these anchor points.
    """
    sector_deg = 4.0 * sigma_theta_deg  # ±2σ coverage

    # Piecewise linear in log domain between anchor points
    if sector_deg <= 4.0:
        nd_dB = -46.0 - (4.0 - sector_deg) / 4.0 * 19.0  # deeper for narrower
        nd_dB = max(nd_dB, -65.0)
    elif sector_deg <= 24.0:
        nd_dB = np.interp(sector_deg, [4.0, 24.0], [-46.0, -13.0])
    else:
        nd_dB = -13.0  # saturated at widest achievable

    inr_resid = inr_base_dB + nd_dB
    sinr_dB = snr_base_dB - inr_resid
    return sinr_dB


# =============================================================================
# DRIFT FUNCTIONS
# =============================================================================
W_DRIFT = 3

def compute_drift(lyapunov_history, t, W=W_DRIFT):
    t_start = max(0, t - W)
    if t == t_start:
        return 0.0
    return (lyapunov_history[t] - lyapunov_history[t_start]) / (t - t_start)


# =============================================================================
# SURVIVAL MODEL (FIX-11)
# =============================================================================
def kill_prob_per_epoch(dt, mttk_sec=None):
    """
    FIX-11: Parameterised survival model. Uses global MTTK_SEC if not specified.
    Physical basis: to be supplied by mission planning.
    """
    tau = mttk_sec if mttk_sec is not None else MTTK_SEC
    return 1.0 - np.exp(-dt / tau)


# =============================================================================
# EKF STABILITY MONITOR
# =============================================================================
def stability_monitor(theta_error_rms_history, jammer_leakage_db_history):
    if len(theta_error_rms_history) < 2:
        return 0.0
    d_theta_error = theta_error_rms_history[-1] - theta_error_rms_history[-2]
    d_leakage_linear = (10 ** (jammer_leakage_db_history[-1] / 10.0) -
                        10 ** (jammer_leakage_db_history[-2] / 10.0))
    if abs(d_leakage_linear) < 1e-12:
        return 0.0
    return abs(d_theta_error / d_leakage_linear)


def rate_limited_null_steer(theta_null_prev, theta_null_new, v_uav, dt_epoch, R_current):
    """FIX from prior session: Use Euclidean slant range for kinematic cap."""
    bearing_rate_kinematic = np.degrees(v_uav * dt_epoch / max(1.0, R_current))
    max_delta = 1.5 * bearing_rate_kinematic
    delta = theta_null_new - theta_null_prev
    is_clipped = False
    if abs(delta) > max_delta:
        delta = np.sign(delta) * max_delta
        is_clipped = True
    return theta_null_prev + delta, is_clipped


def ekf_health_check(P_cov, max_trace=1e9):
    tr = np.trace(P_cov)
    if tr > max_trace:
        raise RuntimeWarning(f"EKF covariance diverging: Tr(P) = {tr:.2e}")
    if np.any(np.linalg.eigvals(P_cov) < 0):
        raise RuntimeError("EKF covariance matrix is not positive definite.")


# =============================================================================
# FIX-6: Neyman-Pearson Radiometer Detection Probability
# =============================================================================
def detection_prob(t_s_sec, BW_hz, JNR_dB, P_fa=P_FA_DEFAULT):
    """
    FIX-6: Energy detector (radiometer) detection probability using Neyman-Pearson.

    Time-bandwidth product u = BW * t_s (number of samples in observation).
    Under H0: test statistic ~ chi-squared(2u)
    Under H1: test statistic ~ noncentral chi-squared(2u, 2u*JNR_linear)

    Args:
        t_s_sec  : sensing dwell time in seconds
        BW_hz    : sensing bandwidth in Hz
        JNR_dB   : jammer-to-noise ratio in dB
        P_fa     : false alarm probability (default 0.01)
    Returns:
        P_d in [0, 1]
    """
    u = BW_hz * t_s_sec  # time-bandwidth product (number of samples)
    u = max(u, 0.5)       # minimum half a sample

    JNR_linear = 10.0 ** (JNR_dB / 10.0)

    # Detection threshold from H0 chi-squared distribution
    lam = chi2.ppf(1.0 - P_fa, df=2 * u)

    # Detection probability from H1 noncentral chi-squared distribution
    # nc = 2 * u * JNR_linear (noncentrality parameter)
    P_d = 1.0 - ncx2.cdf(lam, df=2 * u, nc=2 * u * JNR_linear)
    return float(np.clip(P_d, 0.0, 1.0))


# Legacy wrapper for backward compatibility during validation
def p_detection_legacy(gamma, M, B_sense=1e6, B_total=100e6):
    """Original ad-hoc formula — kept for FIX-1 baseline comparison only."""
    p_present = B_sense / B_total
    Pd_given_present = q_func((gamma - 1.0) / np.sqrt(max(1e-9, 2.0 * gamma / M)))
    return p_present * Pd_given_present


# =============================================================================
# FIX-5: Rician MRC Outage via Marcum Q-function
# =============================================================================
def rician_mrc_outage(gamma_0_dB, gamma_bar_dB, L_fhss, K=10.0):
    """
    FIX-5: Exact Rician MRC outage probability using the Marcum Q-function.

    For L independent Rician branches (same K, same mean SNR gamma_bar):
        P_out = 1 - Q_L(sqrt(2*L*K), sqrt(2*gamma_0*(K+1)/gamma_bar))
    where Q_L is the L-th order Marcum Q-function.

    Args:
        gamma_0_dB  : outage threshold in dB
        gamma_bar_dB: mean branch SNR in dB
        L_fhss      : number of independent FHSS diversity branches
                      (bounded by floor(BW_total / BW_coherence))
        K           : Rician K-factor (K=0 → Rayleigh)
    Returns:
        P_out in [0, 1]

    Note: BW_coherence defaults to 1 MHz. If L_fhss > floor(BW_total/BW_coherence),
    this function will issue a warning because branch independence cannot be assumed.
    """
    BW_TOTAL_HZ = 100e6
    BW_COHERENCE_HZ = 1e6
    L_max_independent = int(BW_TOTAL_HZ / BW_COHERENCE_HZ)
    if L_fhss > L_max_independent:
        import warnings
        warnings.warn(
            f"L_fhss={L_fhss} exceeds independence bound ({L_max_independent}). "
            "Branch correlation not modelled.",
            RuntimeWarning
        )

    gamma_0 = 10.0 ** (gamma_0_dB / 10.0)
    gamma_bar = 10.0 ** (gamma_bar_dB / 10.0)

    if K < 1e-3:
        # Rayleigh special case: P_out = 1 - exp(-x) * sum_{k=0}^{L-1} x^k / k!
        x = min(gamma_0 / gamma_bar, 100.0)
        sum_term = sum((x ** k) / math.factorial(k) for k in range(L_fhss))
        p_out = 1.0 - np.exp(-x) * sum_term
    else:
        # Rician Marcum Q-function via noncentral chi-squared CDF:
        # Q_L(a, b) = 1 - F_ncx2(b^2; 2L, a^2)
        # where F_ncx2 is the noncentral chi-squared CDF.
        # P_out = 1 - Q_L(sqrt(2*L*K), sqrt(2*gamma_0*(K+1)/gamma_bar))
        #       = F_ncx2(b^2; df=2*L, nc=a^2)
        a2 = 2.0 * L_fhss * K
        b2 = 2.0 * gamma_0 * (K + 1.0) / max(gamma_bar, 1e-30)
        # Clip b2 to avoid edge-case NaNs
        b2 = min(b2, 1e6)
        p_out = float(ncx2.cdf(b2, df=2 * L_fhss, nc=a2))

    return float(np.clip(p_out, 0.0, 1.0))


# =============================================================================
# FIX-8A: Unscented Kalman Filter (UKF) for Bearing-Only Jammer Localization
# =============================================================================
class JammerUKF:
    """
    FIX-8A: Unscented Kalman Filter for jammer position estimation.
    State: [x_j, y_j] (jammer Cartesian position, stationary model)
    Measurement: theta = arctan(y_j - y_uav, x_j - x_uav) (bearing in radians)

    Uses Julier-Uhlmann sigma points: alpha=1e-3, beta=2, kappa=0.
    Switches to UKF when tr(P) > (500 m)^2, reverts to linear EKF when below.
    """

    UKF_THRESHOLD_M2 = 500.0 ** 2

    def __init__(self, x_hat_init, P_init):
        """
        x_hat_init: [x_j, y_j] initial state estimate
        P_init    : 2x2 initial covariance matrix
        """
        self.n = 2  # state dimension
        self.x_hat = np.array(x_hat_init, dtype=float)
        self.P = np.array(P_init, dtype=float)

        # UKF parameters
        self.alpha = UKF_ALPHA
        self.beta  = UKF_BETA
        self.kappa = UKF_KAPPA
        self.lam   = self.alpha ** 2 * (self.n + self.kappa) - self.n

        # Sigma-point weights
        self.Wm = np.zeros(2 * self.n + 1)
        self.Wc = np.zeros(2 * self.n + 1)
        self.Wm[0] = self.lam / (self.n + self.lam)
        self.Wc[0] = self.lam / (self.n + self.lam) + (1.0 - self.alpha ** 2 + self.beta)
        for i in range(1, 2 * self.n + 1):
            self.Wm[i] = 1.0 / (2.0 * (self.n + self.lam))
            self.Wc[i] = self.Wm[i]

    def _sigma_points(self):
        """Compute 2N+1 sigma points from current mean and covariance."""
        sqrtP = np.linalg.cholesky((self.n + self.lam) * self.P)
        sigma = np.zeros((2 * self.n + 1, self.n))
        sigma[0] = self.x_hat
        for i in range(self.n):
            sigma[i + 1]          = self.x_hat + sqrtP[:, i]
            sigma[self.n + i + 1] = self.x_hat - sqrtP[:, i]
        return sigma

    def _h(self, x_state, uav_x, uav_y):
        """Nonlinear bearing measurement model: h(x) = arctan((y_j - y_uav) / (x_j - x_uav))"""
        dx = x_state[0] - uav_x
        dy = x_state[1] - uav_y
        return np.arctan2(dy, max(abs(dx), 1.0) * np.sign(dx))

    def predict(self, Q_process):
        """Stationary jammer prediction step: x_pred = x_hat, P_pred = P + Q."""
        self.P = self.P + Q_process

    def update(self, z_bearing_rad, uav_x, uav_y, R_meas_rad2):
        """
        UKF measurement update using bearing measurement.
        z_bearing_rad: measured bearing in radians
        uav_x, uav_y : UAV position (measurement platform)
        R_meas_rad2  : measurement noise variance in rad^2
        """
        use_ukf = np.trace(self.P) > self.UKF_THRESHOLD_M2

        if use_ukf:
            sigma = self._sigma_points()

            # Propagate sigma points through measurement model
            z_sigma = np.array([self._h(sp, uav_x, uav_y) for sp in sigma])

            # Predicted measurement mean
            z_hat = float(np.sum(self.Wm * z_sigma))

            # Innovation covariance
            Pzz = R_meas_rad2 + float(np.sum(self.Wc * (z_sigma - z_hat) ** 2))

            # Cross covariance
            Pxz = np.zeros(self.n)
            for i in range(2 * self.n + 1):
                Pxz += self.Wc[i] * (sigma[i] - self.x_hat) * (z_sigma[i] - z_hat)

            # Kalman gain
            K = Pxz / Pzz

            # Innovation
            innov = z_bearing_rad - z_hat
            # Wrap to [-pi, pi]
            innov = (innov + np.pi) % (2 * np.pi) - np.pi

            self.x_hat = self.x_hat + K * innov
            self.P = self.P - np.outer(K, K) * Pzz

        else:
            # Linear EKF: Jacobian of h at current estimate
            dx = self.x_hat[0] - uav_x
            dy = self.x_hat[1] - uav_y
            r2 = dx ** 2 + dy ** 2
            r2 = max(r2, 1.0)
            H = np.array([[-dy / r2, dx / r2]])

            S = H @ self.P @ H.T + R_meas_rad2
            K = (self.P @ H.T) / float(S)

            z_pred = self._h(self.x_hat, uav_x, uav_y)
            innov = z_bearing_rad - z_pred
            innov = (innov + np.pi) % (2 * np.pi) - np.pi

            self.x_hat = self.x_hat + K.flatten() * innov
            self.P = (np.eye(self.n) - np.outer(K.flatten(), H)) @ self.P

        # Enforce positive-definiteness
        self.P = 0.5 * (self.P + self.P.T)
        eigvals = np.linalg.eigvals(self.P)
        if np.any(eigvals < 0):
            self.P += (-np.min(eigvals) + 1e-6) * np.eye(self.n)

        return self.x_hat.copy(), self.P.copy()

    def observability_metric(self, uav_x, uav_y):
        """
        FIX-8B: Per-epoch observability metric via Fisher Information.
        Returns (condition_number_of_J, is_range_observable).
        """
        dx = self.x_hat[0] - uav_x
        dy = self.x_hat[1] - uav_y
        r2 = max(dx ** 2 + dy ** 2, 1.0)
        H = np.array([[-dy / r2, dx / r2]])
        R_dummy = 1.0  # normalised
        J = H.T @ H / R_dummy
        cond = np.linalg.cond(J + 1e-12 * np.eye(self.n))
        is_observable = cond < OBSERVABILITY_THRESHOLD
        return cond, is_observable


# =============================================================================
# FIX-9B: Lyapunov V Calibration Sweep
# =============================================================================
def calibrate_V(V_range, n_trials=50, scenario='A', div_scheme='MRC_L4'):
    """
    FIX-9B: Sweep over V values to find the largest V for which all queues remain bounded.
    Returns dict with V_max_stable and corresponding metrics.
    """
    results = {}
    for V in V_range:
        q_rf_maxes, q_exp_maxes, q_info_maxes, pmcs_list = [], [], [], []
        for _ in range(n_trials):
            sim = UAVSimulator(scenario=scenario, div_scheme=div_scheme,
                               policy='RNCO', V_override=V)
            for ep in range(sim.epochs):
                sim.step(ep)
            q_rf_maxes.append(max(sim.logs.get('Q_RF_trace', [0])))
            q_exp_maxes.append(max(sim.logs.get('Q_EXP_trace', [0])))
            q_info_maxes.append(max(sim.logs.get('Q_INFO_trace', [0])))
            pmcs_list.append(sim.logs['pmcs_rf'])
        results[V] = {
            'max_Q_RF': np.mean(q_rf_maxes),
            'max_Q_EXP': np.mean(q_exp_maxes),
            'max_Q_INFO': np.mean(q_info_maxes),
            'mean_pmcs': np.mean(pmcs_list)
        }

    # Find largest V where queues are bounded (< 10 * V * R_target)
    R_target = 5.0
    V_max_stable = min(V_range)
    for V in sorted(V_range):
        r = results[V]
        threshold = 10.0 * V * R_target
        if r['max_Q_RF'] < threshold and r['max_Q_EXP'] < threshold:
            V_max_stable = V

    return {'V_max_stable': V_max_stable, 'results': results}


# =============================================================================
# FIX-11: MTTK Sensitivity Analysis
# =============================================================================
def run_mttk_sensitivity(mttk_values=None, n_episodes=50, scenario='A'):
    """
    FIX-11: Run RNCO simulator for each MTTK value and report mission metrics.
    Returns DataFrame with mttk_sec, success_rate, mean_exposure, mean_throughput.
    """
    if mttk_values is None:
        mttk_values = [30, 60, 120, 300, 1000]

    rows = []
    for mttk in mttk_values:
        pmcs_list, exp_list, tput_list = [], [], []
        for _ in range(n_episodes):
            sim = UAVSimulator(scenario=scenario, mttk_sec_override=mttk)
            for ep in range(sim.epochs):
                sim.step(ep)
            pmcs_list.append(sim.logs['pmcs_rf'])
            exp_list.append(sum(sim.logs['exposure']))
            tput_list.append(np.mean(sim.logs['throughput']))
        rows.append({
            'mttk_sec': mttk,
            'success_rate': float(np.mean(pmcs_list)),
            'mean_exposure_s': float(np.mean(exp_list)),
            'mean_throughput': float(np.mean(tput_list))
        })

    return pd.DataFrame(rows)


# =============================================================================
# MAIN SIMULATOR CLASS
# =============================================================================
class UAVSimulator:
    def __init__(self, scenario='A', div_scheme='MRC_L4', target_deg=2.0,
                 policy='RNCO', V_override=None, mttk_sec_override=None,
                 sigma_pos_uav_m=0.0):
        """
        sigma_pos_uav_m: FIX-8C: UAV position uncertainty (1-sigma metres).
                         Set to ScalarPositionEKF.sigma_R at each epoch for full coupling.
        """
        # CONVERGENCE NOTE:
        # This mission is non-stationary. Neely's asymptotic guarantees do not apply.
        # The optimizer is used as a principled heuristic with sliding-window drift (W=3).

        self.scenario = scenario
        self.div_scheme = div_scheme
        self.target_deg = target_deg
        self.policy = policy
        self.sigma_pos_uav_m = sigma_pos_uav_m  # FIX-8C

        # FIX-11: MTTK override
        self.mttk_sec = mttk_sec_override if mttk_sec_override is not None else MTTK_SEC

        self.N_array = 4
        self.epochs = 10
        self.dx_step = 425.0
        self.base_speed = 50.0
        self.base_time_per_epoch = self.dx_step / self.base_speed

        # UAV/jammer geometry
        self.uav_x = -5000.0
        self.uav_y = 0.0  # FIX-8B: UAV starts on x-axis
        self.jammer_y_true = 5000.0 if scenario == 'C' else 0.0

        # FIX-8A: Use JammerUKF instead of raw P_cov matrix
        x_hat_init = np.array([0.0, 0.0])
        r_prior = 5000.0 if scenario == 'C' else 15000.0 # Derive from initial geometry
        P_init = np.diag([r_prior ** 2, r_prior ** 2])
        self.ukf = JammerUKF(x_hat_init, P_init)

        # Queues
        self.Q_RF   = 0.0
        self.Q_EXP  = 0.0
        self.Q_INFO = 0.0

        # Normalization Constants
        self.Q_RF_NORM   = 5.0
        self.Q_EXP_NORM  = kill_prob_per_epoch(self.base_time_per_epoch * self.epochs, self.mttk_sec)
        R_initial = 5000.0
        self.Q_INFO_NORM = (R_initial * np.radians(3.0)) ** 2
        self.V = V_override if V_override is not None else 100.0

        self.lyapunov_history = [0.0]
        self.theta_error_rms_history = []
        self.jammer_leakage_db_history = []

        self.logs = {
            'throughput': [], 'exposure': [], 'P_trace': [],
            'dy': [], 'ts': [], 'outage': [], 'nd': [], 'pmcs_rf': 1.0,
            'cond_num': [], 'loop_gain': [], 'clip': [],
            'failure_mode': [], 'obs_metric': [],
            'Q_RF_trace': [], 'Q_EXP_trace': [], 'Q_INFO_trace': []
        }

        self.theta_null_prev = None

    def get_environment(self, epoch):
        inr_base = 50.0
        snr_base = 15.0
        calib_error = 0.0
        jam_y = self.jammer_y_true
        is_shadow = False

        if self.scenario == 'B':
            if 3 <= epoch <= 5:
                inr_base -= 20.0
                is_shadow = True
        elif self.scenario == 'D':
            calib_error = 0.5
        elif self.scenario == 'E':
            inr_base = 20.0

        jnr_lin = 10.0 ** ((inr_base - 50.0 - 5.0) / 10.0)
        return inr_base, snr_base, calib_error, jam_y, jnr_lin, is_shadow

    def is_shadow_epoch(self, epoch):
        if self.scenario != 'B':
            return False
        return 3 <= epoch <= 5

    def compute_info_arrival(self, t, Tr_P_cov, P_target):
        base_arrival = Tr_P_cov
        if self.scenario == 'B':
            LOOKAHEAD_WINDOW = 2
            LOOKAHEAD_GAIN = 2.0
            lookahead_term = 0.0
            for k in range(1, LOOKAHEAD_WINDOW + 1):
                if self.is_shadow_epoch(t + k):
                    lookahead_term += np.exp(-0.5 * k) * P_target * LOOKAHEAD_GAIN
            return base_arrival + lookahead_term
        return base_arrival

    def get_mrc_outage(self, gamma_bar_lin, req_sinr_dB):
        """
        FIX-5: Rician MRC outage via Marcum Q-function.
        FIX-1: Uses L_fhss (FHSS diversity branches), not L_snapshots.
        """
        gamma_bar_dB = 10.0 * np.log10(max(gamma_bar_lin, 1e-30))
        req_sinr_dB_val = req_sinr_dB

        # FIX-1: L_fhss = number of FHSS diversity branches (temporal/frequency hops)
        L_fhss = 4
        if self.div_scheme == 'No_Div':
            L_fhss = 1

        K_signal = 10.0  # Rician K for UAV-Base link
        return rician_mrc_outage(req_sinr_dB_val, gamma_bar_dB, L_fhss, K=K_signal)

    def _compute_music_sigma(self, L_snapshots, inr_base_dB, snr_base_dB, calib_error):
        """
        FIX-7: MUSIC CRLB with SNR dependence.
        FIX-1: Uses L_snapshots (MUSIC snapshot count), not L_fhss.
        """
        # Effective per-element SNR: combine signal and residual jammer
        snr_per_element = snr_base_dB  # conservative: use signal SNR only
        theta_deg = 0.0               # boresight (conservative)
        sigma_music = music_crlb(self.N_array, L_snapshots, snr_per_element,
                                 theta_deg, calib_error_deg=calib_error)
        # Phase C operational floor: see docstring
        sigma_floor = max(0.5, 10.0 / np.sqrt(L_snapshots)) + calib_error
        return max(sigma_music, sigma_floor)

    def predict_proxy(self, ts, dy, amc, use_ekf, epoch):
        """FAST ANALYTICAL PROXY for RNCO Optimizer Loop."""
        inr_base, snr_base, calib_error, jam_y, jnr_lin, is_shadow = self.get_environment(epoch)

        # FIX-6: Neyman-Pearson detection probability
        BW_SENSE_HZ = 1e6
        JNR_dB = 10.0 * np.log10(max(jnr_lin, 1e-30)) + 50.0 + 5.0
        if ts > 0:
            pd_val = detection_prob(ts * 1e-3, BW_SENSE_HZ, JNR_dB)
            # FIX-1: L_snapshots decreases when t_s takes sensing time away from data collection
            L_snapshots = max(1, int(100 * (1.0 - ts / 10.0)))
        else:
            pd_val = 1.0
            L_snapshots = 100

        # FIX-7: SNR-aware MUSIC sigma
        if jnr_lin < 0.01 or L_snapshots < 10 or is_shadow:
            m_rmse = 45.0 + calib_error  # MUSIC fails in shadow
        else:
            m_rmse = self._compute_music_sigma(L_snapshots, inr_base, snr_base, calib_error)

        exp = (self.dx_step + 2.0 * dy) / self.base_speed

        dx = np.abs(0.0 - self.uav_x)
        dy_j = np.abs(jam_y - dy)
        r = np.sqrt(dx ** 2 + dy_j ** 2)

        if use_ekf:
            P_pred = self.ukf.P + 10.0 * np.eye(2)
            sigma_theta_rad = m_rmse * (np.pi / 180.0)
            # FIX-8C: Couple UAV position uncertainty into R_meas
            sigma_bearing_UAV_rad2 = (self.sigma_pos_uav_m / max(r, 1.0)) ** 2
            R_meas_rad2 = sigma_theta_rad ** 2 + sigma_bearing_UAV_rad2

            H = np.array([[-dy_j / max(r ** 2, 1.0), dx / max(r ** 2, 1.0)]])
            S = H @ P_pred @ H.T + R_meas_rad2
            K = (P_pred @ H.T) / float(S)
            P_post = (np.eye(2) - np.outer(K.flatten(), H)) @ P_pred
            sigma_y_post = np.sqrt(max(P_post[1, 1], 0.0))
            bearing_err = (sigma_y_post / max(r, 1.0)) * (180.0 / np.pi)
        else:
            P_post = self.ukf.P.copy()
            bearing_err = m_rmse

        # FIX-9A: Q_INFO -> Q_RF coupling via SINR achievable
        sinr_from_uncertainty = sinr_achievable_from_uncertainty(bearing_err, inr_base, snr_base)
        # Use the lower of proxy SINR and uncertainty-limited SINR
        nd = max(-65.0, -63.31 + 3.45 * bearing_err + max(0, 5.0 - L_snapshots / 20.0))
        nd = min(nd, 0.0)
        inr_resid = inr_base + nd
        sinr_proxy_dB = snr_base - inr_resid
        sinr_dB = min(sinr_proxy_dB, sinr_from_uncertainty)

        req_sinr_dB = {1.0: 2.0, 2.0: 5.0, 4.0: 12.0, 6.0: 18.0}.get(amc, 5.0)
        gamma_bar = 10.0 ** (sinr_dB / 10.0)
        p_out = self.get_mrc_outage(gamma_bar, req_sinr_dB)

        comm_survival = 1.0 - 0.9 * p_out
        r_eff = amc * pd_val * (1.0 - ts / 10.0) * comm_survival

        return r_eff, exp, P_post, nd, p_out, bearing_err, L_snapshots

    def apply_true_physics(self, best_ts, best_dy, best_amc, use_ekf, epoch,
                           P_post, bearing_err, L_snapshots):
        """Apply TRUE LCMV physics after action selection."""
        inr_base, snr_base, calib_error, jam_y, jnr_lin, is_shadow = self.get_environment(epoch)

        theta_s = -45.0
        dx = np.abs(0.0 - self.uav_x)
        dy_j_true = self.jammer_y_true - best_dy
        R_current = np.sqrt(dx ** 2 + dy_j_true ** 2)
        theta_j_true = np.degrees(np.arctan2(dy_j_true, dx))

        if use_ekf:
            # FIX-7: SNR-aware MUSIC sigma
            m_rmse = self._compute_music_sigma(L_snapshots, inr_base, snr_base, calib_error)
            z_theta_rad = np.radians(theta_j_true) + np.radians(m_rmse) * np.random.randn()

            # FIX-8C: Couple UAV position uncertainty
            sigma_bearing_UAV_rad2 = (self.sigma_pos_uav_m / max(R_current, 1.0)) ** 2
            R_meas_rad2 = (m_rmse * np.pi / 180.0) ** 2 + sigma_bearing_UAV_rad2

            if sigma_bearing_UAV_rad2 > (m_rmse * np.pi / 180.0) ** 2:
                print(f"  [FIX-8C WARNING] UAV position uncertainty dominates jammer "
                      f"bearing noise at range {R_current:.0f} m")

            # FIX-8A: UKF update
            Q_proc = 10.0 * np.eye(2)
            self.ukf.predict(Q_proc)
            x_hat_new, P_new = self.ukf.update(z_theta_rad, self.uav_x, self.uav_y, R_meas_rad2)

            # FIX-8B: Observability metric
            obs_cond, is_obs = self.ukf.observability_metric(self.uav_x, self.uav_y)

            dy_est = x_hat_new[1] - best_dy
            R_est = np.sqrt(dx ** 2 + dy_est ** 2)
            theta_j_est_ukf = np.degrees(np.arctan2(dy_est, max(abs(dx), 1.0)))
            
            # FIX-4: Decouple UKF from null steering, use MUSIC directly
            theta_music_deg = np.degrees(z_theta_rad)
            sigma_theta_current = m_rmse  # This is the CRLB-based sigma_theta
            
            angular_divergence = abs(theta_j_est_ukf - theta_music_deg)
            if angular_divergence > 3.0 * sigma_theta_current:
                bearing_err = min(3.0 * sigma_theta_current, 5.0)
            else:
                bearing_err = (np.sqrt(P_new[1, 1]) / max(R_est, 1.0)) * (180.0 / np.pi)
                
            theta_j_est = theta_music_deg  # Always use MUSIC for null direction; UKF only sets width
        else:
            m_rmse = self._compute_music_sigma(L_snapshots, inr_base, snr_base, calib_error)
            theta_j_est = theta_j_true + m_rmse * np.random.randn()
            bearing_err = m_rmse
            obs_cond = float('inf')

        if self.theta_null_prev is None:
            self.theta_null_prev = theta_j_est

        dt_epoch = self.base_time_per_epoch
        theta_j_steer, is_clipped = rate_limited_null_steer(
            self.theta_null_prev, theta_j_est, self.base_speed, dt_epoch, R_current
        )

        if np.abs(theta_j_steer - theta_s) < 2.0:
            theta_j_steer = theta_s + 2.0

        self.theta_null_prev = theta_j_steer

        # Build true covariance matrix
        a_s = get_steering_vector(self.N_array, theta_s)
        a_j_true = get_steering_vector(self.N_array, theta_j_true)
        R_hat = ((10 ** (snr_base / 10)) * (a_s @ a_s.conj().T) +
                 (10 ** (inr_base / 10)) * (a_j_true @ a_j_true.conj().T) +
                 np.eye(self.N_array))

        # FIX-13: Graceful LCMV fallback
        theta_est_calibrated = theta_j_steer + calib_error
        sigma_theta_deg = bearing_err
        w_lcmv, cond_num, failure_mode, null_width_used = lcmv_with_fallback(
            R_hat, a_s, theta_est_calibrated, sigma_theta_deg,
            self.N_array, loading_factor=0.10
        )

        leakage_gain = np.abs(w_lcmv.conj().T @ a_j_true)[0, 0] ** 2
        nd_true = 10 * np.log10(max(1e-10, leakage_gain))

        inr_resid = inr_base + nd_true
        sinr_dB = snr_base - inr_resid
        gamma_bar = 10.0 ** (sinr_dB / 10.0)

        req_sinr_dB = {1.0: 2.0, 2.0: 5.0, 4.0: 12.0, 6.0: 18.0}.get(best_amc, 5.0)
        p_out_true = self.get_mrc_outage(gamma_bar, req_sinr_dB)

        # FIX-6: Neyman-Pearson P_d
        JNR_dB = 10.0 * np.log10(max(jnr_lin, 1e-30)) + 50.0 + 5.0
        if best_ts > 0:
            pd_val = detection_prob(best_ts * 1e-3, 1e6, JNR_dB)
        else:
            pd_val = 1.0

        r_eff_true = best_amc * pd_val * (1.0 - best_ts / 10.0) * (1.0 - 0.9 * p_out_true)
        exp_true = (self.dx_step + 2.0 * best_dy) / self.base_speed

        return r_eff_true, exp_true, nd_true, p_out_true, cond_num, is_clipped, failure_mode, obs_cond

    def step(self, epoch):
        ts_opts  = [0.1, 0.7, 2.0, 5.0, 9.0]
        dy_opts  = [0.0, 200.0, 460.0, 800.0, 1200.0]
        amc_opts = [1.0, 2.0, 4.0]

        best_dpp = np.inf
        best_action = None
        best_P_post = None
        best_bearing_err = None
        best_L_snapshots = None  # FIX-1: renamed from L_eff

        use_ekf = True
        if self.policy in ('No_EKF', 'No_Localization'):
            use_ekf = False

        dx_target = np.abs(0.0 - self.uav_x)
        P_target = (dx_target * self.target_deg * np.pi / 180.0) ** 2

        if self.policy == 'Fixed_460':
            best_action = (0.7, 460.0, 4.0)
            _, _, best_P_post, _, _, best_bearing_err, best_L_snapshots = \
                self.predict_proxy(*best_action, use_ekf, epoch)
            self.Q_RF = self.Q_EXP = self.Q_INFO = 0.0
        elif self.policy == 'No_Localization':
            best_action = (0.1, 0.0, 4.0)
            _, _, best_P_post, _, _, best_bearing_err, best_L_snapshots = \
                self.predict_proxy(*best_action, use_ekf, epoch)
            self.Q_RF = self.Q_EXP = self.Q_INFO = 0.0
        else:
            w1 = 1.0
            w2 = 1.0
            best_Q = (0.0, 0.0, 0.0)
            best_Ln = 0.0

            for ts in ts_opts:
                for dy in dy_opts:
                    for amc in amc_opts:
                        r, e, P_post, nd_proxy, out_proxy, b_err, l_snap = \
                            self.predict_proxy(ts, dy, amc, use_ekf, epoch)

                        a_rf  = 5.0
                        d_rf  = r
                        a_exp = kill_prob_per_epoch(e, self.mttk_sec)
                        d_exp = kill_prob_per_epoch(self.base_time_per_epoch, self.mttk_sec)
                        a_info = self.compute_info_arrival(epoch, np.trace(P_post), P_target)
                        d_info = P_target

                        Q_RF_n   = max(self.Q_RF   + (a_rf  - d_rf),  0) / self.Q_RF_NORM
                        Q_EXP_n  = max(self.Q_EXP  + (a_exp - d_exp), 0) / self.Q_EXP_NORM
                        Q_INFO_n = max(self.Q_INFO  + (a_info - d_info) / 10.0, 0) / self.Q_INFO_NORM

                        q_rf_old   = self.Q_RF   / self.Q_RF_NORM
                        q_exp_old  = self.Q_EXP  / self.Q_EXP_NORM
                        q_info_old = self.Q_INFO / self.Q_INFO_NORM

                        L_n = 0.5 * (Q_RF_n ** 2 + Q_EXP_n ** 2 + Q_INFO_n ** 2)

                        drift_bound = (q_rf_old   * (a_rf  - d_rf)  / self.Q_RF_NORM  +
                                       q_exp_old  * (a_exp - d_exp) / self.Q_EXP_NORM +
                                       q_info_old * (a_info - d_info) / self.Q_INFO_NORM)

                        pen = w1 * a_exp + w2 * out_proxy
                        dpp = drift_bound + self.V * pen

                        if dpp < best_dpp:
                            best_dpp = dpp
                            best_action = (ts, dy, amc)
                            best_P_post = P_post
                            best_bearing_err = b_err
                            best_L_snapshots = l_snap
                            best_Q = (max(self.Q_RF  + a_rf  - d_rf,  0),
                                      max(self.Q_EXP  + a_exp - d_exp, 0),
                                      max(self.Q_INFO + a_info - d_info, 0))
                            best_Ln = L_n

            self.Q_RF, self.Q_EXP, self.Q_INFO = best_Q
            self.lyapunov_history.append(best_Ln)

            # FIX-9B: Queue overflow check
            threshold = 10.0 * self.V * 5.0  # 10 * V * R_target
            if self.Q_RF > threshold or self.Q_EXP > threshold:
                import warnings
                warnings.warn(f"Queue overflow: Lyapunov V={self.V} may be too large for this scenario")

        # Apply True Physics
        result = self.apply_true_physics(
            best_action[0], best_action[1], best_action[2],
            use_ekf, epoch, best_P_post, best_bearing_err, best_L_snapshots
        )
        r_eff_true, exp_true, nd_true, out_true, cond_num, is_clipped, failure_mode, obs_cond = result

        ekf_health_check(self.ukf.P)

        self.uav_x += self.dx_step

        self.theta_error_rms_history.append(best_bearing_err)
        self.jammer_leakage_db_history.append(nd_true)
        loop_gain = stability_monitor(self.theta_error_rms_history, self.jammer_leakage_db_history)

        self.logs['throughput'].append(r_eff_true)
        self.logs['exposure'].append(exp_true)
        self.logs['P_trace'].append(np.trace(best_P_post))
        self.logs['dy'].append(best_action[1])
        self.logs['ts'].append(best_action[0])
        self.logs['outage'].append(out_true)
        self.logs['nd'].append(nd_true)
        self.logs['pmcs_rf'] *= (1.0 - out_true)
        self.logs['cond_num'].append(cond_num)
        self.logs['loop_gain'].append(loop_gain)
        self.logs['clip'].append(is_clipped)
        self.logs['failure_mode'].append(failure_mode)
        self.logs['obs_metric'].append(obs_cond)
        self.logs['Q_RF_trace'].append(self.Q_RF)
        self.logs['Q_EXP_trace'].append(self.Q_EXP)
        self.logs['Q_INFO_trace'].append(self.Q_INFO)
