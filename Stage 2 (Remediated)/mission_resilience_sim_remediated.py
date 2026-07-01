"""
mission_resilience_sim_remediated.py
Stage 1: 5-Layer Mission Resilience Simulator — Full Remediation

Fixes applied in this file:
  FIX-2 : FSPL unit enforcement — runtime assertions + unit labels in link budgets
  FIX-3 : INS accelerometer bias — full 3-state matrix EKF replacing scalar P_x
  FIX-4A: Jammer Rician K-factor corrected: K_JAMMER_LOS = 12.0
  FIX-10: BEE classifier — drop correlated FHSS_hit_rate feature (Option B)
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.stats import norm, gamma, beta
    from scipy.special import chndtr
except Exception:  # pragma: no cover
    norm = gamma = beta = chndtr = None  # type: ignore


# =============================================================================
# PHYSICAL / ELECTRICAL CONSTANTS
# =============================================================================
G    = 9.81        # m/s^2
C    = 299_792_458.0  # m/s

# FIX-4A: Rician K-factors as named constants
K_SIGNAL_LOS  = 10.0   # UAV-to-base link (strong LOS, low multipath)
K_JAMMER_LOS  = 12.0   # Ground jammer-to-UAV (directional LOS emitter,
                        # ~10.8 dB, consistent with minor ground reflection)
                        # Changed from 3.0: K=3 implied substantial multipath
                        # which intermittently underestimated received jammer power.

# =============================================================================
# BEE STATE AND TRANSITION MODEL
# =============================================================================
STATE_NAMES  = ("NJ", "NB", "BR", "FL", "GS")
STATE_INDEX  = {s: i for i, s in enumerate(STATE_NAMES)}

TRANSITION_MATRIX = np.array(
    [
        [0.90, 0.06, 0.02, 0.01, 0.01],
        [0.05, 0.75, 0.12, 0.05, 0.03],
        [0.03, 0.10, 0.75, 0.08, 0.04],
        [0.02, 0.05, 0.15, 0.75, 0.03],
        [0.05, 0.05, 0.05, 0.05, 0.80],
    ],
    dtype=float,
)

# FIX-10: Dropped FHSS_hit_rate (correlated with SINR_mean through BEP curve).
# Using Option B: remove correlated feature, document simplification.
# Removed feature: "FHSS_hit_rate" — packet drop rate is causally determined by SNR.
# Including it as an independent Bayesian feature double-counts evidence.
BEE_LIKELIHOODS = {
    "SINR_mean": {
        "NJ": ("norm", 35.0, 5.0),
        "NB": ("norm", 32.0, 9.0),
        "BR": ("norm", 10.0, 4.0),
        "FL": ("norm",  5.0, 6.0),
        "GS": ("norm", 35.0, 5.0),
    },
    "SINR_var_freq": {
        "NJ": ("gamma", 13.0, 1.15),
        "NB": ("gamma", 10.0, 3.0),
        "BR": ("gamma",  2.0, 2.0),
        "FL": ("gamma",  5.0, 3.0),
        "GS": ("gamma", 13.0, 1.15),
    },
    # FIX-10: FHSS_hit_rate REMOVED (correlated feature)
    "GPS_INS_div_rate": {
        "NJ": ("norm",    0.0, 1.0),
        "NB": ("norm",    0.0, 1.5),
        "BR": ("norm",    0.0, 2.0),
        "FL": ("norm",    0.0, 1.5),
        "GS": ("absnorm", 5.0, 2.0),
    },
    "RSSI_spatial_var": {
        "NJ": ("gamma",  2.0, 1.0),
        "NB": ("gamma", 10.0, 2.0),
        "BR": ("gamma",  3.0, 2.0),
        "FL": ("gamma",  6.0, 2.0),
        "GS": ("gamma",  2.0, 1.0),
    },
}

JAMMER_CLASSES = {
    "I":  {"p_tx_dbm": 70.0, "g_tx_dbi": 3.0, "standoff_km": 15.0},
    "II": {"p_tx_dbm": 70.0, "g_tx_dbi": 3.0, "standoff_km":  5.0},
}

COMM_BW_KHZ               = 500.0
FHSS_BW_MHZ               = 40.0
FHSS_PROCESSING_GAIN_DB   = 10.0 * math.log10((FHSS_BW_MHZ * 1e6) / (COMM_BW_KHZ * 1e3))
FHSS_JM_DB                = FHSS_PROCESSING_GAIN_DB - (9.6 + 3.0)


# =============================================================================
# FIX-2: FSPL WITH RUNTIME UNIT ASSERTIONS
# =============================================================================

def fspl_db(d_km: float, f_GHz: float) -> float:
    """
    FIX-2: Free-space path loss in dB.

    Formula: FSPL = 92.45 + 20*log10(d_km) + 20*log10(f_GHz)
    VALID ONLY for d in kilometres and f in GHz. Parameters renamed to enforce units.

    Plausible UAV operating ranges:
      d_km  ∈ (0, 1000) km
      f_GHz ∈ (0.1, 100) GHz

    Args:
        d_km : Range in kilometres (NOT metres!).
        f_GHz: Frequency in GHz (NOT Hz or MHz!).

    Returns:
        Path loss in dB.
    """
    # FIX-2: Runtime assertions for unit validation
    assert 0 < d_km < 1000, (
        f"d_km={d_km:.4f} out of plausible UAV range (0, 1000 km). "
        "Did you pass metres instead of kilometres?"
    )
    assert 0.1 < f_GHz < 100, (
        f"f_GHz={f_GHz:.4f} out of plausible RF range (0.1, 100 GHz). "
        "Did you pass Hz instead of GHz?"
    )
    result = 92.45 + 20.0 * math.log10(d_km) + 20.0 * math.log10(f_GHz)
    return result


def fspl_db_verbose(d_km: float, f_GHz: float) -> Tuple[float, str]:
    """
    FIX-2: FSPL with unit-label output string for link budget reporting.

    Returns:
        (fspl_dB, label_string)
    """
    val = fspl_db(d_km, f_GHz)
    label = f"FSPL = {val:.2f} dB  [d={d_km:.3f} km, f={f_GHz:.3f} GHz]"
    return val, label


# =============================================================================
# INS ANALYTICAL ERROR MODEL
# =============================================================================

def analytical_sigma_x_total(
    t_s: np.ndarray | float,
    sigma_b: float,
    sigma_theta_deg: float,
    sigma_w: float = 0.0,
    sigma_bias_m_s2: float = 0.0,
) -> np.ndarray | float:
    """
    FIX-3: 1σ position error from INS dynamics including accelerometer bias.

    The total error variance is:
        σ²_x(t) = (t²-growth from bias)² + (t³-growth from RWA + tilt)²

    For a constant bias b (m/s²):
        σ²_bias_pos(t) ≈ (0.5 * sigma_bias * t²)²

    Combined:
        σ_x(t) = sqrt((0.5*sigma_eff*t²)² + (G*sigma_theta*t³/6)² + (0.5*sigma_bias*t²)²)

    Args:
        sigma_b          : Accelerometer noise spectral density (m/s^(3/2))
        sigma_theta_deg  : Gyroscope angle noise (degrees)
        sigma_w          : Wind acceleration noise (m/s²)
        sigma_bias_m_s2  : FIX-3: RMS accelerometer bias (m/s²). Default 0 for backward-compat.

    Returns:
        1-sigma position error in metres.
    """
    sigma_theta = math.radians(sigma_theta_deg)
    sigma_eff   = math.sqrt(sigma_b ** 2 + sigma_w ** 2)
    t = np.asarray(t_s, dtype=float)

    # t²-growth: random walk in velocity (bias + wind)
    term_t2 = (0.5 * sigma_eff * t ** 2) ** 2
    # t³-growth: gyroscope tilt coupling
    term_t3 = ((G * sigma_theta * t ** 3) / 6.0) ** 2
    # FIX-3: Additional t²-growth from accelerometer bias
    term_bias = (0.5 * sigma_bias_m_s2 * t ** 2) ** 2

    return np.sqrt(term_t2 + term_t3 + term_bias)


# =============================================================================
# FIX-3: 3-STATE MATRIX EKF FOR INS (replaces ScalarPositionEKF)
# =============================================================================

class NavigationEKF:
    """
    FIX-3: Full 3-state EKF for position + velocity + accelerometer bias estimation.

    State vector: x = [position (m), velocity (m/s), accel_bias (m/s²)]

    Process model (Euler):
        pos[k+1]   = pos[k] + vel[k]*dt + 0.5*a_bias[k]*dt²
        vel[k+1]   = vel[k] + a_bias[k]*dt
        a_bias[k+1]= a_bias[k]*(1 - dt/tau_b) + w_b

    Process noise Q (3×3 diagonal):
        Q[0,0] = 0  (position not driven directly by noise)
        Q[1,1] = q_accel * dt   (velocity random walk from accelerometer noise)
        Q[2,2] = q_bias * dt    (bias drift)

    Measurement model (V-SLAM): z = position, H = [1, 0, 0]
    R_vslam = (k_vo * v * dt)²

    API kept backward-compatible with ScalarPositionEKF:
        .predict() -> float (position variance)
        .update_vslam(d_step_m) -> float (position variance)
        .sigma_R -> float (position 1-sigma in metres)

    Note: P_x is now an alias for P[0,0] to avoid breaking downstream code.
    """

    def __init__(
        self,
        sigma_b: float,
        sigma_theta_deg: float,
        dt: float,
        k_vo: float = 0.02,
        v_mps: float = 50.0,
        P_x: float = 0.0,
        # FIX-3: New parameters
        sigma_bias_m_s2: float = 0.001,  # RMS accel bias (1e-3 m/s² typical MEMS IMU)
        tau_b: float = 300.0,            # Gauss-Markov correlation time (seconds)
        q_bias: float = 1e-8,            # Bias process noise PSD (m²/s⁵)
        q_accel: float = None,           # Velocity random walk PSD (m²/s³); derived from sigma_b
    ):
        self.dt    = dt
        self.k_vo  = k_vo
        self.v_mps = v_mps
        self.tau_b = tau_b

        # Derive q_accel from sigma_b if not provided
        if q_accel is None:
            q_accel = sigma_b ** 2

        self.q_accel       = q_accel
        self.q_bias        = q_bias
        self.sigma_bias_m_s2 = sigma_bias_m_s2
        self.sigma_theta_deg = sigma_theta_deg

        # State vector: [pos, vel, bias]
        self.x = np.zeros(3)  # prior mean = zero for all states

        # FIX-3: 3×3 covariance matrix initialization
        # pos uncertainty from P_x arg (for backward-compat); bias from sigma_bias²
        self.P = np.diag([float(P_x), sigma_b ** 2, sigma_bias_m_s2 ** 2])

        # Keep t for analytical comparison
        self.t = 0.0

    @property
    def P_x(self) -> float:
        """Backward-compatible alias: position variance."""
        return float(self.P[0, 0])

    def predict(self) -> float:
        """
        FIX-3: One predict step using 3-state transition model.
        Returns position variance (for backward compatibility).
        """
        dt = self.dt

        # State transition matrix F
        F = np.array([
            [1.0, dt, 0.5 * dt ** 2],
            [0.0, 1.0, dt],
            [0.0, 0.0, 1.0 - dt / self.tau_b],
        ])

        # Process noise Q
        Q = np.diag([
            0.0,                       # position: no direct noise
            self.q_accel * dt,         # velocity: accelerometer random walk
            self.q_bias * dt,          # bias: Gauss-Markov drift
        ])

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.t += dt

        # Enforce positive-definiteness
        self.P = 0.5 * (self.P + self.P.T)

        return self.P_x

    def update_vslam(self, d_step_m: float | None = None) -> float:
        """
        FIX-3: V-SLAM position measurement update.
        H = [1, 0, 0] (observes position only; velocity and bias are unobserved).
        Returns position variance after update.
        """
        d_step = self.v_mps * self.dt if d_step_m is None else float(d_step_m)
        R_vslam = (self.k_vo * d_step) ** 2

        H = np.array([[1.0, 0.0, 0.0]])
        S = H @ self.P @ H.T + R_vslam
        K = (self.P @ H.T) / float(S)

        # Innovation: V-SLAM measures zero error (perfect origin alignment)
        # In practice, z would be the V-SLAM position fix. Using z=0 for unit tests.
        innov = 0.0 - float(H @ self.x)
        self.x = self.x + K.flatten() * innov
        self.P = (np.eye(3) - np.outer(K.flatten(), H)) @ self.P

        # Enforce positive-definiteness
        self.P = 0.5 * (self.P + self.P.T)
        return self.P_x

    @property
    def sigma_R(self) -> float:
        """Position 1-sigma uncertainty in metres (backward-compatible property)."""
        return math.sqrt(max(self.P_x, 0.0))

    def predicted_analytic_variance(self, t_s: float) -> float:
        """Analytical prediction for backward-compatibility with unit tests."""
        sigma = analytical_sigma_x_total(t_s, math.sqrt(self.q_accel),
                                          self.sigma_theta_deg, 0.0,
                                          self.sigma_bias_m_s2)
        return float(sigma * sigma)


# Backward-compatible alias
ScalarPositionEKF = NavigationEKF


# =============================================================================
# COMPUTE SINR WITH FIX-4A (Rician jammer)
# =============================================================================

def compute_sinr(
    d_uav_km: float,
    jammer_class: str,
    f_ghz: float,
    P_tx_dbm: float,
    G_tx_dbi: float,
    G_rx_dbi: float,
    NF_db: float,
    B_khz: float,
    rng: np.random.Generator | None = None,
    h_uav: float = 100.0,
    h_gs: float = 2.0,
) -> Dict[str, float]:
    """
    FIX-2: Uses fspl_db(d_km, f_GHz) with enforced units.
    FIX-4A: Jammer channel now uses K_JAMMER_LOS = 12.0 Rician fading.
    """
    jammer_class = jammer_class.upper()
    if jammer_class not in JAMMER_CLASSES:
        raise ValueError(f"Unknown jammer class: {jammer_class}")

    d_horizontal = d_uav_km * 1000.0
    dh           = h_uav - h_gs
    d_slant      = math.sqrt(d_horizontal ** 2 + dh ** 2)
    theta_rad    = math.atan(dh / max(d_horizontal, 0.1))

    # FIX-2: explicit km argument
    L_fs = fspl_db(d_slant / 1000.0, f_ghz)

    if rng is not None:
        theta_deg = math.degrees(theta_rad)
        K = 2.3 * math.exp(0.035 * theta_deg)  # signal K-factor (elevation dependent)

        S_mean_dBm = P_tx_dbm + G_tx_dbi + G_rx_dbi - L_fs
        S_mean_lin = 10.0 ** (S_mean_dBm / 10.0)

        s_los   = math.sqrt(K / (K + 1.0))
        sigma_s = math.sqrt(1.0 / (2.0 * (K + 1.0)))
        x = rng.normal(s_los,   sigma_s)
        y = rng.normal(0.0,     sigma_s)
        S_lin = (x ** 2 + y ** 2) * S_mean_lin
        S_dbm = 10.0 * math.log10(max(S_lin, 1e-30))
    else:
        fade_margin_db = 3.0
        S_dbm = P_tx_dbm + G_tx_dbi + G_rx_dbi - L_fs - fade_margin_db
        S_lin = 10.0 ** (S_dbm / 10.0)

    N_dbm = -174.0 + 10.0 * math.log10(B_khz * 1e3) + NF_db

    j = JAMMER_CLASSES[jammer_class]
    # FIX-2: standoff_km is already in km
    L_j   = fspl_db(j["standoff_km"], f_ghz)
    J_dbm_raw = j["p_tx_dbm"] + j["g_tx_dbi"] - L_j

    # FIX-4A: Apply Rician fading to the jammer channel with K_JAMMER_LOS = 12.0
    if rng is not None:
        K_j     = K_JAMMER_LOS
        s_los_j = math.sqrt(K_j / (K_j + 1.0))
        sigma_j = math.sqrt(1.0 / (2.0 * (K_j + 1.0)))
        xj      = rng.normal(s_los_j, sigma_j)
        yj      = rng.normal(0.0,     sigma_j)
        J_fading_gain = xj ** 2 + yj ** 2
        J_dbm = J_dbm_raw + 10.0 * math.log10(max(J_fading_gain, 1e-30))
    else:
        J_dbm = J_dbm_raw  # deterministic mode: no jammer fading

    js_raw_db  = J_dbm - S_dbm
    js_fhss_db = js_raw_db - FHSS_PROCESSING_GAIN_DB

    J_eff_dBm = J_dbm - FHSS_PROCESSING_GAIN_DB
    J_eff_lin = 10.0 ** (J_eff_dBm / 10.0)
    N_lin     = 10.0 ** (N_dbm     / 10.0)
    sinr_lin  = S_lin / (J_eff_lin + N_lin)
    sinr_db   = 10.0 * math.log10(sinr_lin)

    return {
        "fspl_db":                    L_fs,
        "signal_dbm":                 S_dbm,
        "jammer_dbm":                 J_dbm,
        "noise_dbm":                  N_dbm,
        "sinr_db":                    sinr_db,
        "js_raw_db":                  js_raw_db,
        "js_fhss_db":                 js_fhss_db,
        "fhss_processing_gain_db":    FHSS_PROCESSING_GAIN_DB,
        "fhss_jamming_margin_db":     FHSS_JM_DB,
    }


# =============================================================================
# FIX-10: BEE CLASSIFIER (correlated feature removed)
# =============================================================================

class BEEClassifier:
    """
    FIX-10: Bayesian Electromagnetic Environment Estimator.

    Simplification (Option B): FHSS_hit_rate feature REMOVED because it is
    causally correlated with SINR_mean through the bit error rate curve.
    Including it as an independent Bayesian feature double-counts evidence,
    producing an overconfident posterior.

    Remaining features (3): SINR_mean, SINR_var_freq, GPS_INS_div_rate, RSSI_spatial_var.
    All treated as conditionally independent given jammer state.
    """

    def __init__(self, prior: Sequence[float] | None = None,
                 likelihoods: Dict | None = None):
        if prior is None:
            prior = [0.2] * 5
        prior = np.asarray(prior, dtype=float)
        prior = prior / prior.sum()
        self.posterior   = prior.copy()
        self.likelihoods = likelihoods if likelihoods is not None else BEE_LIKELIHOODS

    def _pdf(self, observable: str, state: str, value: float) -> float:
        kind, a, b = self.likelihoods[observable][state]
        if kind == "norm":
            return float(norm.pdf(value, loc=a, scale=b))
        if kind == "gamma":
            return float(gamma.pdf(value, a=a, scale=b))
        if kind == "beta":
            v = min(max(value, 1e-9), 1 - 1e-9)
            return float(beta.pdf(v, a=a, b=b))
        if kind == "absnorm":
            if value < 0:
                return 0.0
            mu, sigma_val = a, b
            return float(norm.pdf(value, loc=mu, scale=sigma_val) +
                         norm.pdf(-value, loc=mu, scale=sigma_val))
        raise ValueError(kind)

    @staticmethod
    def sample_observation(state: str, rng: np.random.Generator | None = None,
                           likelihoods: Dict | None = None) -> Dict[str, float]:
        rng    = np.random.default_rng() if rng is None else rng
        lhoods = likelihoods if likelihoods is not None else BEE_LIKELIHOODS
        obs    = {}
        for observable in lhoods:
            kind, a, b = lhoods[observable][state]
            if kind == "norm":
                obs[observable] = float(rng.normal(a, b))
            elif kind == "gamma":
                obs[observable] = float(rng.gamma(shape=a, scale=b))
            elif kind == "beta":
                obs[observable] = float(rng.beta(a, b))
            elif kind == "absnorm":
                obs[observable] = float(abs(rng.normal(a, b)))
            else:
                raise ValueError(kind)
        return obs

    def step(self, obs: Mapping[str, float]) -> Tuple[np.ndarray, str]:
        likelihood = np.zeros(len(STATE_NAMES), dtype=float)
        for j, state in enumerate(STATE_NAMES):
            lp = 1.0
            for observable, value in obs.items():
                if observable in self.likelihoods:
                    lp *= self._pdf(observable, state, float(value))
                # FIX-10: Silently ignore FHSS_hit_rate if passed from old code
            likelihood[j] = lp

        pred = self.posterior @ TRANSITION_MATRIX
        post_unnorm = likelihood * pred
        z = post_unnorm.sum()
        if z <= 0:
            self.posterior = np.ones(len(STATE_NAMES)) / len(STATE_NAMES)
        else:
            self.posterior = post_unnorm / z
        return self.posterior.copy(), STATE_NAMES[int(np.argmax(self.posterior))]

    def reset(self, prior: Sequence[float] | None = None) -> None:
        if prior is None:
            prior = [0.2] * 5
        prior = np.asarray(prior, dtype=float)
        self.posterior = prior / prior.sum()


# =============================================================================
# ANALYTICAL R_COMM CUTOFF RANGE (FIX-2: validated units)
# =============================================================================

def analytical_r_comm(
    sigma_b: float,
    sigma_theta_deg: float,
    sigma_w: float,
    v: float,
    r_l: float,
    t_max: float = 120.0,
    sigma_bias_m_s2: float = 0.0,  # FIX-3
) -> float:
    """Numerically solve the 3σ success equation (FIX-3 updated analytical formula)."""
    sigma_theta = math.radians(sigma_theta_deg)
    sigma_eff   = math.sqrt(sigma_b ** 2 + sigma_w ** 2)

    def f(t: float) -> float:
        term_rwa  = (0.5 * sigma_eff * t ** 2) ** 2
        term_gyro = ((G * sigma_theta * t ** 3) / 6.0) ** 2
        term_bias = (0.5 * sigma_bias_m_s2 * t ** 2) ** 2
        sigma2 = term_rwa + term_gyro + term_bias
        return 9.0 * sigma2 - r_l ** 2

    lo, hi = 0.0, t_max
    if f(hi) < 0:
        return v * hi
    while f(hi) < 0:
        hi *= 2.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if f(mid) <= 0:
            lo = mid
        else:
            hi = mid
    return v * lo


# =============================================================================
# INS SIMULATION (unchanged, kept for MC tests)
# =============================================================================

def simulate_ins(
    sigma_b: float,
    sigma_theta_deg: float,
    sigma_w: float,
    v: float,
    dt: float,
    T_max: float,
    rng: np.random.Generator | None = None,
    sigma_bias_m_s2: float = 0.0,  # FIX-3
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate one INS error trajectory including optional accelerometer bias."""
    rng         = np.random.default_rng() if rng is None else rng
    sigma_theta = math.radians(sigma_theta_deg)

    b_a    = rng.normal(0.0, sigma_b)
    omega_g = rng.normal(0.0, sigma_theta)
    a_wind = rng.normal(0.0, sigma_w)
    b_bias = rng.normal(0.0, sigma_bias_m_s2) if sigma_bias_m_s2 > 0 else 0.0  # FIX-3

    n = int(round(T_max / dt))
    t = np.linspace(0.0, n * dt, n + 1)

    v_err = np.zeros_like(t)
    x_err = np.zeros_like(t)
    for k in range(n):
        a_false = b_a + a_wind + b_bias + G * omega_g * t[k]  # FIX-3: add bias
        v_err[k + 1] = v_err[k] + a_false * dt
        x_err[k + 1] = x_err[k] + v_err[k + 1] * dt

    return t, x_err


def simulate_ins_final_error(
    t_s: np.ndarray,
    sigma_b: float,
    sigma_theta_deg: float,
    sigma_w: float,
    rng: np.random.Generator | None = None,
    sigma_bias_m_s2: float = 0.0,  # FIX-3
) -> np.ndarray:
    """Fast ensemble helper for unit tests."""
    rng         = np.random.default_rng() if rng is None else rng
    sigma_theta = math.radians(sigma_theta_deg)
    t           = np.asarray(t_s, dtype=float)
    n           = t.shape[0]
    b_a    = rng.normal(0.0, sigma_b,   size=n)
    omega_g = rng.normal(0.0, sigma_theta, size=n)
    a_wind = rng.normal(0.0, sigma_w,   size=n)
    b_bias = rng.normal(0.0, sigma_bias_m_s2, size=n) if sigma_bias_m_s2 > 0 else np.zeros(n)
    return (0.5 * (b_a + a_wind + b_bias) * t ** 2 +
            (G * omega_g * t ** 3) / 6.0)


# =============================================================================
# RICIAN GENERATOR VALIDATION (unchanged; jammer K now K_JAMMER_LOS = 12.0)
# =============================================================================

def validate_rician_generator(n_samples: int = 100000) -> Dict[str, float]:
    """Verify Rician generator matches analytical expectations."""
    rng        = np.random.default_rng(42)
    d_uav_km   = 1.037
    dh         = 98.0
    d_horizontal = d_uav_km * 1000.0
    theta_rad  = math.atan(dh / max(d_horizontal, 0.1))
    theta_deg  = math.degrees(theta_rad)
    K          = 2.3 * math.exp(0.035 * theta_deg)

    d_slant    = math.sqrt(d_horizontal ** 2 + dh ** 2)
    L_fs       = fspl_db(d_slant / 1000.0, 2.4)
    S_mean_dBm = 37.0 + 3.0 + 10.0 - L_fs
    S_mean_lin = 10.0 ** (S_mean_dBm / 10.0)

    s_los   = math.sqrt(K / (K + 1.0))
    sigma_s = math.sqrt(1.0 / (2.0 * (K + 1.0)))
    x = rng.normal(s_los,   sigma_s, n_samples)
    y = rng.normal(0.0,     sigma_s, n_samples)
    S_lin_samples = (x ** 2 + y ** 2) * S_mean_lin

    mc_mean         = float(np.mean(S_lin_samples))
    rel_error_mean  = abs(mc_mean - S_mean_lin) / S_mean_lin
    mc_var          = float(np.var(S_lin_samples))
    analytical_var  = S_mean_lin ** 2 * (2.0 * K + 1.0) / (K + 1.0) ** 2
    rel_error_var   = abs(mc_var - analytical_var) / analytical_var
    passed          = (rel_error_mean <= 0.01) and (rel_error_var <= 0.05)

    return {
        "analytical_mean":  S_mean_lin,
        "mc_mean":          mc_mean,
        "relative_error":   rel_error_mean,
        "analytical_var":   analytical_var,
        "mc_var":           mc_var,
        "var_relative_error": rel_error_var,
        "passed":           passed,
    }


# =============================================================================
# MISSION PHASE LOGIC (unchanged)
# =============================================================================

@dataclass
class HMRSMResult:
    time_s:             np.ndarray
    true_range_m:       np.ndarray
    estimated_range_m:  np.ndarray
    sigma_R_m:          np.ndarray
    phase:              List[int]
    energy_reallocation:List[bool]
    trigger_r_m:        float
    visual_r_m:         float


def phase_from_range(
    d_true_m:   float,
    d_hat_m:    float,
    r_comm_m:   float,
    sigma_R_m:  float,
    r_visual_m: float = 750.0,
    alpha:      float = 1.0,
) -> int:
    r_trigger = r_comm_m + abs(alpha) * sigma_R_m
    if d_true_m > r_comm_m * 1.2:
        return 0
    if d_hat_m <= r_trigger and d_true_m > r_visual_m:
        return 2
    if d_true_m <= r_visual_m:
        return 3
    return 1


def hand_trace_mission(
    r_comm_m:        float,
    r_visual_m:      float = 750.0,
    alpha:           float = 1.0,
    sigma_b:         float = 0.05,
    sigma_theta_deg: float = 0.05,
    v_mps:           float = 50.0,
    dt:              float = 0.5,
    initial_range_m: float = 3000.0,
    vslam_available: bool  = True,
    k_vo:            float = 0.02,
    sigma_bias_m_s2: float = 0.0,  # FIX-3
) -> HMRSMResult:
    """Hand-trace one deterministic mission episode (FIX-3: NavigationEKF)."""
    ekf = NavigationEKF(
        sigma_b=sigma_b,
        sigma_theta_deg=sigma_theta_deg,
        dt=dt,
        k_vo=k_vo,
        v_mps=v_mps,
        sigma_bias_m_s2=sigma_bias_m_s2,
    )

    n         = int(math.ceil(initial_range_m / (v_mps * dt)))
    time_s    = np.arange(n + 1) * dt
    true_range = np.maximum(initial_range_m - v_mps * time_s, 0.0)
    est_range  = np.zeros_like(true_range)
    sigma_R    = np.zeros_like(true_range)
    phases:     List[int]  = []
    energy_flags: List[bool] = []

    prev_phase = 0
    for k, t in enumerate(time_s):
        if k > 0:
            ekf.predict()
            if vslam_available and true_range[k] > r_visual_m:
                ekf.update_vslam(d_step_m=v_mps * dt)
        sigma_R[k]  = ekf.sigma_R
        est_range[k] = true_range[k] + sigma_R[k]
        phase = phase_from_range(true_range[k], est_range[k],
                                  r_comm_m, sigma_R[k], r_visual_m, alpha)
        phases.append(phase)
        energy_flags.append(prev_phase == 1 and phase == 2)
        prev_phase = phase

    trigger_idx = next((i for i, p in enumerate(phases) if p == 2), None)
    trigger_r_m = r_comm_m + abs(alpha) * float(sigma_R[trigger_idx]) if trigger_idx else r_comm_m

    return HMRSMResult(
        time_s=time_s,
        true_range_m=true_range,
        estimated_range_m=est_range,
        sigma_R_m=sigma_R,
        phase=phases,
        energy_reallocation=energy_flags,
        trigger_r_m=trigger_r_m,
        visual_r_m=r_visual_m,
    )


# =============================================================================
# MISSION SUCCESS PROBABILITY (unchanged logic; picks up FIX-4A via compute_sinr)
# =============================================================================

def mission_success_probability(
    R_comm_m: float,
    sigma_b: float,
    sigma_theta_deg: float,
    sigma_w: float,
    v_mps: float,
    r_l_m: float,
    rng: np.random.Generator,
    n_trials: int = 1000,
    initial_range_m: float = 5000.0,
    jammer_class: str | None = None,
    f_ghz: float = 2.4,
    P_tx_dbm: float = 37.0,
    G_tx_dbi: float = 3.0,
    G_rx_dbi: float = 10.0,
    NF_db: float = 6.0,
    B_khz: float = 500.0,
    channel_model: str = "legacy",
    sigma_bias_m_s2: float = 0.0,  # FIX-3
) -> Tuple[float, Dict[str, float]]:
    """Mission success probability (FIX-4A jammer fading applied via compute_sinr)."""
    if jammer_class is None:
        jammer_class = rng.choice(["I", "II"], p=[0.45, 0.55])

    exposure_time = max((initial_range_m - R_comm_m) / v_mps, 0.0)

    t_silent = R_comm_m / v_mps
    sigma    = float(analytical_sigma_x_total(t_silent, sigma_b, sigma_theta_deg,
                                               sigma_w, sigma_bias_m_s2))
    p_nav    = float(math.erf(r_l_m / (math.sqrt(2.0) * max(sigma, 1e-12))))
    p_nav    = max(0.0, min(1.0, p_nav))

    if channel_model == "legacy":
        link = compute_sinr(
            d_uav_km=max(initial_range_m / 1000.0, 0.001),
            jammer_class=jammer_class, f_ghz=f_ghz,
            P_tx_dbm=P_tx_dbm, G_tx_dbi=G_tx_dbi, G_rx_dbi=G_rx_dbi,
            NF_db=NF_db, B_khz=B_khz, rng=None,
        )
        margin_db  = link["fhss_jamming_margin_db"] - link["js_fhss_db"]
        p_link_pure = 1.0 / (1.0 + math.exp(-0.9 * margin_db))
        p_exposure  = math.exp(-exposure_time / 120.0)
        successes   = sum(
            1 for _ in range(n_trials)
            if (rng.random() < p_link_pure and
                rng.random() < p_exposure and
                rng.random() < p_nav)
        )
        return successes / n_trials, {
            "p_link": p_link_pure, "p_exposure": p_exposure, "p_nav": p_nav,
            "link_margin_db": margin_db, "sigma_silent_m": sigma,
            "jammer_class": jammer_class, "js_fhss_db": link["js_fhss_db"],
        }

    elif channel_model in ("rician_start", "rician_handover"):
        d_eval_km = (max(initial_range_m / 1000.0, 0.001) if channel_model == "rician_start"
                     else max(R_comm_m / 1000.0, 0.001))
        successes = 0
        p_links   = []
        for _ in range(n_trials):
            link         = compute_sinr(d_eval_km, jammer_class, f_ghz, P_tx_dbm,
                                        G_tx_dbi, G_rx_dbi, NF_db, B_khz, rng=rng)
            link_success = link["sinr_db"] >= 10.0
            exp_success  = rng.random() < math.exp(-exposure_time / 120.0)
            nav_success  = rng.random() < p_nav
            if link_success and exp_success and nav_success:
                successes += 1
            p_links.append(float(link_success))
        return successes / n_trials, {
            "p_link": float(np.mean(p_links)),
            "p_exposure": math.exp(-exposure_time / 120.0),
            "p_nav": p_nav, "link_margin_db": 0.0, "sigma_silent_m": sigma,
            "jammer_class": jammer_class, "js_fhss_db": 0.0,
        }
    else:
        raise ValueError(f"Unknown channel_model: {channel_model}")
