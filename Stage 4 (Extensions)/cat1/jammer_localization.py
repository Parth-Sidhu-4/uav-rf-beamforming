"""
cat1/jammer_localization.py
Extension 1a: Jammer Localization via Triangulation + Anti-Drone Beam Tracker

Implements (Section 3.1.1 of Extension Plan):
  - Two-UAV closed-form ray intersection
  - N>=3 UAV weighted least-squares (WLS) localization
  - Cramér-Rao Lower Bound (CRLB) for position error
  - AntidroneBridgeTracker: Kalman filter + directed beam gain model
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Data container: single DOA observation from one UAV position
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class UAVObservation:
    """
    Single MUSIC DOA observation from one UAV position.

    Attributes
    ----------
    position : ndarray (2,) — UAV [x, y] in metres (2-D plane)
    doa_rad  : float        — MUSIC DOA estimate (radians from +x axis)
    doa_var  : float        — angle variance (rad²), from MUSIC CRLB
    """
    position: np.ndarray
    doa_rad:  float
    doa_var:  float = 1e-4    # default: 0.01 rad standard deviation


# ─────────────────────────────────────────────────────────────────────────────
# 1a.1  Two-UAV closed-form triangulation
# ─────────────────────────────────────────────────────────────────────────────
def triangulate_jammer_2uav(obs1: UAVObservation,
                             obs2: UAVObservation,
                             ill_cond_thresh: float = 1e3
                             ) -> Tuple[np.ndarray, float, bool]:
    """
    Exact 2-D jammer location via ray intersection of two MUSIC DOA bearings.

    Math (Section 3.1.1):
        A = [[cos(θ1), -cos(θ2)], [sin(θ1), -sin(θ2)]]
        b = p2 - p1
        [t1, t2] = A⁻¹ b
        p_J = p1 + t1 * [cos(θ1), sin(θ1)]

    Returns
    -------
    p_J         : ndarray (2,) — estimated jammer [x, y]
    cond_number : float        — condition number of 2×2 system
    reliable    : bool         — False if geometry is near-degenerate
    """
    A = np.array([[np.cos(obs1.doa_rad), -np.cos(obs2.doa_rad)],
                  [np.sin(obs1.doa_rad), -np.sin(obs2.doa_rad)]])
    b = obs2.position - obs1.position

    t, _, _, sv = np.linalg.lstsq(A, b, rcond=None)
    cond = float(sv[0] / (sv[-1] + 1e-15))

    # Jammer position along ray 1
    p_J = obs1.position + t[0] * np.array([np.cos(obs1.doa_rad),
                                            np.sin(obs1.doa_rad)])
    reliable = cond < ill_cond_thresh
    return p_J, cond, reliable


# ─────────────────────────────────────────────────────────────────────────────
# 1a.2  N>=3 UAV Weighted Least-Squares localization
# ─────────────────────────────────────────────────────────────────────────────
def triangulate_jammer_ls(observations: List[UAVObservation]
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Weighted least-squares jammer localization for N >= 2 UAVs.

    Each ray defines a line:
        sin(θ_i)·x − cos(θ_i)·y = sin(θ_i)·x_i − cos(θ_i)·y_i

    Weighted by w_i = 1/σ²_θ_i (inverse DOA variance).

    Returns
    -------
    p_J   : ndarray (2,)   — estimated jammer position [x, y] metres
    cov_J : ndarray (2, 2) — estimated position covariance matrix
    """
    N = len(observations)
    if N < 2:
        raise ValueError("Need at least 2 observations.")

    A_ls = np.zeros((N, 2))
    b_ls = np.zeros(N)
    W    = np.zeros(N)

    for i, obs in enumerate(observations):
        s = np.sin(obs.doa_rad)
        c = np.cos(obs.doa_rad)
        A_ls[i] = [s, -c]
        b_ls[i] = s * obs.position[0] - c * obs.position[1]
        W[i]    = 1.0 / (obs.doa_var + 1e-20)

    W_mat = np.diag(W)
    M     = A_ls.T @ W_mat @ A_ls
    v     = A_ls.T @ W_mat @ b_ls
    p_J   = np.linalg.solve(M, v)

    residuals = A_ls @ p_J - b_ls
    dof       = max(N - 2, 1)
    sigma2    = float(residuals @ W_mat @ residuals) / dof
    cov_J     = sigma2 * np.linalg.inv(M)
    return p_J, cov_J


# ─────────────────────────────────────────────────────────────────────────────
# 1a.3  Cramér-Rao Lower Bound
# ─────────────────────────────────────────────────────────────────────────────
def compute_crlb(observations: List[UAVObservation],
                 p_J_true: np.ndarray) -> np.ndarray:
    """
    Compute Fisher Information Matrix → CRLB for jammer position.

    Gradient of θ_i(p_J) = atan2(y_J - y_i, x_J - x_i) w.r.t. p_J:
        ∂θ/∂x = -(y_J - y_i)/r²
        ∂θ/∂y =  (x_J - x_i)/r²

    Returns
    -------
    crlb : ndarray (2, 2) — lower bound on position error covariance
    """
    F = np.zeros((2, 2))
    for obs in observations:
        delta = p_J_true - obs.position
        r2    = float(delta @ delta) + 1e-10
        grad  = np.array([-delta[1] / r2, delta[0] / r2])   # ∂θ/∂p_J
        F    += np.outer(grad, grad) / obs.doa_var
    return np.linalg.inv(F)


# ─────────────────────────────────────────────────────────────────────────────
# 1a.4  Anti-Drone Bridge Tracker
# ─────────────────────────────────────────────────────────────────────────────
class AntidroneBridgeTracker:
    """
    Ground-based UAV tracking + directed jammer beam simulation.

    State vector: x = [φ_az, dφ_az/dt, φ_el, dφ_el/dt]
    (azimuth and elevation angles + their angular rates)

    Measurement: z = [φ_az_meas, φ_el_meas] from MUSIC.

    The Kalman filter reduces pointing error, then we compute the
    focused jamming beam gain toward the estimated UAV bearing.
    """

    def __init__(self,
                 dt:                 float = 0.1,
                 meas_noise_std:     float = 0.01,   # MUSIC DOA noise (rad)
                 process_noise_std:  float = 1e-3,   # angular acceleration noise
                 g_max_dbi:          float = 20.0,   # peak beam gain (dBi)
                 bw_3db_rad:         float = 0.05):  # 3-dB beamwidth (rad)

        self.dt = dt

        # Constant angular velocity state transition
        self.F = np.array([[1, dt, 0,  0],
                           [0,  1, 0,  0],
                           [0,  0, 1, dt],
                           [0,  0, 0,  1]])

        # Measurement extracts azimuth and elevation only
        self.H = np.array([[1, 0, 0, 0],
                           [0, 0, 1, 0]])

        q = process_noise_std ** 2
        self.Q = q * np.eye(4)

        r = meas_noise_std ** 2
        self.R = r * np.eye(2)

        # Initial state and covariance
        self.x = np.zeros(4)
        self.P = np.eye(4) * 1.0

        self.g_max_lin = 10 ** (g_max_dbi / 10)
        self.bw_3db    = bw_3db_rad

        # Logs for analysis
        self.state_log:     List[np.ndarray] = []
        self.jam_gain_log:  List[float]      = []
        self.innovation_log: List[np.ndarray] = []

    def update(self, z_meas: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Kalman predict + update step.

        Parameters
        ----------
        z_meas : ndarray (2,) — measured [φ_az, φ_el] from MUSIC.

        Returns
        -------
        x_hat       : ndarray (4,) — filtered state estimate
        jam_gain_db : float        — jamming gain toward estimated UAV bearing
        """
        # ── Predict ──────────────────────────────────────────────────────────
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        # ── Update ───────────────────────────────────────────────────────────
        S    = self.H @ self.P @ self.H.T + self.R
        K    = self.P @ self.H.T @ np.linalg.inv(S)
        innov = z_meas - self.H @ self.x
        self.x = self.x + K @ innov
        self.P = (np.eye(4) - K @ self.H) @ self.P

        # ── Directed beam gain (Gaussian roll-off model) ──────────────────
        delta_phi   = np.linalg.norm(innov)
        g_lin       = self.g_max_lin * np.exp(
            -4 * np.log(2) * (delta_phi / self.bw_3db) ** 2)
        jam_gain_db = float(10 * np.log10(g_lin + 1e-30))

        self.state_log.append(self.x.copy())
        self.jam_gain_log.append(jam_gain_db)
        self.innovation_log.append(innov.copy())
        return self.x.copy(), jam_gain_db

    def reset(self):
        self.x = np.zeros(4)
        self.P = np.eye(4) * 1.0
        self.state_log.clear()
        self.jam_gain_log.clear()
        self.innovation_log.clear()
