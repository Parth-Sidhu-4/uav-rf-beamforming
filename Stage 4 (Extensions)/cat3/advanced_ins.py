"""
cat3/advanced_ins.py
Extension 3c: Advanced INS Error Model (Temperature Drift + Schuler Oscillation)

Mathematical basis (Section 5.3 of Extension Plan):
  Accelerometer bias model:
    b(t) = b0 + k_T * (T(t) - T0) + white_noise_in_run

  Schuler dynamics (orbital resonance with Earth's gravity pendulum):
    omega_S = sqrt(g / R_earth) = 2*pi / T_S where T_S = 84.4 min
    Schuler loop couples position error back to tilt, creating bounded oscillation.

  Augmented EKF state (Stage 2 JammerUKF extended):
    x = [x_j, y_j, b_x, b_y, delta_T]
    b_x, b_y = IMU accelerometer bias (m/s^2)
    delta_T  = temperature deviation from nominal (deg C)

Integration:
  - Extends the Stage 2 JammerUKF (same interface, augmented state)
  - Uses rician_mrc_outage() to show how INS position error
    increases mission outage probability
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))

import numpy as np
from typing import Optional
from scipy.linalg import block_diag

from channel_bridge import (
    rician_mrc_outage, compute_crlb_doa_snr_aware,
    SINR_THRESH_DB, RICIAN_K, L_MRC, N_ARRAY, L_SNAPSHOTS
)


# ─────────────────────────────────────────────────────────────────────────────
# Physical constants
# ─────────────────────────────────────────────────────────────────────────────
G_MS2     = 9.81            # gravity (m/s^2)
R_EARTH   = 6.371e6         # Earth radius (m)
OMEGA_S   = np.sqrt(G_MS2 / R_EARTH)    # Schuler frequency (rad/s)
T_SCHULER = 2 * np.pi / OMEGA_S         # = 5057 s ≈ 84.3 min


# ─────────────────────────────────────────────────────────────────────────────
# Advanced INS Error Model
# ─────────────────────────────────────────────────────────────────────────────
class AdvancedINS:
    """
    Extended INS error model adding:
      1. Temperature-dependent accelerometer bias
      2. Schuler oscillation dynamics
      3. Augmented 5-state EKF: [x, y, b_x, b_y, delta_T]

    Used to show how thermal environment and Schuler resonance degrade
    UAV position accuracy beyond the Dryden wind model alone.
    """

    # Typical MEMS IMU temperature coefficients
    K_TEMP = 1e-4           # m/s^2 per degC
    B0     = 5e-4           # m/s^2 initial bias at T0
    T0     = 25.0           # degC nominal temperature
    SIGMA_BIAS_WALK = 5e-5  # m/s^2/sqrt(s) bias random walk

    def __init__(self,
                 dt_s:        float = 0.1,
                 sigma_accel: float = 0.01,    # m/s^2 white noise (1-sigma)
                 sigma_gps:   float = 2.0,     # m GPS aiding noise (1-sigma)
                 use_schuler: bool  = True):
        self.dt      = dt_s
        self.sig_a   = sigma_accel
        self.sig_gps = sigma_gps
        self.schuler = use_schuler

        # Augmented state: [x, y, vx, vy, bx, by, dT]
        # x,y = position error (m); vx,vy = velocity error (m/s)
        # bx,by = accel bias (m/s^2); dT = temperature deviation (degC)
        n = 7
        self.x  = np.zeros(n)           # state vector
        self.P  = np.diag([1.0, 1.0,    # pos (m^2)
                           0.01, 0.01,  # vel
                           1e-6, 1e-6,  # bias (m/s^2)^2
                           4.0])        # temperature (degC^2)
        self.n  = n
        self._build_matrices()

    def _build_matrices(self):
        dt = self.dt
        ws = OMEGA_S

        # State transition: position/velocity with Schuler coupling
        # Schuler: x'' = -omega_S^2 * x  (from tilt-gravity coupling)
        if self.schuler:
            c  = np.cos(ws * dt)
            s_ = np.sin(ws * dt)
            # 4x4 Schuler-coupled position/velocity block
            F_pv = np.array([[c,      0, s_/ws,   0   ],
                             [0,      c,   0,   s_/ws  ],
                             [-ws*s_, 0,   c,     0    ],
                             [0, -ws*s_, 0,       c    ]])
        else:
            F_pv = np.array([[1, 0, dt, 0 ],
                             [0, 1, 0,  dt],
                             [0, 0, 1,  0 ],
                             [0, 0, 0,  1 ]])

        # Bias integration: b_k+1 = b_k (random walk)
        F_bias = np.eye(2)

        # Temperature 1st-order Markov with 5-minute correlation time
        tau_T  = 300.0
        F_temp = np.array([[np.exp(-dt / tau_T)]])

        # Full state transition
        self.F = block_diag(F_pv, F_bias, F_temp)
        # Bias drives velocity (dt integration)
        self.F[2, 4] = dt   # vx ← bx
        self.F[3, 5] = dt   # vy ← by
        # Temperature drives bias statically, we apply it in predict()
        # self.F[4, 6] = self.K_TEMP  # bx ← dT
        # self.F[5, 6] = self.K_TEMP

        # Process noise
        q_pos  = (self.sig_a * dt**2 / 2) ** 2
        q_vel  = (self.sig_a * dt) ** 2
        q_bias = (self.SIGMA_BIAS_WALK * np.sqrt(dt)) ** 2
        q_temp = (0.5 * np.sqrt(dt)) ** 2   # temperature drift 0.5 degC/sqrt(s)
        self.Q = np.diag([q_pos, q_pos, q_vel, q_vel, q_bias, q_bias, q_temp])

        # GPS measurement matrix: observe position only
        self.H = np.zeros((2, 7))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.R = np.eye(2) * self.sig_gps ** 2

    def predict(self, inject_bias: bool = False) -> None:
        if inject_bias:
            delta_ba_hz = 9.81e-4
            self.x[2] += delta_ba_hz * self.dt
            
        # Add static thermal bias mapping to velocity
        thermal_bias = self.K_TEMP * self.x[6]
        self.x[2] += thermal_bias * self.dt
        self.x[3] += thermal_bias * self.dt
        
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update_gps(self, gps_meas_m: np.ndarray) -> None:
        """EKF update with GPS position fix."""
        y = gps_meas_m - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(self.n) - K @ self.H) @ self.P

    def position_error_m(self) -> np.ndarray:
        """Current estimated position error (from state vector)."""
        return self.x[:2]

    def position_rmse_m(self) -> float:
        """RMS position error from covariance diagonal."""
        return float(np.sqrt(self.P[0, 0] + self.P[1, 1]))

    def temperature_deviation_degc(self) -> float:
        return float(self.x[6])

    def bias_norm_ms2(self) -> float:
        return float(np.linalg.norm(self.x[4:6]))


def simulate_ins_trajectory(duration_s: float = 300.0,
                             dt_s:       float = 0.1,
                             temperature_profile: Optional[np.ndarray] = None,
                             gps_update_rate_hz: float = 0.0,
                             use_schuler: bool = True,
                             inject_bias: bool = False,
                             seed: int = 42) -> dict:
    """
    Simulate INS error over time with temperature drift and Schuler oscillation.

    Parameters
    ----------
    temperature_profile : ndarray (T,) — temperature vs time in degC.
                          If None, uses a slow ramp from 25 to 45 degC.

    Returns
    -------
    dict with arrays: times, pos_rmse, bias_norm, temp_dev, doa_sigma
    """
    rng  = np.random.default_rng(seed)
    ins  = AdvancedINS(dt_s=dt_s, use_schuler=use_schuler)
    N_t  = int(duration_s / dt_s)
    t    = np.arange(N_t) * dt_s

    if temperature_profile is None:
        # Slow heat-soak: 25 → 45 degC over 5 minutes
        temperature_profile = 25.0 + 20.0 * np.clip(t / 300.0, 0, 1)

    if gps_update_rate_hz > 0:
        gps_interval = max(1, int(1.0 / (gps_update_rate_hz * dt_s)))
    else:
        gps_interval = int(1e9) # Effectively infinite

    pos_rmse_arr  = np.zeros(N_t)
    bias_norm_arr = np.zeros(N_t)
    temp_dev_arr  = np.zeros(N_t)
    doa_sigma_arr = np.zeros(N_t)

    for k in range(N_t):
        # Inject temperature state
        ins.x[6] = temperature_profile[k] - AdvancedINS.T0

        ins.predict(inject_bias=inject_bias)

        # GPS aiding at the specified rate
        if gps_update_rate_hz > 0 and k % gps_interval == 0:
            gps_noise = rng.normal(0, ins.sig_gps, size=2)
            ins.update_gps(gps_noise)

        # Total error = deterministic instantaneous error magnitude
        rmse = float(np.linalg.norm(ins.position_error_m()))
        pos_rmse_arr[k]  = rmse
        bias_norm_arr[k] = ins.bias_norm_ms2()
        temp_dev_arr[k]  = ins.temperature_deviation_degc()

        # DOA sigma degradation from position error (same formula as wind_robustness)
        sigma_music = compute_crlb_doa_snr_aware(N_ARRAY, L_SNAPSHOTS, 15.0, 30.0)
        sigma_wind_deg = float(np.degrees(np.arctan2(rmse, 1000.0)))
        doa_sigma_arr[k] = float(np.sqrt(sigma_music**2 + sigma_wind_deg**2))

    return {
        'times_s':     t,
        'pos_rmse_m':  pos_rmse_arr,
        'bias_norm':   bias_norm_arr,
        'temp_dev':    temp_dev_arr,
        'doa_sigma':   doa_sigma_arr,
    }

