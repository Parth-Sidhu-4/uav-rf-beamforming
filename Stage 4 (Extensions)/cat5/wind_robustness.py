"""
cat5/wind_robustness.py
Extension 5b: Wind Robustness via Dryden Gust Model

Mathematical basis (Section 7.2 of Extension Plan):
  - Drag force:  F = 0.5 * rho * C_D * A * V_w^2
  - Steady displacement: delta_r = F / k_c
  - Dryden PSD: Phi(omega) = sigma_u^2 * (2*L_u/(pi*V)) / (1 + (L_u*omega/V)^2)
  - INS position error: double integral of gust acceleration

Wiring to Stage 2:
  - Uses rician_mrc_outage() to convert INS position error into
    increased pointing uncertainty, which degrades the MUSIC DOA sigma,
    which degrades triangulation accuracy (Cat 1 → Cat 5b chain)
  - Uses compute_crlb_doa_snr_aware() to show degraded CRLB under wind
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))

import numpy as np
from dataclasses import dataclass
from scipy.signal import lti, lsim

from channel_bridge import (
    rician_mrc_outage, compute_crlb_doa_snr_aware,
    SINR_THRESH_DB, RICIAN_K, N_ARRAY, L_SNAPSHOTS, L_MRC
)


@dataclass
class WindConfig:
    v_wind_ms:             float = 10.0    # steady crosswind (m/s)
    altitude_m:            float = 50.0    # UAV AGL (m)
    airspeed_ms:           float = 15.0    # UAV airspeed (m/s)
    C_D:                   float = 1.2     # drag coefficient (multirotor)
    A_eff_m2:              float = 0.08    # effective cross-section
    mass_kg:               float = 1.5     # UAV mass
    control_stiffness_Npm: float = 5.0     # PD controller stiffness (N/m)
    dt_s:                  float = 0.01    # simulation timestep


class DrydenWindModel:
    """
    Dryden gust model (MIL-SPEC-1797B) generating correlated wind
    turbulence time series and computing resulting INS position errors.
    """

    def __init__(self, config: WindConfig):
        self.cfg = config
        rho_air = 1.225
        c = config
        self.F_drag_N = 0.5 * rho_air * c.C_D * c.A_eff_m2 * c.v_wind_ms ** 2
        self.delta_r_ss_m = self.F_drag_N / c.control_stiffness_Npm

    def steady_displacement_m(self) -> float:
        """Quasi-static crosswind displacement at steady state."""
        return self.delta_r_ss_m

    def generate_gust_series(self, t_total_s: float,
                              seed: int = 0) -> np.ndarray:
        """
        Generate Dryden turbulence acceleration time series (m/s^2).
        Uses first-order Dryden shaping filter.

        Returns
        -------
        a_gust : ndarray (N_steps,)  wind acceleration in m/s^2
        """
        cfg = self.cfg
        N_steps = int(t_total_s / cfg.dt_s)
        t_arr   = np.arange(N_steps) * cfg.dt_s

        # Dryden length scale and intensity
        L_u     = cfg.altitude_m / 0.177
        sigma_u = 0.1 * cfg.v_wind_ms

        # Dryden shaping filter: H(s) = gain / (tau*s + 1)
        tau   = L_u / cfg.airspeed_ms
        gain  = sigma_u * np.sqrt(2 * L_u / (np.pi * cfg.airspeed_ms))
        system = lti([gain], [tau, 1.0])

        rng = np.random.default_rng(seed)
        white_noise = rng.standard_normal(N_steps) / np.sqrt(cfg.dt_s)
        _, y_gust, _ = lsim(system, white_noise, t_arr)

        # Wind speed gust -> acceleration (linearised Newton's 2nd law)
        rho_air = 1.225
        a_gust = (cfg.C_D * rho_air * cfg.A_eff_m2
                  * cfg.v_wind_ms / cfg.mass_kg) * y_gust
        return a_gust

    def ins_position_error(self, a_gust: np.ndarray) -> np.ndarray:
        """
        Double-integrate wind acceleration to get INS position error.
        (Assumes INS cannot distinguish wind from platform motion.)

        Returns
        -------
        pos_error : ndarray  cumulative position error in metres
        """
        dt = self.cfg.dt_s
        vel_error = np.cumsum(a_gust) * dt
        pos_error = np.cumsum(vel_error) * dt
        return pos_error

    def degraded_doa_sigma_deg(self,
                                pos_error_m: float,
                                range_m:     float = 1000.0,
                                snr_db:      float = 15.0,
                                theta_deg:   float = 30.0) -> float:
        """
        Compute MUSIC CRLB with wind-induced pointing uncertainty added in RSS.

        The INS position error at the UAV translates to an angular uncertainty
        of sigma_theta_wind = arctan(pos_error / range) fed back into the DOA model.

        Returns
        -------
        total_sigma_deg : float  RSS of MUSIC CRLB + wind pointing error
        """
        # Stage 2 MUSIC CRLB (no wind)
        sigma_music_deg = compute_crlb_doa_snr_aware(
            N_ARRAY, L_SNAPSHOTS, snr_db, theta_deg, calib_error_deg=0.0)

        # Wind-induced angular uncertainty (geometric)
        sigma_wind_rad = np.arctan2(abs(pos_error_m), range_m)
        sigma_wind_deg = float(np.degrees(sigma_wind_rad))

        return float(np.sqrt(sigma_music_deg ** 2 + sigma_wind_deg ** 2))


def wind_speed_sweep(v_wind_values: list = None,
                     t_total_s: float = 60.0,
                     range_m: float = 1000.0,
                     snr_db:  float = 15.0) -> dict:
    """
    Sweep wind speed and report steady displacement, gust RMS, and
    degraded DOA sigma for each wind speed.

    Returns
    -------
    dict {v_wind: {delta_r_ss, gust_rms_ms2, sigma_pos_rms_m,
                   doa_sigma_no_wind_deg, doa_sigma_wind_deg}}
    """
    if v_wind_values is None:
        v_wind_values = [0, 5, 10, 15, 20]

    results = {}
    for v_w in v_wind_values:
        cfg   = WindConfig(v_wind_ms=max(v_w, 0.01))
        model = DrydenWindModel(cfg)
        gust  = model.generate_gust_series(t_total_s, seed=42)
        pos_e = model.ins_position_error(gust)

        sigma_pos_rms = float(np.std(pos_e))
        worst_pos     = float(np.percentile(np.abs(pos_e), 95))

        doa_no_wind = float(compute_crlb_doa_snr_aware(
            N_ARRAY, L_SNAPSHOTS, snr_db, 30.0))
        doa_wind    = model.degraded_doa_sigma_deg(worst_pos, range_m, snr_db)

        # Outage degradation at 1 sigma position error
        p_out_base = rician_mrc_outage(SINR_THRESH_DB, snr_db, L_MRC, RICIAN_K)

        results[v_w] = {
            'delta_r_ss_m':          model.steady_displacement_m(),
            'gust_rms_ms2':          float(np.std(gust)),
            'sigma_pos_rms_m':       sigma_pos_rms,
            'worst_pos_error_m':     worst_pos,
            'doa_sigma_no_wind_deg': doa_no_wind,
            'doa_sigma_wind_deg':    doa_wind,
            'p_out_base':            p_out_base,
        }

    return results
