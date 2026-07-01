"""
cat3/knife_edge.py
Extension 3b: Knife-Edge Terrain Diffraction Loss (ITU-R P.526-15)

Mathematical basis (Section 5.2 of Extension Plan):
  Fresnel-Kirchhoff diffraction parameter:
    nu = h_eff * sqrt(2*(d1+d2) / (lambda*d1*d2))

  ITU-R P.526 approximate formula for L(nu):
    nu <= -0.7 : L = 0 dB
    -0.7 < nu <= 0:  L = 20*log10(0.5 - 0.62*nu)
    0 < nu <= 1:     L = 20*log10(0.5*exp(-0.95*nu))
    1 < nu <= 2.4:   L = 20*log10(0.4 - sqrt(0.1184-(0.38-0.1*nu)^2))
    nu > 2.4:        L = 20*log10(0.225/nu)

  Multiple obstacles: Bullington equivalent single-edge method.

Integration with Stage 4:
  - Produces additive loss L_diff fed into band_sweep.py (5a enrichment)
  - Used by JAPP (3a) to determine terrain-masked waypoints
  - Uses fspl_db() from channel_bridge for frequency parametrization
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

from channel_bridge import fspl_db, BANDS_HZ  # noqa: F401


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic terrain profile generators (no SRTM dependency)
# ─────────────────────────────────────────────────────────────────────────────
def gaussian_hill_profile(n_points: int = 200,
                           path_length_m: float = 2000.0,
                           peak_heights_m: Tuple = (60.0, 40.0),
                           peak_positions: Tuple = (0.35, 0.65)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic 2-D terrain profile with Gaussian hills.

    Returns
    -------
    x_m   : ndarray (n_points,)  — horizontal distances in metres
    h_m   : ndarray (n_points,)  — terrain elevation in metres
    """
    x = np.linspace(0, path_length_m, n_points)
    h = np.zeros(n_points)
    sigma = path_length_m * 0.08    # hill width = 8% of path length
    for height, pos in zip(peak_heights_m, peak_positions):
        centre = pos * path_length_m
        h     += height * np.exp(-0.5 * ((x - centre) / sigma) ** 2)
    return x, h


# ─────────────────────────────────────────────────────────────────────────────
# Core ITU-R P.526 knife-edge formula
# ─────────────────────────────────────────────────────────────────────────────
def knife_edge_loss_db(nu: float) -> float:
    """
    ITU-R P.526-15 knife-edge diffraction loss L(nu) in dB.
    Returns POSITIVE dB (loss/attenuation). nu=-inf means no loss (0 dB).
    """
    if nu <= -0.7:
        return 0.0
    elif nu <= 0:
        return -float(20 * np.log10(0.5 - 0.62 * nu))
    elif nu <= 1:
        return -float(20 * np.log10(0.5 * np.exp(-0.95 * nu)))
    elif nu <= 2.4:
        inner = 0.1184 - (0.38 - 0.1 * nu) ** 2
        if inner < 0:
            inner = 0.0
        val = 0.4 - np.sqrt(inner)
        if val <= 0:
            return 50.0   # full obstruction cap
        return -float(20 * np.log10(val))
    else:
        return -float(20 * np.log10(0.225 / nu))


def fresnel_nu(h_eff_m: float, d1_m: float, d2_m: float,
               freq_hz: float) -> float:
    """
    Fresnel-Kirchhoff parameter nu for a single obstacle.

    Parameters
    ----------
    h_eff_m : float — height of obstacle above direct TX–RX path (m)
    d1_m    : float — TX to obstacle distance (m)
    d2_m    : float — obstacle to RX distance (m)
    freq_hz : float — carrier frequency (Hz)
    """
    lam = 3e8 / freq_hz
    return h_eff_m * np.sqrt(2 * (d1_m + d2_m) / (lam * d1_m * d2_m + 1e-30))


# ─────────────────────────────────────────────────────────────────────────────
# Multiple-obstacle: Bullington equivalent single-edge
# ─────────────────────────────────────────────────────────────────────────────
def bullington_equivalent_nu(x_m: np.ndarray,
                              h_m: np.ndarray,
                              h_tx_m: float,
                              h_rx_m: float,
                              freq_hz: float) -> Tuple[float, int]:
    """
    Bullington equivalent knife-edge method for a terrain profile.

    Draws lines from TX tip and RX tip and finds the obstacle that
    protrudes most above the direct line (Bullington's worst-case knife edge).

    Parameters
    ----------
    x_m   : ndarray — horizontal positions (m), x[0]=TX, x[-1]=RX
    h_m   : ndarray — terrain height (m) at each point
    h_tx_m : float  — TX antenna height above x[0] terrain
    h_rx_m : float  — RX antenna height above x[-1] terrain

    Returns
    -------
    nu_equiv   : float — equivalent Fresnel parameter for Bullington obstacle
    peak_index : int   — index of the dominant obstacle
    """
    d_total = x_m[-1] - x_m[0]
    h_tx_abs = h_m[0]  + h_tx_m
    h_rx_abs = h_m[-1] + h_rx_m

    # Height of the direct TX–RX ray at each intermediate point
    ray_h = h_tx_abs + (h_rx_abs - h_tx_abs) * (x_m - x_m[0]) / d_total

    # Effective clearance: terrain above the ray (positive = obstacle)
    h_eff_arr = h_m - ray_h
    peak_idx  = int(np.argmax(h_eff_arr))
    h_eff     = float(h_eff_arr[peak_idx])

    d1 = float(x_m[peak_idx] - x_m[0])
    d2 = float(x_m[-1] - x_m[peak_idx])
    if d1 < 1.0: d1 = 1.0
    if d2 < 1.0: d2 = 1.0

    nu = fresnel_nu(h_eff, d1, d2, freq_hz)
    return nu, peak_idx


# ─────────────────────────────────────────────────────────────────────────────
# Top-level: total diffraction loss for a terrain profile
# ─────────────────────────────────────────────────────────────────────────────
def knife_edge_total_loss_db(x_m: np.ndarray,
                              h_m: np.ndarray,
                              h_tx_m: float = 2.0,
                              h_rx_m: float = 2.0,
                              freq_hz: float = 2.4e9) -> float:
    """
    Compute total terrain diffraction loss using Bullington method.

    Parameters
    ----------
    x_m    : ndarray — horizontal range grid (m)
    h_m    : ndarray — terrain elevation (m)
    h_tx_m : float   — TX antenna height AGL (m)
    h_rx_m : float   — RX antenna height AGL (m)
    freq_hz: float   — carrier frequency (Hz)

    Returns
    -------
    L_diff : float — diffraction attenuation in dB (positive = loss)
    """
    nu, _ = bullington_equivalent_nu(x_m, h_m, h_tx_m, h_rx_m, freq_hz)
    return knife_edge_loss_db(nu)


def band_diffraction_comparison(x_m: np.ndarray,
                                 h_m: np.ndarray,
                                 h_tx_m: float = 2.0,
                                 h_rx_m: float = 2.0) -> dict:
    """
    Compute diffraction loss for all three standard bands.
    Directly shows 900 MHz advantage over terrain.

    Returns
    -------
    dict {band_name: L_diff_dB}
    """
    return {
        name: knife_edge_total_loss_db(x_m, h_m, h_tx_m, h_rx_m, freq)
        for name, freq in BANDS_HZ.items()
    }


def fresnel_zone_clearance_m(d1_m: float, d2_m: float,
                              freq_hz: float, n: int = 1) -> float:
    """
    First Fresnel zone radius at the obstacle (n=1 by convention).
    Rule of thumb: need 0.6*r1 clearance for negligible diffraction loss.
    """
    lam = 3e8 / freq_hz
    return np.sqrt(n * lam * d1_m * d2_m / (d1_m + d2_m))


# ─────────────────────────────────────────────────────────────────────────────
# Multiple-obstacle: Epstein-Peterson method
# ─────────────────────────────────────────────────────────────────────────────
def epstein_peterson_loss_db(x_m: np.ndarray,
                              h_m: np.ndarray,
                              h_tx_m: float = 2.0,
                              h_rx_m: float = 2.0,
                              freq_hz: float = 2.4e9,
                              min_peak_separation_m: float = 50.0) -> float:
    """
    Epstein-Peterson multiple knife-edge diffraction loss (ITU-R P.526-15).

    Algorithm (from image):
      1. Find all distinct obstacle peaks where terrain is above direct TX-RX ray.
      2. For obstacle k=1: compute nu_1 using actual TX, obstacle 1, obstacle 2 geometry.
      3. Replace TX with a virtual source at the TOP of obstacle k; iterate to RX.
      4. Sum individual losses: L_total = sum_k L_d(nu_k).

    More accurate than Bullington for two well-separated obstacles because
    it accounts for both peaks independently rather than finding one equivalent.

    Parameters
    ----------
    x_m                  : ndarray — horizontal positions (m), x[0]=TX, x[-1]=RX
    h_m                  : ndarray — terrain elevation (m)
    h_tx_m               : float   — TX antenna AGL (m)
    h_rx_m               : float   — RX antenna AGL (m)
    freq_hz              : float   — carrier frequency (Hz)
    min_peak_separation_m: float   — minimum distance between peaks to count separately

    Returns
    -------
    L_total : float — total diffraction loss (dB, positive = attenuation)
    """
    d_total  = float(x_m[-1] - x_m[0])
    h_tx_abs = float(h_m[0]  + h_tx_m)
    h_rx_abs = float(h_m[-1] + h_rx_m)

    # ── Step 1: direct TX-RX ray and effective obstacle heights ──────────────
    ray_h   = h_tx_abs + (h_rx_abs - h_tx_abs) * (x_m - x_m[0]) / d_total
    h_eff   = h_m - ray_h   # positive where terrain protrudes above ray

    # ── Step 2: find distinct obstacle peaks (local maxima of h_eff > 0) ─────
    obstacle_indices = []
    for i in range(1, len(h_eff) - 1):
        if h_eff[i] > 0 and h_eff[i] >= h_eff[i - 1] and h_eff[i] >= h_eff[i + 1]:
            obstacle_indices.append(i)

    # Merge peaks that are too close together (keep the taller one)
    if len(obstacle_indices) > 1:
        merged = [obstacle_indices[0]]
        for idx in obstacle_indices[1:]:
            if (x_m[idx] - x_m[merged[-1]]) < min_peak_separation_m:
                # keep the taller peak
                if h_m[idx] > h_m[merged[-1]]:
                    merged[-1] = idx
            else:
                merged.append(idx)
        obstacle_indices = merged

    if not obstacle_indices:
        return 0.0   # no obstacles above direct ray

    # ── Step 3: iterate, replacing TX with virtual source at each obstacle ────
    total_loss   = 0.0
    virt_tx_x    = float(x_m[0])
    virt_tx_h_abs = h_tx_abs

    for k, obs_idx in enumerate(obstacle_indices):
        obs_x   = float(x_m[obs_idx])
        obs_h   = float(h_m[obs_idx])   # terrain absolute height at obstacle

        # Next reference point: top of next obstacle, or actual RX
        if k + 1 < len(obstacle_indices):
            nxt_idx = obstacle_indices[k + 1]
            nxt_x   = float(x_m[nxt_idx])
            nxt_h   = float(h_m[nxt_idx])
        else:
            nxt_x = float(x_m[-1])
            nxt_h = h_rx_abs

        # Ray from virtual TX to next reference point
        span = nxt_x - virt_tx_x
        if span < 1.0:
            span = 1.0
        ray_at_obs = virt_tx_h_abs + (nxt_h - virt_tx_h_abs) * (obs_x - virt_tx_x) / span

        h_eff_k = obs_h - ray_at_obs   # effective height above this sub-path ray
        d1 = max(1.0, obs_x - virt_tx_x)
        d2 = max(1.0, nxt_x  - obs_x)

        nu_k = fresnel_nu(h_eff_k, d1, d2, freq_hz)
        total_loss += knife_edge_loss_db(nu_k)

        # Virtual TX advances to the crest of current obstacle
        virt_tx_x     = obs_x
        virt_tx_h_abs = obs_h

    return total_loss


def band_ep_comparison(x_m: np.ndarray,
                        h_m: np.ndarray,
                        h_tx_m: float = 2.0,
                        h_rx_m: float = 2.0) -> dict:
    """
    Epstein-Peterson diffraction loss for all three standard bands.
    Use alongside band_diffraction_comparison() to show method accuracy difference.
    """
    return {
        name: epstein_peterson_loss_db(x_m, h_m, h_tx_m, h_rx_m, freq)
        for name, freq in BANDS_HZ.items()
    }

