"""
phase_b_beamforming_remediated.py
Phase B: Array Signal Processing — Full Remediation

Fixes applied:
  FIX-4B: Rician spatial covariance matrix (LOS + scatter decomposition)
  FIX-7 : MUSIC CRLB with SNR dependence and RSS calibration error
  FIX-12: 3D ULA steering vector with azimuth + elevation
"""

import numpy as np
import scipy.linalg as la
from typing import Optional

# =============================================================================
# FIX-12: 3D ULA STEERING VECTOR
# =============================================================================

def ula_steering_vector_3d(N: int, theta_az_deg: float, theta_el_deg: float = 0.0,
                            d_lambda: float = 0.5) -> np.ndarray:
    """
    FIX-12: Compute the steering vector for a horizontally-oriented N-element ULA
    in 3D geometry using azimuth and elevation angles.

    For a horizontally-mounted ULA, the direction cosine projected onto the array axis is:
        u = sin(phi) * cos(epsilon)
    where phi is azimuth and epsilon is elevation.

    The steering vector element n is: exp(j * 2*pi * d_lambda * n * u)
    For d_lambda = 0.5 (half-wavelength): exp(j * pi * n * sin(phi) * cos(epsilon))

    Args:
        N           : Number of array elements.
        theta_az_deg: Azimuth angle in degrees (relative to array boresight).
        theta_el_deg: Elevation angle in degrees above horizontal (default 0.0).
        d_lambda    : Element spacing in wavelengths (default 0.5).

    Returns:
        Complex steering vector of shape (N,).

    Note: At theta_el_deg = 0 (2D degenerate case), reduces exactly to the 2D
    steering vector ula_steering_vector(N, theta_az_deg).
    """
    az_rad = np.radians(theta_az_deg)
    el_rad = np.radians(theta_el_deg)
    # Direction cosine projection onto array axis
    u = np.sin(az_rad) * np.cos(el_rad)
    n = np.arange(N)
    return np.exp(1j * 2.0 * np.pi * d_lambda * n * u)



def ula_steering_vector(N: int, theta_rad: float) -> np.ndarray:
    """2D ULA steering vector (λ/2 spacing, half-wavelength, backward-compatible)."""
    n = np.arange(N)
    return np.exp(1j * n * np.pi * np.sin(theta_rad))


def steering_vector_derivative(N: int, theta_rad: float) -> np.ndarray:
    """Derivative of the steering vector with respect to theta (radians)."""
    n = np.arange(N)
    a = ula_steering_vector(N, theta_rad)
    return 1j * n * np.pi * np.cos(theta_rad) * a


# =============================================================================
# FIX-4B: RICIAN SPATIAL COVARIANCE MATRIX (LOS + SCATTER DECOMPOSITION)
# =============================================================================

def rician_source_covariance(N: int, theta_rad: float, sigma2_total: float, K: float) -> np.ndarray:
    """
    FIX-4B: Compute the spatial covariance contribution for a single Rician source.

    Decomposes into:
      - LOS component:     sigma2_LOS  = sigma2_total * K / (K + 1)
      - Scatter component: sigma2_scat = sigma2_total / (K + 1)

    Full covariance: R_source = sigma2_LOS * a * a^H + sigma2_scat * I_N
    (The scatter term is approximated as isotropic — appropriate first-order correction.)

    Args:
        N          : Number of array elements.
        theta_rad  : Source angle in radians.
        sigma2_total: Total source power (linear).
        K          : Rician K-factor (K=0 → Rayleigh, K→∞ → pure LOS).

    Returns:
        R_source   : N×N complex covariance matrix.
    """
    a = ula_steering_vector(N, theta_rad)
    sigma2_LOS  = sigma2_total * K / (K + 1.0)
    sigma2_scat = sigma2_total / (K + 1.0)
    R_LOS  = sigma2_LOS * np.outer(a, np.conj(a))
    R_scat = sigma2_scat * np.eye(N, dtype=complex)
    return R_LOS + R_scat


def build_rician_covariance_matrix(N: int,
                                    theta_s_rad: float, sigma2_s: float, K_signal: float,
                                    theta_j_rad: float, sigma2_j: float, K_jammer: float,
                                    sigma2_noise: float = 1.0) -> np.ndarray:
    """
    FIX-4B: Build the full spatial covariance matrix with Rician LOS + scatter decomposition.

    R_xx = R_signal + R_jammer + sigma2_noise * I_N

    The effective noise floor is:
        sigma2_noise + sigma2_s_scatter + sigma2_j_scatter
    (scatter terms add to the thermal noise floor).

    Args:
        N            : Array elements.
        theta_s_rad  : Signal angle (rad).
        sigma2_s     : Signal total power (linear).
        K_signal     : Signal Rician K-factor.
        theta_j_rad  : Jammer angle (rad).
        sigma2_j     : Jammer total power (linear).
        K_jammer     : Jammer Rician K-factor.
        sigma2_noise : Thermal noise power per element (default 1.0).

    Returns:
        R_xx: N×N complex covariance matrix.

    Validation note: As K_signal → ∞ and K_jammer → ∞, this converges to the
    deterministic model: sigma2_s * a_s*a_s^H + sigma2_j * a_j*a_j^H + sigma2_noise * I.
    """
    R_signal = rician_source_covariance(N, theta_s_rad, sigma2_s, K_signal)
    R_jammer = rician_source_covariance(N, theta_j_rad, sigma2_j, K_jammer)
    R_noise   = sigma2_noise * np.eye(N, dtype=complex)
    return R_signal + R_jammer + R_noise


# =============================================================================
# FIX-7: MUSIC CRLB WITH SNR DEPENDENCE AND RSS CALIBRATION ERROR
# =============================================================================

def compute_crlb_doa_snr_aware(N: int, L_snapshots: int, SNR_dB_per_element: float,
                                 theta_deg: float, calib_error_deg: float = 0.5) -> float:
    """
    FIX-7: Compute ULA CRB on DOA estimation with SNR dependence and RSS calibration error.

    Formula (ULA, λ/2 spacing, far-field):
        sigma²_CRB = 6 / (L_snapshots * rho * N * (N²-1) * pi² * cos²(theta))
    where rho = per-element SNR * N (coherent combining gain).

    Hardware floor: sigma_music_deg = max(0.5°, sigma_CRB_deg)
    Calibration RSS: sigma_total = sqrt(sigma_music² + calib_error²)

    Args:
        N                   : Number of array elements.
        L_snapshots         : Number of MUSIC snapshots (FIX-1: not L_fhss).
        SNR_dB_per_element  : Per-element SNR in dB.
        theta_deg           : Source angle in degrees.
        calib_error_deg     : Hardware calibration error (1-sigma, degrees). Default 0.5°.

    Returns:
        Total sigma in degrees (statistical + calibration combined via RSS).
    """
    theta_rad = np.radians(theta_deg)
    # Array output SNR (coherent combining gain: N × per-element SNR)
    rho = 10.0 ** (SNR_dB_per_element / 10.0) * N
    cos2 = max(np.cos(theta_rad) ** 2, 1e-6)

    var_crb_rad2 = 6.0 / (L_snapshots * rho * N * (N ** 2 - 1) * np.pi ** 2 * cos2)
    sigma_crb_deg = np.degrees(np.sqrt(var_crb_rad2))

    # Hardware resolution floor
    sigma_music_deg = max(0.5, sigma_crb_deg)

    # FIX-7: RSS combination (not additive!)
    sigma_total_deg = np.sqrt(sigma_music_deg ** 2 + calib_error_deg ** 2)
    return float(sigma_total_deg)


def compute_crlb_doa_rad(N: int, L: int, SNR_dB: float, theta_rad: float) -> float:
    """
    Original CRLB formula (backward-compatible, no calibration RSS).
    Kept for regression testing. FIX-7 uses compute_crlb_doa_snr_aware instead.
    """
    SNR_lin = 10.0 ** (SNR_dB / 10.0)
    a = ula_steering_vector(N, theta_rad)
    da = steering_vector_derivative(N, theta_rad)
    P_a_orth = np.eye(N) - np.outer(a, np.conj(a)) / N
    denom_term = np.real(np.conj(da).T @ P_a_orth @ da)
    return 1.0 / (2.0 * L * SNR_lin * denom_term)


# =============================================================================
# SIGNAL GENERATION (updated for Rician covariance)
# =============================================================================

def generate_received_signal_rician(
    N: int,
    theta_s_deg: float,
    theta_j_list_deg: list,
    SNR_dB: float,
    INR_dB_list: list,
    L_snapshots: int,  # FIX-1: renamed from L
    K_signal: float = 10.0,
    K_jammer: float = 12.0,  # FIX-4A: K_JAMMER_LOS = 12.0
    rng: Optional[np.random.Generator] = None
):
    """
    FIX-4B: Generate signal snapshots using Rician-corrected covariance.
    FIX-1: Parameter renamed to L_snapshots (snapshot count, distinct from L_fhss).
    FIX-4A: Default K_jammer raised from 3.0 to 12.0 (directional LOS jammer).
    """
    if rng is None:
        rng = np.random.default_rng()

    sigma_n2 = 1.0
    sigma_s2 = 10.0 ** (SNR_dB / 10.0)

    theta_s_rad = np.radians(theta_s_deg)
    a_s = ula_steering_vector(N, theta_s_rad)

    # Signal source
    s = (rng.normal(0, np.sqrt(sigma_s2 / 2.0), L_snapshots) +
         1j * rng.normal(0, np.sqrt(sigma_s2 / 2.0), L_snapshots))
    X_s = np.outer(a_s, s)

    # Jammer sources
    X_j = np.zeros((N, L_snapshots), dtype=complex)
    for theta_j_deg, INR_dB in zip(theta_j_list_deg, INR_dB_list):
        sigma_j2 = 10.0 ** (INR_dB / 10.0)
        theta_j_rad = np.radians(theta_j_deg)
        j_sig = (rng.normal(0, np.sqrt(sigma_j2 / 2.0), L_snapshots) +
                 1j * rng.normal(0, np.sqrt(sigma_j2 / 2.0), L_snapshots))
        a_j = ula_steering_vector(N, theta_j_rad)
        X_j += np.outer(a_j, j_sig)

    # Thermal noise
    noise = (rng.normal(0, np.sqrt(sigma_n2 / 2.0), (N, L_snapshots)) +
             1j * rng.normal(0, np.sqrt(sigma_n2 / 2.0), (N, L_snapshots)))

    X = X_s + X_j + noise

    # FIX-4B: Rician true covariance (replaces deterministic plane-wave model)
    theta_j_rads = [np.radians(t) for t in theta_j_list_deg]
    R_xx_true = build_rician_covariance_matrix(
        N=N,
        theta_s_rad=theta_s_rad,
        sigma2_s=sigma_s2,
        K_signal=K_signal,
        theta_j_rad=theta_j_rads[0] if theta_j_rads else 0.0,
        sigma2_j=10.0 ** (INR_dB_list[0] / 10.0) if INR_dB_list else 0.0,
        K_jammer=K_jammer,
        sigma2_noise=sigma_n2
    )

    R_xx_sample = (X @ np.conj(X).T) / L_snapshots
    return X, R_xx_sample, R_xx_true


# =============================================================================
# BEAMFORMERS (unchanged logic; updated to use FIX-4B R_xx)
# =============================================================================

def mvdr_beamformer(R_xx: np.ndarray, theta_s: float) -> np.ndarray:
    """MVDR beamformer (unchanged logic, now receives FIX-4B corrected R_xx)."""
    N = R_xx.shape[0]
    a_s = ula_steering_vector(N, theta_s)
    R_inv = la.inv(R_xx)
    numerator = R_inv @ a_s
    denominator = np.conj(a_s).T @ numerator
    return numerator / denominator


def lcmv_beamformer(R_xx: np.ndarray, theta_s: float,
                    theta_j_list: list, loading_factor: float = 0.10) -> np.ndarray:
    """
    LCMV beamformer with diagonal loading (unchanged logic).
    Now receives FIX-4B corrected R_xx.
    """
    N = R_xx.shape[0]
    M = len(theta_j_list)

    # Diagonal loading
    delta = loading_factor * np.real(np.trace(R_xx)) / N
    R_loaded = R_xx + delta * np.eye(N, dtype=complex)

    a_s = ula_steering_vector(N, theta_s)
    C = np.zeros((N, M + 1), dtype=complex)
    C[:, 0] = a_s
    for idx, theta_j in enumerate(theta_j_list):
        C[:, idx + 1] = ula_steering_vector(N, theta_j)

    f = np.zeros(M + 1, dtype=complex)
    f[0] = 1.0

    R_inv = la.inv(R_loaded)
    temp1 = R_inv @ C
    temp2 = np.conj(C).T @ temp1
    return temp1 @ la.inv(temp2) @ f


def compute_output_sinr(w: np.ndarray, R_jn: np.ndarray,
                         SNR_dB: float, theta_s: float) -> float:
    """Output SINR (unchanged logic)."""
    N = R_jn.shape[0]
    sigma_s2 = 10.0 ** (SNR_dB / 10.0)
    a_s = ula_steering_vector(N, theta_s)
    signal_power_out = sigma_s2 * (np.abs(np.conj(w).T @ a_s) ** 2)
    interference_noise_power_out = np.real(np.conj(w).T @ R_jn @ w)
    return 10.0 * np.log10(signal_power_out / max(interference_noise_power_out, 1e-15))


# =============================================================================
# MUSIC DOA (unchanged logic)
# =============================================================================

def music_doa(R_xx: np.ndarray, num_sources: int,
              scan_resolution_deg: float = 0.1) -> tuple:
    """MUSIC DOA estimator (unchanged logic)."""
    N = R_xx.shape[0]
    eigenvals, eigenvecs = la.eigh(R_xx)
    idx = np.argsort(eigenvals)
    eigenvecs = eigenvecs[:, idx]
    E_n = eigenvecs[:, :N - num_sources]

    scan_angles_deg = np.arange(-90.0, 90.0 + scan_resolution_deg, scan_resolution_deg)
    pseudo_spectrum = np.zeros_like(scan_angles_deg)

    E_n_H = np.conj(E_n).T
    E_n_proj = E_n @ E_n_H

    for idx_angle, angle_deg in enumerate(scan_angles_deg):
        angle_rad = np.radians(angle_deg)
        a = ula_steering_vector(N, angle_rad)
        denom = np.real(np.conj(a).T @ E_n_proj @ a)
        pseudo_spectrum[idx_angle] = 1.0 / max(denom, 1e-15)

    return scan_angles_deg, pseudo_spectrum


def find_music_peaks(scan_angles_deg: np.ndarray,
                     pseudo_spectrum: np.ndarray,
                     num_sources: int) -> np.ndarray:
    """Find MUSIC pseudo-spectrum peaks (unchanged logic)."""
    peaks_idx = [i for i in range(1, len(pseudo_spectrum) - 1)
                 if pseudo_spectrum[i] > pseudo_spectrum[i - 1]
                 and pseudo_spectrum[i] > pseudo_spectrum[i + 1]]
    peaks_idx = np.array(peaks_idx)
    if len(peaks_idx) == 0:
        return np.array([])
    sorted_peaks_idx = peaks_idx[np.argsort(pseudo_spectrum[peaks_idx])][::-1]
    est_angles = scan_angles_deg[sorted_peaks_idx[:num_sources]]
    return np.sort(est_angles)


# =============================================================================
# FIX-12: 3D VALIDATION HELPERS
# =============================================================================

def validate_3d_steering_reduces_to_2d(N: int, theta_az_deg: float, tol: float = 1e-10) -> bool:
    """
    FIX-12 validation: At elevation = 0°, 3D steering vector must equal 2D vector.
    Returns True if max element-wise difference < tol.
    """
    theta_az_rad = np.radians(theta_az_deg)
    v_2d = ula_steering_vector(N, theta_az_rad)
    v_3d = ula_steering_vector_3d(N, theta_az_deg, theta_el_deg=0.0)
    return float(np.max(np.abs(v_2d - v_3d))) < tol


def null_depth_vs_elevation(N: int, theta_s_deg: float, theta_j_az_deg: float,
                              snr_dB: float, inr_dB: float,
                              el_range_deg: Optional[list] = None,
                              K_signal: float = 10.0, K_jammer: float = 12.0) -> dict:
    """
    FIX-12: Compute null depth vs. jammer elevation angle.
    Reports the elevation at which null degrades by 6 dB from peak (null 'width').

    Args:
        N              : Array elements.
        theta_s_deg    : Signal azimuth (degrees).
        theta_j_az_deg : Jammer azimuth (degrees).
        snr_dB         : Input SNR (dB).
        inr_dB         : Input INR (dB).
        el_range_deg   : List of elevation angles to test (default 0-20 in steps of 1).
        K_signal/K_jammer: Rician K-factors.

    Returns:
        dict with 'elevation_deg', 'null_depth_dB', '6dB_elevation_deg'.
    """
    if el_range_deg is None:
        el_range_deg = list(range(0, 21, 1))

    theta_s_rad = np.radians(theta_s_deg)
    theta_j_az_rad = np.radians(theta_j_az_deg)
    sigma2_s = 10.0 ** (snr_dB / 10.0)
    sigma2_j = 10.0 ** (inr_dB / 10.0)

    null_depths = []
    for el_deg in el_range_deg:
        # 3D steering vectors for signal and jammer
        a_s = ula_steering_vector_3d(N, theta_s_deg, 0.0)
        a_j = ula_steering_vector_3d(N, theta_j_az_deg, el_deg)

        # FIX-4B Rician covariance (use azimuth-only approximation for R_xx)
        R_xx = build_rician_covariance_matrix(
            N, theta_s_rad, sigma2_s, K_signal,
            theta_j_az_rad, sigma2_j, K_jammer
        )
        # Load and invert
        delta = 0.10 * np.real(np.trace(R_xx)) / N
        R_loaded = R_xx + delta * np.eye(N, dtype=complex)
        R_inv = la.inv(R_loaded)

        # LCMV: one signal constraint, one null at true jammer (using 3D steering)
        C = np.column_stack([a_s, a_j]).reshape(N, 2)
        f = np.array([[1.0], [0.0]])
        try:
            w = R_inv @ C @ la.inv(C.conj().T @ R_inv @ C) @ f
            leakage = np.abs(w.conj().T @ a_j.reshape(N, 1))[0, 0] ** 2
            nd = 10 * np.log10(max(leakage, 1e-12))
        except np.linalg.LinAlgError:
            nd = 0.0
        null_depths.append(nd)

    null_depths = np.array(null_depths)
    peak_nd = null_depths[0]  # ε=0 is peak (deepest null)
    target_6dB = peak_nd + 6.0  # 6 dB shallower than peak

    el_6dB = None
    for i, el in enumerate(el_range_deg):
        if null_depths[i] > target_6dB:
            el_6dB = el
            break

    return {
        'elevation_deg': el_range_deg,
        'null_depth_dB': null_depths.tolist(),
        '6dB_elevation_deg': el_6dB
    }
