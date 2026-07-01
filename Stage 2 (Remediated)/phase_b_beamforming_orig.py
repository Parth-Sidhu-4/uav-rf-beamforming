"""Phase B: Array Signal Processing, Beamforming, and DOA Estimation.

This module implements Uniform Linear Array (ULA) steering vector calculations,
MVDR and LCMV beamforming, and MUSIC direction-of-arrival (DOA) estimation.
It also includes analytical Cramer-Rao Lower Bound (CRLB) calculations.
"""

import numpy as np
import scipy.linalg as la

def ula_steering_vector(N: int, theta_rad: float) -> np.ndarray:
    """Compute the steering vector for an N-element Uniform Linear Array (ULA)
    with half-wavelength element spacing (d = lambda/2).
    
    Args:
        N: Number of array elements.
        theta_rad: Look angle relative to array broadside in radians.
        
    Returns:
        Complex steering vector of shape (N,).
    """
    n = np.arange(N)
    # phase shift = 2 * pi * (d / lambda) * sin(theta) = pi * sin(theta) for d = lambda/2
    return np.exp(1j * n * np.pi * np.sin(theta_rad))

def steering_vector_derivative(N: int, theta_rad: float) -> np.ndarray:
    """Compute the derivative of the steering vector with respect to theta (in radians).
    
    Args:
        N: Number of array elements.
        theta_rad: Look angle relative to array broadside in radians.
        
    Returns:
        Complex derivative vector of shape (N,).
    """
    n = np.arange(N)
    # d(e^(j*n*pi*sin(theta)))/d(theta) = j * n * pi * cos(theta) * e^(j*n*pi*sin(theta))
    a = ula_steering_vector(N, theta_rad)
    return 1j * n * np.pi * np.cos(theta_rad) * a

def generate_received_signal(
    N: int,
    theta_s: float,
    theta_j_list: list[float],
    SNR_dB: float,
    INR_dB_list: list[float],
    L: int,
    rng: np.random.Generator | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate signal snapshots received by the ULA.
    
    Args:
        N: Number of array elements.
        theta_s: Angle of desired signal in radians.
        theta_j_list: List of angles for jammers in radians.
        SNR_dB: Input signal-to-noise ratio in dB.
        INR_dB_list: List of input jammer-to-noise ratios in dB.
        L: Number of snapshots (samples).
        rng: Random generator.
        
    Returns:
        X: Received signal matrix of shape (N, L).
        R_xx_est: Estimated sample covariance matrix of shape (N, N).
        R_jn_true: True jammer-plus-noise covariance matrix of shape (N, N).
    """
    if rng is None:
        rng = np.random.default_rng()
        
    # Noise power normalized to 1.0 (linear scale)
    sigma_n2 = 1.0
    sigma_s2 = 10.0 ** (SNR_dB / 10.0)
    
    # Generate desired signal source (complex Gaussian)
    s = (rng.normal(0, np.sqrt(sigma_s2/2.0), L) + 
         1j * rng.normal(0, np.sqrt(sigma_s2/2.0), L))
    a_s = ula_steering_vector(N, theta_s)
    
    # Desired signal component at array
    X_s = np.outer(a_s, s)
    
    # Generate jammers and build true jammer + noise covariance
    X_j = np.zeros((N, L), dtype=complex)
    R_jn_true = sigma_n2 * np.eye(N, dtype=complex)
    
    for theta_j, INR_dB in zip(theta_j_list, INR_dB_list):
        sigma_j2 = 10.0 ** (INR_dB / 10.0)
        j = (rng.normal(0, np.sqrt(sigma_j2/2.0), L) + 
             1j * rng.normal(0, np.sqrt(sigma_j2/2.0), L))
        a_j = ula_steering_vector(N, theta_j)
        X_j += np.outer(a_j, j)
        R_jn_true += sigma_j2 * np.outer(a_j, np.conj(a_j))
        
    # Generate thermal noise
    noise = (rng.normal(0, np.sqrt(sigma_n2/2.0), (N, L)) + 
             1j * rng.normal(0, np.sqrt(sigma_n2/2.0), (N, L)))
    
    # Total received signal
    X = X_s + X_j + noise
    
    # Sample covariance matrix
    R_xx_est = (X @ np.conj(X).T) / L
    
    return X, R_xx_est, R_jn_true

def mvdr_beamformer(R_xx: np.ndarray, theta_s: float) -> np.ndarray:
    """Minimum Variance Distortionless Response (MVDR) beamformer.
    
    Args:
        R_xx: Covariance matrix (estimated or true) of shape (N, N).
        theta_s: Look direction of desired signal in radians.
        
    Returns:
        Weight vector w of shape (N,).
    """
    N = R_xx.shape[0]
    a_s = ula_steering_vector(N, theta_s)
    
    R_inv = la.inv(R_xx)
    numerator = R_inv @ a_s
    denominator = np.conj(a_s).T @ numerator
    
    return numerator / denominator

def lcmv_beamformer(R_xx: np.ndarray, theta_s: float, theta_j_list: list[float]) -> np.ndarray:
    """Linearly Constrained Minimum Variance (LCMV) beamformer.
    Enforces unit gain in the desired signal direction, and zero gain (nulls)
    in the directions of specified jammers.
    
    Args:
        R_xx: Covariance matrix (estimated or true) of shape (N, N).
        theta_s: Look direction of desired signal in radians.
        theta_j_list: List of jammer directions to place nulls in.
        
    Returns:
        Weight vector w of shape (N,).
    """
    N = R_xx.shape[0]
    M = len(theta_j_list)
    
    # Construct constraint matrix C of shape (N, M + 1)
    a_s = ula_steering_vector(N, theta_s)
    C = np.zeros((N, M + 1), dtype=complex)
    C[:, 0] = a_s
    for idx, theta_j in enumerate(theta_j_list):
        C[:, idx + 1] = ula_steering_vector(N, theta_j)
        
    # Constraint vector f: [1, 0, ..., 0]^T of shape (M + 1,)
    f = np.zeros(M + 1, dtype=complex)
    f[0] = 1.0
    
    R_inv = la.inv(R_xx)
    
    # Closed-form LCMV solution: w = R_inv * C * (C^H * R_inv * C)^-1 * f
    temp1 = R_inv @ C
    temp2 = np.conj(C).T @ temp1
    temp2_inv = la.inv(temp2)
    
    return temp1 @ temp2_inv @ f

def compute_output_sinr(w: np.ndarray, R_jn: np.ndarray, SNR_dB: float, theta_s: float) -> float:
    """Compute output SINR for the beamformer weight vector w.
    
    Args:
        w: Weight vector of shape (N,).
        R_jn: Jammer-plus-noise covariance matrix of shape (N, N).
        SNR_dB: Input SNR in dB.
        theta_s: Direction of desired signal in radians.
        
    Returns:
        Output SINR in dB.
    """
    N = R_jn.shape[0]
    sigma_s2 = 10.0 ** (SNR_dB / 10.0)
    a_s = ula_steering_vector(N, theta_s)
    
    signal_power_out = sigma_s2 * (np.abs(np.conj(w).T @ a_s) ** 2)
    interference_noise_power_out = np.real(np.conj(w).T @ R_jn @ w)
    
    return 10.0 * np.log10(signal_power_out / max(interference_noise_power_out, 1e-15))

def music_doa(R_xx: np.ndarray, num_sources: int, scan_resolution_deg: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """MUSIC (Multiple Signal Classification) Direction-of-Arrival (DOA) estimator.
    
    Args:
        R_xx: Sample covariance matrix of shape (N, N).
        num_sources: Expected number of sources (desired + jammers).
        scan_resolution_deg: Fine grid search resolution in degrees.
        
    Returns:
        scan_angles_deg: Array of angles searched in degrees.
        pseudo_spectrum: MUSIC pseudo-spectrum values.
    """
    N = R_xx.shape[0]
    
    # Eigendecompose covariance matrix
    eigenvals, eigenvecs = la.eigh(R_xx)
    
    # Sort eigenvalues in ascending order
    idx = np.argsort(eigenvals)
    eigenvecs = eigenvecs[:, idx]
    
    # Noise subspace is spanned by eigenvectors corresponding to N - num_sources smallest eigenvalues
    E_n = eigenvecs[:, :N - num_sources]
    
    # Grid search from -90 to 90 degrees
    scan_angles_deg = np.arange(-90.0, 90.0 + scan_resolution_deg, scan_resolution_deg)
    pseudo_spectrum = np.zeros_like(scan_angles_deg)
    
    E_n_H = np.conj(E_n).T
    E_n_proj = E_n @ E_n_H
    
    for idx_angle, angle_deg in enumerate(scan_angles_deg):
        angle_rad = np.radians(angle_deg)
        a = ula_steering_vector(N, angle_rad)
        # Denominator: a^H * E_n * E_n^H * a
        denom = np.real(np.conj(a).T @ E_n_proj @ a)
        pseudo_spectrum[idx_angle] = 1.0 / max(denom, 1e-15)
        
    return scan_angles_deg, pseudo_spectrum

def find_music_peaks(scan_angles_deg: np.ndarray, pseudo_spectrum: np.ndarray, num_sources: int) -> np.ndarray:
    """Find peaks in the MUSIC pseudo-spectrum.
    
    Args:
        scan_angles_deg: Grid of angles in degrees.
        pseudo_spectrum: MUSIC pseudo-spectrum values.
        num_sources: Expected number of sources to locate.
        
    Returns:
        Sorted array of estimated peak angles in degrees.
    """
    # Simple peak detector: value is greater than neighbors
    peaks_idx = []
    for i in range(1, len(pseudo_spectrum) - 1):
        if pseudo_spectrum[i] > pseudo_spectrum[i - 1] and pseudo_spectrum[i] > pseudo_spectrum[i + 1]:
            peaks_idx.append(i)
            
    peaks_idx = np.array(peaks_idx)
    if len(peaks_idx) == 0:
        return np.array([])
        
    # Sort peaks by amplitude descending
    sorted_peaks_idx = peaks_idx[np.argsort(pseudo_spectrum[peaks_idx])][::-1]
    
    # Select top num_sources peaks
    est_angles = scan_angles_deg[sorted_peaks_idx[:num_sources]]
    return np.sort(est_angles)

def compute_crlb_doa_rad(N: int, L: int, SNR_dB: float, theta_rad: float) -> float:
    """Compute the Cramér-Rao Lower Bound (CRLB) on DOA variance (in radians^2).
    
    Formula: CRLB = 1 / (2 * L * SNR * (da^H * P_a_orth * da))
    where P_a_orth = I - a * a^H / N.
    
    Args:
        N: Number of array elements.
        L: Number of snapshots (samples).
        SNR_dB: SNR in dB.
        theta_rad: Source angle in radians.
        
    Returns:
        Lower bound on DOA variance in radians^2.
    """
    SNR_lin = 10.0 ** (SNR_dB / 10.0)
    a = ula_steering_vector(N, theta_rad)
    da = steering_vector_derivative(N, theta_rad)
    
    # Projection matrix onto the orthogonal complement of steering vector space
    P_a_orth = np.eye(N) - np.outer(a, np.conj(a)) / N
    
    denom_term = np.real(np.conj(da).T @ P_a_orth @ da)
    
    return 1.0 / (2.0 * L * SNR_lin * denom_term)
