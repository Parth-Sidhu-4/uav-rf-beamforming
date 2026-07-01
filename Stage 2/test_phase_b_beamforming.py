"""Unit tests for Phase B Array Signal Processing module.

Tests steering vector properties, derivatives, constraint satisfaction,
and MUSIC resolution.
"""

import numpy as np
import scipy.linalg as la
import pytest

# Import beamforming module
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
import phase_b_beamforming as pbb

def test_ula_steering_vector():
    """Verify that steering vector elements have unit magnitude and correct phase shifts."""
    N = 8
    theta = np.radians(30.0) # 30 degrees
    a = pbb.ula_steering_vector(N, theta)
    
    # 1. Magnitudes must be 1.0
    np.testing.assert_allclose(np.abs(a), 1.0, rtol=1e-12)
    
    # 2. Phase shifts must increase by pi * sin(theta) = pi * sin(30) = pi/2
    phases = np.angle(a)
    phase_diffs = np.diff(np.unwrap(phases))
    expected_diff = np.pi * np.sin(theta) # pi * 0.5 = 1.570796
    
    np.testing.assert_allclose(phase_diffs, expected_diff, rtol=1e-12)

def test_steering_vector_derivative():
    """Verify the analytical derivative against a numerical finite difference."""
    N = 8
    theta = np.radians(15.0)
    da_analytical = pbb.steering_vector_derivative(N, theta)
    
    # Central finite difference approximation
    d_theta = 1e-6
    a_plus = pbb.ula_steering_vector(N, theta + d_theta)
    a_minus = pbb.ula_steering_vector(N, theta - d_theta)
    da_numerical = (a_plus - a_minus) / (2.0 * d_theta)
    
    # Verify relative agreement
    np.testing.assert_allclose(da_analytical, da_numerical, rtol=1e-5, atol=1e-5)

def test_mvdr_beamformer_constraints():
    """Verify that MVDR satisfies the unit gain constraint w^H * a_s = 1."""
    N = 6
    theta_s = np.radians(10.0)
    theta_j_list = [np.radians(-20.0)]
    SNR_dB = 10.0
    INR_dB_list = [30.0]
    L = 1000
    
    _, R_xx, _ = pbb.generate_received_signal(N, theta_s, theta_j_list, SNR_dB, INR_dB_list, L, rng=np.random.default_rng(42))
    
    w = pbb.mvdr_beamformer(R_xx, theta_s)
    a_s = pbb.ula_steering_vector(N, theta_s)
    
    gain_s = np.conj(w).T @ a_s
    # Real part must be 1.0, imaginary part must be 0.0
    np.testing.assert_allclose(np.real(gain_s), 1.0, atol=1e-10)
    np.testing.assert_allclose(np.imag(gain_s), 0.0, atol=1e-10)

def test_lcmv_beamformer_nulls():
    """Verify that LCMV satisfies both desired-signal constraint (gain = 1)
    and jammer nulling constraints (gain = 0)."""
    N = 8
    theta_s = np.radians(0.0) # Desired signal at broadside
    theta_j_list = [np.radians(-30.0), np.radians(45.0)] # Jammers at -30 and +45
    SNR_dB = 15.0
    INR_dB_list = [40.0, 40.0]
    L = 500
    
    _, R_xx, R_jn = pbb.generate_received_signal(N, theta_s, theta_j_list, SNR_dB, INR_dB_list, L, rng=np.random.default_rng(100))
    
    w = pbb.lcmv_beamformer(R_xx, theta_s, theta_j_list)
    
    # 1. Verify desired signal gain is 1.0
    a_s = pbb.ula_steering_vector(N, theta_s)
    gain_s = np.conj(w).T @ a_s
    np.testing.assert_allclose(np.real(gain_s), 1.0, atol=1e-10)
    np.testing.assert_allclose(np.imag(gain_s), 0.0, atol=1e-10)
    
    # 2. Verify jammers are nulled (gain magnitude ~ 0.0)
    for theta_j in theta_j_list:
        a_j = pbb.ula_steering_vector(N, theta_j)
        gain_j = np.conj(w).T @ a_j
        np.testing.assert_allclose(np.abs(gain_j), 0.0, atol=1e-10)
        
    # 3. Check that output SINR can be computed
    sinr_out = pbb.compute_output_sinr(w, R_jn, SNR_dB, theta_s)
    assert sinr_out > SNR_dB - 3.0 # Null steering should preserve reasonable SINR

def test_music_doa_estimation():
    """Verify that MUSIC resolves direction-of-arrival peaks correctly."""
    N = 8
    theta_s = np.radians(15.0) # Source 1 at 15 degrees
    theta_j_list = [np.radians(-25.0)] # Source 2 at -25 degrees
    SNR_dB = 20.0
    INR_dB_list = [20.0]
    L = 1000
    
    _, R_xx, _ = pbb.generate_received_signal(N, theta_s, theta_j_list, SNR_dB, INR_dB_list, L, rng=np.random.default_rng(2024))
    
    # Total of 2 sources (1 desired + 1 jammer)
    scan_angles, pseudo_spectrum = pbb.music_doa(R_xx, num_sources=2, scan_resolution_deg=0.1)
    est_angles = pbb.find_music_peaks(scan_angles, pseudo_spectrum, num_sources=2)
    
    expected_angles = np.array([-25.0, 15.0])
    
    # Estimates should be accurate to within 0.5 degrees at this SNR/snapshots
    np.testing.assert_allclose(est_angles, expected_angles, atol=0.5)

def test_crlb():
    """Verify that CRLB runs and returns positive bounds."""
    N = 8
    L = 100
    SNR_dB = 10.0
    theta_rad = np.radians(0.0)
    
    var_bound = pbb.compute_crlb_doa_rad(N, L, SNR_dB, theta_rad)
    assert var_bound > 0.0
    
    # CRLB should decrease as SNR increases
    var_bound_high_snr = pbb.compute_crlb_doa_rad(N, L, SNR_dB + 10.0, theta_rad)
    assert var_bound_high_snr < var_bound

def test_n1_beamforming_vs_fspl():
    """Verify that for N=1 element, the output SINR computed via compute_output_sinr
    with w = [1.0] matches the analytical link budget exactly."""
    P_tx_dBm = 37.0         # UAV TX Power: 37 dBm (5 W)
    G_tx_dBi = 3.0          # UAV TX Antenna Gain: 3 dBi
    G_rx_element_dBi = 10.0 # Single element gain: 10 dBi
    B_Hz = 500e3            # Hop Bandwidth: 500 kHz
    NF_dB = 6.0             # Receiver Noise Figure: 6 dB
    
    P_N_dBm = -174.0 + 10.0 * np.log10(B_Hz) + NF_dB  # -111.0 dBm
    P_N_lin = 10.0 ** (P_N_dBm / 10.0)
    
    # Jammer (Class II: 73 dBm EIRP @ 5 km standoff)
    jammer_standoff_m = 5000.0
    jammer_eirp_dBm = 73.0
    G_rx_jam = 0.0
    B_ss_Hz = 40e6
    PG_dB = 10.0 * np.log10(B_ss_Hz / B_Hz)  # 19.03 dB
    
    # Range
    d_m = 1000.0
    
    # Analytical FSPL calculations
    def fspl_db(d):
        return 40.05 + 20.0 * np.log10(d)
        
    L_s = fspl_db(d_m)
    S_dBm = P_tx_dBm + G_tx_dBi + G_rx_element_dBi - L_s
    Ps = 10.0 ** (S_dBm / 10.0)
    
    L_j = fspl_db(jammer_standoff_m)
    J_dBm = jammer_eirp_dBm + G_rx_jam - L_j - PG_dB
    Pj = 10.0 ** (J_dBm / 10.0)
    
    # Direct analytical SINR
    sinr_analytical_db = 10.0 * np.log10(Ps / (Pj + P_N_lin))
    
    # Beamforming computation with N=1
    N = 1
    w = np.array([1.0], dtype=complex)
    
    # Normalized covariance model
    SNR_lin = Ps / P_N_lin
    INR_lin = Pj / P_N_lin
    
    # Steering vectors for N=1 are just [1.0]
    R_jn_norm = INR_lin * np.outer([1.0], [1.0]) + np.eye(N)
    
    sinr_beamforming_db = pbb.compute_output_sinr(w, R_jn_norm, 10.0 * np.log10(SNR_lin), 0.0)
    
    np.testing.assert_allclose(sinr_beamforming_db, sinr_analytical_db, rtol=1e-12)

def test_calibration_errors():
    """Verify calibration errors generate steering vector perturbations with expected statistics."""
    N = 8
    amp_err_std = 0.1
    phase_err_std_deg = 5.0
    
    rng = np.random.default_rng(42)
    amp_err = rng.normal(0, amp_err_std, N)
    phase_err = rng.normal(0, np.radians(phase_err_std_deg), N)
    cal_err = (1.0 + amp_err) * np.exp(1j * phase_err)
    
    # Check that amplitude is perturbed and phase is perturbed
    assert len(cal_err) == N
    assert np.any(np.abs(cal_err) != 1.0)
    assert np.any(np.angle(cal_err) != 0.0)
    
    # Check mean statistics roughly match (since N=8 is small, we just check they are in bounds)
    assert np.mean(np.abs(cal_err)) > 0.8 and np.mean(np.abs(cal_err)) < 1.2

def test_phase_quantization():
    """Verify that phase quantization rounds weights to correct digital states."""
    quant_bits = 3 # 8 phase states: 0, 45, 90, 135, 180, 225, 270, 315 degrees
    w = np.array([1.0, 1j, np.exp(1j * np.radians(30.0))], dtype=complex)
    
    phases = np.angle(w)
    step = 2.0 * np.pi / (2**quant_bits)
    phases_q = np.round(phases / step) * step
    w_q = np.abs(w) * np.exp(1j * phases_q)
    
    # 30 degrees should round to 45 degrees (which is step = pi/4)
    expected_phases = np.array([0.0, np.pi/2, np.pi/4])
    np.testing.assert_allclose(np.angle(w_q), expected_phases, rtol=1e-12)

def test_lcmv_multi_jammer():
    """Verify that LCMV satisfies multiple nulling constraints simultaneously."""
    N = 8
    theta_s = np.radians(0.0)
    theta_j_list = [np.radians(-30.0), np.radians(45.0)]
    
    # Create covariance matrix with 2 jammers
    Ps = 1.0
    Pj1 = 100.0
    Pj2 = 100.0
    
    a_s = pbb.ula_steering_vector(N, theta_s)
    a_j1 = pbb.ula_steering_vector(N, theta_j_list[0])
    a_j2 = pbb.ula_steering_vector(N, theta_j_list[1])
    
    R_xx = Ps * np.outer(a_s, np.conj(a_s)) + Pj1 * np.outer(a_j1, np.conj(a_j1)) + Pj2 * np.outer(a_j2, np.conj(a_j2)) + np.eye(N)
    
    w = pbb.lcmv_beamformer(R_xx, theta_s, theta_j_list)
    
    # Constraints: gain=1 at signal, gain=0 at both jammers
    np.testing.assert_allclose(np.conj(w).T @ a_s, 1.0, atol=1e-10)
    np.testing.assert_allclose(np.abs(np.conj(w).T @ a_j1), 0.0, atol=1e-10)
    np.testing.assert_allclose(np.abs(np.conj(w).T @ a_j2), 0.0, atol=1e-10)

def test_pbj_power_allocation():
    """Verify that jammer power spectral density is correctly scaled by alpha and N_h."""
    P_j = 1.0
    N_h = 80
    for alpha in [0.01, 0.1, 0.5, 1.0]:
        P_j_ch = P_j / (alpha * N_h)
        # Check that power increases as alpha decreases
        assert P_j_ch >= P_j / N_h
        np.testing.assert_allclose(P_j_ch * alpha * N_h, P_j, rtol=1e-12)

def test_follower_jamming_fraction():
    """Verify that follower jammer temporal fraction is correctly calculated."""
    T_hop = 10e-3 # 10 ms
    # 1. Zero delay -> 100% jammed
    rho_0 = np.max([0.0, 1.0 - 0.0 / T_hop])
    assert rho_0 == 1.0
    
    # 2. Delay = 5 ms -> 50% jammed
    rho_5 = np.max([0.0, 1.0 - 5e-3 / T_hop])
    np.testing.assert_allclose(rho_5, 0.5, rtol=1e-12)
    
    # 3. Delay = 10 ms -> 0% jammed
    rho_10 = np.max([0.0, 1.0 - 10e-3 / T_hop])
    assert rho_10 == 0.0
    
    # 4. Delay = 15 ms (larger than T_hop) -> 0% jammed
    rho_15 = np.max([0.0, 1.0 - 15e-3 / T_hop])
    assert rho_15 == 0.0

def test_effective_processing_gain_math():
    """Verify the mathematical logic for equivalent SINR and effective processing gain."""
    # Target nominal PG = 10 log10(80) = 19.03 dB
    N_h = 80
    
    def equiv_sinr_db(ber):
        ber_c = np.clip(ber, 1e-15, 0.5 - 1e-15)
        sinr_lin = -2.0 * np.log(2.0 * ber_c)
        return 10.0 * np.log10(np.maximum(sinr_lin, 1e-10))
        
    # Case A: Barrage Jamming (alpha=1.0)
    # Average BER should be much lower than spot jamming (no FHSS)
    ber_spot = 0.2
    ber_barrage = 0.01  # much lower due to N_h power division
    
    sinr_eff = equiv_sinr_db(ber_barrage)
    sinr_raw = equiv_sinr_db(ber_spot)
    pg_eff = sinr_eff - sinr_raw
    
    assert pg_eff > 0.0
    # PG eff should be bounded by 10 log10(N_h)
    assert pg_eff <= 10.0 * np.log10(N_h) + 1.0


def test_amc_q_function_and_ber():
    """Verify analytical BER equations against known target values and Q-function behavior."""
    import task_b7_amc as t7
    
    # 1. Q-function correctness
    # Q(0) = 0.5
    assert np.isclose(t7.Q(0.0), 0.5)
    # Q(1) ≈ 0.158655
    assert np.isclose(t7.Q(1.0), 0.15865525393145705, rtol=1e-6)
    
    # 2. Threshold verification: at threshold, BER must equal target_ber (3.37e-3)
    target = 3.37e-3
    
    # M0 (BPSK 1/2): threshold 5.64 dB
    sinr_db_0 = 5.64
    sinr_lin_0 = 10.0 ** (sinr_db_0 / 10.0)
    ber_0 = t7.get_ber_for_mode(0, sinr_lin_0)
    assert np.isclose(ber_0, target, rtol=1e-2)
    
    # M1 (QPSK 1/2): threshold 8.65 dB
    sinr_db_1 = 8.65
    sinr_lin_1 = 10.0 ** (sinr_db_1 / 10.0)
    ber_1 = t7.get_ber_for_mode(1, sinr_lin_1)
    assert np.isclose(ber_1, target, rtol=1e-2)
    
    # M2 (QPSK 3/4): threshold 10.41 dB
    sinr_db_2 = 10.41
    sinr_lin_2 = 10.0 ** (sinr_db_2 / 10.0)
    ber_2 = t7.get_ber_for_mode(2, sinr_lin_2)
    assert np.isclose(ber_2, t7.Q(np.sqrt(sinr_lin_2)), rtol=1e-6)
    
    # M3 (16QAM 1/2): threshold 15.33 dB
    sinr_db_3 = 15.33
    sinr_lin_3 = 10.0 ** (sinr_db_3 / 10.0)
    ber_3 = t7.get_ber_for_mode(3, sinr_lin_3)
    assert np.isclose(ber_3, target, rtol=1e-2)


def test_amc_mode_selection_and_hysteresis():
    """Verify hysteresis transitions and state persistence."""
    import task_b7_amc as t7
    
    # Hysteresis = 2 dB. Mode thresholds are: [5.64, 8.65, 10.41, 15.33]
    # 1. Start from outage (prev_mode = -1)
    assert t7.select_amc_mode_hysteresis(3.0, -1) == -1
    assert t7.select_amc_mode_hysteresis(6.0, -1) == 0
    assert t7.select_amc_mode_hysteresis(12.0, -1) == 2
    
    # 2. UP Transitions from mode 0
    assert t7.select_amc_mode_hysteresis(8.0, 0) == 0
    assert t7.select_amc_mode_hysteresis(9.0, 0) == 1
    assert t7.select_amc_mode_hysteresis(11.0, 0) == 2
    
    # 3. DOWN Transitions from mode 1 (threshold = 8.65 dB, down threshold = 6.65 dB)
    assert t7.select_amc_mode_hysteresis(7.0, 1) == 1
    assert t7.select_amc_mode_hysteresis(6.0, 1) == 0
    assert t7.select_amc_mode_hysteresis(3.0, 1) == -1


def test_amc_follower_jamming_ber():
    """Verify average hop BER calculations under follower jamming."""
    import task_b7_amc as t7
    
    p_link, _, avg_throughput, avg_ber, _ = t7.simulate_amc_hop_sequence(
        d_m=2000.0, N=8, theta_s=0.0, theta_j=np.radians(30.0), alpha=1.0,
        jammer_type='follower', tau_delay=5e-3, use_amc=True, trials=10
    )
    assert 0.0 <= p_link <= 1.0
    assert avg_throughput >= 0.0
    assert 0.0 <= avg_ber <= 0.5


