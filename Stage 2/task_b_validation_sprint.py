"""Phase B Validation Sprint: Auditing Spatial Processing Assumptions.

This script implements the physical audits for the GCS phased array receiver
under realistic limitations: multiple jammers, multipath spread, calibration errors,
phase quantization, MVDR vs LCMV formulations, and correlated fading.
"""

import os
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

# Import beamforming module
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
import phase_b_beamforming as pbb

# ==============================================================================
# 1. SYSTEM BASELINE PARAMETERS
# ==============================================================================
P_tx_dBm = 37.0         # UAV TX Power: 37 dBm (5 W)
G_tx_dBi = 3.0          # UAV TX Antenna Gain: 3 dBi
G_rx_element_dBi = 10.0 # Single element gain: 10 dBi
B_Hz = 500e3            # Receiver/Hop Bandwidth: 500 kHz
NF_dB = 6.0             # Receiver Noise Figure: 6 dB
f_c = 2.4e9             # Carrier Frequency: 2.4 GHz
gamma_th_dB = 10.0      # Demodulation SNR threshold: 10.0 dB

P_N_dBm = -174.0 + 10 * np.log10(B_Hz) + NF_dB  # Noise floor: -111.0 dBm
P_N_lin = 10 ** (P_N_dBm / 10.0)
gamma_th_lin = 10 ** (gamma_th_dB / 10.0)

# Antenna Heights & Ground properties
h_GS = 2.0              # Ground GCS height: 2 m
h_UAV = 100.0           # UAV altitude: 100 m
epsilon_r = 15.0        # Relative permittivity

# Jammer Standoff coordinates
jammer_standoff_m = 5000.0
jammer_eirp_dBm = 73.0
G_rx_jam = 0.0
B_ss_Hz = 40e6
PG_dB = 10 * np.log10(B_ss_Hz / B_Hz)  # 19.03 dB

# ==============================================================================
# 2. PATH LOSS MODELS
# ==============================================================================
def fspl_db(d_m):
    return 40.05 + 20.0 * np.log10(np.maximum(d_m, 0.1))

def two_ray_path_loss(d_horizontal, h_gs=2.0, h_uav=100.0, f=2.4e9, eps_r=15.0, polarization='vertical'):
    c = 3.0e8
    wl = c / f
    k = 2.0 * np.pi / wl
    
    d_dir = np.sqrt(d_horizontal**2 + (h_uav - h_gs)**2)
    d_ref = np.sqrt(d_horizontal**2 + (h_uav + h_gs)**2)
    
    sin_theta = (h_uav + h_gs) / d_ref
    cos_theta = d_horizontal / d_ref
    
    sqrt_term = np.sqrt(eps_r - cos_theta**2 + 0j)
    
    if polarization == 'horizontal':
        Gamma = (sin_theta - sqrt_term) / (sin_theta + sqrt_term)
    else:  # vertical
        Gamma = (eps_r * sin_theta - sqrt_term) / (eps_r * sin_theta + sqrt_term)
        
    delta_phi = k * (d_ref - d_dir)
    interference = np.abs(1.0 + Gamma * np.exp(-1j * delta_phi))**2
    fs_amp = wl / (4.0 * np.pi * d_dir)
    pl_db = -10.0 * np.log10(fs_amp**2 * interference + 1e-20)
    return pl_db

def get_received_signal_power(d_m, path_loss_model='two-ray'):
    if path_loss_model == 'two-ray':
        L_s = two_ray_path_loss(d_m, h_gs=h_GS, h_uav=h_UAV, f=f_c, eps_r=epsilon_r, polarization='vertical')
    else:
        L_s = fspl_db(d_m)
    S_dBm = P_tx_dBm + G_tx_dBi + G_rx_element_dBi - L_s
    return 10 ** (S_dBm / 10.0)

def get_received_jammer_power(path_loss_model='fspl'):
    if path_loss_model == 'two-ray':
        L_j = two_ray_path_loss(jammer_standoff_m, h_gs=h_GS, h_uav=h_GS, f=f_c, eps_r=epsilon_r, polarization='vertical')
    else:
        L_j = fspl_db(jammer_standoff_m)
    J_dBm = jammer_eirp_dBm + G_rx_jam - L_j - PG_dB
    return 10 ** (J_dBm / 10.0)

# ==============================================================================
# 3. COMPREHENSIVE AUDIT METRICS EXTRACTOR
# ==============================================================================
def get_audit_metrics(d_m, N, theta_s, theta_j_list, sigma_theta_deg, L, path_loss_model='two-ray',
                      amp_err_std=0.0, phase_err_std_deg=0.0, quant_bits=None, angular_spread_deg=0.0,
                      use_mvdr=False, max_null_depth_db=None, sig_mismatch_deg=0.0):
    
    Ps = get_received_signal_power(d_m, path_loss_model)
    Pj_list = [get_received_jammer_power('fspl') for _ in theta_j_list]
    
    # Generate calibration error vectors (amplitude and phase errors)
    rng = np.random.default_rng(42)
    amp_err = rng.normal(0, amp_err_std, N)
    phase_err = rng.normal(0, np.radians(phase_err_std_deg), N)
    cal_err = (1.0 + amp_err) * np.exp(1j * phase_err)
    
    # Steering vectors for signal (with optional signal look angle mismatch)
    theta_s_est = theta_s + np.radians(sig_mismatch_deg)
    a_s_nom = pbb.ula_steering_vector(N, theta_s_est)
    a_s_true = pbb.ula_steering_vector(N, theta_s) * cal_err
    
    # Build Jammer Covariance Component
    R_j_total_nom = np.zeros((N, N), dtype=complex)
    R_j_total_true = np.zeros((N, N), dtype=complex)
    
    a_j_nom_list = []
    a_j_est_list = []
    
    for theta_j, Pj in zip(theta_j_list, Pj_list):
        INR_lin = Pj / P_N_lin
        
        # Nominal jammer direction
        a_j_nom = pbb.ula_steering_vector(N, theta_j)
        a_j_nom_list.append(a_j_nom)
        
        # Estimated jammer direction (with pointing error)
        e_theta = rng.normal(0, np.radians(sigma_theta_deg))
        theta_j_est = theta_j + e_theta
        a_j_est = pbb.ula_steering_vector(N, theta_j_est)
        a_j_est_list.append(a_j_est)
        
        # Multipath angular spread (spread centered at theta_j)
        if angular_spread_deg > 0.0:
            K_paths = 50
            angles = rng.normal(theta_j, np.radians(angular_spread_deg), K_paths)
            R_j_nom = np.zeros((N, N), dtype=complex)
            R_j_true = np.zeros((N, N), dtype=complex)
            for ang in angles:
                vec = pbb.ula_steering_vector(N, ang)
                R_j_nom += (INR_lin / K_paths) * np.outer(vec, np.conj(vec))
                R_j_true += (INR_lin / K_paths) * np.outer(vec * cal_err, np.conj(vec * cal_err))
            R_j_total_nom += R_j_nom
            R_j_total_true += R_j_true
        else:
            R_j_total_nom += INR_lin * np.outer(a_j_nom, np.conj(a_j_nom))
            R_j_total_true += INR_lin * np.outer(a_j_nom * cal_err, np.conj(a_j_nom * cal_err))

    # Signal component
    SNR_lin = Ps / P_N_lin
    # If desired signal look angle mismatch is present, the true signal is present
    # in the covariance matrix, which triggers self-nulling. Otherwise, we isolate
    # other impairments by using the nominal look vector in the covariance matrix.
    if sig_mismatch_deg != 0.0:
        R_s = SNR_lin * np.outer(a_s_true, np.conj(a_s_true))
    else:
        R_s = SNR_lin * np.outer(a_s_nom, np.conj(a_s_nom))
        
    R_xx_dl = R_s + R_j_total_nom + np.eye(N) + 1e-5 * np.eye(N)
    
    # Compute beamforming weight vector w
    if use_mvdr:
        w = pbb.mvdr_beamformer(R_xx_dl, theta_s_est)
    else:
        # LCMV nulling
        if N <= len(theta_j_list):
            # Fallback to MVDR when jammers exhaust array degrees of freedom
            w = pbb.mvdr_beamformer(R_xx_dl, theta_s_est)
        else:
            # LCMV with estimated jammer angles
            w = pbb.lcmv_beamformer(R_xx_dl, theta_s_est, [theta_j + rng.normal(0, np.radians(sigma_theta_deg)) for theta_j in theta_j_list])
            
    # Apply weight quantization (Phase quantization)
    if quant_bits is not None:
        phases = np.angle(w)
        step = 2.0 * np.pi / (2**quant_bits)
        phases_q = np.round(phases / step) * step
        w = np.abs(w) * np.exp(1j * phases_q)
        
    # Physical output calculations
    gain_s = np.conj(w).T @ a_s_true
    sig_power_out = SNR_lin * (np.abs(gain_s) ** 2)
    
    # True jammer + noise covariance matrix
    R_jn_true = R_j_total_true + np.eye(N)
    thermal_noise_out = np.sum(np.abs(w)**2)
    jammer_leakage_out = np.real(np.conj(w).T @ R_j_total_true @ w)
    
    # Apply synthetic null depth limit if specified
    if max_null_depth_db is not None:
        min_leakage_lin = INR_lin * (10.0 ** (-max_null_depth_db / 10.0)) * thermal_noise_out
        jammer_leakage_out = max(jammer_leakage_out, min_leakage_lin)
        
    jn_power_out = jammer_leakage_out + thermal_noise_out
    sinr_out_db = 10.0 * np.log10(sig_power_out / max(jn_power_out, 1e-15))
    
    # Array Gain Loss
    array_gain_db = 10.0 * np.log10(np.abs(gain_s) ** 2 / np.sum(np.abs(w)**2))
    gain_loss_db = 10.0 * np.log10(N) - array_gain_db
    
    # Achieved Null Depth for Jammer 1 (relative to signal gain)
    gain_j = np.conj(w).T @ (a_j_nom_list[0] * cal_err)
    null_depth_db = 10.0 * np.log10(np.abs(gain_j)**2 / max(np.abs(gain_s)**2, 1e-15) + 1e-20)
    
    # Residual jammer power (in dBm)
    Pj_res_dbm = 10.0 * np.log10(Pj_list[0]) + 10.0 * np.log10(np.abs(gain_j)**2 + 1e-20)
    
    if max_null_depth_db is not None:
        # If synthetic null cap is applied, override the null depth and residual power to reflect the cap
        null_depth_db = max(null_depth_db, -max_null_depth_db)
        Pj_res_dbm = max(Pj_res_dbm, 10.0 * np.log10(Pj_list[0]) - max_null_depth_db)
        
    # Identify Dominant Failure Mechanism
    if np.abs(gain_s)**2 < 0.01:
        mechanism = "Signal Self-Nulling"
    elif jammer_leakage_out > 10.0 * thermal_noise_out:
        mechanism = "Jammer Leakage"
    elif thermal_noise_out > 2.0:  # Noise enhancement
        mechanism = "Noise Enhancement"
    else:
        mechanism = "Thermal Noise"
        
    return sinr_out_db, Pj_res_dbm, null_depth_db, gain_loss_db, mechanism

def get_audit_row_metrics(N, theta_s, theta_j_list, sigma_theta_deg, L, path_loss_model='two-ray',
                          amp_err_std=0.0, phase_err_std_deg=0.0, quant_bits=None, angular_spread_deg=0.0,
                          use_mvdr=False, max_null_depth_db=None, sig_mismatch_deg=0.0):
    # Solve for defeat range
    r_def = solve_defeat_range_val(N, theta_s, theta_j_list, sigma_theta_deg, L, path_loss_model,
                                   amp_err_std, phase_err_std_deg, quant_bits, angular_spread_deg,
                                   use_mvdr, max_null_depth_db, sig_mismatch_deg)
    
    # Get metrics at solved defeat range
    _, pj_res, null_d, gain_l, mech = get_audit_metrics(r_def, N, theta_s, theta_j_list, sigma_theta_deg, L, path_loss_model,
                                                        amp_err_std, phase_err_std_deg, quant_bits, angular_spread_deg,
                                                        use_mvdr, max_null_depth_db, sig_mismatch_deg)
    
    # Nominal range map for vertical polarization Two-Ray unperturbed cases
    r_nom_map = {1: 1364.4, 4: 118978.4, 8: 141555.2, 16: 168402.2, 32: 200331.4}
    r_nom = r_nom_map.get(N, 141555.2)
    
    # Evaluate at nominal boundary
    sinr_nom, _, _, _, _ = get_audit_metrics(r_nom, N, theta_s, theta_j_list, sigma_theta_deg, L, path_loss_model,
                                             amp_err_std, phase_err_std_deg, quant_bits, angular_spread_deg,
                                             use_mvdr, max_null_depth_db, sig_mismatch_deg)
    nom_margin = sinr_nom - 10.0
    
    return r_def, pj_res, null_d, gain_l, nom_margin, mech

def solve_defeat_range_val(N, theta_s, theta_j_list, sigma_theta_deg, L, path_loss_model='two-ray',
                           amp_err_std=0.0, phase_err_std_deg=0.0, quant_bits=None, angular_spread_deg=0.0,
                           use_mvdr=False, max_null_depth_db=None, sig_mismatch_deg=0.0):
    lo, hi = 100.0, 3000000.0
    for _ in range(30):
        mid = (lo + hi) / 2.0
        sinr, _, _, _, _ = get_audit_metrics(mid, N, theta_s, theta_j_list, sigma_theta_deg, L, path_loss_model,
                                             amp_err_std, phase_err_std_deg, quant_bits, angular_spread_deg,
                                             use_mvdr, max_null_depth_db, sig_mismatch_deg)
        if sinr >= 10.0:
            lo = mid
        else:
            hi = mid
    return lo

# ==============================================================================
# 4. CORRELATED Rician FADING PATH SIMULATOR
# ==============================================================================
def run_correlated_fading_mc(N, theta_s, theta_j, L, rho, trials=1000):
    """Simulate path survival probability along ingress trajectory with correlated Rician fading."""
    # Horizontal trajectory steps: 5000 m down to 100 m with 25 m step size
    ranges = np.arange(5000.0, 99.0, -25.0)
    
    # Handover boundary under Two-Ray
    r_defeat = solve_defeat_range_val(N, theta_s, [theta_j], 0.0, L, 'two-ray')
    
    successes = 0
    rng = np.random.default_rng(2026)
    
    for _ in range(trials):
        # Initialize diffuse fading states
        u_s = (rng.normal(0, np.sqrt(0.5)) + 1j * rng.normal(0, np.sqrt(0.5)))
        u_j = (rng.normal(0, np.sqrt(0.5)) + 1j * rng.normal(0, np.sqrt(0.5)))
        
        path_ok = True
        
        # We only evaluate communication within the active comm window: from r_defeat down to 100m
        active_ranges = ranges[ranges <= r_defeat]
        
        if len(active_ranges) == 0:
            path_ok = False
            
        for r in active_ranges:
            # Elevation dependent Rician K-factor for signal
            theta_elev = np.arctan(h_UAV / r)
            K_s_db = 10.0 + 10.0 * np.sin(theta_elev)
            K_s = 10 ** (K_s_db / 10.0)
            
            # Ground-to-ground link for jammer has Rayleigh fading (K_j = 0)
            K_j = 0.0
            
            # Autoregressive diffuse fading step
            z_s = (rng.normal(0, np.sqrt(0.5)) + 1j * rng.normal(0, np.sqrt(0.5)))
            z_j = (rng.normal(0, np.sqrt(0.5)) + 1j * rng.normal(0, np.sqrt(0.5)))
            
            u_s = rho * u_s + np.sqrt(1.0 - rho**2) * z_s
            u_j = rho * u_j + np.sqrt(1.0 - rho**2) * z_j
            
            h_s = np.sqrt(K_s / (K_s + 1.0)) * np.exp(1j * rng.uniform(-np.pi, np.pi)) + np.sqrt(1.0 / (K_s + 1.0)) * u_s
            h_j = np.sqrt(K_j / (K_j + 1.0)) * np.exp(1j * rng.uniform(-np.pi, np.pi)) + np.sqrt(1.0 / (K_j + 1.0)) * u_j
            
            # Received powers at step
            Ps = get_received_signal_power(r, 'two-ray')
            Pj = get_received_jammer_power('fspl')
            
            # Apply beamforming output formula with fading coefficients
            a_s = pbb.ula_steering_vector(N, theta_s)
            a_j = pbb.ula_steering_vector(N, theta_j)
            
            SNR_lin = Ps / P_N_lin
            INR_lin = Pj / P_N_lin
            
            R_xx_dl = SNR_lin * np.outer(a_s, np.conj(a_s)) + INR_lin * np.outer(a_j, np.conj(a_j)) + np.eye(N) + 1e-5 * np.eye(N)
            w = pbb.lcmv_beamformer(R_xx_dl, theta_s, [theta_j]) if N > 1 else np.array([1.0])
            
            sig_power_out = SNR_lin * (np.abs(h_s)**2) * (np.abs(np.conj(w).T @ a_s) ** 2)
            noise_out = np.sum(np.abs(w)**2)
            jammer_out = INR_lin * (np.abs(h_j)**2) * (np.abs(np.conj(w).T @ a_j) ** 2)
            
            sinr_db = 10.0 * np.log10(sig_power_out / max(jammer_out + noise_out, 1e-15))
            
            if sinr_db < 10.0:
                path_ok = False
                break
                
        if path_ok and len(active_ranges) > 0:
            successes += 1
            
    return successes / trials

# ==============================================================================
# 5. AUDIT EXECUTION
# ==============================================================================
def execute_validation_sprint():
    print("=" * 120)
    print("                 PHASE B VALIDATION SPRINT: AUDITING ASSUMPTIONS (DETAILED REPORTS)")
    print("=" * 120)
    
    theta_s = np.radians(0.0)
    theta_j = np.radians(30.0)
    L = 500
    
    def print_unified_header(title):
        print("\n" + "="*120)
        print(f" {title}")
        print("="*120)
        print(f"{'Configuration':<25} | {'R_defeat (m)':^12} | {'Resid Jam (dBm)':^16} | {'Null Depth (dB)':^16} | {'Gain Loss (dB)':^15} | {'Nom Margin (dB)':^15} | {'Dominant Failure Mechanism':<25}")
        print("-" * 120)

    def print_unified_row(config_str, r_def, pj_res, null_d, gain_l, nom_margin, mech):
        print(f"{config_str:<25} | {r_def:^12.1f} | {pj_res:^16.2f} | {null_d:^16.2f} | {gain_l:^15.2f} | {nom_margin:^15.2f} | {mech:<25}")
    
    # --------------------------------------------------------------------------
    # AUDIT #1: MULTIPLE JAMMERS (Tier 1)
    # --------------------------------------------------------------------------
    print_unified_header("[AUDIT #1] Multiple Jammers Sweep (N elements vs M Jammers)")
    
    jammer_scenarios = [
        [np.radians(30.0)],
        [np.radians(30.0), np.radians(-45.0)],
        [np.radians(30.0), np.radians(-45.0), np.radians(60.0)]
    ]
    
    for N in [4, 8, 16]:
        for idx, j_list in enumerate(jammer_scenarios):
            angles_str = ", ".join([f"{np.degrees(a):.0f}°" for a in j_list])
            config_str = f"N={N}, M={len(j_list)} ({angles_str})"
            r_def, pj_res, null_d, gain_l, nom_margin, mech = get_audit_row_metrics(N, theta_s, j_list, 0.0, L)
            print_unified_row(config_str, r_def, pj_res, null_d, gain_l, nom_margin, mech)
    print("-" * 120)
    
    # --------------------------------------------------------------------------
    # AUDIT #2: ANGULAR SPREAD / COHERENCE LOSS (Tier 1)
    # --------------------------------------------------------------------------
    print_unified_header("[AUDIT #2] Multipath Angular Spread Audit (N=8 Elements, 1 Jammer at 30°)")
    
    for spread in [0.0, 0.5, 1.0, 2.0, 5.0]:
        config_str = f"Spread = {spread:.1f}°"
        r_def, pj_res, null_d, gain_l, nom_margin, mech = get_audit_row_metrics(8, theta_s, [theta_j], 0.0, L, angular_spread_deg=spread)
        print_unified_row(config_str, r_def, pj_res, null_d, gain_l, nom_margin, mech)
    print("-" * 120)
    
    # --------------------------------------------------------------------------
    # AUDIT #3: CALIBRATION MISMATCH (Tier 1)
    # --------------------------------------------------------------------------
    print_unified_header("[AUDIT #3] Array Calibration Mismatch (N=8 Elements, 1 Jammer at 30°)")
    
    cal_configs = [
        (0.0, 0.0),
        (0.05, 2.0),
        (0.1, 5.0),
        (0.2, 10.0),
        (0.2, 15.0)
    ]
    
    for amp, ph in cal_configs:
        config_str = f"Amp={amp:.2f}, Ph={ph:.1f}°"
        r_def, pj_res, null_d, gain_l, nom_margin, mech = get_audit_row_metrics(8, theta_s, [theta_j], 0.0, L, amp_err_std=amp, phase_err_std_deg=ph)
        print_unified_row(config_str, r_def, pj_res, null_d, gain_l, nom_margin, mech)
    print("-" * 120)
    
    # --------------------------------------------------------------------------
    # AUDIT #4: SYNTHETIC NULL DEPTH SWEET (Tier 2)
    # --------------------------------------------------------------------------
    print_unified_header("[AUDIT #4] Synthetic Null Depth Cap Audit (N=8 Elements)")
    
    for depth in [20, 30, 40, 50, None]:
        depth_str = f"{depth} dB" if depth is not None else "Infinite"
        config_str = f"Cap = {depth_str}"
        r_def, pj_res, null_d, gain_l, nom_margin, mech = get_audit_row_metrics(8, theta_s, [theta_j], 0.0, L, max_null_depth_db=depth)
        print_unified_row(config_str, r_def, pj_res, null_d, gain_l, nom_margin, mech)
    print("-" * 120)
    
    # --------------------------------------------------------------------------
    # AUDIT #5: MVDR VS LCMV FORMULATION COMPARISON (Tier 2)
    # --------------------------------------------------------------------------
    print("\n" + "="*120)
    print(" [AUDIT #5] MVDR vs LCMV Formulation & Mismatch Audit (N=8)")
    print("="*120)
    print(f"{'Pointing Mismatch':<25} | {'MVDR R_def (m)':^14} | {'LCMV R_def (m)':^14} | {'MVDR Mech':^18} | {'LCMV Mech':^18} | {'MVDR Marg (dB)':^14} | {'LCMV Marg (dB)':^14}")
    print("-" * 120)
    
    mismatch_configs = [
        ("No mismatch", 0.0, 0.0),
        ("Jammer Mismatch (2°)", 2.0, 0.0),
        ("Signal Mismatch (1°)", 0.0, 1.0)
    ]
    
    for label, j_err, s_err in mismatch_configs:
        r_def_mvdr = solve_defeat_range_val(8, theta_s, [theta_j], j_err, L, use_mvdr=True, sig_mismatch_deg=s_err)
        _, pj_res_m, null_d_m, gain_l_m, mech_mvdr = get_audit_metrics(r_def_mvdr, 8, theta_s, [theta_j], j_err, L, use_mvdr=True, sig_mismatch_deg=s_err)
        sinr_nom_mvdr, _, _, _, _ = get_audit_metrics(141555.2, 8, theta_s, [theta_j], j_err, L, use_mvdr=True, sig_mismatch_deg=s_err)
        margin_mvdr = sinr_nom_mvdr - 10.0
        
        r_def_lcmv = solve_defeat_range_val(8, theta_s, [theta_j], j_err, L, use_mvdr=False, sig_mismatch_deg=s_err)
        _, pj_res_l, null_d_l, gain_l_l, mech_lcmv = get_audit_metrics(r_def_lcmv, 8, theta_s, [theta_j], j_err, L, use_mvdr=False, sig_mismatch_deg=s_err)
        sinr_nom_lcmv, _, _, _, _ = get_audit_metrics(141555.2, 8, theta_s, [theta_j], j_err, L, use_mvdr=False, sig_mismatch_deg=s_err)
        margin_lcmv = sinr_nom_lcmv - 10.0
        
        print(f"{label:<25} | {r_def_mvdr:^14.1f} | {r_def_lcmv:^14.1f} | {mech_mvdr:^18} | {mech_lcmv:^18} | {margin_mvdr:^14.2f} | {margin_lcmv:^14.2f}")
    print("-" * 120)
    
    # --------------------------------------------------------------------------
    # AUDIT #6: PHASE QUANTIZATION (Tier 3)
    # --------------------------------------------------------------------------
    print_unified_header("[AUDIT #6] Digital Phase Quantization Audit (N=8 Elements)")
    
    for bits in [3, 4, 5, 6, None]:
        bits_str = f"{bits} bits" if bits is not None else "Infinite"
        config_str = f"Quant = {bits_str}"
        r_def, pj_res, null_d, gain_l, nom_margin, mech = get_audit_row_metrics(8, theta_s, [theta_j], 0.0, L, quant_bits=bits)
        print_unified_row(config_str, r_def, pj_res, null_d, gain_l, nom_margin, mech)
    print("-" * 120)
    
    # --------------------------------------------------------------------------
    # AUDIT #7: CORRELATED FADING (Additional Audit)
    # --------------------------------------------------------------------------
    print_unified_header("[AUDIT #7] Correlated Rician Fading SINR & Outage Statistics (trials=1000)")
    
    fading_cases = [
        (1, 1000.0, "N=1 at 1.0 km"),
        (4, 100000.0, "N=4 at 100.0 km")
    ]
    
    for N_f, d_m_f, label_f in fading_cases:
        for rho_f in [0.0, 0.5, 0.9]:
            # Generate fading statistics
            rng_fad = np.random.default_rng(2026)
            sinrs_fad = []
            
            Ps_fad = get_received_signal_power(d_m_f, 'two-ray')
            Pj_fad = get_received_jammer_power('fspl')
            SNR_lin_fad = Ps_fad / P_N_lin
            INR_lin_fad = Pj_fad / P_N_lin
            
            a_s_fad = pbb.ula_steering_vector(N_f, theta_s)
            a_j_fad = pbb.ula_steering_vector(N_f, theta_j)
            
            R_xx_dl_fad = SNR_lin_fad * np.outer(a_s_fad, np.conj(a_s_fad)) + INR_lin_fad * np.outer(a_j_fad, np.conj(a_j_fad)) + np.eye(N_f) + 1e-5 * np.eye(N_f)
            w_fad = pbb.lcmv_beamformer(R_xx_dl_fad, theta_s, [theta_j]) if N_f > 1 else np.array([1.0])
            
            u_s = (rng_fad.normal(0, np.sqrt(0.5)) + 1j * rng_fad.normal(0, np.sqrt(0.5)))
            u_j = (rng_fad.normal(0, np.sqrt(0.5)) + 1j * rng_fad.normal(0, np.sqrt(0.5)))
            
            theta_elev = np.arctan(h_UAV / d_m_f)
            K_s_db = 10.0 + 10.0 * np.sin(theta_elev)
            K_s = 10 ** (K_s_db / 10.0)
            K_j = 0.0
            
            for _ in range(1000):
                z_s = (rng_fad.normal(0, np.sqrt(0.5)) + 1j * rng_fad.normal(0, np.sqrt(0.5)))
                z_j = (rng_fad.normal(0, np.sqrt(0.5)) + 1j * rng_fad.normal(0, np.sqrt(0.5)))
                
                u_s = rho_f * u_s + np.sqrt(1.0 - rho_f**2) * z_s
                u_j = rho_f * u_j + np.sqrt(1.0 - rho_f**2) * z_j
                
                h_s = np.sqrt(K_s / (K_s + 1.0)) * np.exp(1j * rng_fad.uniform(-np.pi, np.pi)) + np.sqrt(1.0 / (K_s + 1.0)) * u_s
                h_j = np.sqrt(K_j / (K_j + 1.0)) * np.exp(1j * rng_fad.uniform(-np.pi, np.pi)) + np.sqrt(1.0 / (K_j + 1.0)) * u_j
                
                sig_power_out = SNR_lin_fad * (np.abs(h_s)**2) * (np.abs(np.conj(w_fad).T @ a_s_fad) ** 2)
                noise_out = np.sum(np.abs(w_fad)**2)
                jammer_out = INR_lin_fad * (np.abs(h_j)**2) * (np.abs(np.conj(w_fad).T @ a_j_fad) ** 2)
                
                sinr_db = 10.0 * np.log10(sig_power_out / max(jammer_out + noise_out, 1e-15))
                sinrs_fad.append(sinr_db)
                
            sinrs_fad = np.array(sinrs_fad)
            mean_sinr = np.mean(sinrs_fad)
            p5_sinr = np.percentile(sinrs_fad, 5)
            outage_prob = np.mean(sinrs_fad < 10.0)
            
            print_unified_row(f"{label_f} (rho={rho_f:.2f})", d_m_f, 10.0 * np.log10(Pj_fad), p5_sinr, mean_sinr - 10.0, outage_prob, "Thermal/Fading" if N_f > 1 else "Jammer/Fading")
    print("-" * 120)
    
    # --------------------------------------------------------------------------
    # AUDIT #8: JOINT CALIBRATION MISMATCH & MULTIPLE JAMMERS (Final Audit)
    # --------------------------------------------------------------------------
    print_unified_header("[AUDIT #8] Joint Calibration Mismatch & Multiple Jammers Audit")
    
    jammer_scenarios_combined = {
        1: [np.radians(30.0)],
        2: [np.radians(30.0), np.radians(-45.0)],
        3: [np.radians(30.0), np.radians(-45.0), np.radians(60.0)]
    }
    
    for N_c in [4, 8, 16]:
        for M_c in [1, 2, 3]:
            j_list_c = jammer_scenarios_combined[M_c]
            for pe in [0.0, 2.0, 5.0, 10.0]:
                ae = 0.01 * pe
                config_str = f"N={N_c}, M={M_c}, PhErr={pe:.1f}°"
                r_def, pj_res, null_d, gain_l, nom_margin, mech = get_audit_row_metrics(N_c, theta_s, j_list_c, 0.0, L, amp_err_std=ae, phase_err_std_deg=pe)
                print_unified_row(config_str, r_def, pj_res, null_d, gain_l, nom_margin, mech)
            print("-" * 120)
    
    # --------------------------------------------------------------------------
    # 6. Generate 6-Panel Visualization Sweeps Plot
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    
    # Panel 1: R_defeat vs N & Number of Jammers
    ax1 = axes[0, 0]
    N_list = [4, 8, 16, 32]
    for idx, j_list in enumerate(jammer_scenarios):
        r_defs = [solve_defeat_range_val(N, theta_s, j_list, 0.0, L) / 1000.0 for N in N_list]
        ax1.plot(N_list, r_defs, marker="o", label=f"{idx+1} Jammer(s)")
    ax1.set_title("1. Defeat Boundary vs. Array Size & Jammer Count", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Number of Array Elements $N$", fontsize=10)
    ax1.set_ylabel("Defeat Range $R_{defeat}$ [km]", fontsize=10)
    ax1.set_xscale("log")
    ax1.set_xticks(N_list)
    ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left", frameon=True)
    
    # Panel 2: R_defeat vs Multipath Angular Spread
    ax2 = axes[0, 1]
    spreads = np.linspace(0.0, 5.0, 15)
    for N in [4, 8, 16]:
        r_defs = [solve_defeat_range_val(N, theta_s, [theta_j], 0.0, L, angular_spread_deg=s) / 1000.0 for s in spreads]
        ax2.plot(spreads, r_defs, label=f"$N = {N}$")
    ax2.set_title("2. Angular Spread Sensitivity (Coherence Loss)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Multipath Angular Spread $\\sigma_{spread}$ [degrees]", fontsize=10)
    ax2.set_ylabel("Defeat Range $R_{defeat}$ [km]", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", frameon=True)
    
    # Panel 3: R_defeat vs Calibration Phase Error
    ax3 = axes[1, 0]
    phase_errs = np.linspace(0.0, 15.0, 15)
    for N in [4, 8, 16]:
        # Assume amplitude error scales linearly with phase error: amp_err = 0.01 * phase_err
        r_defs = [solve_defeat_range_val(N, theta_s, [theta_j], 0.0, L, amp_err_std=0.01*pe, phase_err_std_deg=pe) / 1000.0 for pe in phase_errs]
        ax3.plot(phase_errs, r_defs, label=f"$N = {N}$")
    ax3.set_title("3. Calibration Error Sweep (Phase & Amp Mismatch)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Phase Mismatch $\\sigma_\\phi$ [degrees]", fontsize=10)
    ax3.set_ylabel("Defeat Range $R_{defeat}$ [km]", fontsize=10)
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(loc="upper right", frameon=True)
    
    # Panel 4: R_defeat vs Synthetic Null Depth Cap
    ax4 = axes[1, 1]
    depths = np.linspace(20, 60, 15)
    for N in [4, 8, 16]:
        r_defs = [solve_defeat_range_val(N, theta_s, [theta_j], 0.0, L, max_null_depth_db=d) / 1000.0 for d in depths]
        ax4.plot(depths, r_defs, label=f"$N = {N}$")
    ax4.set_title("4. Synthetic Null Depth Cap Sweep", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Capped Null Depth Limit [dB]", fontsize=10)
    ax4.set_ylabel("Defeat Range $R_{defeat}$ [km]", fontsize=10)
    ax4.grid(True, linestyle=":", alpha=0.6)
    ax4.legend(loc="lower right", frameon=True)
    
    # Panel 5: MVDR vs LCMV pointing error sweep
    ax5 = axes[2, 0]
    pointing_errors = np.linspace(0.0, 10.0, 15)
    r_mvdr = [solve_defeat_range_val(8, theta_s, [theta_j], pe, L, use_mvdr=True) / 1000.0 for pe in pointing_errors]
    r_lcmv = [solve_defeat_range_val(8, theta_s, [theta_j], pe, L, use_mvdr=False) / 1000.0 for pe in pointing_errors]
    ax5.plot(pointing_errors, r_mvdr, color="red", label="MVDR")
    ax5.plot(pointing_errors, r_lcmv, color="blue", linestyle="--", label="LCMV")
    ax5.set_title("5. Jammer Pointing Mismatch: MVDR vs. LCMV ($N=8$)", fontsize=11, fontweight="bold")
    ax5.set_xlabel("DOA Pointing Error $\\sigma_\\theta$ [degrees]", fontsize=10)
    ax5.set_ylabel("Defeat Range $R_{defeat}$ [km]", fontsize=10)
    ax5.grid(True, linestyle=":", alpha=0.6)
    ax5.legend(loc="upper right", frameon=True)
    
    # Panel 6: Correlated Fading path survival
    ax6 = axes[2, 1]
    rhos = np.linspace(0.0, 0.95, 10)
    prob_N1 = [run_correlated_fading_mc(1, theta_s, theta_j, L, r, trials=300) for r in rhos]
    prob_N4 = [run_correlated_fading_mc(4, theta_s, theta_j, L, r, trials=300) for r in rhos]
    ax6.plot(rhos, prob_N1, marker="o", color="black", label="$N=1$ (Baseline)")
    ax6.plot(rhos, prob_N4, marker="s", color="green", label="$N=4$ Beamforming")
    ax6.set_title("6. Path Survival Probability vs. Fading Correlation $\\rho$", fontsize=11, fontweight="bold")
    ax6.set_xlabel("Fading Correlation Coefficient $\\rho$", fontsize=10)
    ax6.set_ylabel("Path Survival Probability", fontsize=10)
    ax6.grid(True, linestyle=":", alpha=0.6)
    ax6.legend(loc="upper left", frameon=True)
    
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Stage 2", "phase_b_validation_sprint.png")
    plt.savefig(output_path, dpi=300)
    print(f"\n[VALIDATION SUCCESS] Completed sweeps and saved plots to: {output_path}")

if __name__ == "__main__":
    execute_validation_sprint()
