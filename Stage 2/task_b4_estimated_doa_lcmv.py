"""Stage 2 Task B4: Estimated-DOA LCMV Integration.

This script evaluates the impact of DOA estimation errors on the LCMV 
beamformer's null depth and the resulting UAV defeat range.
It compares the perfect oracle against MUSIC and static pointing errors.
"""

import os
import sys
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(__file__))
import phase_b_beamforming as pbb
import task_b_jamming_threats as jts

def run_lcmv_sim(doa_mode, N=8, theta_s=0.0, theta_j_true=np.radians(30.0), d_m=2000.0, trials=500):
    """
    doa_mode: 'oracle', 'music', or a float representing static error in degrees.
    Returns probability of link success for BPSK (gamma > 5.64 dB).
    """
    Ps = jts.get_received_signal_power(d_m, 'two-ray')
    P_j = jts.get_received_jammer_power('fspl') # Follower / Barrage power
    
    SNR_lin = Ps / jts.P_N_lin
    INR_lin = P_j / jts.P_N_lin
    SNR_dB = 10.0 * np.log10(SNR_lin)
    INR_dB = 10.0 * np.log10(INR_lin)
    
    theta_elev = np.arctan(jts.h_UAV / d_m)
    K_s_db = 10.0 + 10.0 * np.sin(theta_elev)
    K_s = 10 ** (K_s_db / 10.0)
    
    rng = np.random.default_rng(42)
    
    success_count = 0
    L = 100 # Snapshots for MUSIC
    
    los_s = np.sqrt(K_s / (K_s + 1.0))
    diff_s = np.sqrt(1.0 / (K_s + 1.0))
    
    avg_null_depth_lin = 0.0
    
    for _ in range(trials):
        # Array calibration errors (Operational point)
        amp_err = np.clip(rng.normal(0, 0.05, N), -0.9, 0.9)
        phase_err = rng.normal(0, np.radians(5.0), N)
        cal_err = (1.0 + amp_err) * np.exp(1j * phase_err)
        
        a_s_nom = pbb.ula_steering_vector(N, theta_s)
        a_j_nom = pbb.ula_steering_vector(N, theta_j_true)
        a_s_true = a_s_nom * cal_err
        a_j_true = a_j_nom * cal_err
        
        # ALWAYS generate snapshots to keep RNG aligned
        s_snap = rng.normal(0, np.sqrt(SNR_lin/2.0), L) + 1j * rng.normal(0, np.sqrt(SNR_lin/2.0), L)
        j_snap = rng.normal(0, np.sqrt(INR_lin/2.0), L) + 1j * rng.normal(0, np.sqrt(INR_lin/2.0), L)
        n_snap = rng.normal(0, np.sqrt(1.0/2.0), (N, L)) + 1j * rng.normal(0, np.sqrt(1.0/2.0), (N, L))
        
        if doa_mode == 'oracle':
            hat_theta_j = theta_j_true
        elif doa_mode == 'music':
            # Generate sample covariance over L snapshots
            X = np.outer(a_s_true, s_snap) + np.outer(a_j_true, j_snap) + n_snap
            R_xx = (X @ np.conj(X).T) / L
            
            scan_angles, pseudo_spectrum = pbb.music_doa(R_xx, num_sources=2, scan_resolution_deg=0.1)
            peaks = pbb.find_music_peaks(scan_angles, pseudo_spectrum, num_sources=2)
            
            if len(peaks) > 0:
                best_peak = peaks[np.argmin(np.abs(peaks - np.degrees(theta_j_true)))]
                hat_theta_j = np.radians(best_peak)
            else:
                hat_theta_j = theta_j_true # Fallback
        else:
            hat_theta_j = theta_j_true + np.radians(float(doa_mode))
            
        # LCMV Design (based on nominal manifold but with estimated angle)
        R_xx_dl = SNR_lin * np.outer(a_s_nom, np.conj(a_s_nom)) + INR_lin * np.outer(pbb.ula_steering_vector(N, hat_theta_j), np.conj(pbb.ula_steering_vector(N, hat_theta_j))) + np.eye(N)
        w = pbb.lcmv_beamformer(R_xx_dl, theta_s, [hat_theta_j])
        
        # Apply to fading channel
        phi_s = rng.uniform(-np.pi, np.pi)
        u_s = rng.normal(0, np.sqrt(0.5)) + 1j * rng.normal(0, np.sqrt(0.5))
        h_s = los_s * np.exp(1j * phi_s) + diff_s * u_s
        h_j = rng.normal(0, np.sqrt(0.5)) + 1j * rng.normal(0, np.sqrt(0.5))
        
        noise_out = np.sum(np.abs(w)**2)
        sig_out = SNR_lin * (np.abs(h_s)**2) * (np.abs(np.conj(w).T @ a_s_true) ** 2)
        jam_out = INR_lin * (np.abs(h_j)**2) * (np.abs(np.conj(w).T @ a_j_true) ** 2)
        
        avg_null_depth_lin += (np.abs(np.conj(w).T @ a_j_true) ** 2)
        
        gamma = sig_out / (noise_out + jam_out)
        gamma_db = 10.0 * np.log10(gamma)
        
        if gamma_db >= 5.64:
            success_count += 1
            
    p_success = success_count / trials
    mean_null_depth_db = 10.0 * np.log10(max(avg_null_depth_lin / trials, 1e-15))
    return p_success, mean_null_depth_db

def solve_defeat_range(doa_mode):
    d_min = 100.0
    d_max = 20000.0
    d_opt = d_max
    
    p_link_max, _ = run_lcmv_sim(doa_mode, d_m=d_max, trials=500)
    if 1.0 - p_link_max < 0.10:
        return d_max
        
    for _ in range(10):
        d_mid = (d_min + d_max) / 2.0
        p_link, _ = run_lcmv_sim(doa_mode, d_m=d_mid, trials=500)
        if 1.0 - p_link >= 0.10:
            d_max = d_mid
        else:
            d_min = d_mid
            d_opt = d_mid
    return d_opt

def run_b4_comparison():
    print("=======================================================")
    print(" TASK B4: DEFEAT RANGE VS DOA ERROR ")
    print("=======================================================")
    
    modes = ['oracle', 'music', 1.0, 2.0, 5.0, 10.0]
    
    print(f"{'Case':<12} | {'DOA Error [deg]':<18} | {'Avg Null Depth [dB]':<20} | {'Defeat Range [km]':<18}")
    print("-" * 75)
    
    for mode in modes:
        dr = solve_defeat_range(mode)
        # Re-run at Defeat Range to get the average null depth
        _, null_depth = run_lcmv_sim(mode, d_m=dr, trials=1000)
        
        if mode == 'oracle':
            err_str = "0.00"
            case_str = "Oracle"
        elif mode == 'music':
            err_str = "Estimated"
            case_str = "MUSIC"
        else:
            err_str = f"+{mode:.1f}"
            case_str = f"Static Error"
            
        print(f"{case_str:<12} | {err_str:<18} | {null_depth:<20.2f} | {dr/1000.0:<18.2f}")

if __name__ == "__main__":
    run_b4_comparison()
