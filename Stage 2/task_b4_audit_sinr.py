"""Stage 2 Task B4: Oracle vs MUSIC SINR Audit

This script runs a detailed diagnostic at d = 2.49 km to understand why 
MUSIC achieves a massive defeat range extension compared to the Geometric Oracle.
It compares three controllers:
1. Geometric Oracle
2. MUSIC
3. True Manifold Oracle
"""

import os
import sys
import numpy as np
import scipy.linalg as la

sys.path.append(os.path.dirname(__file__))
import phase_b_beamforming as pbb
import task_b_jamming_threats as jts

def lcmv_beamformer_true_manifold(R_xx: np.ndarray, a_s_true: np.ndarray, a_j_true: np.ndarray) -> np.ndarray:
    N = R_xx.shape[0]
    C = np.zeros((N, 2), dtype=complex)
    C[:, 0] = a_s_true
    C[:, 1] = a_j_true
    
    f = np.zeros(2, dtype=complex)
    f[0] = 1.0
    
    R_inv = la.inv(R_xx)
    temp1 = R_inv @ C
    temp2 = np.conj(C).T @ temp1
    temp2_inv = la.inv(temp2)
    return temp1 @ temp2_inv @ f

def audit_sinr_distributions():
    N = 8
    d_m = 2490.0
    theta_s = 0.0
    theta_j_true = np.radians(30.0)
    L = 100
    trials = 5000
    
    # Follower jammer hits exactly 50% of the hop (tau = 5ms, T_hop = 10ms)
    rho = 0.5
    
    Ps = jts.get_received_signal_power(d_m, 'two-ray')
    P_j = jts.get_received_jammer_power('fspl')
    
    SNR_lin = Ps / jts.P_N_lin
    INR_lin = P_j / jts.P_N_lin
    
    theta_elev = np.arctan(jts.h_UAV / d_m)
    K_s_db = 10.0 + 10.0 * np.sin(theta_elev)
    K_s = 10 ** (K_s_db / 10.0)
    los_s = np.sqrt(K_s / (K_s + 1.0))
    diff_s = np.sqrt(1.0 / (K_s + 1.0))
    
    rng = np.random.default_rng(2026)
    
    modes = ['Geometric Oracle', 'MUSIC', 'True Manifold Oracle']
    results = {m: {'null_depths': [], 'res_jam_pwr': [], 'sinrs': [], 'outages': 0} for m in modes}
    
    for _ in range(trials):
        # Array calibration errors
        amp_err = np.clip(rng.normal(0, 0.05, N), -0.9, 0.9)
        phase_err = rng.normal(0, np.radians(5.0), N)
        cal_err = (1.0 + amp_err) * np.exp(1j * phase_err)
        
        a_s_nom = pbb.ula_steering_vector(N, theta_s)
        a_j_nom = pbb.ula_steering_vector(N, theta_j_true)
        a_s_true = a_s_nom * cal_err
        a_j_true = a_j_nom * cal_err
        
        # MUSIC Estimation Data
        s_snap = rng.normal(0, np.sqrt(SNR_lin/2.0), L) + 1j * rng.normal(0, np.sqrt(SNR_lin/2.0), L)
        j_snap = rng.normal(0, np.sqrt(INR_lin/2.0), L) + 1j * rng.normal(0, np.sqrt(INR_lin/2.0), L)
        n_snap = rng.normal(0, np.sqrt(1.0/2.0), (N, L)) + 1j * rng.normal(0, np.sqrt(1.0/2.0), (N, L))
        X = np.outer(a_s_true, s_snap) + np.outer(a_j_true, j_snap) + n_snap
        R_xx_music = (X @ np.conj(X).T) / L
        
        scan_angles, pseudo_spectrum = pbb.music_doa(R_xx_music, num_sources=2, scan_resolution_deg=0.1)
        peaks = pbb.find_music_peaks(scan_angles, pseudo_spectrum, num_sources=2)
        if len(peaks) > 0:
            best_peak = peaks[np.argmin(np.abs(peaks - 30.0))]
            hat_theta_j = np.radians(best_peak)
        else:
            hat_theta_j = theta_j_true
            
        # Fast fading channel
        phi_s = rng.uniform(-np.pi, np.pi)
        u_s = rng.normal(0, np.sqrt(0.5)) + 1j * rng.normal(0, np.sqrt(0.5))
        h_s = los_s * np.exp(1j * phi_s) + diff_s * u_s
        h_j = rng.normal(0, np.sqrt(0.5)) + 1j * rng.normal(0, np.sqrt(0.5))
        
        for mode in modes:
            if mode == 'Geometric Oracle':
                R_xx_dl = SNR_lin * np.outer(a_s_nom, np.conj(a_s_nom)) + INR_lin * np.outer(a_j_nom, np.conj(a_j_nom)) + np.eye(N)
                w = pbb.lcmv_beamformer(R_xx_dl, theta_s, [theta_j_true])
            elif mode == 'MUSIC':
                a_j_music = pbb.ula_steering_vector(N, hat_theta_j)
                R_xx_dl = SNR_lin * np.outer(a_s_nom, np.conj(a_s_nom)) + INR_lin * np.outer(a_j_music, np.conj(a_j_music)) + np.eye(N)
                w = pbb.lcmv_beamformer(R_xx_dl, theta_s, [hat_theta_j])
            else: # True Manifold Oracle
                R_xx_dl = SNR_lin * np.outer(a_s_true, np.conj(a_s_true)) + INR_lin * np.outer(a_j_true, np.conj(a_j_true)) + np.eye(N)
                w = lcmv_beamformer_true_manifold(R_xx_dl, a_s_true, a_j_true)
                
            noise_out = np.sum(np.abs(w)**2)
            sig_out = SNR_lin * (np.abs(h_s)**2) * (np.abs(np.conj(w).T @ a_s_true) ** 2)
            jam_out = INR_lin * (np.abs(h_j)**2) * (np.abs(np.conj(w).T @ a_j_true) ** 2)
            
            null_depth = np.abs(np.conj(w).T @ a_j_true) ** 2
            
            # Follower outage happens when jammed SINR is below 5.64 dB
            # The outage over the hop is determined by whether the BER exceeds target.
            # But here we just compute the raw jammed SINR for the distribution audit.
            gamma_jammed = sig_out / (noise_out + jam_out)
            
            results[mode]['null_depths'].append(null_depth)
            results[mode]['res_jam_pwr'].append(jam_out)
            results[mode]['sinrs'].append(10.0 * np.log10(max(gamma_jammed, 1e-15)))
            
            if 10.0 * np.log10(gamma_jammed) < 5.64:
                results[mode]['outages'] += 1
                
    print(f"| {'Metric':<30} | {'Geometric Oracle':<18} | {'MUSIC':<18} | {'True Oracle':<18} |")
    print("|" + "-"*32 + "|" + "-"*20 + "|" + "-"*20 + "|" + "-"*20 + "|")
    
    def format_row(metric, key, is_db=True):
        v_geo = np.mean(results['Geometric Oracle'][key])
        v_mus = np.mean(results['MUSIC'][key])
        v_tru = np.mean(results['True Manifold Oracle'][key])
        
        if is_db:
            v_geo = 10.0 * np.log10(max(v_geo, 1e-15))
            v_mus = 10.0 * np.log10(max(v_mus, 1e-15))
            v_tru = 10.0 * np.log10(max(v_tru, 1e-15))
            
        print(f"| {metric:<30} | {v_geo:<18.2f} | {v_mus:<18.2f} | {v_tru:<18.2f} |")

    format_row('Mean null depth (dB)', 'null_depths')
    format_row('Mean residual jammer (dB)', 'res_jam_pwr')
    
    s_geo = np.mean(results['Geometric Oracle']['sinrs'])
    s_mus = np.mean(results['MUSIC']['sinrs'])
    s_tru = np.mean(results['True Manifold Oracle']['sinrs'])
    print(f"| {'Mean post-BF SINR (dB)':<30} | {s_geo:<18.2f} | {s_mus:<18.2f} | {s_tru:<18.2f} |")
    
    p10_geo = np.percentile(results['Geometric Oracle']['sinrs'], 10)
    p10_mus = np.percentile(results['MUSIC']['sinrs'], 10)
    p10_tru = np.percentile(results['True Manifold Oracle']['sinrs'], 10)
    print(f"| {'10th percentile SINR (dB)':<30} | {p10_geo:<18.2f} | {p10_mus:<18.2f} | {p10_tru:<18.2f} |")
    
    o_geo = results['Geometric Oracle']['outages'] / trials * 100
    o_mus = results['MUSIC']['outages'] / trials * 100
    o_tru = results['True Manifold Oracle']['outages'] / trials * 100
    print(f"| {'Outage probability (%)':<30} | {o_geo:<18.2f} | {o_mus:<18.2f} | {o_tru:<18.2f} |")

if __name__ == "__main__":
    audit_sinr_distributions()
