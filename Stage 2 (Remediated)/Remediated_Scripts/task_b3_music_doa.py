import sys
import os
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, r'd:\UAV Internship project\Stage 2')

"""Stage 2 Task B3: MUSIC DOA Estimation & Validation.

This script evaluates the MUSIC algorithm for estimating the jammer's
Direction of Arrival (DOA), validates the RMSE against the Cramér-Rao 
Lower Bound (CRLB), and assesses operational estimation outages.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(__file__))
import phase_b_beamforming_remediated as pbb
import task_b_jamming_threats as jts

def generate_received_signal_with_cal(N, theta_s, theta_j, SNR_dB, INR_dB, L, sigma_A=0.0, sigma_phi=0.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()
        
    sigma_n2 = 1.0
    sigma_s2 = 10.0 ** (SNR_dB / 10.0)
    sigma_j2 = 10.0 ** (INR_dB / 10.0)
    
    # Generate sources
    s = rng.normal(0, np.sqrt(sigma_s2/2.0), L) + 1j * rng.normal(0, np.sqrt(sigma_s2/2.0), L)
    j = rng.normal(0, np.sqrt(sigma_j2/2.0), L) + 1j * rng.normal(0, np.sqrt(sigma_j2/2.0), L)
    
    # Array manifold
    a_s = pbb.ula_steering_vector(N, theta_s)
    a_j = pbb.ula_steering_vector(N, theta_j)
    
    # Calibration errors
    amp_err = np.clip(rng.normal(0, sigma_A, N), -0.9, 0.9)
    phase_err = rng.normal(0, np.radians(sigma_phi), N)
    cal_err = (1.0 + amp_err) * np.exp(1j * phase_err)
    
    a_s_true = a_s * cal_err
    a_j_true = a_j * cal_err
    
    X_s = np.outer(a_s_true, s)
    X_j = np.outer(a_j_true, j)
    noise = rng.normal(0, np.sqrt(sigma_n2/2.0), (N, L)) + 1j * rng.normal(0, np.sqrt(sigma_n2/2.0), (N, L))
    
    X = X_s + X_j + noise
    R_xx_est = (X @ np.conj(X).T) / L
    
    return R_xx_est

def run_monte_carlo_music(N, L, SNR_dB, INR_dB, theta_s_deg, theta_j_deg, sigma_A=0.0, sigma_phi=0.0, trials=1000):
    theta_s = np.radians(theta_s_deg)
    theta_j = np.radians(theta_j_deg)
    
    rng = np.random.default_rng(2026)
    
    crlb_rad = pbb.compute_crlb_doa_rad(N, L, INR_dB, np.radians(theta_j_deg))
    crlb_deg = np.degrees(crlb_rad)
    
    squared_errors = []
    errors_deg = []
    
    for _ in range(trials):
        R_xx = generate_received_signal_with_cal(N, theta_s, theta_j, SNR_dB, INR_dB, L, sigma_A, sigma_phi, rng)
        
        # FIX-1: Dynamic grid resolution to prevent sub-physical precision
        res_deg = max(0.05, crlb_deg / 5.0)
        scan_angles, pseudo_spectrum = pbb.music_doa(R_xx, num_sources=2, scan_resolution_deg=res_deg)
        peaks = pbb.find_music_peaks(scan_angles, pseudo_spectrum, num_sources=2)
        
        if len(peaks) > 0:
            # Pick the peak closest to the true jammer angle
            best_peak = peaks[np.argmin(np.abs(peaks - theta_j_deg))]
            err = best_peak - theta_j_deg
            squared_errors.append(err**2)
            errors_deg.append(err)
        else:
            squared_errors.append(90.0**2) # Failed to find any peak
            errors_deg.append(90.0)
            
    rmse_deg = np.sqrt(np.mean(squared_errors))
    
    # FIX-1: CRLB Guard assertion
    assert rmse_deg >= 0.8 * crlb_deg, (
        f"MUSIC RMSE {rmse_deg:.3e} deg fell below 0.8xCRLB {crlb_deg:.3e} deg "
        f"at INR={INR_dB} dB, N={N}, L={L}. "
        "This indicates non-physical precision."
    )
    
    return rmse_deg, errors_deg

def plot_sweeps():
    print("Running MUSIC Sweeps...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # Fixed baselines
    N_base = 8
    L_base = 100
    SNR_base = 0.0
    INR_base = 10.0
    theta_s_base = 0.0
    theta_j_base = 30.0123
    trials = 500
    
    # 1. INR Sweep
    print("1. Sweeping INR...")
    inr_vals = np.arange(-20, 45, 5)
    rmse_inr = []
    crlb_inr = []
    for inr in inr_vals:
        rmse, _ = run_monte_carlo_music(N_base, L_base, SNR_base, inr, theta_s_base, theta_j_base, trials=trials)
        rmse_inr.append(rmse)
        crlb_rad2 = pbb.compute_crlb_doa_rad(N_base, L_base, inr, np.radians(theta_j_base))
        crlb_deg = np.degrees(np.sqrt(crlb_rad2))
        crlb_inr.append(crlb_deg)
        
    axes[0].plot(inr_vals, rmse_inr, 'bo-', label='MUSIC RMSE')
    axes[0].plot(inr_vals, crlb_inr, 'r--', label='CRLB')
    axes[0].set_yscale('log')
    axes[0].set_title("RMSE vs Jammer INR (N=8, L=100)", fontweight='bold')
    axes[0].set_xlabel("INR [dB]")
    axes[0].set_ylabel("RMSE [degrees]")
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].legend()
    
    # 2. L Sweep
    print("2. Sweeping Snapshots (L)...")
    L_vals = [10, 20, 50, 100, 200, 500]
    rmse_L = []
    crlb_L = []
    for l in L_vals:
        rmse, _ = run_monte_carlo_music(N_base, l, SNR_base, INR_base, theta_s_base, theta_j_base, trials=trials)
        rmse_L.append(rmse)
        crlb_rad2 = pbb.compute_crlb_doa_rad(N_base, l, INR_base, np.radians(theta_j_base))
        crlb_L.append(np.degrees(np.sqrt(crlb_rad2)))
        
    axes[1].plot(L_vals, rmse_L, 'go-', label='MUSIC RMSE')
    axes[1].plot(L_vals, crlb_L, 'r--', label='CRLB')
    axes[1].set_yscale('log')
    axes[1].set_xscale('log')
    axes[1].set_title("RMSE vs Snapshots L (N=8, INR=10dB)", fontweight='bold')
    axes[1].set_xlabel("Snapshots (L)")
    axes[1].set_ylabel("RMSE [degrees]")
    axes[1].grid(True, linestyle=':', alpha=0.6)
    axes[1].legend()
    
    # 3. N Sweep
    print("3. Sweeping Array Size (N)...")
    N_vals = [4, 8, 16, 32]
    rmse_N = []
    crlb_N = []
    for n in N_vals:
        rmse, _ = run_monte_carlo_music(n, L_base, SNR_base, INR_base, theta_s_base, theta_j_base, trials=trials)
        rmse_N.append(rmse)
        crlb_rad2 = pbb.compute_crlb_doa_rad(n, L_base, INR_base, np.radians(theta_j_base))
        crlb_N.append(np.degrees(np.sqrt(crlb_rad2)))
        
    axes[2].plot(N_vals, rmse_N, 'mo-', label='MUSIC RMSE')
    axes[2].plot(N_vals, crlb_N, 'r--', label='CRLB')
    axes[2].set_yscale('log')
    axes[2].set_title("RMSE vs Array Size N (L=100, INR=10dB)", fontweight='bold')
    axes[2].set_xlabel("Array Elements (N)")
    axes[2].set_ylabel("RMSE [degrees]")
    axes[2].grid(True, linestyle=':', alpha=0.6)
    axes[2].set_xticks(N_vals)
    axes[2].legend()
    
    # 4. Angular Separation Sweep
    print("4. Sweeping Angular Separation...")
    delta_vals = [5, 10, 15, 20, 30, 45]
    rmse_delta = []
    crlb_delta = []
    for d_theta in delta_vals:
        tj = theta_s_base + d_theta
        rmse, _ = run_monte_carlo_music(N_base, L_base, SNR_base, INR_base, theta_s_base, tj, trials=trials)
        rmse_delta.append(rmse)
        crlb_rad2 = pbb.compute_crlb_doa_rad(N_base, L_base, INR_base, np.radians(tj))
        crlb_delta.append(np.degrees(np.sqrt(crlb_rad2)))
        
    axes[3].plot(delta_vals, rmse_delta, 'co-', label='MUSIC RMSE')
    axes[3].plot(delta_vals, crlb_delta, 'r--', label='CRLB')
    axes[3].set_yscale('log')
    axes[3].set_title("RMSE vs Ang. Separation (N=8, INR=10dB)", fontweight='bold')
    axes[3].set_xlabel("Separation |theta_j - theta_s| [deg]")
    axes[3].set_ylabel("RMSE [degrees]")
    axes[3].grid(True, linestyle=':', alpha=0.6)
    axes[3].legend()
    
    plt.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(__file__), "task_b3_music_sweeps.png"), dpi=300)
    print("Saved task_b3_music_sweeps.png")

def evaluate_operational_exceedance():
    print("\\n=======================================================")
    print(" OPERATIONAL OUTAGE PROBABILITY (d = 2.49 km) ")
    print("=======================================================")
    N = 8
    L = 100
    sigma_A = 0.05
    sigma_phi = 5.0
    d_m = 2490.0
    theta_s = 0.0
    theta_j = 30.0
    
    Ps = jts.get_received_signal_power(d_m, 'two-ray')
    P_j = jts.get_received_jammer_power('fspl')
    
    SNR_dB = 10.0 * np.log10(Ps / jts.P_N_lin)
    INR_dB = 10.0 * np.log10(P_j / jts.P_N_lin)
    
    print(f"Operational Signal SNR: {SNR_dB:.2f} dB")
    print(f"Operational Jammer INR: {INR_dB:.2f} dB")
    
    rmse, errors = run_monte_carlo_music(N, L, SNR_dB, INR_dB, theta_s, theta_j, sigma_A, sigma_phi, trials=5000)
    
    errors = np.abs(errors)
    p_1 = np.mean(errors > 1.0) * 100
    p_2 = np.mean(errors > 2.0) * 100
    p_5 = np.mean(errors > 5.0) * 100
    p_10 = np.mean(errors > 10.0) * 100
    
    print(f"MUSIC RMSE at Operational Point: {rmse:.4f} deg")
    print(f"P(|err| > 1 deg) : {p_1:.2f}%")
    print(f"P(|err| > 2 deg) : {p_2:.2f}%")
    print(f"P(|err| > 5 deg) : {p_5:.2f}%")
    print(f"P(|err| > 10 deg): {p_10:.2f}%")

if __name__ == "__main__":
    plot_sweeps()
    evaluate_operational_exceedance()
