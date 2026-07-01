"""Stage 2 Task B8: Beam Pattern Plotter under LCMV Beamforming.

This script plots the GCS phased array beam patterns, comparing:
1. Ideal array (no calibration mismatch).
2. Realistic array (with amplitude and phase mismatch: std_A = 0.05, std_phi = 5 deg).
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Ensure Stage 2 directory is in path
sys.path.append(os.path.dirname(__file__))
import phase_b_beamforming as pbb
import task_b_jamming_threats as jts

def generate_beam_patterns():
    print("=" * 80)
    print("         PHASE B SPRINT 3: GCS PHASED ARRAY BEAM PATTERN ANALYSIS")
    print("=" * 80)
    
    # Parameters
    N = 8
    theta_s = np.radians(0.0)
    theta_j = np.radians(30.0)
    
    # Received powers at d = 2.49 km
    d_ref_m = 2490.0
    Ps = jts.get_received_signal_power(d_ref_m, 'two-ray')
    P_j_total = jts.get_received_jammer_power('fspl')
    
    SNR_lin = Ps / jts.P_N_lin
    INR_lin = P_j_total / jts.P_N_lin
    
    # 1. Ideal Weights
    a_s_nom = pbb.ula_steering_vector(N, theta_s)
    a_j_nom = pbb.ula_steering_vector(N, theta_j)
    
    R_xx_ideal = SNR_lin * np.outer(a_s_nom, np.conj(a_s_nom)) + INR_lin * np.outer(a_j_nom, np.conj(a_j_nom)) + np.eye(N) + 1e-5 * np.eye(N)
    w_ideal = pbb.lcmv_beamformer(R_xx_ideal, theta_s, [theta_j])
    
    # 2. Calibration Mismatch Weights
    rng_cal = np.random.default_rng(42)
    amp_err = rng_cal.normal(0, jts.amp_err_std, N)
    amp_err = np.clip(amp_err, -0.9, 0.9)
    phase_err = rng_cal.normal(0, np.radians(jts.phase_err_std_deg), N)
    cal_err = (1.0 + amp_err) * np.exp(1j * phase_err)
    
    a_s_true = a_s_nom * cal_err
    a_j_true = a_j_nom * cal_err
    
    R_xx_real = SNR_lin * np.outer(a_s_nom, np.conj(a_s_nom)) + INR_lin * np.outer(a_j_nom, np.conj(a_j_nom)) + np.eye(N) + 1e-5 * np.eye(N)
    w_real = pbb.lcmv_beamformer(R_xx_real, theta_s, [theta_j])
    
    # Scan angles
    scan_deg = np.linspace(-90.0, 90.0, 500)
    scan_rad = np.radians(scan_deg)
    
    gain_ideal = []
    gain_real = []
    
    for theta in scan_rad:
        a = pbb.ula_steering_vector(N, theta)
        
        # Gain is |w^H * a|^2
        g_id = np.abs(np.conj(w_ideal).T @ a) ** 2
        # For the real array, the actual steering vector includes calibration error
        g_re = np.abs(np.conj(w_real).T @ (a * cal_err)) ** 2
        
        gain_ideal.append(10.0 * np.log10(g_id + 1e-20))
        gain_real.append(10.0 * np.log10(g_re + 1e-20))
        
    gain_ideal = np.array(gain_ideal)
    gain_real = np.array(gain_real)
    
    # Find gains at constraints
    gain_s_ideal = np.abs(np.conj(w_ideal).T @ a_s_nom) ** 2
    gain_j_ideal = np.abs(np.conj(w_ideal).T @ a_j_nom) ** 2
    gain_s_real = np.abs(np.conj(w_real).T @ a_s_true) ** 2
    gain_j_real = np.abs(np.conj(w_real).T @ a_j_true) ** 2
    
    print(f"Ideal Array Gain: Signal ({theta_s:.1f} rad) = {10*np.log10(gain_s_ideal):.2f} dB, Jammer ({theta_j:.1f} rad) = {10*np.log10(gain_j_ideal+1e-30):.2f} dB")
    print(f"Real Array Gain:  Signal ({theta_s:.1f} rad) = {10*np.log10(gain_s_real):.2f} dB, Jammer ({theta_j:.1f} rad) = {10*np.log10(gain_j_real+1e-30):.2f} dB")
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(scan_deg, gain_ideal, label="Ideal Phased Array (No Mismatch)", color="blue", linewidth=2.0)
    plt.plot(scan_deg, gain_real, label="Real Phased Array (std_A=0.05, std_phi=5°)", color="green", linewidth=2.0)
    
    plt.axvline(0.0, color="red", linestyle="--", label="Desired Signal (0°)")
    plt.axvline(30.0, color="orange", linestyle="--", label="Jammer (30°)")
    
    plt.title("GCS Phased Array Beam Pattern under LCMV Spatial Processing (N = 8)", fontsize=12, fontweight="bold")
    plt.xlabel("Azimuth Angle [degrees]", fontsize=10)
    plt.ylabel("Array Gain [dB]", fontsize=10)
    plt.ylim([-50, 25])
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", frameon=True)
    
    plot_path = os.path.join(os.path.dirname(__file__), "phase_b_beam_patterns.png")
    plt.savefig(plot_path, dpi=300)
    print(f"\n[SUCCESS] Saved beam patterns to: {plot_path}")

if __name__ == "__main__":
    generate_beam_patterns()
