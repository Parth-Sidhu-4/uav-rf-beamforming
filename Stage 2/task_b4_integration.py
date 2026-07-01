"""Task B4: Phased Array Beamforming Integration & Defeat Boundary Sweeps.

This script integrates the Phase B array processing module into the communication
link budget and jamming model. It computes the output SINR, searches numerically
for the communication defeat boundary, and sweeps sensitivities over array size,
pointing error, snapshot count, and angular separation under both FSPL and Two-Ray models.
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
# 1. SYSTEM PARAMETERS (FROZEN BASELINE)
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

# Antenna Heights & Ground properties (from Task A2 baseline)
h_GS = 2.0              # Ground station antenna height: 2 m
h_UAV = 100.0           # UAV altitude: 100 m
epsilon_r = 15.0        # Relative permittivity

# Jammer (Class II: 73 dBm EIRP @ 5 km standoff)
jammer_standoff_m = 5000.0
jammer_eirp_dBm = 73.0
G_rx_jam = 0.0          # Isotropic jammer sidelobe coupling
B_ss_Hz = 40e6
PG_dB = 10 * np.log10(B_ss_Hz / B_Hz)  # 19.03 dB

# ==============================================================================
# 2. PATH LOSS MODELS & POWER CALCULATIONS
# ==============================================================================
def fspl_db(d_m):
    """Free Space Path Loss in dB."""
    return 40.05 + 20.0 * np.log10(np.maximum(d_m, 0.1))

def two_ray_path_loss(d_horizontal, h_gs=2.0, h_uav=100.0, f=2.4e9, eps_r=15.0, polarization='vertical'):
    """Calculate the Two-Ray Ground Reflection Path Loss (Vertical polarization)."""
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

def get_received_signal_power(d_m, path_loss_model='fspl'):
    """Signal power at a single array element (in Watts)."""
    if path_loss_model == 'two-ray':
        L_s = two_ray_path_loss(d_m, h_gs=h_GS, h_uav=h_UAV, f=f_c, eps_r=epsilon_r, polarization='vertical')
    else:
        L_s = fspl_db(d_m)
    S_dBm = P_tx_dBm + G_tx_dBi + G_rx_element_dBi - L_s
    return 10 ** (S_dBm / 10.0)

def get_received_jammer_power(path_loss_model='fspl'):
    """Jammer power at a single element after FHSS processing gain (in Watts)."""
    # Jammer path loss is modeled as FSPL for conservative baseline comparison
    if path_loss_model == 'two-ray':
        L_j = two_ray_path_loss(jammer_standoff_m, h_gs=h_GS, h_uav=h_GS, f=f_c, eps_r=epsilon_r, polarization='vertical')
    else:
        L_j = fspl_db(jammer_standoff_m)
    J_dBm = jammer_eirp_dBm + G_rx_jam - L_j - PG_dB
    return 10 ** (J_dBm / 10.0)

# ==============================================================================
# 3. NUMERICAL SOLVERS
# ==============================================================================
def evaluate_beamformer_sinr(d_m, N, theta_s, theta_j, sigma_theta_deg, L, delta_dl=1e-5, path_loss_model='fspl'):
    """Generate signal/jammer, apply diagonal loading LCMV, and compute output SINR."""
    Ps = get_received_signal_power(d_m, path_loss_model)
    Pj = get_received_jammer_power('fspl') # Jammer uses FSPL as conservative threat model
    
    if N == 1:
        # Single element case (no beamforming possible)
        return 10.0 * np.log10(Ps / (Pj + P_N_lin))

    # Generate covariance matrix (analytical true covariance for smooth sweeps)
    a_s = pbb.ula_steering_vector(N, theta_s)
    a_j = pbb.ula_steering_vector(N, theta_j)
    
    # Normalize powers by noise floor for consistent scaling in compute_output_sinr
    SNR_lin = Ps / P_N_lin
    INR_lin = Pj / P_N_lin
    
    R_xx_norm = SNR_lin * np.outer(a_s, np.conj(a_s)) + INR_lin * np.outer(a_j, np.conj(a_j)) + np.eye(N)
    
    # Apply diagonal loading for stability (relative to normalized noise floor of 1.0)
    R_xx_dl = R_xx_norm + delta_dl * np.eye(N)
    
    # Pointing error in estimated jammer angle
    e_theta = np.random.normal(0, np.radians(sigma_theta_deg))
    theta_j_est = theta_j + e_theta
    
    # LCMV constraints (unit gain at signal, null at estimated jammer direction)
    w = pbb.lcmv_beamformer(R_xx_dl, theta_s, [theta_j_est])
    
    # Output SINR using true signal/jammer bearings (normalized by P_N_lin)
    R_jn_true_norm = INR_lin * np.outer(a_j, np.conj(a_j)) + np.eye(N)
    
    return pbb.compute_output_sinr(w, R_jn_true_norm, 10.0 * np.log10(SNR_lin), theta_s)

def solve_defeat_range(N, theta_s, theta_j, sigma_theta_deg, L, d_min=100.0, d_max=3000000.0, path_loss_model='fspl'):
    """Find horizontal range in meters where output SINR equals 10 dB."""
    lo, hi = d_min, d_max
    
    # Check boundaries
    sinr_lo = evaluate_beamformer_sinr(lo, N, theta_s, theta_j, sigma_theta_deg, L, path_loss_model=path_loss_model)
    sinr_hi = evaluate_beamformer_sinr(hi, N, theta_s, theta_j, sigma_theta_deg, L, path_loss_model=path_loss_model)
    
    if sinr_hi >= 10.0:
        return hi
    if sinr_lo <= 10.0:
        return lo
        
    for _ in range(30):
        mid = (lo + hi) / 2.0
        sinr_mid = evaluate_beamformer_sinr(mid, N, theta_s, theta_j, sigma_theta_deg, L, path_loss_model=path_loss_model)
        if sinr_mid >= 10.0:
            lo = mid
        else:
            hi = mid
            
    return lo

def solve_noise_only_defeat_range(N, path_loss_model='fspl'):
    """Find horizontal range in meters where noise-only SINR (no jammer) equals 10 dB."""
    lo, hi = 100.0, 3000000.0
    for _ in range(30):
        mid = (lo + hi) / 2.0
        Ps = get_received_signal_power(mid, path_loss_model)
        # Coherent array gain gives N * Ps received signal power at output
        snr_out = 10.0 * np.log10(N * Ps / P_N_lin)
        if snr_out >= 10.0:
            lo = mid
        else:
            hi = mid
    return lo

# ==============================================================================
# 4. VERIFICATION SWEEPS
# ==============================================================================
def run_integration_sweeps():
    print("=" * 80)
    print("                 PHASE B SPRINT 1: TASK B4 BEAMFORMING INTEGRATION")
    print("=" * 80)
    
    theta_s = np.radians(0.0)   # Signal at broadside
    theta_j = np.radians(30.0)  # Jammer at 30 degrees
    L = 500                     # Default snapshots
    N_list = [2, 4, 8, 16, 32]
    
    # --------------------------------------------------------------------------
    # AUDIT #1: FSPL Baseline (Noise-Limited Ceiling Verification)
    # --------------------------------------------------------------------------
    print("\n[AUDIT #1] FSPL Propagation Model Defeat Boundaries:")
    print("-" * 110)
    print(f"{'N elements':<12} | {'Actual R_defeat (m)':^22} | {'Perfect-Jammer Boundary (m)':^28} | {'% Achieved Improvement':^25}")
    print("-" * 110)
    
    r_defeat_N1_fspl = solve_defeat_range(1, theta_s, theta_j, 0.0, L, path_loss_model='fspl')
    
    for N in [1] + N_list:
        r_def = solve_defeat_range(N, theta_s, theta_j, 0.0, L, path_loss_model='fspl')
        r_perfect = solve_noise_only_defeat_range(N, path_loss_model='fspl')
        pct_imp = 0.0 if N == 1 else (r_def - r_defeat_N1_fspl) / (r_perfect - r_defeat_N1_fspl) * 100.0
        print(f"{N:<12d} | {r_def:^22.1f} | {r_perfect:^28.1f} | {pct_imp:^25.4f}%")
    print("-" * 110)
    
    # --------------------------------------------------------------------------
    # AUDIT #2: Two-Ray Ground Reflection Model (Physical Realism Verification)
    # --------------------------------------------------------------------------
    print("\n[AUDIT #2] Two-Ray Ground Reflection Model Defeat Boundaries (Vertical Polarization):")
    print("-" * 110)
    print(f"{'N elements':<12} | {'Actual R_defeat (m)':^22} | {'Perfect-Jammer Boundary (m)':^28} | {'% Achieved Improvement':^25}")
    print("-" * 110)
    
    r_defeat_N1_2ray = solve_defeat_range(1, theta_s, theta_j, 0.0, L, path_loss_model='two-ray')
    
    for N in [1] + N_list:
        r_def = solve_defeat_range(N, theta_s, theta_j, 0.0, L, path_loss_model='two-ray')
        r_perfect = solve_noise_only_defeat_range(N, path_loss_model='two-ray')
        pct_imp = 0.0 if N == 1 else (r_def - r_defeat_N1_2ray) / (r_perfect - r_defeat_N1_2ray) * 100.0
        print(f"{N:<12d} | {r_def:^22.1f} | {r_perfect:^28.1f} | {pct_imp:^25.4f}%")
    print("-" * 110)
    
    # --------------------------------------------------------------------------
    # AUDIT #3: Link Budget Diagnostics (Check 1)
    # --------------------------------------------------------------------------
    print("\n[AUDIT #3] Physical Link Budget Diagnostics (at FSPL Defeat Boundaries):")
    print("-" * 120)
    print(f"{'N':<4} | {'Range (km)':^12} | {'FSPL PL (dB)':^14} | {'Rx Sig Power (dBm)':^20} | {'Thermal Noise (dBm)':^20} | {'Array Gain (dB)':^16} | {'SINR (dB)':^10}")
    print("-" * 120)
    
    for N in [4, 8, 16, 32]:
        r_def = solve_defeat_range(N, theta_s, theta_j, 0.0, L, path_loss_model='fspl')
        pl_fspl = fspl_db(r_def)
        sig_power_dbm = P_tx_dBm + G_tx_dBi + G_rx_element_dBi - pl_fspl
        array_gain_db = 10 * np.log10(N)
        sinr = sig_power_dbm - P_N_dBm + array_gain_db
        print(f"{N:<4d} | {r_def/1000:^12.2f} | {pl_fspl:^14.2f} | {sig_power_dbm:^20.2f} | {P_N_dBm:^20.2f} | {array_gain_db:^16.2f} | {sinr:^10.2f}")
    print("-" * 120)
    
    # --------------------------------------------------------------------------
    # 5. Generate Sensitivity Plots (4 Panels) under Two-Ray Model
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Panel A: R_defeat vs. N for different pointing errors (DOA mismatch)
    print("\n[5] Running Sensitivity Sweeps under Two-Ray model...")
    ax_a = axes[0, 0]
    N_sweep = np.array([1, 2, 4, 8, 16, 32])
    pointing_errors = [0.0, 1.0, 2.0, 5.0, 10.0]
    colors = ["blue", "green", "orange", "red", "purple"]
    
    for p_err, col in zip(pointing_errors, colors):
        r_defs = []
        for N in N_sweep:
            r_def_trials = [solve_defeat_range(N, theta_s, theta_j, p_err, L, path_loss_model='two-ray') for _ in range(10)]
            r_defs.append(np.mean(r_def_trials))
        ax_a.plot(N_sweep, np.array(r_defs)/1000.0, marker="o", color=col, label=f"$\\sigma_\\theta = {p_err:.0f}^\\circ$")
        
    ax_a.set_title("A. Defeat Range vs. Array Size & DOA Mismatch (Two-Ray)", fontsize=11, fontweight="bold")
    ax_a.set_xlabel("Number of Array Elements $N$", fontsize=10)
    ax_a.set_ylabel("Defeat Range $R_{defeat}$ [km]", fontsize=10)
    ax_a.grid(True, linestyle=":", alpha=0.6)
    ax_a.set_xscale("log")
    ax_a.set_xticks(N_sweep)
    ax_a.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax_a.legend(loc="upper left", frameon=True)
    
    # Panel B: R_defeat vs. DOA estimation error (Robustness Sweep)
    ax_b = axes[0, 1]
    err_sweep = np.linspace(0.0, 15.0, 30)
    for N in [2, 4, 8, 16]:
        r_defs = []
        for err in err_sweep:
            r_def_trials = [solve_defeat_range(N, theta_s, theta_j, err, L, path_loss_model='two-ray') for _ in range(15)]
            r_defs.append(np.mean(r_def_trials))
        ax_b.plot(err_sweep, np.array(r_defs)/1000.0, label=f"$N = {N}$")
        
    ax_b.set_title("B. DOA Error Robustness (Defeat boundary collapse, Two-Ray)", fontsize=11, fontweight="bold")
    ax_b.set_xlabel("DOA Estimation Error $\\sigma_\\theta$ [degrees]", fontsize=10)
    ax_b.set_ylabel("Defeat Range $R_{defeat}$ [km]", fontsize=10)
    ax_b.grid(True, linestyle=":", alpha=0.6)
    ax_b.legend(loc="upper right", frameon=True)
    
    # Panel C: R_defeat vs. Snapshot Count L
    ax_c = axes[1, 0]
    L_sweep = np.logspace(1.0, 3.0, 15, dtype=int)
    for N in [4, 8, 16]:
        r_defs = []
        for L_val in L_sweep:
            r_def_trials = []
            for _ in range(10):
                def solve_defeat_range_finite(N, theta_s, theta_j, L_val):
                    lo, hi = 100.0, 3000000.0
                    for _ in range(30):
                        mid = (lo + hi) / 2.0
                        Ps = get_received_signal_power(mid, 'two-ray')
                        Pj = get_received_jammer_power('fspl')
                        _, R_xx_est, R_jn = pbb.generate_received_signal(N, theta_s, [theta_j], 10.0*np.log10(Ps/P_N_lin), [10.0*np.log10(Pj/P_N_lin)], L_val)
                        R_xx_est_dl = R_xx_est + 1e-5 * np.eye(N)
                        w = pbb.lcmv_beamformer(R_xx_est_dl, theta_s, [theta_j])
                        sinr_mid = pbb.compute_output_sinr(w, R_jn, 10.0*np.log10(Ps/P_N_lin), theta_s)
                        if sinr_mid >= 10.0:
                            lo = mid
                        else:
                            hi = mid
                    return lo
                r_def_trials.append(solve_defeat_range_finite(N, theta_s, theta_j, L_val))
            r_defs.append(np.mean(r_def_trials))
        ax_c.plot(L_sweep, np.array(r_defs)/1000.0, marker="s", label=f"$N = {N}$")
        
    ax_c.set_title("C. Finite Snapshot Effect (Two-Ray)", fontsize=11, fontweight="bold")
    ax_c.set_xlabel("Number of snapshots $L$", fontsize=10)
    ax_c.set_ylabel("Defeat Range $R_{defeat}$ [km]", fontsize=10)
    ax_c.grid(True, linestyle=":", alpha=0.6)
    ax_c.set_xscale("log")
    ax_c.legend(loc="lower right", frameon=True)
    
    # Panel D: R_defeat vs. Angular Separation (Resolution Valley)
    ax_d = axes[1, 1]
    sep_sweep_deg = np.linspace(1.0, 60.0, 30)
    for N in [4, 8, 16]:
        r_defs = []
        for sep_deg in sep_sweep_deg:
            theta_j_sep = np.radians(sep_deg)
            r_defs.append(solve_defeat_range(N, theta_s, theta_j_sep, 0.0, L, path_loss_model='two-ray'))
        ax_d.plot(sep_sweep_deg, np.array(r_defs)/1000.0, label=f"$N = {N}$")
        
    ax_d.set_title("D. Angular Separation Limit (Resolution Valley, Two-Ray)", fontsize=11, fontweight="bold")
    ax_d.set_xlabel("Desired-Jammer Angular Separation $\\Delta\\theta$ [degrees]", fontsize=10)
    ax_d.set_ylabel("Defeat Range $R_{defeat}$ [km]", fontsize=10)
    ax_d.grid(True, linestyle=":", alpha=0.6)
    ax_d.legend(loc="lower right", frameon=True)
    
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Stage 2", "phase_b_task_b4_sweeps.png")
    plt.savefig(output_path, dpi=300)
    print(f"\n[B4 SUCCESS] Completed sweeps and saved plots to: {output_path}")

if __name__ == "__main__":
    run_integration_sweeps()
