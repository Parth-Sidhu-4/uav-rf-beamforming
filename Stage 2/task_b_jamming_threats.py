"""Stage 2 Task B5 & B6: Advanced Jamming Threat Models.

This script implements:
1. Task B5 (Partial-Band Jamming): out-of-band FHSS processing gain degradation
   and outage probability vs. bandwidth occupancy fraction alpha.
2. Task B6 (Follower / Reactive Jamming): temporal overlap and link survival
   vs. follower reaction delay ratio tau_delay / T_hop.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Ensure Stage 2 directory is in path for imports
sys.path.append(os.path.dirname(__file__))
import phase_b_beamforming as pbb

# ==============================================================================
# 1. SYSTEM BASELINE PARAMETERS
# ==============================================================================
P_tx_dBm = 37.0         # UAV TX Power: 37 dBm (5 W)
G_tx_dBi = 3.0          # UAV TX Antenna Gain: 3 dBi
G_rx_element_dBi = 10.0 # Single GCS element gain: 10 dBi
B_Hz = 500e3            # Receiver/Hop Bandwidth: 500 kHz
NF_dB = 6.0             # Receiver Noise Figure: 6 dB
f_c = 2.4e9             # Carrier Frequency: 2.4 GHz
gamma_th_dB = 10.0      # Demodulation SNR threshold: 10.0 dB

P_N_dBm = -174.0 + 10 * np.log10(B_Hz) + NF_dB  # Noise floor: -111.0 dBm
P_N_lin = 10 ** (P_N_dBm / 10.0)
ber_th = 0.5 * np.exp(-10.0 / 2.0)  # FSK BER threshold corresponding to 10 dB SINR

# FHSS parameters
N_h = 80                # Number of channels (40 MHz / 500 kHz)
T_hop = 10e-3           # Hop duration: 10 ms (100 hops/s)

# Antenna Heights & Ground properties
h_GS = 2.0              # Ground GCS height: 2 m
h_UAV = 100.0           # UAV altitude: 100 m
epsilon_r = 15.0        # Relative permittivity

# Jammer Standoff coordinates
jammer_standoff_m = 5000.0
jammer_eirp_dBm = 73.0
G_rx_jam = 0.0

# Validated impairments from Sprint 1
amp_err_std = 0.05
phase_err_std_deg = 5.0

# Fixed operating range for sensitivity curves
d_fixed_m = 15000.0     # 15 km

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
    J_dBm = jammer_eirp_dBm + G_rx_jam - L_j
    return 10 ** (J_dBm / 10.0)

# ==============================================================================
# 3. CORE THREAT MODEL SIMULATOR
# ==============================================================================
def simulate_link_performance(d_m, N, theta_s, theta_j, alpha, jammer_type='pbj', tau_delay=0.0, trials=1000):
    """Simulate link outage probability and BER under PBJ or Follower jamming,
    inheriting array calibration errors and Rician A2G fading.
    """
    # 1. Received Powers
    Ps = get_received_signal_power(d_m, 'two-ray')
    P_j_total = get_received_jammer_power('fspl')
    
    # 2. Jammer Power Per Channel
    if jammer_type == 'pbj':
        P_j_ch = P_j_total / (alpha * N_h)
    else:
        # Follower jammer acts as spot jammer during the jammed portion
        P_j_ch = P_j_total
        
    SNR_lin = Ps / P_N_lin
    INR_lin = P_j_ch / P_N_lin
    
    # 3. Fading K-factor
    theta_elev = np.arctan(h_UAV / d_m)
    K_s_db = 10.0 + 10.0 * np.sin(theta_elev)
    K_s = 10 ** (K_s_db / 10.0)
    K_j = 0.0  # Rayleigh fading for ground-to-ground jammer
    
    # 4. Calibration Mismatch (Static Hardware Property)
    rng_cal = np.random.default_rng(42)
    amp_err = rng_cal.normal(0, amp_err_std, N)
    phase_err = rng_cal.normal(0, np.radians(phase_err_std_deg), N)
    cal_err = (1.0 + amp_err) * np.exp(1j * phase_err)
    
    # Steering Vectors
    a_s_nom = pbb.ula_steering_vector(N, theta_s)
    a_j_nom = pbb.ula_steering_vector(N, theta_j)
    a_s_true = a_s_nom * cal_err
    a_j_true = a_j_nom * cal_err
    
    # 5. Phased Array Spatial Nulling Weights w
    if N > 1:
        # Build design covariance matrix using nominal directions to avoid self-nulling
        R_xx_dl = SNR_lin * np.outer(a_s_nom, np.conj(a_s_nom)) + INR_lin * np.outer(a_j_nom, np.conj(a_j_nom)) + np.eye(N) + 1e-5 * np.eye(N)
        w = pbb.lcmv_beamformer(R_xx_dl, theta_s, [theta_j])
    else:
        w = np.array([1.0], dtype=complex)
        
    # 6. Generate Fading Trials
    rng_fad = np.random.default_rng(2026)
    los_s = np.sqrt(K_s / (K_s + 1.0))
    diff_s = np.sqrt(1.0 / (K_s + 1.0))
    diff_j = 1.0  # Since K_j = 0
    
    ber_list = []
    
    for _ in range(trials):
        # Desired signal fading
        phi_s = rng_fad.uniform(-np.pi, np.pi)
        u_s = rng_fad.normal(0, np.sqrt(0.5)) + 1j * rng_fad.normal(0, np.sqrt(0.5))
        h_s = los_s * np.exp(1j * phi_s) + diff_s * u_s
        
        # Jammer fading
        u_j = rng_fad.normal(0, np.sqrt(0.5)) + 1j * rng_fad.normal(0, np.sqrt(0.5))
        h_j = diff_j * u_j
        
        # Output signal and noise power
        sig_out = SNR_lin * (np.abs(h_s)**2) * (np.abs(np.conj(w).T @ a_s_true) ** 2)
        noise_out = np.sum(np.abs(w)**2)
        
        # Output jammer power
        jam_out_jammed = INR_lin * (np.abs(h_j)**2) * (np.abs(np.conj(w).T @ a_j_true) ** 2)
        
        # Output SINRs
        sinr_jammed = sig_out / (noise_out + jam_out_jammed)
        sinr_unjammed = sig_out / noise_out
        
        # BFSK BERs
        ber_jammed = 0.5 * np.exp(-sinr_jammed / 2.0)
        ber_unjammed = 0.5 * np.exp(-sinr_unjammed / 2.0)
        
        # Hop BER-weighted averaging
        if jammer_type == 'pbj':
            ber_trial = alpha * ber_jammed + (1.0 - alpha) * ber_unjammed
        else:  # follower
            rho = np.max([0.0, 1.0 - tau_delay / T_hop])
            ber_trial = rho * ber_jammed + (1.0 - rho) * ber_unjammed
            
        ber_list.append(ber_trial)
        
    ber_list = np.array(ber_list)
    ber_avg = np.mean(ber_list)
    outage_prob = np.mean(ber_list > ber_th)
    
    return outage_prob, ber_avg

# ==============================================================================
# 4. EFFECTIVE PROCESSING-GAIN EVALUATOR
# ==============================================================================
def get_effective_processing_gain(d_m, N, theta_s, theta_j, alpha, trials=1000):
    """Calculate the effective FHSS processing gain surviving PBJ."""
    # 1. Average BER under PBJ with fraction alpha
    _, ber_pbj = simulate_link_performance(d_m, N, theta_s, theta_j, alpha, 'pbj', trials=trials)
    
    # 2. Average BER under raw spot jamming (no FHSS, jammer power concentrated, no N_h division)
    _, ber_raw = simulate_link_performance(d_m, N, theta_s, theta_j, 1.0, 'follower', tau_delay=0.0, trials=trials)
    
    # 3. Equivalent SINRs (BPSK/BFSK reference)
    ber_pbj_clipped = np.clip(ber_pbj, 1e-15, 0.5 - 1e-15)
    ber_raw_clipped = np.clip(ber_raw, 1e-15, 0.5 - 1e-15)
    
    sinr_eff_lin = -2.0 * np.log(2.0 * ber_pbj_clipped)
    sinr_raw_lin = -2.0 * np.log(2.0 * ber_raw_clipped)
    
    sinr_eff_db = 10.0 * np.log10(np.maximum(sinr_eff_lin, 1e-10))
    sinr_raw_db = 10.0 * np.log10(np.maximum(sinr_raw_lin, 1e-10))
    
    # PG is the ratio of effective SINR with FHSS to raw SINR without FHSS
    pg_eff_db = sinr_eff_db - sinr_raw_db
    nominal_pg_db = 10.0 * np.log10(N_h)  # 19.03 dB
    
    # Keep within physically bounds: max is nominal PG, min is 0 dB
    pg_eff_db = np.clip(pg_eff_db, 0.0, nominal_pg_db)
    pg_eff_norm = pg_eff_db / nominal_pg_db
    
    return pg_eff_db, pg_eff_norm

# ==============================================================================
# 5. DEFEAT RANGE BISECTION SOLVER
# ==============================================================================
def solve_defeat_range(N, theta_s, theta_j, alpha, jammer_type='pbj', tau_delay=0.0, trials=300):
    """Solve for the defeat boundary (range where fading outage probability reaches 0.10)."""
    lo, hi = 100.0, 250000.0
    for _ in range(20):
        mid = (lo + hi) / 2.0
        outage, _ = simulate_link_performance(mid, N, theta_s, theta_j, alpha, jammer_type, tau_delay, trials)
        if outage <= 0.10:
            lo = mid
        else:
            hi = mid
    return lo

# ==============================================================================
# 6. RUN SWEEPS AND GENERATE RESULTS
# ==============================================================================
def execute_threat_sweeps():
    print("=" * 80)
    print("        PHASE B SPRINT 2: ADVANCED JAMMING THREAT SIMULATION & ANALYSIS")
    print("=" * 80)
    
    theta_s = np.radians(0.0)      # Broadside desired signal
    theta_j = np.radians(30.0)     # Jammer at 30 degrees
    
    # Arrays to sweep
    n_elements = [1, 4, 8, 16]
    
    # --------------------------------------------------------------------------
    # Task B5: Partial-Band Jamming Sweep
    # --------------------------------------------------------------------------
    print("\n--- Running Task B5: Partial-Band Jamming Sweeps ---")
    alphas = np.linspace(0.01, 1.0, 20)
    
    pbj_outages = {N: [] for N in n_elements}
    pbj_bers = {N: [] for N in n_elements}
    pbj_pgs_db = {N: [] for N in n_elements}
    pbj_pgs_norm = {N: [] for N in n_elements}
    pbj_defeat_ranges = {N: [] for N in n_elements}
    
    for N in n_elements:
        print(f"Simulating N={N}...")
        for alpha in alphas:
            outage, ber = simulate_link_performance(d_fixed_m, N, theta_s, theta_j, alpha, 'pbj', trials=500)
            pg_db, pg_norm = get_effective_processing_gain(d_fixed_m, N, theta_s, theta_j, alpha, trials=500)
            pbj_outages[N].append(outage)
            pbj_bers[N].append(ber)
            pbj_pgs_db[N].append(pg_db)
            pbj_pgs_norm[N].append(pg_norm)
            
        # Defeat ranges vs. alpha
        # We sweep fewer points for defeat range to keep execution time fast
        alphas_dr = [0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 1.0]
        for a_dr in alphas_dr:
            dr = solve_defeat_range(N, theta_s, theta_j, a_dr, 'pbj', trials=200)
            pbj_defeat_ranges[N].append((a_dr, dr / 1000.0))
            
    # --------------------------------------------------------------------------
    # Task B6: Follower Jammer Sweep
    # --------------------------------------------------------------------------
    print("\n--- Running Task B6: Follower Jammer Sweeps ---")
    delays_ms = np.linspace(0.0, 12.0, 20)
    
    fol_outages = {N: [] for N in n_elements}
    fol_bers = {N: [] for N in n_elements}
    fol_defeat_ranges = {N: [] for N in n_elements}
    
    for N in n_elements:
        print(f"Simulating N={N}...")
        for delay in delays_ms:
            outage, ber = simulate_link_performance(d_fixed_m, N, theta_s, theta_j, 1.0, 'follower', tau_delay=delay*1e-3, trials=500)
            fol_outages[N].append(outage)
            fol_bers[N].append(ber)
            
        # Defeat ranges vs. delay
        delays_dr = [0.0, 2.0, 5.0, 8.0, 10.0, 12.0]
        for d_dr in delays_dr:
            dr = solve_defeat_range(N, theta_s, theta_j, 1.0, 'follower', tau_delay=d_dr*1e-3, trials=200)
            fol_defeat_ranges[N].append((d_dr, dr / 1000.0))
            
    # ==============================================================================
    # PRINT SUMMARY DIAGNOSTIC TABLE
    # ==============================================================================
    print("\n" + "=" * 80)
    print("                    PBJ SENSITIVITY SUMMARY (d = 15 km)")
    print("=" * 80)
    print(f"{'Array Size':<12} | {'Worst Alpha':^12} | {'Max Outage':^12} | {'Max BER':^12} | {'Min PG_eff (dB)':^16}")
    print("-" * 80)
    for N in n_elements:
        idx_worst = np.argmax(pbj_bers[N])
        worst_alpha = alphas[idx_worst]
        max_outage = pbj_outages[N][idx_worst]
        max_ber = pbj_bers[N][idx_worst]
        min_pg = pbj_pgs_db[N][idx_worst]
        print(f"N={N:<10} | {worst_alpha:^12.2f} | {max_outage:^12.4f} | {max_ber:^12.4e} | {min_pg:^16.2f}")
    print("=" * 80)
    
    print("\n" + "=" * 80)
    print("                FOLLOWER JAMMER SENSITIVITY SUMMARY (d = 15 km)")
    print("=" * 80)
    print(f"{'Array Size':<12} | {'Delay = 0ms':^13} | {'Delay = 5ms':^13} | {'Delay = 10ms':^13} | {'Delay = 12ms':^13}")
    print("-" * 80)
    for N in n_elements:
        out_0 = fol_outages[N][0]
        out_5 = fol_outages[N][np.argmin(np.abs(delays_ms - 5.0))]
        out_10 = fol_outages[N][np.argmin(np.abs(delays_ms - 10.0))]
        out_12 = fol_outages[N][np.argmin(np.abs(delays_ms - 12.0))]
        print(f"N={N:<10} | {out_0:^13.4f} | {out_5:^13.4f} | {out_10:^13.4f} | {out_12:^13.4f}")
    print("=" * 80)

    # ==============================================================================
    # 7. GENERATE 4-PANEL VISUALIZATION PLOT
    # ==============================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Panel 1: Outage Probability vs. PBJ Bandwidth Fraction alpha
    ax1 = axes[0, 0]
    for N in n_elements:
        ax1.plot(alphas, pbj_outages[N], label=f"N={N}", linewidth=2)
    ax1.set_title("1. Link Outage Probability vs. PBJ Bandwidth Fraction $\\alpha$\n(at fixed range $d=15$ km)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("PBJ Bandwidth Occupancy Fraction $\\alpha$", fontsize=10)
    ax1.set_ylabel("Outage Probability (BER > $3.37\\times 10^{-3}$)", fontsize=10)
    ax1.set_xlim([0.01, 1.0])
    ax1.set_ylim([-0.05, 1.05])
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", frameon=True)
    
    # Panel 2: Effective Processing Gain vs. PBJ Bandwidth Fraction alpha
    ax2 = axes[0, 1]
    for N in n_elements:
        ax2.plot(alphas, pbj_pgs_db[N], label=f"N={N}", linewidth=2)
    ax2.set_title("2. Effective FHSS Processing Gain $PG_{eff}$ vs. $\\alpha$\n(at fixed range $d=15$ km)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("PBJ Bandwidth Occupancy Fraction $\\alpha$", fontsize=10)
    ax2.set_ylabel("Effective Processing Gain [dB] (Max: 19.03 dB)", fontsize=10)
    ax2.set_xlim([0.01, 1.0])
    ax2.set_ylim([-1, 20])
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower right", frameon=True)
    
    # Panel 3: Outage Probability vs. Follower Delay Ratio
    ax3 = axes[1, 0]
    for N in n_elements:
        ax3.plot(delays_ms, fol_outages[N], label=f"N={N}", linewidth=2)
    # Highlight delay regimes
    ax3.axvspan(0.0, 0.1, color='red', alpha=0.1, label='Ideal Reactive')
    ax3.axvspan(1.0, 2.0, color='orange', alpha=0.1, label='SDR Follower')
    ax3.axvspan(5.0, 10.0, color='yellow', alpha=0.1, label='Practical Implementation')
    ax3.axvspan(10.0, 12.0, color='green', alpha=0.1, label='Jammer Misses Hop')
    ax3.set_title("3. Link Outage Probability vs. Follower Jammer Reaction Delay\n(at fixed range $d=15$ km)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Jammer Reaction Delay $\\tau_{delay}$ [ms] (Hop Dwell $T_{hop}=10$ ms)", fontsize=10)
    ax3.set_ylabel("Outage Probability (BER > $3.37\\times 10^{-3}$)", fontsize=10)
    ax3.set_xlim([0.0, 12.0])
    ax3.set_ylim([-0.05, 1.05])
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(loc="lower left", frameon=True, fontsize=8)
    
    # Panel 4: Defeat Range vs. Follower Delay Ratio
    ax4 = axes[1, 1]
    for N in n_elements:
        d_x = [pt[0] for pt in fol_defeat_ranges[N]]
        d_y = [pt[1] for pt in fol_defeat_ranges[N]]
        ax4.plot(d_x, d_y, marker='o', label=f"N={N}", linewidth=2)
    ax4.axvspan(0.0, 0.1, color='red', alpha=0.1)
    ax4.axvspan(1.0, 2.0, color='orange', alpha=0.1)
    ax4.axvspan(5.0, 10.0, color='yellow', alpha=0.1)
    ax4.axvspan(10.0, 12.0, color='green', alpha=0.1)
    ax4.set_title("4. Defeat Boundary $R_{defeat}$ vs. Follower Jammer Reaction Delay\n(Outage Prob = 0.10 Threshold)", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Jammer Reaction Delay $\\tau_{delay}$ [ms]", fontsize=10)
    ax4.set_ylabel("Defeat Range $R_{defeat}$ [km]", fontsize=10)
    ax4.set_xlim([0.0, 12.0])
    ax4.grid(True, linestyle=":", alpha=0.6)
    ax4.legend(loc="upper left", frameon=True)
    
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), "phase_b_jamming_threats.png")
    plt.savefig(output_path, dpi=300)
    print(f"\n[SUCCESS] Completed sweeps and saved plots to: {output_path}")

if __name__ == "__main__":
    execute_threat_sweeps()
