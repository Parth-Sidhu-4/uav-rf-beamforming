"""Stage 2 Task B7: Adaptive Modulation & Coding (AMC) Link Simulation.

This script implements:
1. 4-mode AMC state machine with 2 dB hysteresis.
2. Comparative threat simulation under Barrage, PBJ (worst-case alpha = 0.11),
   and Follower (reaction delay = 5 ms) jamming.
3. Comparative Threat vs. Adaptive Mode Matrix at reference defeat range d = 2.49 km.
4. Publication-ready visualization (Link Reliability & Throughput vs. Distance).
"""

import os
import sys
import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt

# Ensure Stage 2 directory is in path for imports
sys.path.append(os.path.dirname(__file__))
import phase_b_beamforming as pbb
import task_b_jamming_threats as jts

# ==============================================================================
# 1. AMC CONFIGURATION & DERIVED THRESHOLDS
# ==============================================================================
# Target BER constraint: 3.37e-3
target_ber = 3.37e-3

# Mode set:
# M0: BPSK 1/2   (Threshold: 5.64 dB, Spectral Efficiency: 0.5)
# M1: QPSK 1/2   (Threshold: 8.65 dB, Spectral Efficiency: 1.0)
# M2: QPSK 3/4   (Threshold: 10.41 dB, Spectral Efficiency: 1.5 - Engineering Approx)
# M3: 16QAM 1/2  (Threshold: 15.33 dB, Spectral Efficiency: 2.0)
amc_thresholds_db = np.array([5.64, 8.65, 10.41, 15.33])
amc_thresholds_lin = 10.0 ** (amc_thresholds_db / 10.0)
amc_spectral_efficiencies = np.array([0.5, 1.0, 1.5, 2.0])

def Q(x):
    """Q-function using complementary error function."""
    return 0.5 * sp.erfc(x / np.sqrt(2.0))

def get_ber_for_mode(mode_idx, sinr_lin):
    """Compute analytical BER for a given mode at a given SINR (linear scale)."""
    sinr_lin = np.maximum(sinr_lin, 1e-15)
    
    if mode_idx == 0:    # BPSK 1/2
        return Q(np.sqrt(2.0 * sinr_lin))
    elif mode_idx == 1 or mode_idx == 2:  # QPSK 1/2 or QPSK 3/4
        return Q(np.sqrt(sinr_lin))
    elif mode_idx == 3:  # 16QAM 1/2
        return 0.75 * Q(np.sqrt(sinr_lin / 5.0))
    else:
        raise ValueError(f"Invalid AMC mode index: {mode_idx}")

def select_amc_mode_hysteresis(sinr_db, prev_mode, hyst_db=2.0):
    """Select the AMC mode index using a 2 dB hysteresis state machine."""
    if prev_mode == -1:
        # Outage / Initialization state: find highest mode where SINR >= threshold
        for idx in range(3, -1, -1):
            if sinr_db >= amc_thresholds_db[idx]:
                return idx
        return -1
    
    # Check UP transitions (requires SINR to meet the next threshold)
    highest_up = -1
    for idx in range(3, prev_mode, -1):
        if sinr_db >= amc_thresholds_db[idx]:
            highest_up = idx
            break
    if highest_up != -1:
        return highest_up
        
    # Check DOWN transitions (occurs only if SINR drops below threshold - hysteresis)
    if sinr_db < amc_thresholds_db[prev_mode] - hyst_db:
        for idx in range(prev_mode - 1, -1, -1):
            if sinr_db >= amc_thresholds_db[idx]:
                return idx
        return -1  # Outage
        
    # Maintain current mode
    return prev_mode

# ==============================================================================
# 2. CORE AMC LINK SIMULATOR
# ==============================================================================
def simulate_amc_hop_sequence(d_m, N, theta_s, theta_j, alpha, jammer_type='pbj', tau_delay=0.0, use_amc=True, trials=10000, seed=2026):
    """Simulate a sequential hop time-series at range d_m under fading and jamming."""
    # 1. Nominal received powers (without fast fading)
    Ps = jts.get_received_signal_power(d_m, 'two-ray')
    P_j_total = jts.get_received_jammer_power('fspl')
    
    if jammer_type == 'pbj':
        P_j_ch = P_j_total / (alpha * jts.N_h)
    else:
        P_j_ch = P_j_total
        
    SNR_lin = Ps / jts.P_N_lin
    INR_lin = P_j_ch / jts.P_N_lin
    
    # 2. Fading K-factor (desired: elevation-dependent Rician, jammer: Rayleigh)
    theta_elev = np.arctan(jts.h_UAV / d_m)
    K_s_db = 10.0 + 10.0 * np.sin(theta_elev)
    K_s = 10 ** (K_s_db / 10.0)
    
    # 3. Phased Array Calibration mismatch
    rng_cal = np.random.default_rng(42)
    amp_err = rng_cal.normal(0, jts.amp_err_std, N)
    amp_err = np.clip(amp_err, -0.9, 0.9)
    phase_err = rng_cal.normal(0, np.radians(jts.phase_err_std_deg), N)
    cal_err = (1.0 + amp_err) * np.exp(1j * phase_err)
    
    # Steering vectors
    a_s_nom = pbb.ula_steering_vector(N, theta_s)
    a_j_nom = pbb.ula_steering_vector(N, theta_j)
    a_s_true = a_s_nom * cal_err
    a_j_true = a_j_nom * cal_err
    
    # 4. LCMV Spatial nulling weights w
    if N > 1:
        R_xx_dl = SNR_lin * np.outer(a_s_nom, np.conj(a_s_nom)) + INR_lin * np.outer(a_j_nom, np.conj(a_j_nom)) + np.eye(N) + 1e-5 * np.eye(N)
        w = pbb.lcmv_beamformer(R_xx_dl, theta_s, [theta_j])
    else:
        w = np.array([1.0], dtype=complex)
        
    noise_out = np.sum(np.abs(w)**2)
    
    # 5. Temporal jamming fraction
    if jammer_type == 'pbj':
        p_jam = alpha
    elif jammer_type == 'follower':
        p_jam = np.max([0.0, 1.0 - tau_delay / jts.T_hop])
    else:  # barrage
        p_jam = 1.0
        
    # 6. Initialize mode selection
    # For AMC: initialize based on nominal jammed SINR
    sig_out_nom = SNR_lin * (np.abs(np.conj(w).T @ a_s_true) ** 2)
    jam_out_jammed_nom = INR_lin * (np.abs(np.conj(w).T @ a_j_true) ** 2)
    sinr_jammed_nom_db = 10.0 * np.log10(sig_out_nom / (noise_out + jam_out_jammed_nom))
    
    prev_mode = -1
    if use_amc:
        for idx in range(3, -1, -1):
            if sinr_jammed_nom_db >= amc_thresholds_db[idx]:
                prev_mode = idx
                break
    else:
        # Legacy Fixed BFSK: threshold is 10 dB
        prev_mode = 0  # We represent BFSK as mode 0 internally for structure, but with 10 dB threshold
        
    # 7. Time-series loop
    rng_fad = np.random.default_rng(seed)
    los_s = np.sqrt(K_s / (K_s + 1.0))
    diff_s = np.sqrt(1.0 / (K_s + 1.0))
    
    success_ber_count = 0
    success_sinr_count = 0
    total_throughput = 0.0
    total_ber = 0.0
    modes_selected = []
    
    for k in range(trials):
        # Desired signal fading
        phi_s = rng_fad.uniform(-np.pi, np.pi)
        u_s = rng_fad.normal(0, np.sqrt(0.5)) + 1j * rng_fad.normal(0, np.sqrt(0.5))
        h_s = los_s * np.exp(1j * phi_s) + diff_s * u_s
        
        # Jammer fading
        u_j = rng_fad.normal(0, np.sqrt(0.5)) + 1j * rng_fad.normal(0, np.sqrt(0.5))
        h_j = u_j
        
        # Instantaneous output power
        sig_out = SNR_lin * (np.abs(h_s)**2) * (np.abs(np.conj(w).T @ a_s_true) ** 2)
        jam_out_jammed = INR_lin * (np.abs(h_j)**2) * (np.abs(np.conj(w).T @ a_j_true) ** 2)
        
        # Instantaneous output SINRs
        sinr_jammed = sig_out / (noise_out + jam_out_jammed)
        sinr_unjammed = sig_out / noise_out
        
        sinr_jammed_db = 10.0 * np.log10(sinr_jammed)
        
        if use_amc:
            # Mode selection based on measured jammed SINR from previous hop
            mode = select_amc_mode_hysteresis(sinr_jammed_db, prev_mode, hyst_db=2.0)
            prev_mode = mode
        else:
            mode = 0  # Fixed BFSK
            
        if mode == -1:
            # Outage (no mode selected)
            total_ber += 0.5
            modes_selected.append(-1)
            continue
            
        # Compute hop BER
        if use_amc:
            # BPSK, QPSK, 16QAM BER formulas (mode indices map to BPSK, QPSK, QPSK, 16QAM)
            ber_jammed = get_ber_for_mode(mode, sinr_jammed)
            ber_unjammed = get_ber_for_mode(mode, sinr_unjammed)
        else:
            # Legacy Fixed BFSK BER formula
            ber_jammed = 0.5 * np.exp(-sinr_jammed / 2.0)
            ber_unjammed = 0.5 * np.exp(-sinr_unjammed / 2.0)
            
        if jammer_type == 'pbj':
            is_jammed = (rng_fad.random() < alpha)
            ber_hop = ber_jammed if is_jammed else ber_unjammed
        elif jammer_type == 'follower':
            ber_hop = p_jam * ber_jammed + (1.0 - p_jam) * ber_unjammed
        else:  # barrage
            ber_hop = ber_jammed
            
        # 1. BER-based outage
        ok_ber = (ber_hop <= target_ber)
        
        # 2. SINR-based outage
        if use_amc:
            # Successful if SINR is above the switch-down threshold (threshold - hysteresis)
            ok_sinr = (sinr_jammed_db >= amc_thresholds_db[mode] - 2.0)
        else:
            # Fixed BFSK has fixed 10 dB threshold
            ok_sinr = (sinr_jammed_db >= 10.0)
            
        if ok_ber:
            success_ber_count += 1
        if ok_sinr:
            success_sinr_count += 1
            
        total_ber += ber_hop
        modes_selected.append(mode)
        
        # Throughput calculations (in bit/s/Hz)
        if use_amc:
            eff_rate = amc_spectral_efficiencies[mode]
        else:
            eff_rate = 0.5 # Fixed BFSK efficiency
            
        total_throughput += eff_rate * (1.0 - ber_hop) if ok_ber else 0.0
        
    p_link_ber = success_ber_count / trials
    p_link_sinr = success_sinr_count / trials
    avg_throughput = total_throughput / trials
    avg_ber = total_ber / trials
    
    return p_link_ber, p_link_sinr, avg_throughput, avg_ber, modes_selected

# ==============================================================================
# 3. RUN SPRINT 3 COMPARISON STUDY
# ==============================================================================
def run_amc_comparison():
    print("=" * 80)
    print("         PHASE B SPRINT 3: ADAPTIVE MODULATION & CODING LINK RESILIENCE")
    print("=" * 80)
    
    d_ref_m = 2490.0  # Reference defeat range
    N = 8
    theta_s = np.radians(0.0)
    theta_j = np.radians(30.0)
    
    threats = [
        ('Barrage', 1.0, 'pbj', 0.0),
        ('PBJ', 0.11, 'pbj', 0.0),
        ('Follower', 1.0, 'follower', 5e-3)
    ]
    
    results = {}
    
    print(f"\n--- Simulating GCS Link at d = {d_ref_m/1000.0:.3f} km (10,000 Hops) ---")
    
    for name, alpha, j_type, delay in threats:
        print(f"Running Threat: {name}...")
        # Fixed BFSK
        p_link_b, p_link_s, r_eff, avg_ber, _ = simulate_amc_hop_sequence(
            d_ref_m, N, theta_s, theta_j, alpha, j_type, delay, use_amc=False, trials=10000
        )
        results[(name, 'Fixed')] = {
            'p_link_ber': p_link_b, 'p_link_sinr': p_link_s, 'r_eff': r_eff, 'ber': avg_ber
        }
        
        # AMC
        p_link_b_amc, p_link_s_amc, r_eff_amc, avg_ber_amc, modes = simulate_amc_hop_sequence(
            d_ref_m, N, theta_s, theta_j, alpha, j_type, delay, use_amc=True, trials=10000
        )
        results[(name, 'AMC')] = {
            'p_link_ber': p_link_b_amc, 'p_link_sinr': p_link_s_amc, 'r_eff': r_eff_amc, 'ber': avg_ber_amc, 'modes': modes
        }
        
    # Print results table
    print("\n" + "=" * 110)
    print("                     THREAT VS. ADAPTIVE MODE PERFORMANCE MATRIX (d = 2.49 km)")
    print("=" * 110)
    print(f"{'Threat':<10} | {'Scheme':<10} | {'P_link (BER)':^14} | {'P_link (SINR)':^14} | {'P_out (BER)':^14} | {'Throughput':^16} | {'Avg BER':^12}")
    print("-" * 110)
    for name, _, _, _ in threats:
        res_fs = results[(name, 'Fixed')]
        res_amc = results[(name, 'AMC')]
        
        p_out_fs = 1.0 - res_fs['p_link_ber']
        p_out_amc = 1.0 - res_amc['p_link_ber']
        
        print(f"{name:<10} | {'Fixed':<10} | {res_fs['p_link_ber']*100.0:12.2f}% | {res_fs['p_link_sinr']*100.0:12.2f}% | {p_out_fs*100.0:12.2f}% | {res_fs['r_eff']:11.4f} b/s/Hz | {res_fs['ber']:^12.4e}")
        print(f"{'':<10} | {'AMC':<10} | {res_amc['p_link_ber']*100.0:12.2f}% | {res_amc['p_link_sinr']*100.0:12.2f}% | {p_out_amc*100.0:12.2f}% | {res_amc['r_eff']:11.4f} b/s/Hz | {res_amc['ber']:^12.4e}")
        print("-" * 110)
    print("=" * 110)

    # ==============================================================================
    # 4. GENERATE PLOTS (RELIABILITY & THROUGHPUT VS DISTANCE)
    # ==============================================================================
    print("\n--- Running Distance Sweep for Follower Jammer Plot ---")
    distances = np.linspace(100.0, 15000.0, 30)
    
    fs_reliability = []
    fs_throughput = []
    amc_reliability = []
    amc_throughput = []
    
    # Extract follower parameters
    _, alpha_f, j_type_f, delay_f = threats[2]
    
    for dist in distances:
        # Fixed
        p_b, _, r_eff, _, _ = simulate_amc_hop_sequence(dist, N, theta_s, theta_j, alpha_f, j_type_f, delay_f, use_amc=False, trials=2000)
        fs_reliability.append(p_b)
        fs_throughput.append(r_eff)
        
        # AMC
        p_b_amc, _, r_eff_amc, _, _ = simulate_amc_hop_sequence(dist, N, theta_s, theta_j, alpha_f, j_type_f, delay_f, use_amc=True, trials=2000)
        amc_reliability.append(p_b_amc)
        amc_throughput.append(r_eff_amc)
        
    # Generate 2-panel plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Panel 1: Reliability vs Distance
    ax1.plot(distances / 1000.0, np.array(fs_reliability) * 100.0, label="Fixed BFSK", color="blue", linewidth=2.5)
    ax1.plot(distances / 1000.0, np.array(amc_reliability) * 100.0, label="AMC (BPSK/QPSK/16QAM)", color="green", linewidth=2.5)
    ax1.axvline(d_ref_m / 1000.0, color="red", linestyle="--", label="Reference Defeat Range (2.49 km)")
    ax1.set_title("1. Link Reliability vs. Distance\n(Follower Jammer, 5 ms delay)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Link Range [km]", fontsize=10)
    ax1.set_ylabel("Link Reliability (P_link) [%]", fontsize=10)
    ax1.set_ylim([-5, 105])
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower left", frameon=True)
    
    # Panel 2: Throughput vs Distance
    ax2.plot(distances / 1000.0, fs_throughput, label="Fixed BFSK", color="blue", linewidth=2.5)
    ax2.plot(distances / 1000.0, amc_throughput, label="AMC (BPSK/QPSK/16QAM)", color="green", linewidth=2.5)
    ax2.axvline(d_ref_m / 1000.0, color="red", linestyle="--")
    ax2.set_title("2. Effective Link Throughput vs. Distance\n(Follower Jammer, 5 ms delay)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Link Range [km]", fontsize=10)
    ax2.set_ylabel("Effective Throughput [bit/s/Hz]", fontsize=10)
    ax2.set_ylim([-0.1, 2.1])
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", frameon=True)
    
    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), "phase_b_amc_results.png")
    plt.savefig(plot_path, dpi=300)
    print(f"\n[SUCCESS] Completed AMC simulation and saved plots to: {plot_path}")
    
    return results

if __name__ == "__main__":
    run_amc_comparison()
