import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
import sys

sys.path.append(os.path.dirname(__file__))
import phase_b_beamforming as pbb

def q_func(x):
    return 1.0 - stats.norm.cdf(x)

def q_inv(x):
    return stats.norm.ppf(1.0 - x)

def compute_pd(tau_s, fs, p_fa, snr_lin):
    # Liang et al. formulation for Energy Detection
    N_samples = tau_s * fs
    if N_samples < 1:
        return 0.0
        
    thresh_term = q_inv(p_fa)
    # Pd = Q( ( Q^-1(Pfa) - sqrt(N)*SNR ) / sqrt(1 + 2*SNR) )
    # Assuming complex Gaussian signal and noise
    arg = (thresh_term - np.sqrt(N_samples) * snr_lin) / np.sqrt(1.0 + 2.0 * snr_lin)
    return q_func(arg)

def generate_received_signal(N, theta_j, INR_dB, L, rng=None):
    if rng is None: rng = np.random.default_rng()
    sigma_n2 = 1.0
    sigma_j2 = 10.0 ** (INR_dB / 10.0)
    j = rng.normal(0, np.sqrt(sigma_j2/2.0), L) + 1j * rng.normal(0, np.sqrt(sigma_j2/2.0), L)
    a_j = pbb.ula_steering_vector(N, theta_j)
    X_j = np.outer(a_j, j)
    noise = rng.normal(0, np.sqrt(sigma_n2/2.0), (N, L)) + 1j * rng.normal(0, np.sqrt(sigma_n2/2.0), (N, L))
    X = X_j + noise
    return (X @ X.conj().T) / max(L, 1)

def run_c3_sensing_tradeoff():
    print("Running Task C3: Cognitive Sensing Tradeoff...")
    
    # System parameters
    T_f = 10e-3       # 10 ms frame
    fs = 1e6          # 1 MHz sampling rate
    p_fa = 0.05       # 5% false alarm rate
    jnr_dB = -5.0     # Jammer is at -5dB per channel (requires integration to detect)
    jnr_lin = 10.0**(jnr_dB/10.0)
    
    L_max = 100       # Coherent snapshots available if 0 sensing time
    N_array = 8
    theta_s = 0.0
    theta_j = np.radians(30.0)
    
    tau_s_ms_sweep = np.linspace(0.1, 9.0, 15)
    
    throughput_list = []
    l_eff_list = []
    rmse_list = []
    depth_list = []
    
    print("| ts (ms) | Throughput Norm | P_D | L_eff | MUSIC RMSE (deg) | Null Depth (dB) |")
    print("| ------- | --------------- | --- | ----- | ---------------- | --------------- |")
    
    np.random.seed(42)
    
    for tau_s_ms in tau_s_ms_sweep:
        tau_s = tau_s_ms * 1e-3
        
        # 1. Sensing Performance
        pd = compute_pd(tau_s, fs, p_fa, jnr_lin)
        
        # Throughput = (1 - tau_s / T_f) * P_D
        # (Assuming we only successfully communicate if we detect the jammer and avoid it)
        throughput = (1.0 - tau_s / T_f) * pd
        
        # 2. Spatial Processing Penalty
        L_eff = max(int(L_max * (1.0 - tau_s / T_f)), 1)
        
        # Run Monte Carlo for MUSIC and LCMV
        trials = 200
        sq_err = []
        depths = []
        
        for _ in range(trials):
            # To isolate L_eff impact, we simulate just jammer + noise
            R_xx = generate_received_signal(N_array, theta_j, 10.0, L_eff) # Use 10dB INR for tracking
            scan_angles, P_mu = pbb.music_doa(R_xx, num_sources=1, scan_resolution_deg=0.5)
            peaks = pbb.find_music_peaks(scan_angles, P_mu, num_sources=1)
            
            if len(peaks) > 0:
                best_peak = peaks[np.argmin(np.abs(peaks - 30.0))]
                err = best_peak - 30.0
            else:
                best_peak = 90.0
                err = 60.0
                
            sq_err.append(err**2)
            
            # Beamforming Null Depth
            w = pbb.lcmv_beamformer(np.eye(N_array), theta_s, [np.radians(best_peak)])
            a_j_true = pbb.ula_steering_vector(N_array, theta_j)
            d = 10 * np.log10(np.abs(w.conj().T @ a_j_true)**2 + 1e-12)
            depths.append(d)
            
        rmse = np.sqrt(np.mean(sq_err))
        avg_depth = np.mean(depths)
        
        throughput_list.append(throughput)
        l_eff_list.append(L_eff)
        rmse_list.append(rmse)
        depth_list.append(avg_depth)
        
        print(f"| {tau_s_ms:4.1f}    | {throughput:14.3f}  | {pd:.2f}| {L_eff:4d}  | {rmse:13.2f}  | {avg_depth:14.1f}  |")

    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel('Sensing Time $\\tau_s$ (ms)')
    ax1.set_ylabel('Normalized Throughput', color='tab:blue')
    ax1.plot(tau_s_ms_sweep, throughput_list, 'b-o', linewidth=2, label='Throughput')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    
    # Identify optimum
    opt_idx = np.argmax(throughput_list)
    opt_tau = tau_s_ms_sweep[opt_idx]
    ax1.axvline(opt_tau, color='b', linestyle='--', alpha=0.5)

    ax2 = ax1.twinx()
    ax2.set_ylabel('LCMV Null Depth (dB)', color='tab:red')
    ax2.plot(tau_s_ms_sweep, depth_list, 'r-s', linewidth=2, label='Null Depth')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    fig.suptitle('Cognitive Sensing vs Spatial Processing Tradeoff', fontweight='bold')
    fig.tight_layout()
    plt.grid(True, alpha=0.3)
    plt.savefig('task_c3_sensing_tradeoff.png', dpi=300)
    print("\\nSaved tradeoff plot to task_c3_sensing_tradeoff.png")

if __name__ == "__main__":
    run_c3_sensing_tradeoff()
