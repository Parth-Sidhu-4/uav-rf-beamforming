import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Ensure imports work from the Phase 2 Track 1 and Stage 8 folders
sys.path.append(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)")
sys.path.append(r"D:\UAV Internship project\Phase 2 Track 1")

from mesh_loader import load_uav_mesh
from conformal_array import get_conformal_array
from mesh_aware_lcmv import get_steering_vector, compute_reduced_lcmv, LAMBDA_M

def main():
    mesh_path = Path(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone_prepared.stl")
    mesh = load_uav_mesh(mesh_path)
    antennas_body, normals = get_conformal_array(mesh)
    
    k = 2 * np.pi / LAMBDA_M
    N_ant = len(antennas_body)
    
    # Fixed geometry: GCS directly below, Jammer to the right
    gcs_body = np.array([0.0, 0.0, -1.0])
    jam_body = np.array([0.0, 1.0, 0.0])
    
    # Ideal (perceived) steering vectors
    a_sig_perc = get_steering_vector(antennas_body, normals, k * gcs_body, add_noise=False)
    a_jam_perc = get_steering_vector(antennas_body, normals, k * jam_body, add_noise=False)
    
    N_counts = np.arange(3, 17)
    sigmas = np.logspace(-3, 0, 15)  # 15 points from 0.001 to 1.0
    sigmas[10] = 0.1  # Force exact value 0.1 into the array for plotting label
    target_idx = 10
    num_trials = 5000
    num_batches = 5
    
    mean_nd_matrix = np.zeros((len(N_counts), len(sigmas)))
    std_nd_matrix = np.zeros((len(N_counts), len(sigmas)))
    mean_sinr_matrix = np.zeros((len(N_counts), len(sigmas)))
    se_nd_matrix = np.zeros((len(N_counts), len(sigmas)))
    
    print(f"Starting 2D Monte Carlo Sweep (N vs Sigma)...")
    print(f"Running {num_batches} batches of {num_trials} trials each.")
    
    for i, count in enumerate(N_counts):
        print(f"Running N = {count}...")
        active_elements = np.zeros(N_ant, dtype=bool)
        active_elements[:count] = True
        
        for j, sigma in enumerate(sigmas):
            batch_mean_nds = []
            batch_mean_sinrs = []
            
            for b in range(num_batches):
                # Set a unique seed per batch for reproducibility
                np.random.seed(42 + b * 1000 + i * 100 + j)
                
                nd_linear_trials = np.zeros(num_trials)
                sinr_linear_trials = np.zeros(num_trials)
                
                for t in range(num_trials):
                    phase_error = np.random.normal(0, sigma, N_ant)
                    phase_rotation = np.exp(1j * phase_error)
                    
                    a_sig_true = a_sig_perc * phase_rotation
                    a_jam_true = a_jam_perc * phase_rotation
                    
                    w, sinr_db, nd_db = compute_reduced_lcmv(
                        a_sig_perc, a_jam_perc,
                        a_sig_true, a_jam_true,
                        active_elements
                    )
                    
                    nd_linear_trials[t] = 10**(nd_db / 10.0)
                    sinr_linear_trials[t] = 10**(sinr_db / 10.0)
                    
                batch_mean_nds.append(10 * np.log10(np.mean(nd_linear_trials)))
                batch_mean_sinrs.append(10 * np.log10(np.mean(sinr_linear_trials)))
                
            mean_nd_matrix[i, j] = np.mean(batch_mean_nds)
            std_nd_matrix[i, j] = np.std(batch_mean_nds) # For error bands
            se_nd_matrix[i, j] = np.std(batch_mean_nds) / np.sqrt(num_batches)
            mean_sinr_matrix[i, j] = np.mean(batch_mean_sinrs)
            
    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Null Depth vs Sigma
    for i, count in enumerate(N_counts):
        if count in [3, 6, 8, 12, 16]:
            ax1.plot(sigmas, mean_nd_matrix[i, :], label=f'N={count}')
            if count == 16:
                ax1.fill_between(sigmas, 
                                 mean_nd_matrix[i, :] - std_nd_matrix[i, :],
                                 mean_nd_matrix[i, :] + std_nd_matrix[i, :],
                                 alpha=0.2, color='C4')
                
    # Theoretical approx
    approx_nd = 10 * np.log10(sigmas**2 + 1e-12)
    ax1.plot(sigmas, approx_nd, 'k--', linewidth=2, label='10*log10(σ²)')
    ax1.axvspan(0.5, 1.0, color='red', alpha=0.1, label='Small Angle Breakdown')
    
    ax1.set_xscale('log')
    ax1.set_xlabel('Phase Noise Std Dev σ (rad)')
    ax1.set_ylabel('Mean Null Depth (dB)')
    ax1.set_title('Null Depth vs Phase Noise (Linear Avg)')
    ax1.legend()
    
    ax1.grid(True)
    
    # Plot 2: Null Depth vs N at sigma = 0.1
    ax2.plot(N_counts, mean_nd_matrix[:, target_idx], 'b-o', label=f'Null Depth (σ={sigmas[target_idx]:.3f})')
    ax2.fill_between(N_counts,
                     mean_nd_matrix[:, target_idx] - se_nd_matrix[:, target_idx],
                     mean_nd_matrix[:, target_idx] + se_nd_matrix[:, target_idx],
                     alpha=0.3, color='blue', label='Standard Error')
    ax2.set_xlabel('Active Element Count (N)')
    ax2.set_ylabel('Mean Null Depth (dB)')
    ax2.set_title('Does N deepen the noise floor?')
    ax2.grid(True)
    
    ax2_sinr = ax2.twinx()
    ax2_sinr.plot(N_counts, mean_sinr_matrix[:, target_idx], 'g-s', label='SINR')
    ax2_sinr.set_ylabel('Mean SINR (dB)')
    
    lines_1, labels_1 = ax2.get_legend_handles_labels()
    lines_2, labels_2 = ax2_sinr.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right')
    
    fig.tight_layout()
    out_path = r"C:\Users\parth\.gemini\antigravity\brain\262e56d0-07ef-4f86-81fb-69ef858784e6\static_noise_scan_results.png"
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")
    
    # Output the required quantitative formula metrics
    idx_6 = np.where(N_counts == 6)[0][0]
    sinr_floor_6 = mean_sinr_matrix[idx_6, idx_01]
    nd_6 = mean_nd_matrix[idx_6, idx_01]
    
    print("\n--- Quantitative Threshold Results (sigma=0.1 rad, Linear Avg) ---")
    print(f"Legacy Cognitive (N=6): Null Depth = {nd_6:.2f} dB, SINR_floor = {sinr_floor_6:.2f} dB")
    for i, count in enumerate(N_counts):
        if count > 6:
            sinr_proj = mean_sinr_matrix[i, idx_01]
            nd_imp = mean_nd_matrix[i, idx_01] - nd_6
            se = se_nd_matrix[i, idx_01]
            print(f"Candidate N={count:2d} | Null Improv: {nd_imp:6.2f} dB (\u00B1{se:.2f} SE) | Projected SINR: {sinr_proj:6.2f} dB")

if __name__ == "__main__":
    main()
