import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.append(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)")
sys.path.append(r"D:\UAV Internship project\Phase 2 Track 1")

from mesh_loader import load_uav_mesh
from conformal_array import get_conformal_array
from mesh_aware_lcmv import get_steering_vector, compute_reduced_lcmv, LAMBDA_M, P_JAM

def main():
    mesh_path = Path(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone_prepared.stl")
    mesh = load_uav_mesh(mesh_path)
    antennas_body, normals = get_conformal_array(mesh)
    
    k = 2 * np.pi / LAMBDA_M
    N_ant = len(antennas_body)
    
    gcs_body = np.array([0.0, 0.0, -1.0])
    jam_body = np.array([0.0, 1.0, 0.0])
    
    a_sig_perc = get_steering_vector(antennas_body, normals, k * gcs_body, add_noise=False)
    a_jam_perc = get_steering_vector(antennas_body, normals, k * jam_body, add_noise=False)
    
    count = 10
    sigma = 0.1
    num_trials = 5000
    
    active_elements = np.zeros(N_ant, dtype=bool)
    active_elements[:count] = True
    
    nd_db_trials = []
    nd_linear_trials = []
    
    print(f"Running {num_trials} trials for N={count}, sigma={sigma}...")
    for _ in range(num_trials):
        phase_error = np.random.normal(0, sigma, N_ant)
        phase_rotation = np.exp(1j * phase_error)
        
        a_sig_true = a_sig_perc * phase_rotation
        a_jam_true = a_jam_perc * phase_rotation
        
        w, sinr, nd_db = compute_reduced_lcmv(
            a_sig_perc, a_jam_perc,
            a_sig_true, a_jam_true,
            active_elements
        )
        
        # Recalculate linear leakage from dB (nd_db = 10*log10(leakage))
        nd_linear = 10**(nd_db / 10.0)
        
        nd_db_trials.append(nd_db)
        nd_linear_trials.append(nd_linear)
        
    nd_db_trials = np.array(nd_db_trials)
    nd_linear_trials = np.array(nd_linear_trials)
    
    mean_db = np.mean(nd_db_trials)
    mean_linear = np.mean(nd_linear_trials)
    db_of_mean_linear = 10 * np.log10(mean_linear)
    median_db = np.median(nd_db_trials)
    
    print(f"Mean of dB: {mean_db:.2f} dB")
    print(f"Median of dB: {median_db:.2f} dB")
    print(f"10*log10(Mean of Linear): {db_of_mean_linear:.2f} dB")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.hist(nd_linear_trials, bins=50, color='blue', alpha=0.7)
    ax1.axvline(mean_linear, color='red', linestyle='dashed', linewidth=2, label=f'Mean = {mean_linear:.2e}')
    ax1.set_title('Histogram of Linear Leakage Power')
    ax1.set_xlabel('Linear Power')
    ax1.legend()
    
    ax2.hist(nd_db_trials, bins=50, color='green', alpha=0.7)
    ax2.axvline(mean_db, color='red', linestyle='dashed', linewidth=2, label=f'Mean = {mean_db:.2f} dB')
    ax2.axvline(db_of_mean_linear, color='purple', linestyle='dashed', linewidth=2, label=f'10*log10(Mean_Lin) = {db_of_mean_linear:.2f} dB')
    ax2.axvline(median_db, color='black', linestyle='dashed', linewidth=2, label=f'Median = {median_db:.2f} dB')
    ax2.set_title('Histogram of Null Depth (dB)')
    ax2.set_xlabel('Null Depth (dB)')
    ax2.legend()
    
    plt.tight_layout()
    out_path = r"C:\Users\parth\.gemini\antigravity\brain\262e56d0-07ef-4f86-81fb-69ef858784e6\diagnostic_histogram.png"
    plt.savefig(out_path)
    print(f"Saved histogram to {out_path}")

if __name__ == "__main__":
    main()
