import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

from mesh_loader import load_uav_mesh
from conformal_array import get_conformal_array, get_conformal_array_parametric
from attitude import euler_to_quaternion, rotate_points
from shadow_engine import compute_shadow_mask
from lcmv_stage8 import compute_lcmv_weights, get_steering_vector

def main():
    logging.basicConfig(level=logging.INFO)
    mesh_path = Path(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone_prepared.stl")
    mesh = load_uav_mesh(mesh_path)
    
    # Baseline Check
    baseline_antennas, _ = get_conformal_array(mesh)
    parametric_16, _ = get_conformal_array_parametric(mesh, 16)
    assert len(baseline_antennas) == len(parametric_16), "Zone mismatch"
    print(f"Baseline vs parametric L2 distance: "
          f"{np.mean(np.linalg.norm(baseline_antennas - parametric_16, axis=1)):.4f}m")
    
    freq, c = 2.4e9, 3e8
    k = 2 * np.pi / (c / freq)
    
    jammer_world_dir = np.array([0.0, 1.0, 0.0])
    signal_world_dir = np.array([1.0, 0.0, 0.0])
    
    noise_power = 1.0
    signal_power = noise_power * (10 ** (20 / 10))
    jammer_power = noise_power * (10 ** (40 / 10))
    
    fixed_roll = 15.0
    q = euler_to_quaternion(np.deg2rad(fixed_roll), 0.0, 0.0)
    q_inv = q.conjugate()
    
    jammer_body_dir = rotate_points(jammer_world_dir.reshape(1, 3), q_inv)[0]
    signal_body_dir = rotate_points(signal_world_dir.reshape(1, 3), q_inv)[0]
    
    N_values = [4, 8, 12, 16, 24, 32]
    sinrs = []
    actives = []
    
    for N in N_values:
        logging.info(f"Running sweep for N={N}...")
        antennas_body, normals = get_conformal_array_parametric(mesh, N)
        
        mask = compute_shadow_mask(mesh, antennas_body, normals, jammer_body_dir)
        active_count = np.sum(mask)
        actives.append(active_count)
        
        if active_count < 3:
            sinrs.append(np.nan)
            continue
            
        k_sig, k_jam = k * signal_body_dir, k * jammer_body_dir
        a_sig = get_steering_vector(antennas_body, k_sig)
        a_jam = get_steering_vector(antennas_body, k_jam)
        
        R_xx = (signal_power * np.outer(a_sig, np.conj(a_sig)) + 
                jammer_power * np.outer(a_jam, np.conj(a_jam)) + 
                noise_power * np.eye(N))
        
        w = compute_lcmv_weights(R_xx, a_sig, mask)
        
        P_s = signal_power * np.abs(np.conj(w).T @ a_sig)**2
        P_j = jammer_power * np.abs(np.conj(w).T @ a_jam)**2
        P_n = noise_power * np.linalg.norm(w)**2
        
        if (P_j + P_n) < 1e-12 or P_s < 1e-12:
            sinrs.append(0.0)
        else:
            sinrs.append(10 * np.log10(P_s / (P_j + P_n)))

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    x_positions = range(len(N_values))
    
    ax1.plot(x_positions, sinrs, 'b-o', linewidth=2)
    ax1.set_ylabel('LCMV SINR (dB)')
    ax1.set_title(f'Sweep 4: Hardware Cost of Resilience at {fixed_roll}° Roll')
    ax1.grid(True)
    ax1.text(0.02, 0.95, "N=4 is placement-dependent (shown for uniform distribution)", 
             transform=ax1.transAxes, fontsize=10, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax2.plot(x_positions, actives, 'g-o', linewidth=2)
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels([str(n) for n in N_values])
    ax2.set_xlabel('Number of Elements (N)')
    ax2.set_ylabel('Active Element Count')
    ax2.axhline(3, color='orange', linestyle='--', label='Minimum for beamformer validity')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(Path(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\sweep4_results.png"))
    logging.info("Saved plot to sweep4_results.png")

if __name__ == "__main__":
    main()
