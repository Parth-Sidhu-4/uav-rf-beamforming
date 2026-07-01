import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Ensure imports work from the Phase 2 Track 1 folder
sys.path.append(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)")

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
    
    a_sig = get_steering_vector(antennas_body, normals, k * gcs_body, add_noise=False)
    a_jam = get_steering_vector(antennas_body, normals, k * jam_body, add_noise=False)
    
    active_counts = np.arange(16, 0, -1)
    null_depths = []
    sinrs = []
    
    for count in active_counts:
        # Mask out elements to simulate occlusion, keeping the first `count` elements
        active_elements = np.zeros(N_ant, dtype=bool)
        active_elements[:count] = True
        
        # We need at least 1 active element for MF, and at least 3 for LCMV (2 constraints + 1 DOF)
        if count == 0:
            null_depths.append(0.0)
            sinrs.append(0.0)
            continue
            
        w, sinr, nd = compute_reduced_lcmv(a_sig, a_jam, a_sig, a_jam, active_elements)
        null_depths.append(nd)
        sinrs.append(sinr)
        
    # Plot results
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    ax1.plot(active_counts, null_depths, 'b-o', label='Null Depth')
    ax1.set_xlabel('Active Elements')
    ax1.set_ylabel('Null Depth (dB)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_ylim(-350, 10)
    ax1.axhline(y=-40, color='purple', linestyle=':', label='Target Null Depth')
    ax1.grid(True)
    
    ax2 = ax1.twinx()
    ax2.plot(active_counts, sinrs, 'g-s', label='SINR')
    ax2.set_ylabel('SINR (dB)', color='g')
    ax2.tick_params(axis='y', labelcolor='g')
    
    plt.title('Static Beamformer Performance vs Active Elements (Ideal Noiseless)')
    fig.tight_layout()
    plt.savefig('static_dof_sweep.png')
    print("Saved sweep plot to static_dof_sweep.png")
    
    for c, nd, s in zip(active_counts, null_depths, sinrs):
        print(f"Elements: {c:2d} | Null Depth: {nd:8.2f} dB | SINR: {s:6.2f} dB")

if __name__ == "__main__":
    main()
