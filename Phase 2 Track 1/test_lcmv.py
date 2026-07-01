import numpy as np
from pathlib import Path
import trimesh

from mesh_loader import load_uav_mesh
from conformal_array import get_conformal_array
from mesh_aware_lcmv import get_steering_vector, compute_reduced_lcmv

LAMBDA_M = 0.3

def main():
    mesh_path = Path(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone_prepared.stl")
    mesh = load_uav_mesh(mesh_path)
    antennas_body, normals = get_conformal_array(mesh)
    
    k = 2 * np.pi / LAMBDA_M
    
    # 16 active elements
    active_elements = np.ones(16, dtype=bool)
    
    # Fixed geometry: GCS directly below (0,0,-1), Jammer directly right (0,1,0)
    gcs_body = np.array([0.0, 0.0, -1.0])
    jam_body = np.array([0.0, 1.0, 0.0])
    
    # No phase noise for ideal case
    a_sig_true = get_steering_vector(antennas_body, normals, k * gcs_body, add_noise=False)
    a_jam_true = get_steering_vector(antennas_body, normals, k * jam_body, add_noise=False)
    
    w, sinr, nd = compute_reduced_lcmv(a_sig_true, a_jam_true, a_sig_true, a_jam_true, active_elements)
    
    print(f"Ideal 16-element case:")
    print(f"Null Depth: {nd:.2f} dB")
    print(f"SINR: {sinr:.2f} dB")
    
    # With Phase Noise
    a_sig_perc = get_steering_vector(antennas_body, normals, k * gcs_body, add_noise=True)
    a_jam_perc = get_steering_vector(antennas_body, normals, k * jam_body, add_noise=True)
    
    w_noisy, sinr_noisy, nd_noisy = compute_reduced_lcmv(a_sig_perc, a_jam_perc, a_sig_true, a_jam_true, active_elements)
    
    print(f"Noisy 16-element case:")
    print(f"Null Depth: {nd_noisy:.2f} dB")
    print(f"SINR: {sinr_noisy:.2f} dB")

if __name__ == "__main__":
    main()
