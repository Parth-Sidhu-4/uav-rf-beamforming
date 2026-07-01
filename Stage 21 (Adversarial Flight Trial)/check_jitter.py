import numpy as np
import trimesh
import sys, os
from pathlib import Path

sys.path.append(os.path.abspath(r'D:\UAV Internship project\Stage 12 (Operational Analysis)'))
sys.path.append(os.path.abspath(r'D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)'))
sys.path.append(os.path.abspath(r'D:\UAV Internship project\Stage 13 (AI Beamforming)'))

from stage21_adversarial_flight_trial import FallbackCache
from shadow_engine_batched import compute_shadow_mask_batched
from conformal_array import get_conformal_array_parametric
from mesh_loader import load_uav_mesh

physics = FallbackCache('true_raytraced_grid.npz')
mesh = load_uav_mesh(Path(r'D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone_prepared.stl'))
pos_body, normals_body = get_conformal_array_parametric(mesh, N=32)
pos_lambda = pos_body / 0.125

target_h = 180.0
target_e = 0.0

# Base grid node
v_grid = physics.get_ground_truth(np.array([np.radians(target_h)]), np.array([np.radians(target_e)]))[0]

jitters = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

print("=== SENSITIVITY TO JITTER AROUND (180, 0) ===")
for j in jitters:
    # Add jitter to azimuth
    az = np.radians(target_h + j)
    el = np.radians(target_e + j)
    
    dx = np.sin(np.pi/2 - el) * np.cos(az)
    dy = np.sin(np.pi/2 - el) * np.sin(az)
    dz = np.cos(np.pi/2 - el)
    jam_body = np.array([[dx, dy, dz]])
    
    g_batch = compute_shadow_mask_batched(mesh, pos_body, normals_body, jam_body)
    v_ideal = np.exp(1j * 2.0 * np.pi * (jam_body[0] @ pos_lambda.T))
    v_cont = g_batch[0] * v_ideal
    
    # Compute MVDR weights using the grid vector (Nearest Neighbor)
    P_J = 10000.0
    sigma2 = 1.0
    R_grid = P_J * np.outer(v_grid, v_grid.conj()) + (sigma2 + 390.0) * np.eye(32)
    
    # Target signal is at (0, 0)
    sig_az = 0.0
    sig_el = 0.0
    sig_dx = np.sin(np.pi/2 - sig_el) * np.cos(sig_az)
    sig_dy = np.sin(np.pi/2 - sig_el) * np.sin(sig_az)
    sig_dz = np.cos(np.pi/2 - sig_el)
    v_sig = np.exp(1j * 2.0 * np.pi * (np.array([sig_dx, sig_dy, sig_dz]) @ pos_lambda.T))
    
    w = np.linalg.solve(R_grid, v_sig)
    w = w / (v_sig.conj() @ w)
    
    # Evaluate leakage using the TRUE continuous vector
    w_H_v_cont = np.abs(np.vdot(w, v_cont))**2
    leakage = P_J * w_H_v_cont
    
    # Compute SINR
    sig_power = 100.0 * np.abs(np.vdot(w, v_sig))**2
    noise_power = sigma2 * np.linalg.norm(w)**2
    sinr = sig_power / (leakage + noise_power)
    sinr_db = 10 * np.log10(sinr)
    
    print(f"Jitter {j:5.2f} deg -> Jammer Leakage: {leakage:7.2f}, SINR: {sinr_db:6.2f} dB, Null Depth: {w_H_v_cont:.2e}")
