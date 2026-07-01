import numpy as np
import trimesh
import sys, os

sys.path.append(os.path.abspath(r'D:\UAV Internship project\Stage 12 (Operational Analysis)'))
sys.path.append(os.path.abspath(r'D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)'))
sys.path.append(os.path.abspath(r'D:\UAV Internship project\Stage 13 (AI Beamforming)'))

from stage21_adversarial_flight_trial import FallbackCache
from shadow_engine_batched import compute_shadow_mask_batched
from conformal_array import get_conformal_array_parametric
from mesh_loader import load_uav_mesh

physics = FallbackCache('true_raytraced_grid.npz')

# 1. Get from grid
h_deg = 180.0
e_deg = 0.0
v_grid = physics.get_ground_truth(np.array([np.radians(h_deg)]), np.array([np.radians(e_deg)]))[0]

# 2. Get from continuous
from pathlib import Path
mesh = load_uav_mesh(Path(r'D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone_prepared.stl'))
pos_body, normals_body = get_conformal_array_parametric(mesh, N=32)
pos_lambda = pos_body / 0.125

az = np.radians(h_deg)
el = np.radians(e_deg)
dx = np.sin(np.pi/2 - el) * np.cos(az)
dy = np.sin(np.pi/2 - el) * np.sin(az)
dz = np.cos(np.pi/2 - el)
jam_body = np.array([[dx, dy, dz]])

g_batch = compute_shadow_mask_batched(mesh, pos_body, normals_body, jam_body)
v_ideal = np.exp(1j * 2.0 * np.pi * (jam_body[0] @ pos_lambda.T))
v_continuous = g_batch[0] * v_ideal

print("v_grid (first 5 elements):", v_grid[:5])
print("v_cont (first 5 elements):", v_continuous[:5])

diff = np.abs(v_grid - v_continuous)
print("Max absolute difference:", np.max(diff))
print("Mean absolute difference:", np.mean(diff))

# What happens to SINR if we use v_grid to null v_cont?
P_J = 10000.0
sigma2 = 1.0
R_grid = P_J * np.outer(v_grid, v_grid.conj()) + (sigma2 + 390.0) * np.eye(32)

sig_az = np.radians(0)
sig_el = np.radians(0)
sig_dx = np.sin(np.pi/2 - sig_el) * np.cos(sig_az)
sig_dy = np.sin(np.pi/2 - sig_el) * np.sin(sig_az)
sig_dz = np.cos(np.pi/2 - sig_el)
v_sig_ideal = np.exp(1j * 2.0 * np.pi * (np.array([sig_dx, sig_dy, sig_dz]) @ pos_lambda.T))
# assume open space for signal
w = np.linalg.solve(R_grid, v_sig_ideal)
w = w / (v_sig_ideal.conj() @ w)

w_H_v_cont = np.abs(np.vdot(w, v_continuous))**2
w_H_v_grid = np.abs(np.vdot(w, v_grid))**2

print(f"Jammer Leakage (using v_cont): {P_J * w_H_v_cont:.2f}")
print(f"Jammer Leakage (using v_grid): {P_J * w_H_v_grid:.2f}")
