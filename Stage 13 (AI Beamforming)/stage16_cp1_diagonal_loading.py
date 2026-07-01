import os
import sys
import numpy as np
from pathlib import Path

sys.path.append(os.path.abspath(r'D:\UAV Internship project\Phase 2 Track 1\em_realism'))
sys.path.append(os.path.abspath(r'D:\UAV Internship project\Phase 2 Track 2\array_structures'))
sys.path.append(os.path.abspath(r'D:\UAV Internship project\Stage 12 (Operational Analysis)'))
sys.path.append(os.path.abspath(r'D:\UAV Internship project\Stage 13 (AI Beamforming)'))

from mesh_loader import load_uav_mesh
from conformal_array import get_conformal_array_parametric
from lcmv_stage8 import get_steering_vector
from attitude import euler_to_quaternion, rotate_points
from shadow_engine_batched import compute_shadow_mask_batched

def main():
    print("Loading mesh...")
    MESH_PATH = Path(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone_prepared.stl")
    mesh = load_uav_mesh(MESH_PATH)
    pos_body, normals_body = get_conformal_array_parametric(mesh, N=32)
    
    # 720 evaluation angles on the 15 degree cone (0.5 deg resolution)
    N_POINTS = 720
    headings = np.linspace(0, 360, N_POINTS, endpoint=False)
    jam_bodies = np.zeros((N_POINTS, 3))
    
    q_inv = euler_to_quaternion(np.deg2rad(15.0), 0, 0).conjugate()
    
    for i, h in enumerate(headings):
        jam_world = np.array([np.cos(np.deg2rad(h)), np.sin(np.deg2rad(h)), 0.0])
        jam_bodies[i] = rotate_points(jam_world.reshape(1, 3), q_inv)[0]
        
    print(f"Pre-computing exact physics for {N_POINTS} points...")
    g_exact = np.zeros((N_POINTS, 32), dtype=np.complex128)
    chunk_size = 36
    for i in range(0, N_POINTS, chunk_size):
        chunk = jam_bodies[i:i+chunk_size]
        g_exact[i:i+chunk_size] = compute_shadow_mask_batched(mesh, pos_body, normals_body, chunk)
        
    # Signal definition
    sig_world = np.array([1.0, 0.0, 0.0])
    sig_body = rotate_points(sig_world.reshape(1, 3), q_inv)[0]
    K = 2.0 * np.pi / 0.15
    v_sig = get_steering_vector(pos_body, K * sig_body)
    
    P_S = 100.0
    P_J = 10000.0
    sigma2 = 1.0
    alpha = 390.0
    MSE = 0.004
    
    R_s = P_S * np.outer(v_sig, np.conj(v_sig))
    R_n = sigma2 * np.eye(32)
    
    min_sinr_overall = float('inf')
    
    print("\n--- Running Monte Carlo Diagonal Loading Test ---")
    for i in range(N_POINTS):
        v_true = g_exact[i] * get_steering_vector(pos_body, jam_bodies[i] * K)
        R_j = P_J * np.outer(v_true, np.conj(v_true))
        
        min_sinr_mc = float('inf')
        for mc in range(500):
            # Generate random complex perturbation with ||delta||^2 = MSE
            delta = np.random.randn(32) + 1j * np.random.randn(32)
            delta = delta / np.linalg.norm(delta) * np.sqrt(MSE)
            
            v_hat = v_true + delta
            
            # Diagonal Loading MVDR
            R_hat = P_J * np.outer(v_hat, np.conj(v_hat)) + (sigma2 + alpha) * np.eye(32)
            R_hat_inv = np.linalg.inv(R_hat)
            
            w = R_hat_inv @ v_sig
            w = w / np.real(np.conj(v_sig) @ R_hat_inv @ v_sig)
            
            # True evaluation
            S = np.real(np.conj(w) @ R_s @ w)
            NJ = np.real(np.conj(w) @ (R_j + R_n) @ w)
            sinr = 10 * np.log10(S / max(NJ, 1e-12))
            
            min_sinr_mc = min(min_sinr_mc, sinr)
            
        min_sinr_overall = min(min_sinr_overall, min_sinr_mc)
        if i % 72 == 0:
            print(f"Sweep Progress: {i/N_POINTS*100:.0f}%, Current Worst-Case: {min_sinr_overall:.2f} dB")
        
    print(f"\nWorst-case SINR over 360-deg sweep (500 Monte Carlo draws/angle): {min_sinr_overall:.2f} dB")

if __name__ == '__main__':
    main()
