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

def get_rank_k_covariance(theta_j, phi_j, pos_lambda, v_hat, P_J, K=7, delta_max=0.035, sigma2=1.0, alpha=390.0):
    N = pos_lambda.shape[0]
    psi = np.linspace(0, 2*np.pi, K-1, endpoint=False)
    delta = np.zeros(K)
    delta[1:] = delta_max # 1 center point, 6 points on the ring
    
    psi = np.concatenate(([0], psi))
    
    R_interf = np.zeros((N, N), dtype=complex)
    
    u_center = np.array([np.sin(theta_j)*np.cos(phi_j),
                         np.sin(theta_j)*np.sin(phi_j),
                         np.cos(theta_j)])
    
    for k in range(K):
        th_k = theta_j + delta[k] * np.cos(psi[k])
        ph_k = phi_j   + delta[k] * np.sin(psi[k])
        
        u_k = np.array([np.sin(th_k)*np.cos(ph_k),
                        np.sin(th_k)*np.sin(ph_k),
                        np.cos(th_k)])
        
        # Perturb v_hat by the relative phase difference between u_k and u_center
        phase_diff = 2j * np.pi * (pos_lambda @ (u_k - u_center))
        v_k = v_hat * np.exp(phase_diff)
        
        R_interf += (P_J / K) * np.outer(v_k, np.conj(v_k))
        
    return R_interf + (sigma2 + alpha) * np.eye(N, dtype=complex)

def main():
    print("Loading mesh...")
    MESH_PATH = Path(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone_prepared.stl")
    mesh = load_uav_mesh(MESH_PATH)
    pos_body, normals_body = get_conformal_array_parametric(mesh, N=32)
    pos_lambda = pos_body / 0.15 # 0.15m is wavelength at 2.4 GHz
    
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
    K_wave = 2.0 * np.pi / 0.15
    v_sig = get_steering_vector(pos_body, K_wave * sig_body)
    
    P_S = 100.0
    P_J = 10000.0
    sigma2 = 1.0
    alpha = 390.0
    MSE = 0.004
    
    R_s = P_S * np.outer(v_sig, np.conj(v_sig))
    R_n = sigma2 * np.eye(32)
    
    min_sinr_overall = float('inf')
    
    print("\n--- Running Monte Carlo DCMVDR Test ---")
    for i in range(N_POINTS):
        v_true = g_exact[i] * get_steering_vector(pos_body, jam_bodies[i] * K_wave)
        R_j = P_J * np.outer(v_true, np.conj(v_true))
        
        # Get spherical coords for the jammer in body frame
        jb = jam_bodies[i]
        r = np.linalg.norm(jb)
        theta_j = np.arccos(jb[2] / r)
        phi_j = np.arctan2(jb[1], jb[0])
        
        # Analytical derivatives of the spatial phase
        du_dth = np.array([np.cos(theta_j)*np.cos(phi_j),
                           np.cos(theta_j)*np.sin(phi_j),
                          -np.sin(theta_j)])
        du_dph = np.array([-np.sin(theta_j)*np.sin(phi_j),
                            np.sin(theta_j)*np.cos(phi_j),
                            0.0])
        
        min_sinr_mc = float('inf')
        for mc in range(500):
            delta = np.random.randn(32) + 1j * np.random.randn(32)
            delta = delta / np.linalg.norm(delta) * np.sqrt(MSE)
            
            v_hat = v_true + delta
            
            R_hat = get_rank_k_covariance(theta_j, phi_j, pos_lambda, v_hat, P_J, K=7, delta_max=0.035, sigma2=sigma2, alpha=alpha)
            R_hat_inv = np.linalg.inv(R_hat)
            
            # Derivative constraints on v_hat
            dv_j_dth  = 2j * np.pi * (pos_lambda @ du_dth) * v_hat
            dv_j_dph  = 2j * np.pi * (pos_lambda @ du_dph) * v_hat
            
            # C = [v_sig | v_hat | dv_hat_dth | dv_hat_dph]
            C = np.column_stack([v_sig, v_hat, dv_j_dth, dv_j_dph])
            f = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
            
            R_inv_C = R_hat_inv @ C
            # Small regularizer on inner matrix to avoid singularity
            CRC = C.conj().T @ R_inv_C
            CRC += 1e-10 * np.eye(4)
            CRC_inv  = np.linalg.inv(CRC)
            
            w = R_inv_C @ (CRC_inv @ f)
            
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
