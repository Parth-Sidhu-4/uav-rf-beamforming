import os
import sys
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path

sys.path.append(os.path.abspath(r'D:\UAV Internship project\Phase 2 Track 1\em_realism'))
sys.path.append(os.path.abspath(r'D:\UAV Internship project\Phase 2 Track 2\array_structures'))
sys.path.append(os.path.abspath(r'D:\UAV Internship project\Stage 12 (Operational Analysis)'))
sys.path.append(os.path.abspath(r'D:\UAV Internship project\Stage 13 (AI Beamforming)'))
sys.path.append(os.path.abspath(r'D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)'))

from mesh_loader import load_uav_mesh
from conformal_array import get_conformal_array_parametric
from lcmv_stage8 import get_steering_vector
from attitude import euler_to_quaternion, rotate_points
from shadow_engine_batched import compute_shadow_mask_batched
from stage18_d3_covariance_train import SIRENCovariancePredictor, d3_mvdr_beamformer, compute_sinr

def get_steering_vector_pt(pos_lambda, theta, phi):
    B = theta.shape[0]
    u = torch.stack([
        torch.sin(theta)*torch.cos(phi),
        torch.sin(theta)*torch.sin(phi),
        torch.cos(theta)
    ], dim=1)
    phase = 2.0 * np.pi * torch.matmul(pos_lambda, u.unsqueeze(-1)).squeeze(-1)
    return torch.exp(1j * phase)

def main():
    device = torch.device('cpu')
    
    print("Loading mesh and network for OOD Elevation Evaluation...")
    MESH_PATH = Path(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone_prepared.stl")
    mesh = load_uav_mesh(MESH_PATH)
    pos_body, normals_body = get_conformal_array_parametric(mesh, N=32)
    pos_lambda_np = pos_body / 0.15
    
    # Use the best K model (K=5)
    model = SIRENCovariancePredictor(w0=30.0, K_rank=5).to(device)
    model.load_state_dict(torch.load("siren_beamformer_d3_tether_K5.pt", map_location=device))
    model.eval()
    
    P_S = 100.0
    P_J = 10000.0
    sigma2 = 1.0
    alpha = 390.0
    K_wave = 2.0 * np.pi / 0.15
    
    q_inv = euler_to_quaternion(np.deg2rad(15.0), 0, 0).conjugate()
    sig_world = np.array([1.0, 0.0, 0.0])
    sig_body = rotate_points(sig_world.reshape(1, 3), q_inv)[0]
    
    g_sig = compute_shadow_mask_batched(mesh, pos_body, normals_body, sig_body.reshape(1, 3))[0]
    v_sig_ideal_np = get_steering_vector(pos_body, K_wave * sig_body)
    v_sig_masked_np = g_sig * v_sig_ideal_np
    v_sig = torch.tensor(v_sig_masked_np, dtype=torch.complex128, device=device)
    
    test_elevations = [10.0, 12.0, 18.0, 20.0]
    headings_full = np.linspace(0, 360, 360, endpoint=False) # 1 degree steps for faster eval
    
    print("\n--- OOD Elevation Sweep Results ---")
    pos_lambda = torch.tensor(pos_lambda_np, dtype=torch.float64, device=device)
    
    for test_elev in test_elevations:
        jam_bodies_list = []
        for h in headings_full:
            dist = min(h, 360 - h)
            if dist > 15.0:
                jam_world = np.array([
                    np.cos(np.deg2rad(h)) * np.cos(np.deg2rad(test_elev)),
                    np.sin(np.deg2rad(h)) * np.cos(np.deg2rad(test_elev)),
                    np.sin(np.deg2rad(test_elev))
                ])
                jam_bodies_list.append(rotate_points(jam_world.reshape(1, 3), q_inv)[0])
                
        if not jam_bodies_list:
            continue
            
        jam_bodies_np = np.array(jam_bodies_list)
        
        # Batch process physical masks
        print(f"\nProcessing Elevation {test_elev} deg (N={len(jam_bodies_np)} points)...")
        # Split into chunks to avoid memory issues with raytracer
        g_exact_list = []
        chunk_size = 50
        for i in tqdm(range(0, len(jam_bodies_np), chunk_size)):
            chunk = jam_bodies_np[i:i+chunk_size]
            g_chunk = compute_shadow_mask_batched(mesh, pos_body, normals_body, chunk)
            g_exact_list.append(g_chunk)
        g_exact_np = np.concatenate(g_exact_list, axis=0)
        
        jam_bodies_t = torch.tensor(jam_bodies_np, dtype=torch.float64, device=device)
        g_exact_t = torch.tensor(g_exact_np, dtype=torch.complex128, device=device)
        
        theta_j_all = torch.acos(jam_bodies_t[:, 2] / torch.norm(jam_bodies_t, dim=1))
        phi_j_all = torch.atan2(jam_bodies_t[:, 1], jam_bodies_t[:, 0])
        
        v_true_all = g_exact_t * get_steering_vector_pt(pos_lambda, theta_j_all, phi_j_all)
        R_j_all = P_J * torch.einsum('bi,bj->bij', v_true_all, torch.conj(v_true_all))
        R_s_all = P_S * torch.einsum('i,j->ij', v_sig, torch.conj(v_sig)).unsqueeze(0).expand(len(jam_bodies_np), -1, -1)
        R_n_all = sigma2 * torch.eye(32, dtype=torch.complex128, device=device).unsqueeze(0).expand(len(jam_bodies_np), -1, -1)
        
        with torch.no_grad():
            xyz_val = jam_bodies_t.float()
            U_val = model(xyz_val).to(torch.complex128)
            
            w_val = d3_mvdr_beamformer(U_val, v_sig, P_J, sigma2, alpha)
            sinr_val = compute_sinr(w_val, R_s_all, R_j_all, R_n_all)
            
            worst_sinr = torch.min(sinr_val).item()
            mean_sinr = torch.mean(sinr_val).item()
            
        print(f"Elevation {test_elev} deg -> Worst SINR: {worst_sinr:.2f} dB, Mean SINR: {mean_sinr:.2f} dB")

if __name__ == '__main__':
    main()
