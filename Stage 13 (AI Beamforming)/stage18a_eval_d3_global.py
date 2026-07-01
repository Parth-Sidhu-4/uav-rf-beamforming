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
    
    print("Loading mesh and network...")
    MESH_PATH = Path(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone_prepared.stl")
    mesh = load_uav_mesh(MESH_PATH)
    pos_body, normals_body = get_conformal_array_parametric(mesh, N=32)
    pos_lambda_np = pos_body / 0.15
    
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
    
    print("Computing mask for signal...")
    g_sig = compute_shadow_mask_batched(mesh, pos_body, normals_body, sig_body.reshape(1, 3))[0]
    v_sig_ideal_np = get_steering_vector(pos_body, K_wave * sig_body)
    v_sig_masked_np = g_sig * v_sig_ideal_np
    v_sig = torch.tensor(v_sig_masked_np, dtype=torch.complex128, device=device)
    
    print("Loading precomputed exact physics for Global Sweep...")
    dataset_path = "dataset_3600_masks.npz"
    data = np.load(dataset_path)
    g_exact_full = data['g_exact']
    
    headings_full = np.linspace(0, 360, 3600, endpoint=False)
    jam_bodies_full = np.zeros((3600, 3))
    for i, h in enumerate(headings_full):
        jam_world = np.array([np.cos(np.deg2rad(h)), np.sin(np.deg2rad(h)), 0.0])
        jam_bodies_full[i] = rotate_points(jam_world.reshape(1, 3), q_inv)[0]
        
    pos_lambda = torch.tensor(pos_lambda_np, dtype=torch.float64, device=device)
    jam_bodies_t = torch.tensor(jam_bodies_full, dtype=torch.float64, device=device)
    g_exact_t = torch.tensor(g_exact_full, dtype=torch.complex128, device=device)
    
    theta_j_all = torch.acos(jam_bodies_t[:, 2] / torch.norm(jam_bodies_t, dim=1))
    phi_j_all = torch.atan2(jam_bodies_t[:, 1], jam_bodies_t[:, 0])
    
    v_true_all = g_exact_t * get_steering_vector_pt(pos_lambda, theta_j_all, phi_j_all)
    R_j_all = P_J * torch.einsum('bi,bj->bij', v_true_all, torch.conj(v_true_all))
    R_s_all = P_S * torch.einsum('i,j->ij', v_sig, torch.conj(v_sig)).unsqueeze(0).expand(3600, -1, -1)
    R_n_all = sigma2 * torch.eye(32, dtype=torch.complex128, device=device).unsqueeze(0).expand(3600, -1, -1)
    
    print("\n--- Part 1: Global Sweep (> 15 deg separation) ---")
    valid_global = []
    for i, h in enumerate(headings_full):
        dist = min(h, 360 - h)
        if dist > 15.0:
            valid_global.append(i)
            
    with torch.no_grad():
        xyz_val = jam_bodies_t[valid_global].float()
        U_val = model(xyz_val).to(torch.complex128)
        
        w_val = d3_mvdr_beamformer(U_val, v_sig, P_J, sigma2, alpha)
        sinr_val = compute_sinr(w_val, R_s_all[valid_global], R_j_all[valid_global], R_n_all[valid_global])
        
        worst_idx = torch.argmin(sinr_val).item()
        worst_sinr = sinr_val[worst_idx].item()
        worst_heading = headings_full[valid_global[worst_idx]]
        
        # Shannon Entropy Effective Rank
        S = torch.linalg.svdvals(U_val)  # [B, K]
        S_norm = S / (S.sum(dim=-1, keepdim=True) + 1e-8)
        entropy = -torch.sum(S_norm * torch.log(S_norm + 1e-8), dim=-1)
        eff_rank = torch.exp(entropy).mean().item()
        
    print(f"Global Sweep Worst SINR: {worst_sinr:.2f} dB (at heading {worst_heading:.2f})")
    print(f"Effective Rank of U (Shannon entropy, out of K=5): {eff_rank:.2f}")

    print("\n--- Part 2: Boundary Annulus Sweep (10 to 15 deg separation) ---")
    valid_boundary = []
    for i, h in enumerate(headings_full):
        dist = min(h, 360 - h)
        if 10.0 <= dist <= 15.0:
            valid_boundary.append(i)
            
    with torch.no_grad():
        xyz_val = jam_bodies_t[valid_boundary].float()
        U_val = model(xyz_val).to(torch.complex128)
        
        w_val = d3_mvdr_beamformer(U_val, v_sig, P_J, sigma2, alpha)
        sinr_val = compute_sinr(w_val, R_s_all[valid_boundary], R_j_all[valid_boundary], R_n_all[valid_boundary])
        
        worst_idx = torch.argmin(sinr_val).item()
        worst_sinr_bound = sinr_val[worst_idx].item()
        worst_heading_bound = headings_full[valid_boundary[worst_idx]]
        
    print(f"Boundary Annulus Worst SINR: {worst_sinr_bound:.2f} dB (at heading {worst_heading_bound:.2f})")
    
    print("\n--- Final Results ---")
    overall_worst = min(worst_sinr, worst_sinr_bound)
    print(f"Overall Worst-Case SINR: {overall_worst:.2f} dB")
    if overall_worst >= 15.0:
        print("RESULT: SUCCESS - Exceeded 15 dB threshold across the entire operational manifold!")
    else:
        print("RESULT: FAIL - Dropped below 15 dB.")

if __name__ == '__main__':
    main()
