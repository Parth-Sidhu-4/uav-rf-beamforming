import os
import sys
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

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
from stage16_cp3b_siren_masked_signal import SIRENShadowMaskPredictor, get_steering_vector_pt, dcmvdr_beamformer_pytorch, compute_sinr

def evaluate_headings(headings, mesh, pos_body, normals_body, model, v_sig_masked, v_sig_masked_t, P_S, P_J, sigma2, alpha, K_wave, q_inv, device, g_exact_precomputed=None):
    N = len(headings)
    jam_bodies = np.zeros((N, 3))
    for i, h in enumerate(headings):
        jam_world = np.array([np.cos(np.deg2rad(h)), np.sin(np.deg2rad(h)), 0.0])
        jam_bodies[i] = rotate_points(jam_world.reshape(1, 3), q_inv)[0]
        
    if g_exact_precomputed is not None:
        g_exact = g_exact_precomputed
    else:
        print(f"Computing exact physics for {N} points...")
        g_exact = np.zeros((N, 32), dtype=np.complex128)
        chunk_size = 36
        for i in tqdm(range(0, N, chunk_size)):
            chunk = jam_bodies[i:i+chunk_size]
            g_exact[i:i+chunk_size] = compute_shadow_mask_batched(mesh, pos_body, normals_body, chunk)
        
    pos_lambda = torch.tensor(pos_body / 0.15, dtype=torch.float64, device=device)
    jam_bodies_t = torch.tensor(jam_bodies, dtype=torch.float64, device=device)
    
    model.eval()
    with torch.no_grad():
        xyz = jam_bodies_t.float()
        g_hat, _ = model(xyz)
        g_hat = g_hat.to(torch.complex128)
        
        theta_j = torch.acos(jam_bodies_t[:, 2] / torch.norm(jam_bodies_t, dim=1))
        phi_j = torch.atan2(jam_bodies_t[:, 1], jam_bodies_t[:, 0])
        v_hat = g_hat * get_steering_vector_pt(pos_lambda, theta_j, phi_j)
        
        w_t = dcmvdr_beamformer_pytorch(theta_j, phi_j, pos_lambda, v_hat, v_sig_masked_t, P_J, P_S, sigma2, alpha)
        w_all = w_t.cpu().numpy()
        
    min_sinr = float('inf')
    worst_h = -1
    
    for i in range(N):
        w = w_all[i]
        v_jam_ex = g_exact[i] * get_steering_vector(pos_body, jam_bodies[i] * K_wave)
        
        R_j_ex = P_J * np.outer(v_jam_ex, np.conj(v_jam_ex))
        R_s_ex = P_S * np.outer(v_sig_masked, np.conj(v_sig_masked))
        R_n_ex = sigma2 * np.eye(32)
        
        S = np.real(np.conj(w) @ R_s_ex @ w)
        NJ = np.real(np.conj(w) @ (R_j_ex + R_n_ex) @ w)
        sinr = 10 * np.log10(S / max(NJ, 1e-12))
        
        if sinr < min_sinr:
            min_sinr = sinr
            worst_h = headings[i]
            
    return min_sinr, worst_h

def main():
    device = torch.device('cpu')
    print("Loading mesh and network...")
    MESH_PATH = Path(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone_prepared.stl")
    mesh = load_uav_mesh(MESH_PATH)
    pos_body, normals_body = get_conformal_array_parametric(mesh, N=32)
    
    model = SIRENShadowMaskPredictor(w0=30.0).to(device)
    model_path = "siren_beamformer_cp3b_masked.pt"
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Train CP3b first!")
        sys.exit(1)
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    
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
    v_sig_masked = g_sig * v_sig_ideal_np
    v_sig_masked_t = torch.tensor(v_sig_masked, dtype=torch.complex128, device=device)
    
    # Load 3600 points precomputed mask
    print("Loading precomputed exact physics for Global Sweep...")
    dataset_path = "dataset_3600_masks.npz"
    if not os.path.exists(dataset_path):
        print("Error: dataset_3600_masks.npz not found.")
        sys.exit(1)
    
    g_exact_full = np.load(dataset_path)['g_exact']
    all_headings = np.linspace(0, 360, 3600, endpoint=False)
    
    global_indices = []
    global_headings = []
    for i, h in enumerate(all_headings):
        if min(h, 360 - h) > 15.0:
            global_indices.append(i)
            global_headings.append(h)
            
    g_exact_global = g_exact_full[global_indices]
    
    # 1. Global Sweep (> 15 deg separation)
    print("\n--- Part 1: Global Sweep (> 15 deg separation) ---")
    min_sinr_global, worst_h_global = evaluate_headings(
        global_headings, mesh, pos_body, normals_body, model, 
        v_sig_masked, v_sig_masked_t, P_S, P_J, sigma2, alpha, K_wave, q_inv, device,
        g_exact_precomputed=g_exact_global
    )
    print(f"Global Sweep Worst SINR: {min_sinr_global:.2f} dB (at heading {worst_h_global:.2f})")
    
    # 2. Boundary Annulus Stress-Test (10 to 15 deg separation)
    # Dense 0.01 deg resolution sweep
    print("\n--- Part 2: Boundary Annulus Sweep (10 to 15 deg separation) ---")
    left_annulus = np.linspace(10.0, 15.0, 500)
    right_annulus = np.linspace(345.0, 350.0, 500)
    boundary_headings = np.concatenate([left_annulus, right_annulus])
    
    min_sinr_bound, worst_h_bound = evaluate_headings(
        boundary_headings, mesh, pos_body, normals_body, model, 
        v_sig_masked, v_sig_masked_t, P_S, P_J, sigma2, alpha, K_wave, q_inv, device
    )
    print(f"Boundary Annulus Worst SINR: {min_sinr_bound:.2f} dB (at heading {worst_h_bound:.2f})")
    
    print("\n--- Final Results ---")
    print(f"Overall Worst-Case SINR: {min(min_sinr_global, min_sinr_bound):.2f} dB")
    if min(min_sinr_global, min_sinr_bound) >= 15.0:
        print("RESULT: PASS - Target > 15 dB achieved.")
    else:
        print("RESULT: FAIL - Dropped below 15 dB.")

if __name__ == '__main__':
    main()
