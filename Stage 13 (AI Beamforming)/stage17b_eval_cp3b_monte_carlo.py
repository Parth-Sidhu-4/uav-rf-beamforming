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
from stage16_cp3b_siren_masked_signal import SIRENShadowMaskPredictor, get_steering_vector_pt, dcmvdr_beamformer_pytorch

def main():
    print("--- Stage 17b: Monte Carlo Perturbation Test ---")
    device = torch.device('cpu')
    MESH_PATH = Path(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone_prepared.stl")
    mesh = load_uav_mesh(MESH_PATH)
    pos_body, normals_body = get_conformal_array_parametric(mesh, N=32)
    
    model = SIRENShadowMaskPredictor(w0=30.0).to(device)
    model.load_state_dict(torch.load("siren_beamformer_cp3b_masked.pt", map_location=device))
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
    v_sig_masked = g_sig * get_steering_vector(pos_body, K_wave * sig_body)
    v_sig_masked_t = torch.tensor(v_sig_masked, dtype=torch.complex128, device=device)
    
    # Select a challenging valid heading: 191 degrees (found in stage17a)
    h_test = 191.0
    jam_world = np.array([np.cos(np.deg2rad(h_test)), np.sin(np.deg2rad(h_test)), 0.0])
    jam_body = rotate_points(jam_world.reshape(1, 3), q_inv)[0]
    
    g_exact = compute_shadow_mask_batched(mesh, pos_body, normals_body, jam_body.reshape(1, 3))[0]
    v_jam_ex = g_exact * get_steering_vector(pos_body, jam_body * K_wave)
    
    jam_bodies_t = torch.tensor([jam_body], dtype=torch.float64, device=device)
    pos_lambda = torch.tensor(pos_body / 0.15, dtype=torch.float64, device=device)
    
    with torch.no_grad():
        g_hat, _ = model(jam_bodies_t.float())
        g_hat = g_hat.to(torch.complex128)
        theta_j = torch.acos(jam_bodies_t[:, 2] / torch.norm(jam_bodies_t, dim=1))
        phi_j = torch.atan2(jam_bodies_t[:, 1], jam_bodies_t[:, 0])
        v_hat = g_hat * get_steering_vector_pt(pos_lambda, theta_j, phi_j)
        
    N_TRIALS = 500
    sinr_results = []
    
    print(f"Running {N_TRIALS} Monte Carlo perturbations (phase noise \u00b110 degrees) on exact physics...")
    for i in tqdm(range(N_TRIALS)):
        # Inject random uniform phase noise to exact physics
        phase_noise = np.exp(1j * np.random.uniform(-10, 10, size=32) * np.pi / 180.0)
        v_jam_noisy = v_jam_ex * phase_noise
        v_sig_noisy = v_sig_masked * phase_noise
        
        with torch.no_grad():
            w_t = dcmvdr_beamformer_pytorch(theta_j, phi_j, pos_lambda, v_hat, v_sig_masked_t, P_J, P_S, sigma2, alpha)
            w = w_t[0].cpu().numpy()
            
        R_j_ex = P_J * np.outer(v_jam_noisy, np.conj(v_jam_noisy))
        R_s_ex = P_S * np.outer(v_sig_noisy, np.conj(v_sig_noisy))
        R_n_ex = sigma2 * np.eye(32)
        
        S = np.real(np.conj(w) @ R_s_ex @ w)
        NJ = np.real(np.conj(w) @ (R_j_ex + R_n_ex) @ w)
        sinr = 10 * np.log10(S / max(NJ, 1e-12))
        sinr_results.append(sinr)
        
    sinr_arr = np.array(sinr_results)
    print("\n--- Monte Carlo Results ---")
    print(f"Heading tested: {h_test} deg")
    print(f"Mean SINR: {np.mean(sinr_arr):.2f} dB")
    print(f"Min SINR: {np.min(sinr_arr):.2f} dB")
    print(f"Max SINR: {np.max(sinr_arr):.2f} dB")
    
if __name__ == '__main__':
    main()
