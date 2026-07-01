import sys
import os
import numpy as np
import torch
from pathlib import Path

sys.path.append(os.path.abspath(r'D:\UAV Internship project\Phase 2 Track 1\em_realism'))
sys.path.append(os.path.abspath(r'D:\UAV Internship project\Phase 2 Track 2\array_structures'))
sys.path.append(os.path.abspath(r'D:\UAV Internship project\Stage 12 (Operational Analysis)'))

from mesh_loader import load_uav_mesh
from conformal_array import get_conformal_array
from shadow_engine_batched import compute_shadow_mask_batched
from attitude import euler_to_quaternion, rotate_points
from lcmv_stage8 import get_steering_vector
from stage15b_train_phase_e import SirenResidualNet
from stage15a_dataset_prep import real_spherical_harmonics

def get_R_matrices(v_sig, v_jam):
    JAM_POW = 10000.0
    NOISE_POW = 1.0
    SIG_POW = 100.0
    R_j = JAM_POW * np.outer(v_jam, np.conj(v_jam))
    R_n = NOISE_POW * np.eye(len(v_jam))
    R_s = SIG_POW * np.outer(v_sig, np.conj(v_sig))
    return R_s, R_j, R_n

def main():
    print("Evaluating Phase E Combined Architecture (shadow_net_phase_e.pt)")
    MESH_PATH = Path(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone_prepared.stl")
    mesh = load_uav_mesh(MESH_PATH)
    pos_body, normals_body = get_conformal_array(mesh)
    
    device = torch.device("cpu")
    model = SirenResidualNet().to(device)
    model.load_state_dict(torch.load("shadow_net_phase_e.pt", map_location=device))
    model.eval()

    headings = np.linspace(0, 360, 360, endpoint=False)
    q_inv = euler_to_quaternion(np.deg2rad(15.0), 0, 0).conjugate()
    jam_bodies = np.zeros((360, 3))
    for i, h in enumerate(headings):
        jam_world = np.array([np.cos(np.deg2rad(h)), np.sin(np.deg2rad(h)), 0.0])
        jam_bodies[i] = rotate_points(jam_world.reshape(1, 3), q_inv)[0]
    
    sig_world = np.array([1.0, 0.0, 0.0])
    sig_body = rotate_points(sig_world.reshape(1, 3), q_inv)[0]

    print("Computing exact physics...")
    g_exact_all = compute_shadow_mask_batched(mesh, pos_body, normals_body, jam_bodies)
    
    print("Computing Spherical Harmonic features...")
    sh_features = real_spherical_harmonics(jam_bodies[:, 0], jam_bodies[:, 1], jam_bodies[:, 2], L=4)
    
    LAM = 0.15
    K = 2 * np.pi / LAM
    v_sig_np = get_steering_vector(pos_body, K * sig_body)
    v_sig = torch.tensor(v_sig_np, dtype=torch.complex64)
    pos_body_t = torch.tensor(pos_body, dtype=torch.float32)

    print("Computing AI predictions...")
    with torch.no_grad():
        sh_t = torch.tensor(sh_features, dtype=torch.float32)
        cart_t = torch.tensor(jam_bodies, dtype=torch.float32)
        
        out = model(sh_t)
        mag_ai = torch.sigmoid(out[:, 0::2])
        phase_pert = torch.pi * torch.tanh(out[:, 1::2])
        
        phases_geo = K * (cart_t @ pos_body_t.T)
        geo_sp = torch.exp(1j * phases_geo)
        
        g_ai = mag_ai * torch.exp(1j * phase_pert)
        v_jam_ai_all = (geo_sp * g_ai).numpy()

    JAM_POW = 10000.0
    NOISE_POW = 1.0
    DL_ALPHA = 0.05

    print("\n--- SINR Evaluation ---")
    min_sinr_0 = float('inf')
    
    for i in range(360):
        v_jam_ai = v_jam_ai_all[i]
        v_jam_ex = g_exact_all[i] * get_steering_vector(pos_body, jam_bodies[i] * K)
        
        # Standard exact MVDR to test for SM bugs
        R_s, R_j, R_n = get_R_matrices(v_sig_np, v_jam_ai)
        
        # DL_ALPHA * trace(R_j + R_n) / 16
        trace_val = np.real(np.trace(R_j + R_n))
        R_xx = R_j + R_n + DL_ALPHA * trace_val / 16.0 * np.eye(16)
        
        R_inv = np.linalg.inv(R_xx)
        num = R_inv @ v_sig_np
        w = num / (np.conj(v_sig_np) @ num)
        
        # Evaluate using EXACT physics
        R_s_ex, R_j_ex, R_n_ex = get_R_matrices(v_sig_np, v_jam_ex)
        S = np.real(np.conj(w) @ R_s_ex @ w)
        NJ = np.real(np.conj(w) @ (R_j_ex + R_n_ex) @ w)
        sinr = 10 * np.log10(S / max(NJ, 1e-12))
        
        if sinr < min_sinr_0:
            min_sinr_0 = sinr

    print(f"Worst-case SINR over 360-deg Azimuth Sweep: {min_sinr_0:.2f} dB")
    print(f"Gain over Phase D-2 champion (13.60 dB): {(min_sinr_0 - 13.60):.2f} dB")

if __name__ == '__main__':
    main()
