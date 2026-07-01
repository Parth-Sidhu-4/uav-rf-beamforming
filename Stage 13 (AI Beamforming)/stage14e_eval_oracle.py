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
from stage14e_oracle_weights import ShadowNetDB

def get_R_matrices(v_sig, v_jam):
    JAM_POW = 10000.0
    NOISE_POW = 1.0
    SIG_POW = 100.0
    R_j = JAM_POW * np.outer(v_jam, np.conj(v_jam))
    R_n = NOISE_POW * np.eye(len(v_jam))
    R_s = SIG_POW * np.outer(v_sig, np.conj(v_sig))
    return R_s, R_j, R_n

def main():
    print("Evaluating Oracle Weight Prediction (shadow_net_oracle.pt)")
    MESH_PATH = Path(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone_prepared.stl")
    mesh = load_uav_mesh(MESH_PATH)
    pos_body, normals_body = get_conformal_array(mesh)
    
    device = torch.device("cpu")
    model = ShadowNetDB().to(device)
    model.load_state_dict(torch.load("shadow_net_oracle.pt", map_location=device))
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
    
    print("Computing AI weight predictions...")
    with torch.no_grad():
        preds = model(torch.tensor(jam_bodies, dtype=torch.float32).to(device)).cpu().numpy()
    w_ai_all = preds[:, 0::2] + 1j * preds[:, 1::2]

    LAM = 0.15
    K = 2 * np.pi / LAM
    v_sig = get_steering_vector(pos_body, K * sig_body)

    print("\n--- SINR Evaluation ---")
    min_sinr_0 = float('inf')
    for i in range(360):
        # We don't need to compute MVDR on the fly! The AI already predicted w.
        w = w_ai_all[i]
        
        # Ensure constraint w^H v_sig = 1 (this prevents arbitrary scaling by AI)
        c = np.conj(v_sig) @ w
        if abs(c) > 1e-12:
            w = w / c
        else:
            w = w / 1e-12
        
        v_jam_ex = g_exact_all[i] * get_steering_vector(pos_body, jam_bodies[i] * K)
        
        R_s_ex, R_j_ex, R_n_ex = get_R_matrices(v_sig, v_jam_ex)
        S = np.real(np.conj(w) @ R_s_ex @ w)
        NJ = np.real(np.conj(w) @ (R_j_ex + R_n_ex) @ w)
        sinr = 10 * np.log10(S / max(NJ, 1e-12))
        
        if sinr < min_sinr_0:
            min_sinr_0 = sinr

    print(f"Worst-case SINR over 360-deg Azimuth Sweep: {min_sinr_0:.2f} dB")
    print(f"Gain over Phase B baseline (12.94 dB): {(min_sinr_0 - 12.94):.2f} dB")

if __name__ == '__main__':
    main()
