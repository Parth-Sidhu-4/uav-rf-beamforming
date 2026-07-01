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
from stage14d_run_d2 import ShadowNet

def get_R_matrices(v_sig, v_jam):
    JAM_POW = 10000.0
    NOISE_POW = 1.0
    SIG_POW = 100.0
    R_j = JAM_POW * np.outer(v_jam, np.conj(v_jam))
    R_n = NOISE_POW * np.eye(len(v_jam))
    R_s = SIG_POW * np.outer(v_sig, np.conj(v_sig))
    return R_s, R_j, R_n

def compute_mvdr_robust(v_sig, v_jam, dl_factor=0.0):
    R_s, R_j, R_n = get_R_matrices(v_sig, v_jam)
    R_xx = R_j + R_n
    
    trace_val = np.real(np.trace(R_xx))
    diag_load = dl_factor * trace_val / R_xx.shape[0] if dl_factor > 0 else 1e-12
    R_reg = R_xx + diag_load * np.eye(R_xx.shape[0])
    
    try:
        R_inv = np.linalg.inv(R_reg)
    except np.linalg.LinAlgError:
        R_inv = np.linalg.pinv(R_reg)

    num = R_inv @ v_sig
    den = np.conj(v_sig) @ num
    w = num / max(abs(den), 1e-12)
    
    S = np.real(np.conj(w) @ R_s @ w)
    N_J = np.real(np.conj(w) @ (R_j + R_n) @ w)
    sinr = S / max(N_J, 1e-12)
    return 10 * np.log10(sinr)

def main():
    print("Evaluating Phase D-2 Ablation Model (shadow_net_d2.pt)")
    MESH_PATH = Path(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone_prepared.stl")
    mesh = load_uav_mesh(MESH_PATH)
    pos_body, normals_body = get_conformal_array(mesh)
    
    model = ShadowNet()
    model.load_state_dict(torch.load("shadow_net_d2.pt", map_location='cpu'))
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
    
    print("Computing AI predictions...")
    with torch.no_grad():
        preds = model(torch.tensor(jam_bodies, dtype=torch.float32)).numpy()
    g_ai_all = preds[:, 0::2] + 1j * preds[:, 1::2]

    LAM = 0.15
    K = 2 * np.pi / LAM
    v_sig = get_steering_vector(pos_body, K * sig_body)

    print("\n--- SINR Evaluation ---")
    min_sinr_0 = float('inf')
    for i in range(360):
        v_jam_ai = g_ai_all[i] * get_steering_vector(pos_body, jam_bodies[i] * K)
        v_jam_ex = g_exact_all[i] * get_steering_vector(pos_body, jam_bodies[i] * K)
        
        # Calculate weights using AI, but evaluate using EXACT physics
        R_s, R_j, R_n = get_R_matrices(v_sig, v_jam_ai)
        R_xx = R_j + R_n + 1e-2 * np.eye(len(v_jam_ai))  # Step 0: Diagonal loading (1e-2 * NOISE_POW)
        try:
            R_inv = np.linalg.inv(R_xx)
        except np.linalg.LinAlgError:
            R_inv = np.linalg.pinv(R_xx)
        num = R_inv @ v_sig
        w = num / max(abs(np.conj(v_sig) @ num), 1e-12)
        
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
