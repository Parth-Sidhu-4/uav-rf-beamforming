import sys
import os
import numpy as np
import torch
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Phase 2 Track 1', 'em_realism')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Phase 2 Track 2', 'array_structures')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Stage 12 (Operational Analysis)')))

from mesh_loader import load_uav_mesh
from conformal_array import get_conformal_array
from shadow_engine_batched import compute_shadow_mask_batched
from attitude import euler_to_quaternion, rotate_points
from lcmv_stage8 import get_steering_vector
from stage13b_train_shadow_net import ShadowNet
from stage13d_diagnostics import compute_mvdr_robust

def main():
    # Load geometry and model
    MESH_PATH = Path(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone_prepared.stl")
    print("Loading mesh...")
    mesh = load_uav_mesh(MESH_PATH)
    pos_body, normals_body = get_conformal_array(mesh)
    
    model = ShadowNet()
    model.load_state_dict(torch.load("shadow_net.pt"))
    model.eval()

    # Generate Test 1 Trajectory
    headings = np.linspace(0, 360, 360, endpoint=False)
    q_inv = euler_to_quaternion(np.deg2rad(15.0), 0, 0).conjugate()
    jam_bodies = np.zeros((360, 3))
    for i, h in enumerate(headings):
        jam_world = np.array([np.cos(np.deg2rad(h)), np.sin(np.deg2rad(h)), 0.0])
        jam_bodies[i] = rotate_points(jam_world.reshape(1, 3), q_inv)[0]
    
    sig_world = np.array([1.0, 0.0, 0.0])
    sig_body = rotate_points(sig_world.reshape(1, 3), q_inv)[0]

    LAM = 0.15
    K = 2 * np.pi / LAM
    v_sig = get_steering_vector(pos_body, K * sig_body)

    print("Computing exact physics (batched)...")
    g_exact_all = compute_shadow_mask_batched(mesh, pos_body, normals_body, jam_bodies)
    
    with torch.no_grad():
        preds = model(torch.tensor(jam_bodies, dtype=torch.float32)).numpy()
    g_ai_all = preds[:, 0::2] + 1j * preds[:, 1::2]

    # Find worst SINR
    worst_sinr = float('inf')
    worst_idx = -1
    
    # We use DL=0.1 to match the robust setup
    for i in range(360):
        v_jam_ai = g_ai_all[i] * get_steering_vector(pos_body, jam_bodies[i] * K)
        w_ai, _ = compute_mvdr_robust(v_sig, v_jam_ai, dl_factor=0.1)
        
        v_jam_exact = g_exact_all[i] * get_steering_vector(pos_body, jam_bodies[i] * K)
        S = np.real(np.conj(w_ai) @ (100.0 * np.outer(v_sig, np.conj(v_sig))) @ w_ai)
        N_J = np.real(np.conj(w_ai) @ (10000.0 * np.outer(v_jam_exact, np.conj(v_jam_exact)) + np.eye(16)) @ w_ai)
        sinr = 10 * np.log10(S / max(N_J, 1e-12))
        
        if sinr < worst_sinr:
            worst_sinr = sinr
            worst_idx = i

    print(f"\n--- WORST SINR HEADING: {headings[worst_idx]:.1f}° (SINR: {worst_sinr:.2f} dB) ---")
    print("Element | Exact |g| | Pred |g|  | Difference (Exact - Pred)")
    print("-" * 60)
    exact_mags = np.abs(g_exact_all[worst_idx])
    pred_mags = np.abs(g_ai_all[worst_idx])
    for j in range(16):
        diff = exact_mags[j] - pred_mags[j]
        # Flag if exact is high but pred is low (false shadow)
        flag = " <--- FALSE SHADOW" if exact_mags[j] > 0.6 and pred_mags[j] < 0.3 else ""
        print(f"   {j:2d}   |  {exact_mags[j]:.3f}  |  {pred_mags[j]:.3f}  |  {diff:+.3f} {flag}")

if __name__ == '__main__':
    main()
