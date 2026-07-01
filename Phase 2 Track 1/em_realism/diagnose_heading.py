import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os
import hashlib
from datetime import datetime
from trimesh.proximity import signed_distance

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Stage 8 (Mesh Ray-Tracing)')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mesh_loader import load_uav_mesh
from conformal_array import get_conformal_array
from attitude import euler_to_quaternion, rotate_points
from em_physics import compute_distances, fresnel_diffraction_gain

def softmin(x, k=100):
    x_min = np.min(x)
    w = np.exp(-k * (x - x_min))
    w = w / np.sum(w)
    return np.sum(w * x), w

def run_diagnostic():
    mesh = load_uav_mesh(Path(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone_prepared.stl"))
    antenna_positions_body, normals_body = get_conformal_array(mesh)
    
    jammer_pos_xy = np.array([0.0, 100000.0])
    jam_world = np.array([jammer_pos_xy[0], jammer_pos_xy[1], 0.0])
    jam_world /= np.linalg.norm(jam_world)
    
    headings = np.linspace(22, 32, 200)
    phi_cmd = 30.0
    lam = 0.125
    
    g_all = np.zeros((len(headings), 16), dtype=complex)
    
    from mesh_aware_lcmv import get_steering_vector, compute_reduced_lcmv
    sinr_all = []
    
    for idx_h, h_ang in enumerate(headings):
        q = euler_to_quaternion(np.deg2rad(phi_cmd), 0, np.deg2rad(h_ang))
        q_inv = q.conjugate()
        jam_body = rotate_points(jam_world.reshape(1,3), q_inv)[0]
        
        # Use all edges to ensure h=0 when ray clears ANY mesh edge
        edges = mesh.face_adjacency_edges
        V1_all = mesh.vertices[edges[:, 0]]
        V2_all = mesh.vertices[edges[:, 1]]
        
        valid_mask = np.linalg.norm(V1_all - V2_all, axis=1) > 0.015
        if not np.any(valid_mask): valid_mask = np.ones(len(edges), dtype=bool)
            
        V1 = V1_all[valid_mask]
        V2 = V2_all[valid_mask]
        
        ray_origins = antenna_positions_body + normals_body * 1e-4
        ray_dirs = np.tile(jam_body, (16, 1))
        
        h, d1, W = compute_distances(ray_origins, jam_body, V1, V2)
        
        # We need the point on the ray closest to the edge for each edge.
        # W = P_ray - P_edge, so P_ray = P_edge + W
        # But we only need to evaluate signed_distance for the selected edges to be fast.
        
        for i in range(16):
            h_idx = h[i, :]
            d1_idx = d1[i, :]
            
            # --- Soft primary-edge selection ---
            # Hard argmin(h_idx) causes a discrete jump whenever two edges swap rank.
            # Replace with softmax weights so each step is a smooth weighted average.
            beta = 500.0   # sharpness: effectively top-K at this scale (~1/beta ≈ 0.002)
            h_min = np.min(h_idx)
            log_w = -beta * (h_idx - h_min)
            log_w -= np.max(log_w)  # numerical stability
            w_soft = np.exp(log_w)
            w_soft /= w_soft.sum()
            
            # Weighted-average primary distance for S_j reference
            edge1_d1 = np.dot(w_soft, d1_idx)  # smooth proxy for "primary edge d1"

            # Separation weight: penalise edges co-located with the primary cluster
            S_j = 1.0 - np.exp(-(d1_idx - edge1_d1)**2 / (2 * 0.2**2))
            h_penalized = h_idx / (S_j + 1e-8)

            # --- Soft primary-edge F1 (top-3 by raw h) ---
            top3 = np.argsort(h_idx)[:3]
            w_top3 = w_soft[top3]; w_top3 /= w_top3.sum()
            F1_soft = 0.0
            for k, ek in enumerate(top3):
                P_ray_k = ray_origins[i] + d1_idx[ek] * jam_body
                sd_k = signed_distance(mesh, [P_ray_k])[0]
                nu_k = sd_k * np.sqrt(2 / (lam * d1_idx[ek]))
                F1_soft += w_top3[k] * fresnel_diffraction_gain(nu_k)

            # --- Soft secondary-edge selection ---
            # Same softmax approach as primary: weight over top-3 candidates
            # ranked by h_penalized (proximity to ray, penalised for being near edge1).
            h_pen_min = np.min(h_penalized)
            log_w2 = -beta * (h_penalized - h_pen_min)
            log_w2 -= np.max(log_w2)
            w_soft2 = np.exp(log_w2); w_soft2 /= w_soft2.sum()
            edge2_d1 = np.dot(w_soft2, d1_idx)  # smooth proxy secondary d1
            weight2  = np.dot(w_soft2, S_j)      # smooth weight2

            top3_2 = np.argsort(h_penalized)[:3]
            w_top3_2 = w_soft2[top3_2]; w_top3_2 /= w_top3_2.sum()

            F2_soft = 0.0
            for k, ek in enumerate(top3_2):
                P_ray_k2 = ray_origins[i] + d1_idx[ek] * jam_body
                sd_k2 = signed_distance(mesh, [P_ray_k2])[0]
                d_eff_k = edge1_d1 + d1_idx[ek]
                nu_k2 = sd_k2 * np.sqrt(2 / (lam * d_eff_k))
                F2_soft += w_top3_2[k] * fresnel_diffraction_gain(nu_k2)

            F2_blended = (1.0 - weight2) * 1.0 + weight2 * F2_soft

            g_all[idx_h, i] = F1_soft * F2_blended

                
        for i in range(16):
            if idx_h > 0:
                prev_gain = 20*np.log10(np.abs(g_all[idx_h-1, i]))
                curr_gain = 20*np.log10(np.abs(g_all[idx_h, i]))
                if abs(curr_gain - prev_gain) > 10.0:
                    print(f"JUMP at Heading {h_ang:.2f} for Element {i}: {prev_gain:.2f} dB -> {curr_gain:.2f} dB")
        
        a_jam_true = get_steering_vector(antenna_positions_body, normals_body, (2 * np.pi / lam) * jam_body)
        a_jam_true_masked = a_jam_true * g_all[idx_h, :]
        jam_power = np.sum(np.abs(a_jam_true_masked)**2)
        sinr_all.append(-10 * np.log10(max(jam_power, 1e-12)))

    # --- Anti-stale-cache verification ---
    data_hash = hashlib.sha256(g_all.tobytes()).hexdigest()[:12]
    run_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[VERIFY] g_all SHA-256 prefix: {data_hash}  |  run timestamp: {run_ts}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    for i in range(16):
        ax1.plot(headings, 20*np.log10(np.abs(g_all[:, i])), label=f'Ant {i}')
    ax1.set_title(f"Individual Element Gains |F| (dB) vs Heading  [run: {run_ts} | hash: {data_hash}]")
    ax1.set_ylabel("Gain (dB)")
    ax1.grid()
    
    ax2.plot(headings, sinr_all, 'k-', linewidth=2)
    ax2.set_title("Aggregate Array Response (Proxy SINR)")
    ax2.set_xlabel("Heading (deg)")
    ax2.grid()
    
    plt.tight_layout()
    plt.savefig("diagnose_heading.png", dpi=120)
    print(f"Saved diagnose_heading.png  |  hash={data_hash}  |  ts={run_ts}")

if __name__ == "__main__":
    run_diagnostic()
