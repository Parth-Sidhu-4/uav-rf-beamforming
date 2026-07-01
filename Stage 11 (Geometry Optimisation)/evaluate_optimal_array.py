import numpy as np
import trimesh
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
from em_physics import compute_distances, fresnel_diffraction_gain, compute_plf
from constants import NACT_THRESHOLD

def softmin(x, k=100):
    x_min = np.min(x)
    w = np.exp(-k * (x - x_min))
    w = w / np.sum(w)
    return np.sum(w * x), w

def run_diagnostic():
    mesh_path = Path(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone_prepared.stl")
    mesh = load_uav_mesh(mesh_path)
    
    # Load optimal genome and candidate pool
    genome = np.load("optimal_genome.npy")
    cand_right = np.load("candidate_pairs_right.npy")
    cand_left = np.load("candidate_pairs_left.npy")
    
    # Extract positions
    pos_right = cand_right[genome]
    pos_left = cand_left[genome]
    antenna_positions_body = np.vstack([pos_right, pos_left])
    
    # Compute normals using closest_point
    _, _, tri_right = trimesh.proximity.closest_point(mesh, pos_right)
    norm_right = mesh.face_normals[tri_right]
    _, _, tri_left = trimesh.proximity.closest_point(mesh, pos_left)
    norm_left = mesh.face_normals[tri_left]
    normals_body = np.vstack([norm_right, norm_left])
    
    jammer_pos_xy = np.array([0.0, 100000.0])
    jam_world = np.array([jammer_pos_xy[0], jammer_pos_xy[1], 0.0])
    jam_world /= np.linalg.norm(jam_world)
    
    headings = np.linspace(0, 360, 360)
    phi_cmd = 30.0
    lam = 0.125
    
    g_all = np.zeros((len(headings), 16), dtype=complex)
    
    from mesh_aware_lcmv import get_steering_vector, compute_reduced_lcmv
    sinr_all = []
    
    jam_pol_world = np.array([0.0, 0.0, -1.0]) # vertical polarization (NED frame)
    
    for idx_h, h_ang in enumerate(headings):
        q = euler_to_quaternion(np.deg2rad(phi_cmd), 0, np.deg2rad(h_ang))
        q_inv = q.conjugate()
        jam_body = rotate_points(jam_world.reshape(1,3), q_inv)[0]
        
        plf_amp = compute_plf(normals_body, q, jam_pol_world)
        
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
        edge_dirs = (V2 - V1) / np.linalg.norm(V2 - V1, axis=1)[:, np.newaxis]
        
        for i in range(16):
            h_idx = h[i, :]
            d1_idx = d1[i, :]
            W_idx = W[i, :, :]
            
            cross_W_D = np.cross(W_idx, jam_body)
            dot_cross = np.sum(cross_W_D * edge_dirs, axis=1)
            sd_i = np.sign(dot_cross) * h_idx
            
            # --- Soft primary-edge selection ---
            # Hard argmin(h_idx) causes a discrete jump whenever two edges swap rank.
            # Replace with softmax weights so each step is a smooth weighted average.
            beta = 500.0   # sharpness: effectively top-K at this scale (~1/beta ≈ 0.002)
            valid_edges = d1_idx > 0.01
            if not np.any(valid_edges):
                g_all[idx_h, i] = 1.0 * plf_amp[i]
                continue
                
            h_valid = h_idx[valid_edges]
            d1_valid = d1_idx[valid_edges]
            edge_idx = np.arange(len(h_idx))[valid_edges]
            sd_valid = sd_i[valid_edges]

            h_min = np.min(h_valid)
            log_w = -beta * (h_valid - h_min)
            log_w -= np.max(log_w)  # numerical stability
            w_soft = np.exp(log_w)
            w_soft /= w_soft.sum()
            # Weighted-average primary distance for S_j reference
            edge1_d1 = np.dot(w_soft, d1_valid)  # smooth proxy for "primary edge d1"

            # Separation weight: penalise edges co-located with the primary cluster
            S_j = 1.0 - np.exp(-(d1_valid - edge1_d1)**2 / (2 * 0.2**2))
            h_penalized = h_valid / (S_j + 1e-8)

            # --- Soft primary-edge F1 (top-3 by raw h) ---
            top3 = np.argsort(h_valid)[:3]
            w_top3 = w_soft[top3]; w_top3 /= w_top3.sum()
            F1_soft = 0.0
            for k, ek_valid in enumerate(top3):
                ek = edge_idx[ek_valid]
                sd_k = sd_i[ek]
                nu_k = sd_k * np.sqrt(2 / (lam * d1_idx[ek]))
                F1_soft += w_top3[k] * fresnel_diffraction_gain(nu_k)

            # --- Soft secondary-edge selection ---
            # Same softmax approach as primary: weight over top-3 candidates
            # ranked by h_penalized (proximity to ray, penalised for being near edge1).
            h_pen_min = np.min(h_penalized)
            log_w2 = -beta * (h_penalized - h_pen_min)
            log_w2 -= np.max(log_w2)
            w_soft2 = np.exp(log_w2); w_soft2 /= w_soft2.sum()
            edge2_d1 = np.dot(w_soft2, d1_valid)  # smooth proxy secondary d1
            weight2  = np.dot(w_soft2, S_j)      # smooth weight2

            top3_2 = np.argsort(h_penalized)[:3]
            w_top3_2 = w_soft2[top3_2]; w_top3_2 /= w_top3_2.sum()

            F2_soft = 0.0
            for k, ek_valid in enumerate(top3_2):
                ek = edge_idx[ek_valid]
                sd_k2 = sd_i[ek]
                d_eff_k = edge1_d1 + d1_idx[ek]
                nu_k2 = sd_k2 * np.sqrt(2 / (lam * d_eff_k))
                F2_soft += w_top3_2[k] * fresnel_diffraction_gain(nu_k2)

            F2_blended = (1.0 - weight2) * 1.0 + weight2 * F2_soft

            g_all[idx_h, i] = F1_soft * F2_blended * plf_amp[i]
                
        for i in range(16):
            if idx_h > 0:
                prev_gain = 20*np.log10(np.abs(g_all[idx_h-1, i]))
                curr_gain = 20*np.log10(np.abs(g_all[idx_h, i]))
                if abs(curr_gain - prev_gain) > 10.0:
                    pass # print(f"JUMP at Heading {h_ang:.2f} for Element {i}: {prev_gain:.2f} dB -> {curr_gain:.2f} dB")
        
        a_jam_true = get_steering_vector(antenna_positions_body, normals_body, (2 * np.pi / lam) * jam_body)
        a_jam_true_masked = a_jam_true * g_all[idx_h, :]
        jam_power = np.sum(np.abs(a_jam_true_masked)**2)
        sinr_all.append(-10 * np.log10(max(jam_power, 1e-12)))

    # Compute N_act profile
    n_act_profile = np.sum(np.abs(g_all) > NACT_THRESHOLD, axis=1)
    n_act_min = np.min(n_act_profile)
    worst_headings = headings[n_act_profile == n_act_min]

    # --- Anti-stale-cache verification ---
    data_hash = hashlib.sha256(g_all.tobytes()).hexdigest()[:12]
    run_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[VERIFY] g_all SHA-256 prefix: {data_hash}  |  run timestamp: {run_ts}")
    print(f"--- Optimal Evaluation (with PLF) ---")
    print(f"N_act,min = {n_act_min}")
    print(f"Worst Headings: {worst_headings}")

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    for i in range(16):
        ax1.plot(headings, 20*np.log10(np.abs(g_all[:, i])), label=f'Ant {i}')
    ax1.set_title(f"Individual Element Gains |F| (dB) vs Heading  [run: {run_ts} | hash: {data_hash}]")
    ax1.set_ylabel("Gain (dB)")
    ax1.grid()
    
    ax2.plot(headings, n_act_profile, 'b-', linewidth=2)
    ax2.axhline(y=15, color='r', linestyle='--', label='Autopilot Threshold (15)')
    ax2.set_title(f"Active Element Count N_act (min = {n_act_min})")
    ax2.set_ylabel("N_act")
    ax2.set_ylim(0, 16.5)
    ax2.grid()
    ax2.legend()
    
    ax3.plot(headings, sinr_all, 'k-', linewidth=2)
    ax3.set_title("Aggregate Array Response (Proxy SINR)")
    ax3.set_xlabel("Heading (deg)")
    ax3.grid()
    
    plt.tight_layout()
    plt.savefig("optimal_reval_results.png", dpi=120)
    print(f"Saved optimal_reval_results.png  |  hash={data_hash}  |  ts={run_ts}")

if __name__ == "__main__":
    run_diagnostic()
