import numpy as np
import sys
import os
import trimesh
from pathlib import Path

# Add Stage 8 paths for LCMV and Mesh Loader
sys.path.append(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)")
from lcmv_stage8 import compute_lcmv_weights, get_steering_vector
from constants import NACT_THRESHOLD
from mesh_loader import load_uav_mesh
from attitude import euler_to_quaternion, rotate_points

# Use Stage 11 physics which returns W for fast sign projection
from em_physics import compute_distances, fresnel_diffraction_gain, compute_plf

def compute_shadow_mask_fast(jam_body, antenna_positions_body, normals_body, V1, V2, edge_dirs):
    N = 16
    lam = 0.125
    
    ray_origins = antenna_positions_body + normals_body * 1e-4
    h_all, d1_all, W_all = compute_distances(ray_origins, jam_body, V1, V2)
    
    g = np.ones(N, dtype=complex)
    
    for i in range(N):
        h_idx = h_all[i, :]
        d1_idx = d1_all[i, :]
        W_i = W_all[i, :, :]
        
        edge1_d1 = d1_idx[np.argmin(h_idx)]
        S_j = 1.0 - np.exp(- (d1_idx - edge1_d1)**2 / (2 * 0.2**2))
        
        h_penalized = h_idx / (S_j + 1e-8)
        idx2 = np.argmin(h_penalized)
        idx1 = np.argmin(h_idx)
        
        # Cross product projection
        cross1 = np.cross(W_i[idx1], jam_body)
        dot1 = np.dot(cross1, edge_dirs[idx1])
        sd1 = np.sign(dot1) * h_idx[idx1]
        
        cross2 = np.cross(W_i[idx2], jam_body)
        dot2 = np.dot(cross2, edge_dirs[idx2])
        sd2 = np.sign(dot2) * h_idx[idx2]
        
        nu1 = sd1 * np.sqrt(2 / (lam * max(d1_idx[idx1], 1e-8)))
        F1 = fresnel_diffraction_gain(nu1)
        
        nu2 = sd2 * np.sqrt(2 / (lam * max(d1_idx[idx2], 1e-8)))
        F2 = fresnel_diffraction_gain(nu2)
        
        weight2 = S_j[idx2]
        F2_blended = (1.0 - weight2) * 1.0 + weight2 * F2
        
        g[i] = F1 * F2_blended
        
    return g

def main():
    if not os.path.exists("top10_genomes.npy"):
        print("top10_genomes.npy not found!")
        return
        
    top10 = np.load("top10_genomes.npy")
    
    mesh_path = Path(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone_prepared.stl")
    mesh = load_uav_mesh(mesh_path)
    
    cand_right = np.load("candidate_pairs_right.npy")
    cand_left = np.load("candidate_pairs_left.npy")
    
    _, _, tri_right = trimesh.proximity.closest_point(mesh, cand_right)
    norm_right = mesh.face_normals[tri_right]
    
    _, _, tri_left = trimesh.proximity.closest_point(mesh, cand_left)
    norm_left = mesh.face_normals[tri_left]
    
    edges = mesh.face_adjacency_edges
    V1_all = mesh.vertices[edges[:, 0]]
    V2_all = mesh.vertices[edges[:, 1]]
    valid_mask = np.linalg.norm(V1_all - V2_all, axis=1) > 0.015
    if not np.any(valid_mask): valid_mask = np.ones(len(edges), dtype=bool)
    V1 = V1_all[valid_mask]
    V2 = V2_all[valid_mask]
    edge_dirs = (V2 - V1) / np.linalg.norm(V2 - V1, axis=1)[:, np.newaxis]
    
    headings = np.arange(0, 360, 1) # Full 360 deg sweep at 1 deg resolution
    
    jam_pos_xy = np.array([0.0, 100000.0])
    jam_world = np.array([jam_pos_xy[0], jam_pos_xy[1], 0.0])
    jam_world /= np.linalg.norm(jam_world)
    jam_power = 10**(60.0/10.0)
    noise_power = 1.0
    
    results = []
    
    for rank, genome in enumerate(top10):
        print(f"\nEvaluating Rank {rank+1} Genome: {genome}")
        
        pos_right = cand_right[genome]
        pos_left = cand_left[genome]
        antenna_positions_body = np.vstack([pos_right, pos_left])
        
        n_right = norm_right[genome]
        n_left = norm_left[genome]
        normals_body = np.vstack([n_right, n_left])
        
        n_act_min = 16
        sinr_min = 1000.0
        
        for h_ang in headings:
            # We must validate at a flight regime the array was trained for (e.g. bank=15)
            # Autopilot logic: Bank TOWARDS the jammer
            actual_phi = 15.0 if h_ang <= 180 else -15.0
            q = euler_to_quaternion(np.deg2rad(actual_phi), 0, np.deg2rad(h_ang))
            q_inv = q.conjugate()
            
            jam_body = rotate_points(jam_world.reshape(1,3), q_inv)[0]
            
            # Fast physics evaluation
            g_diff = compute_shadow_mask_fast(jam_body, antenna_positions_body, normals_body, V1, V2, edge_dirs)
            
            # Need PLF for full gain
            plf_amp = compute_plf(normals_body, q, jam_world)
            g = g_diff * plf_amp
            
            g_amp = np.abs(g)
            n_act = np.sum(g_amp >= NACT_THRESHOLD)
            
            if n_act < n_act_min:
                n_act_min = n_act
                worst_heading_nact = h_ang
                
            # Signal direction
            signal_world_dir = np.array([1.0, 0.0, 0.0])
            signal_body = rotate_points(signal_world_dir.reshape(1, 3), q_inv)[0]
            
            k = 2 * np.pi / (3e8 / 2.4e9)
            k_sig = k * signal_body
            k_jam = k * jam_body
            
            a_sig = get_steering_vector(antenna_positions_body, k_sig)
            a_jam = get_steering_vector(antenna_positions_body, k_jam)
            
            signal_power = 10**(20/10)
            
            R_s = signal_power * np.outer(a_sig, np.conj(a_sig))
            R_j = jam_power * np.outer(a_jam, np.conj(a_jam))
            R_n = noise_power * np.eye(16)
            R_xx = R_s + R_j + R_n
            
            w = compute_lcmv_weights(R_xx, a_sig, g)
            
            P_s = signal_power * np.abs(np.conj(w).T @ a_sig)**2
            P_j = jam_power * np.abs(np.conj(w).T @ a_jam)**2
            P_n = noise_power * np.linalg.norm(w)**2
            
            sinr = 10 * np.log10(max(P_s / (P_j + P_n + 1e-12), 1e-12))
            
            if sinr < sinr_min:
                sinr_min = sinr
                worst_heading_sinr = h_ang
                
        print(f" -> N_act_min: {n_act_min} (at {worst_heading_nact} deg) | SINR trough: {sinr_min:.2f} dB (at {worst_heading_sinr} deg)")
        results.append((genome, n_act_min, sinr_min))
        
    print("\n--- FINAL SELECTION ---")
    valid_genomes = [r for r in results if r[1] >= 13]
    
    if len(valid_genomes) == 0:
        print("WARNING: NO GENOMES CLEARED THE N_ACT_MIN >= 13 TARGET!")
        print("The optimization may be fundamentally restricted by the element count or deduplication radius.")
        best_overall = max(results, key=lambda x: x[2])
        print(f"Best available (failed target): {best_overall[0]} | N_act_min: {best_overall[1]} | SINR trough: {best_overall[2]:.2f} dB")
    else:
        print(f"{len(valid_genomes)} genomes cleared the target!")
        best_overall = max(valid_genomes, key=lambda x: x[2])
        print(f"WINNING GENOME: {best_overall[0]} | N_act_min: {best_overall[1]} | SINR trough: {best_overall[2]:.2f} dB")
        np.save("ultimate_optimal_genome.npy", best_overall[0])
        print("Saved to ultimate_optimal_genome.npy")

if __name__ == "__main__":
    main()
