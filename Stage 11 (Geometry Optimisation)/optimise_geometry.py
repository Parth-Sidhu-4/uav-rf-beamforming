import numpy as np
from pathlib import Path
from scipy.optimize import differential_evolution
import time
import trimesh
import os

from mesh_loader import load_uav_mesh
from attitude import euler_to_quaternion, rotate_points
from em_physics import compute_distances, fresnel_diffraction_gain, compute_plf
from constants import (
    NACT_THRESHOLD, SIGMA_TEMPERATURE, SOFTMIN_BETA,
    MIN_SPACING_M, SPACING_PENALTY_WEIGHT, DE_POPSIZE, DE_MAXITER
)

def softmin_orientations(x, beta=SOFTMIN_BETA):
    x_min = np.min(x)
    return x_min - np.log(np.sum(np.exp(-beta * (x - x_min)))) / beta

def soft_count(gains_amp, tau=NACT_THRESHOLD, T=SIGMA_TEMPERATURE):
    return np.sum(1.0 / (1.0 + np.exp(-(gains_amp - tau) / T)))

# Global cache for the DE
CACHE = {}

def evaluate_genome(indices, headings_to_eval):
    indices = np.round(indices).astype(int)
    
    # Soft penalty for duplicates
    unique_indices = np.unique(indices)
    duplicate_penalty = (len(indices) - len(unique_indices)) * 100.0
    
    pos_right = CACHE['candidates_right'][indices]
    pos_left = CACHE['candidates_left'][indices]
    antenna_positions_body = np.vstack([pos_right, pos_left])
    
    norm_right = CACHE['normals_right'][indices]
    norm_left = CACHE['normals_left'][indices]
    normals_body = np.vstack([norm_right, norm_left])
    
    # Mutual coupling soft penalty
    dist_penalty = 0.0
    for i in range(16):
        for j in range(i+1, 16):
            d = np.linalg.norm(antenna_positions_body[i] - antenna_positions_body[j])
            if d < MIN_SPACING_M:
                dist_penalty += SPACING_PENALTY_WEIGHT * (MIN_SPACING_M - d)**2
                
    # If heavily penalised, return early to save compute
    if duplicate_penalty > 0 or dist_penalty > 1000.0:
        return -(0.0 - duplicate_penalty - dist_penalty) # We want to minimize -fitness
        
    mesh = CACHE['mesh']
    edges = mesh.face_adjacency_edges
    V1 = mesh.vertices[edges[:, 0]]
    V2 = mesh.vertices[edges[:, 1]]
    valid_mask = np.linalg.norm(V1 - V2, axis=1) > 0.015
    if np.any(valid_mask):
        V1 = V1[valid_mask]
        V2 = V2[valid_mask]
        
    n_act_soft_list = []
    
    ray_origins = antenna_positions_body + normals_body * 1e-4
    
    for phi_cmd in CACHE['bank_angles']:
        for h_ang in headings_to_eval:
            # Autopilot logic: Bank TOWARDS the jammer
            actual_phi = phi_cmd if h_ang <= 180 else -phi_cmd
            q = euler_to_quaternion(np.deg2rad(actual_phi), 0, np.deg2rad(h_ang))
            jam_body = rotate_points(CACHE['jam_world'].reshape(1,3), q.conjugate())[0]
            
            plf_amp = compute_plf(normals_body, q, CACHE['jam_pol_world'])
            
            h_all, d1_all, W_all = compute_distances(ray_origins, jam_body, V1, V2)
            edge_dirs = (V2 - V1) / np.linalg.norm(V2 - V1, axis=1)[:, np.newaxis]
            
            gains = np.zeros(16, dtype=complex)
            for i in range(16):
                W_i = W_all[i, :, :]
                d1_idx = d1_all[i, :]
                
                beta = 100.0
                h_idx = h_all[i, :]
                
                valid_edges = d1_idx > 0.01
                if not np.any(valid_edges):
                    gains[i] = 1.0 * plf_amp[i]
                    continue
                
                # Apply mask to distances
                h_valid = h_idx[valid_edges]
                d1_valid = d1_idx[valid_edges]
                edge_idx = np.arange(len(h_idx))[valid_edges]

                h_min = np.min(h_valid)
                log_w = -beta * (h_valid - h_min)
                log_w -= np.max(log_w)
                w_soft = np.exp(log_w)
                w_soft /= w_soft.sum()

                edge1_d1 = np.dot(w_soft, d1_valid)

                S_j = 1.0 - np.exp(-(d1_valid - edge1_d1)**2 / (2 * 0.2**2))
                h_penalized = h_valid / (S_j + 1e-8)

                top3 = np.argsort(h_valid)[:3]
                w_top3 = w_soft[top3]; w_top3 /= w_top3.sum()
                F1_soft = 0.0
                for k, ek_valid in enumerate(top3):
                    ek = edge_idx[ek_valid]
                    
                    # Targeted cross-product projection for sign instead of KDTree
                    W_k = W_i[ek]
                    cross_k = np.cross(W_k, jam_body)
                    dot_k = np.dot(cross_k, edge_dirs[ek])
                    sd_k = np.sign(dot_k) * h_idx[ek]
                    
                    nu_k = sd_k * np.sqrt(2 / (CACHE['lam'] * d1_idx[ek]))
                    F1_soft += w_top3[k] * fresnel_diffraction_gain(nu_k)

                h_pen_min = np.min(h_penalized)
                log_w2 = -beta * (h_penalized - h_pen_min)
                log_w2 -= np.max(log_w2)
                w_soft2 = np.exp(log_w2); w_soft2 /= w_soft2.sum()
                edge2_d1 = np.dot(w_soft2, d1_valid)
                weight2  = np.dot(w_soft2, S_j)

                top3_2 = np.argsort(h_penalized)[:3]
                w_top3_2 = w_soft2[top3_2]; w_top3_2 /= w_top3_2.sum()

                F2_soft = 0.0
                for k, ek_valid in enumerate(top3_2):
                    ek = edge_idx[ek_valid]
                    
                    # Targeted cross-product projection
                    W_k = W_i[ek]
                    cross_k = np.cross(W_k, jam_body)
                    dot_k = np.dot(cross_k, edge_dirs[ek])
                    sd_k2 = np.sign(dot_k) * h_idx[ek]
                    
                    d_eff_k = edge1_d1 + d1_idx[ek]
                    nu_k2 = sd_k2 * np.sqrt(2 / (CACHE['lam'] * d_eff_k))
                    F2_soft += w_top3_2[k] * fresnel_diffraction_gain(nu_k2)

                F2_blended = (1.0 - weight2) * 1.0 + weight2 * F2_soft
                gains[i] = F1_soft * F2_blended * plf_amp[i]
                
            n_act_soft_list.append(soft_count(np.abs(gains)))
            
    # Softmin over all tested orientations
    n_act_min_surrogate = softmin_orientations(n_act_soft_list)
    
    # DE minimizes the objective, so we return negative fitness
    fitness = n_act_min_surrogate - duplicate_penalty - dist_penalty
    fitness_val = -fitness
    
    with open(f"eval_log_{os.getpid()}.txt", "a") as f:
        f.write(f"{indices.tolist()},{fitness_val}\n")
        
    return fitness_val

print("Loading mesh and candidate pool globally for multiprocessing...")
mesh_path = Path(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone_prepared.stl")
mesh = load_uav_mesh(mesh_path)
CACHE['mesh'] = mesh

cand_right = np.load("candidate_pairs_right.npy")
cand_left = np.load("candidate_pairs_left.npy")

print("Computing candidate normals...")
_, _, tri_right = trimesh.proximity.closest_point(mesh, cand_right)
norm_right = mesh.face_normals[tri_right]

_, _, tri_left = trimesh.proximity.closest_point(mesh, cand_left)
norm_left = mesh.face_normals[tri_left]

jam_pos_xy = np.array([0.0, 100000.0])
jam_world = np.array([jam_pos_xy[0], jam_pos_xy[1], 0.0])
jam_world /= np.linalg.norm(jam_world)

CACHE['mesh'] = mesh
CACHE['candidates_right'] = cand_right
CACHE['candidates_left'] = cand_left
CACHE['normals_right'] = norm_right
CACHE['normals_left'] = norm_left
CACHE['jam_world'] = jam_world
CACHE['jam_pol_world'] = np.array([0.0, 0.0, -1.0])
CACHE['lam'] = 0.125
CACHE['bank_angles'] = [15.0, 22.0, 30.0, 38.0, 45.0]
CACHE['headings'] = np.linspace(0, 360, 6, endpoint=False)

import subprocess
import ast

def main():
    K = len(cand_right)
    bounds = [(0, K-1) for _ in range(8)]
    
    headings_to_eval = [0.0, 60.0, 120.0, 180.0, 240.0, 300.0]
    
    max_refinements = 3
    for refinement_loop in range(max_refinements):
        print(f"\n--- Refinement Loop {refinement_loop+1}/{max_refinements} ---")
        print(f"Current Headings Grid: {headings_to_eval}")
        
        # Clear log files
        for f in Path(".").glob("eval_log_*.txt"):
            f.unlink()
            
        print(f"Starting DE with popsize={DE_POPSIZE}, integrality=True, over {K} candidates...")
        
        def callback(xk, convergence):
            obj = evaluate_genome(xk, headings_to_eval)
            print(f"Current best fitness: {-obj:.3f} | Convergence: {convergence:.3f}")
            
        res = differential_evolution(
            evaluate_genome,
            bounds,
            args=(headings_to_eval,),
            integrality=True,
            popsize=DE_POPSIZE,
            maxiter=DE_MAXITER,
            workers=-1,
            callback=callback,
            disp=True
        )
        
        print("\nDE Optimisation Complete!")
        best_genome = np.round(res.x).astype(int)
        print("Best genome:", best_genome)
        print("Best surrogate fitness:", -res.fun)
        
        # Select best genome from logs and validate
        subprocess.run(["python", "select_best_genome.py"], check=True)
        print("\nRunning Validation Gate...")
        val_res = subprocess.run(["python", "validate_top10.py"], capture_output=True, text=True)
        print(val_res.stdout)
        
        if "WARNING: NO GENOMES CLEARED THE N_ACT_MIN >= 13 TARGET!" not in val_res.stdout:
            print("\nValidation Target Cleared! Optimisation successful.")
            break
            
        # Parse worst headings from validation output for the Top 1 genome
        # The output looks like: 
        # Evaluating Rank 1 Genome: ...
        #  -> N_act_min: X (at Y deg) | SINR trough: Z dB (at W deg)
        worst_heading = None
        for line in val_res.stdout.split('\n'):
            if "Evaluating Rank 1 Genome" in line:
                pass
            elif "N_act_min:" in line and "(at " in line:
                # Extract Y from "(at Y deg)"
                import re
                m = re.search(r'\(at (\d+) deg\)', line)
                if m:
                    worst_heading = float(m.group(1))
                break # Only process Rank 1 for feedback
                
        if worst_heading is not None:
            if worst_heading in headings_to_eval:
                print(f"Worst heading {worst_heading} is already in the training grid! We are fundamentally limited.")
                break
            print(f"Injecting worst-case heading {worst_heading} deg into the training grid and restarting DE...")
            headings_to_eval.append(worst_heading)
        else:
            print("Could not parse worst heading. Terminating.")
            break

if __name__ == "__main__":
    main()
