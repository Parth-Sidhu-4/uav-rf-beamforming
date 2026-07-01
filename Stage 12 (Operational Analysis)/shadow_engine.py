import sys
import os
import numpy as np
import trimesh
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Phase 2 Track 1', 'em_realism')))
from em_physics import fresnel_diffraction_gain, compute_distances

def compute_shadow_mask(mesh: trimesh.Trimesh, antenna_positions_body: np.ndarray, 
                        normals: np.ndarray, jammer_dir_body: np.ndarray) -> np.ndarray:
    """
    Computes the complex shadow mask g_i using BVH ray-tracing and Fresnel Knife-Edge Diffraction.
    Uses Epstein-Peterson approximation (Gain Product) for multiple obstructions.
    antenna_positions_body: Nx3 array of antenna coordinates in the drone's body frame.
    jammer_dir_body: 3D vector pointing from the drone TO the jammer in the body frame.
    
    Returns:
    g: Array of length N of complex gains.
    """
    N = antenna_positions_body.shape[0]
    lam = 0.125 # 2.4 GHz
    
    jammer_dir_body = jammer_dir_body / np.linalg.norm(jammer_dir_body)
    
    # Extract silhouette edges
    face_normals = mesh.face_normals
    dot_prod = np.dot(face_normals, jammer_dir_body)
    normals_body = normals
    jam_body = jammer_dir_body / np.linalg.norm(jammer_dir_body)
    
    # Use all edges for finding nearest points
    edges = mesh.face_adjacency_edges
    V1_all = mesh.vertices[edges[:, 0]]
    V2_all = mesh.vertices[edges[:, 1]]
    
    valid_mask = np.linalg.norm(V1_all - V2_all, axis=1) > 0.015
    if not np.any(valid_mask): valid_mask = np.ones(len(edges), dtype=bool)
        
    V1 = V1_all[valid_mask]
    V2 = V2_all[valid_mask]
    
    ray_origins = antenna_positions_body + normals_body * 1e-4
    ray_dirs = np.tile(jam_body, (N, 1))
    
    h, d1, W = compute_distances(ray_origins, jam_body, V1, V2)
    
    from trimesh.proximity import signed_distance
    
    g = np.ones(N, dtype=complex)
    
    for i in range(N):
        h_idx = h[i, :]
        d1_idx = d1[i, :]
        
        edge1_d1 = d1_idx[np.argmin(h_idx)]
        
        # Smooth separation weight for edge 2
        S_j = 1.0 - np.exp(- (d1_idx - edge1_d1)**2 / (2 * 0.2**2))
        
        h_penalized = h_idx / (S_j + 1e-8)
        idx2 = np.argmin(h_penalized)
        idx1 = np.argmin(h_idx)
        
        P_ray_1 = ray_origins[i] + d1_idx[idx1] * jam_body
        P_ray_2 = ray_origins[i] + d1_idx[idx2] * jam_body
        
        # trimesh signed_distance: positive is INSIDE, negative is OUTSIDE.
        # We want nu > 0 for shadowed (INSIDE). So we use it directly!
        sd1 = signed_distance(mesh, [P_ray_1])[0]
        sd2 = signed_distance(mesh, [P_ray_2])[0]
        
        nu1 = sd1 * np.sqrt(2 / (lam * d1_idx[idx1]))
        F1 = fresnel_diffraction_gain(nu1)
        
        nu2 = sd2 * np.sqrt(2 / (lam * d1_idx[idx2]))
        F2 = fresnel_diffraction_gain(nu2)
        
        # Smoothly fade F2 based on separation S_j to avoid any hard cutoffs.
        # If the second edge is very close to the first (S_j -> 0), it collapses to a single edge (F2 -> 1.0).
        weight2 = S_j[idx2]
        F2_blended = (1.0 - weight2) * 1.0 + weight2 * F2
        
        g[i] = F1 * F2_blended
                
    # --- Update Cache ---
            
    return g
