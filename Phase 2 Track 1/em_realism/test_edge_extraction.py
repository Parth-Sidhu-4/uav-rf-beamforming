import numpy as np
import trimesh
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Stage 8 (Mesh Ray-Tracing)')))
from mesh_loader import load_uav_mesh
from conformal_array import get_conformal_array

def compute_distances(P_rays, D_ray, V1, V2):
    """
    Computes shortest distance h and d1 from N rays to M line segments.
    P_rays: (N, 3) origins
    D_ray: (3,) direction, assumed normalized
    V1: (M, 3) start points of segments
    V2: (M, 3) end points of segments
    
    Returns:
    h: (N, M) minimum distances
    d1: (N, M) distance from ray origin to the foot on the segment
    """
    N = P_rays.shape[0]
    M = V1.shape[0]
    
    E = V2 - V1 # (M, 3)
    D = D_ray.reshape(1, 3) # (1, 3)
    
    # Broadcast to (N, M, 3)
    # W0 = P - V1 -> W0[i, j] = P[i] - V1[j]
    P_exp = P_rays[:, np.newaxis, :] # (N, 1, 3)
    V1_exp = V1[np.newaxis, :, :] # (1, M, 3)
    W0 = P_exp - V1_exp # (N, M, 3)
    
    a = np.sum(D**2) # scalar 1.0
    b = np.sum(D * E, axis=1) # (M,)
    c = np.sum(E**2, axis=1) # (M,)
    
    # b can be reshaped to (1, M)
    b = b[np.newaxis, :]
    c = c[np.newaxis, :]
    
    d = np.sum(D[np.newaxis, :, :] * W0, axis=2) # (N, M)
    e = np.sum(E[np.newaxis, :, :] * W0, axis=2) # (N, M)
    
    denom = a * c - b**2 # (1, M)
    # Handle parallel lines
    denom = np.where(denom < 1e-8, 1e-8, denom)
    
    s_c = (a * e - b * d) / denom # (N, M)
    
    # Clamp s_c to [0, 1]
    s_c_clamped = np.clip(s_c, 0.0, 1.0)
    
    # Re-solve t_c
    # a*t - b*s = -d -> t = (b*s - d) / a
    t_c = (b * s_c_clamped - d) / a # (N, M)
    
    # Clamp t_c to >= 0 (ray only goes forward)
    t_c_clamped = np.maximum(t_c, 0.0)
    
    # Re-solve s_c if t_c was clamped
    # We only care about the distance, but let's be rigorous
    # Actually, if t_c was clamped to 0, the closest point on the ray is the origin.
    # The closest point on the segment to the origin P is found by minimizing |P - (V1 + s*E)|
    # Gradient w.r.t s: -E . (P - V1 - sE) = 0 => s = E.(P-V1) / |E|^2
    # E.(P-V1) = -e. |E|^2 = c. So s = -e / c.
    s_c_recalc = np.where(t_c < 0.0, -e / c, s_c_clamped)
    s_c_recalc_clamped = np.clip(s_c_recalc, 0.0, 1.0)
    
    # Final parameters
    s_final = np.where(t_c < 0.0, s_c_recalc_clamped, s_c_clamped)
    t_final = t_c_clamped
    
    # P_edge[i, j] = V1[j] + s_final[i, j] * E[j]
    E_exp = E[np.newaxis, :, :] # (1, M, 3)
    s_final_exp = s_final[:, :, np.newaxis] # (N, M, 1)
    P_edge = V1_exp + s_final_exp * E_exp # (N, M, 3)
    
    # P_ray[i, j] = P[i] + t_final[i, j] * D
    t_final_exp = t_final[:, :, np.newaxis] # (N, M, 1)
    D_exp = D[np.newaxis, :, :] # (1, 1, 3)
    P_ray = P_exp + t_final_exp * D_exp # (N, M, 3)
    
    h = np.linalg.norm(P_ray - P_edge, axis=2) # (N, M)
    d1 = np.linalg.norm(P_edge - P_exp, axis=2) # (N, M)
    
    return h, d1

def test_edge_extraction():
    from pathlib import Path
    mesh = load_uav_mesh(Path(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone_prepared.stl"))
    antenna_positions, normals = get_conformal_array(mesh)
    
    # Jammer at azimuth 90 (right side), elevation 0
    jammer_dir = np.array([0, 1, 0])
    
    # 1. Silhouette extraction
    face_normals = mesh.face_normals # (F, 3)
    dot_prod = np.dot(face_normals, jammer_dir)
    
    # faces pointing towards jammer have dot > 0
    front_facing = dot_prod > 1e-6
    back_facing = dot_prod < -1e-6
    
    # Find edges connecting front and back facing
    adj = mesh.face_adjacency # (K, 2)
    edges = mesh.face_adjacency_edges # (K, 2)
    
    face0_front = front_facing[adj[:, 0]]
    face0_back = back_facing[adj[:, 0]]
    face1_front = front_facing[adj[:, 1]]
    face1_back = back_facing[adj[:, 1]]
    
    # Transition
    is_silhouette = (face0_front & face1_back) | (face0_back & face1_front)
    
    sil_edges = edges[is_silhouette]
    V1 = mesh.vertices[sil_edges[:, 0]]
    V2 = mesh.vertices[sil_edges[:, 1]]
    
    print(f"Total mesh edges: {len(edges)}")
    print(f"Silhouette edges found: {len(sil_edges)}")
    
    # Compute distances for a single antenna
    epsilon = 1e-4
    ray_origins = antenna_positions + normals * epsilon
    
    h, d1 = compute_distances(ray_origins, jammer_dir, V1, V2)
    
    print(f"h shape: {h.shape}")
    print(f"d1 shape: {d1.shape}")
    
    # Find closest edge for each antenna
    min_h_indices = np.argmin(h, axis=1)
    min_h = h[np.arange(len(ray_origins)), min_h_indices]
    min_d1 = d1[np.arange(len(ray_origins)), min_h_indices]
    
    print("Closest edge stats per antenna:")
    for i in range(len(ray_origins)):
        print(f"Antenna {i}: min_h = {min_h[i]:.4f} m, d1 = {min_d1[i]:.4f} m")

if __name__ == "__main__":
    test_edge_extraction()
