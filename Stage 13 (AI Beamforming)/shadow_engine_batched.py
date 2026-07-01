import numpy as np
from em_physics import compute_distances, fresnel_diffraction_gain
from trimesh.proximity import signed_distance

def compute_shadow_mask_batched(mesh, antenna_positions_body, normals_body, jam_body_batch):
    """
    Computes shadow masks for a batch of jammer vectors to avoid trimesh overhead.
    jam_body_batch: (B, 3)
    Returns: (B, 16) complex gains
    """
    B = jam_body_batch.shape[0]
    N = antenna_positions_body.shape[0]
    lam = 3e8 / 2.4e9
    
    edges = mesh.face_adjacency_edges
    V1_all = mesh.vertices[edges[:, 0]]
    V2_all = mesh.vertices[edges[:, 1]]
    valid_mask = np.linalg.norm(V1_all - V2_all, axis=1) > 0.015
    if not np.any(valid_mask): valid_mask = np.ones(len(edges), dtype=bool)
    V1 = V1_all[valid_mask]
    V2 = V2_all[valid_mask]
    
    g_batch = np.ones((B, N), dtype=complex)
    ray_origins = antenna_positions_body + normals_body * 1e-4
    
    # We will collect all signed distance queries
    all_P_rays = np.zeros((B, N, 2, 3))
    idx1_all = np.zeros((B, N), dtype=int)
    idx2_all = np.zeros((B, N), dtype=int)
    weight2_all = np.zeros((B, N))
    d1_idx1_all = np.zeros((B, N))
    d1_idx2_all = np.zeros((B, N))
    
    for b in range(B):
        jam_body = jam_body_batch[b]
        # h, d1 are (N, M)
        h, d1, W = compute_distances(ray_origins, jam_body, V1, V2)
        
        for i in range(N):
            h_idx = h[i, :]
            d1_idx = d1[i, :]
            
            edge1_d1 = d1_idx[np.argmin(h_idx)]
            S_j = 1.0 - np.exp(- (d1_idx - edge1_d1)**2 / (2 * 0.2**2))
            h_penalized = h_idx / (S_j + 1e-8)
            
            idx2 = np.argmin(h_penalized)
            idx1 = np.argmin(h_idx)
            
            idx1_all[b, i] = idx1
            idx2_all[b, i] = idx2
            weight2_all[b, i] = S_j[idx2]
            d1_idx1_all[b, i] = d1_idx[idx1]
            d1_idx2_all[b, i] = d1_idx[idx2]
            
            all_P_rays[b, i, 0, :] = ray_origins[i] + d1_idx[idx1] * jam_body
            all_P_rays[b, i, 1, :] = ray_origins[i] + d1_idx[idx2] * jam_body

    # BATCHED QUERY
    flat_rays = all_P_rays.reshape(-1, 3)
    sd_flat = signed_distance(mesh, flat_rays)
    sd_reshaped = sd_flat.reshape(B, N, 2)
    
    for b in range(B):
        for i in range(N):
            sd1 = sd_reshaped[b, i, 0]
            sd2 = sd_reshaped[b, i, 1]
            
            nu1 = sd1 * np.sqrt(2 / (lam * np.maximum(d1_idx1_all[b, i], 1e-5)))
            F1 = fresnel_diffraction_gain(nu1)
            
            nu2 = sd2 * np.sqrt(2 / (lam * np.maximum(d1_idx2_all[b, i], 1e-5)))
            F2 = fresnel_diffraction_gain(nu2)
            
            weight2 = weight2_all[b, i]
            F2_blended = (1.0 - weight2) * 1.0 + weight2 * F2
            
            g_batch[b, i] = F1 * F2_blended
            
    return g_batch
