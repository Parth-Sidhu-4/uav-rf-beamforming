import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Stage 8 (Mesh Ray-Tracing)')))
from mesh_loader import load_uav_mesh
from conformal_array import get_conformal_array
from attitude import euler_to_quaternion, rotate_vector
from test_edge_extraction import compute_distances
from em_physics import fresnel_diffraction_gain

def softmin(x, k=100):
    """Log-Sum-Exp softmin"""
    # x is (N,)
    # we want sum(x * exp(-k*x)) / sum(exp(-k*x))
    # To avoid overflow, shift by min
    x_min = np.min(x)
    w = np.exp(-k * (x - x_min))
    w = w / np.sum(w)
    return np.sum(w * x), w

def run_blend_test():
    mesh = load_uav_mesh(Path(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone_prepared.stl"))
    antenna_positions_body, normals_body = get_conformal_array(mesh)
    
    jammer_dir_inertial = np.array([0, 1, 0])
    idx = 0
    
    rolls = np.linspace(-5, 2, 140) # fine sweep around the -2 spike
    
    d1_strict = []
    F_strict = []
    
    d1_geom = []
    F_geom = []
    
    F_prod = []
    
    lam = 0.125
    
    for roll in rolls:
        phi = np.radians(roll)
        q = euler_to_quaternion(phi, 0, 0)
        q_inv = q.conjugate()
        jammer_dir_body = rotate_vector(jammer_dir_inertial, q_inv)
        
        face_normals = mesh.face_normals
        dot_prod = np.dot(face_normals, jammer_dir_body)
        
        is_silhouette = (dot_prod[mesh.face_adjacency[:, 0]] > 1e-6) & (dot_prod[mesh.face_adjacency[:, 1]] < -1e-6) | \
                        (dot_prod[mesh.face_adjacency[:, 0]] < -1e-6) & (dot_prod[mesh.face_adjacency[:, 1]] > 1e-6)
                        
        sil_edges = mesh.face_adjacency_edges[is_silhouette]
        V1_all = mesh.vertices[sil_edges[:, 0]]
        V2_all = mesh.vertices[sil_edges[:, 1]]
        
        edge_lens = np.linalg.norm(V1_all - V2_all, axis=1)
        valid = edge_lens > 0.015
        if not np.any(valid): valid = np.ones(len(sil_edges), dtype=bool)
        
        V1 = V1_all[valid]
        V2 = V2_all[valid]
        
        ray_origins = antenna_positions_body + normals_body * 1e-4
        h, d1 = compute_distances(ray_origins, jammer_dir_body, V1, V2)
        
        h_idx = h[idx, :]
        d1_idx = d1[idx, :]
        
        hit = mesh.ray.intersects_any(ray_origins, np.tile(jammer_dir_body, (16, 1)))[idx]
        sign = 1.0 if hit else -1.0
        
        # Method 1: Strict Argmin
        min_i = np.argmin(h_idx)
        min_h = h_idx[min_i]
        min_d1 = d1_idx[min_i]
        nu_strict = sign * min_h * np.sqrt(2 / (lam * min_d1))
        F_s = fresnel_diffraction_gain(nu_strict)
        d1_strict.append(min_d1)
        F_strict.append(np.abs(F_s))
        
        # Identify "local" and "far" edge clusters to avoid double counting same wing
        # Sort by h
        sort_idx = np.argsort(h_idx)
        h_sorted = h_idx[sort_idx]
        d1_sorted = d1_idx[sort_idx]
        
        # Find closest local edge (d1 < 0.5) and closest far edge (d1 > 1.0)
        # We only care if they have competitive h values
        # Let's just use top 2 physically separated edges
        edge1_h = h_sorted[0]
        edge1_d1 = d1_sorted[0]
        
        edge2_h = None
        edge2_d1 = None
        for i in range(1, len(h_sorted)):
            if np.abs(d1_sorted[i] - edge1_d1) > 0.5: # physically separated by 0.5m
                edge2_h = h_sorted[i]
                edge2_d1 = d1_sorted[i]
                break
                
        if edge2_h is None:
            # No physically separated edge found, just fallback to strict
            edge2_h = edge1_h
            edge2_d1 = edge1_d1
            
        # Method 2: Geometric LSE Softmin over the two separated edges
        # weight w = exp(-k * h)
        h_pair = np.array([edge1_h, edge2_h])
        d1_pair = np.array([edge1_d1, edge2_d1])
        h_blend, w = softmin(h_pair, k=100)
        d1_blend = np.sum(w * d1_pair)
        nu_geom = sign * h_blend * np.sqrt(2 / (lam * d1_blend))
        F_g = fresnel_diffraction_gain(nu_geom)
        d1_geom.append(d1_blend)
        F_geom.append(np.abs(F_g))
        
        # Method 3: Gain Product
        # F1 * F2. We only want F2 to reduce gain if it is shadowed. 
        # If the ray clears edge 2, it shouldn't multiply gain by > 1 again (since F(-inf) -> 1).
        # Actually F(-inf) -> 1, so F1 * F2 is well behaved!
        nu1 = sign * edge1_h * np.sqrt(2 / (lam * edge1_d1))
        nu2 = sign * edge2_h * np.sqrt(2 / (lam * edge2_d1))
        
        F1 = fresnel_diffraction_gain(nu1)
        F2 = fresnel_diffraction_gain(nu2)
        
        # To avoid double-counting unobstructed path, F_prod is F1 * F2. 
        # But wait, if both are perfectly illuminated, F1=1, F2=1, F1*F2=1. This is correct!
        # If edge 1 is shadow (F=0.5) and edge 2 is shadow (F=0.5), F1*F2 = 0.25.
        # But are they really two screens? Yes, one on right wing, one on left wing.
        # We need a smooth inclusion weight so the far edge doesn't suddenly "snap" into existence.
        # Actually, if we just multiply F1 * F2 everywhere, it is always smooth!
        F_p = F1 * F2
        F_prod.append(np.abs(F_p))

    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(rolls, d1_strict, label='Strict Argmin')
    plt.plot(rolls, d1_geom, label='Geom LSE', linestyle='--')
    plt.title("d1 vs Roll")
    plt.legend()
    plt.grid()
    
    plt.subplot(1, 3, 2)
    plt.plot(rolls, 20*np.log10(F_strict), label='Strict Argmin')
    plt.plot(rolls, 20*np.log10(F_geom), label='Geom LSE', linestyle='--')
    plt.title("Gain |F| (dB) - Geometry Blend")
    plt.legend()
    plt.grid()
    
    plt.subplot(1, 3, 3)
    plt.plot(rolls, 20*np.log10(F_strict), label='Strict Argmin')
    plt.plot(rolls, 20*np.log10(F_prod), label='Gain Product', linestyle='--')
    plt.title("Gain |F| (dB) - Epstein-Peterson")
    plt.legend()
    plt.grid()
    
    plt.tight_layout()
    plt.savefig('blend_test.png')
    print("Saved blend_test.png")

if __name__ == "__main__":
    run_blend_test()
