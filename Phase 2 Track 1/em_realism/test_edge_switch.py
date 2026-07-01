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

def test_edge_switch():
    mesh = load_uav_mesh(Path(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone_prepared.stl"))
    antenna_positions_body, normals_body = get_conformal_array(mesh)
    
    # Jammer at azimuth 90, elevation 0 (inertial frame)
    jammer_dir_inertial = np.array([0, 1, 0])
    
    rolls = np.linspace(-45, 45, 180) # finer resolution 0.5 degrees
    
    # We will track element 0 (right wing tip)
    idx = 0
    
    h_vals = []
    d1_vals = []
    nu_vals = []
    F_mags = []
    
    lam = 0.125 # 2.4 GHz
    
    for roll in rolls:
        phi = np.radians(roll)
        theta = 0 # pitch
        psi = 0 # yaw
        
        q = euler_to_quaternion(phi, theta, psi)
        
        # Rotate jammer_dir from inertial to body
        q_inv = q.conjugate()
        jammer_dir_body = rotate_vector(jammer_dir_inertial, q_inv)
        
        # 1. Silhouette extraction in body frame
        face_normals = mesh.face_normals
        dot_prod = np.dot(face_normals, jammer_dir_body)
        
        front_facing = dot_prod > 1e-6
        back_facing = dot_prod < -1e-6
        
        adj = mesh.face_adjacency
        edges = mesh.face_adjacency_edges
        
        is_silhouette = (front_facing[adj[:, 0]] & back_facing[adj[:, 1]]) | \
                        (back_facing[adj[:, 0]] & front_facing[adj[:, 1]])
                        
        sil_edges = edges[is_silhouette]
        V1_all = mesh.vertices[sil_edges[:, 0]]
        V2_all = mesh.vertices[sil_edges[:, 1]]
        
        # Length filter > 1.5 cm
        edge_lens = np.linalg.norm(V1_all - V2_all, axis=1)
        valid_mask = edge_lens > 0.015
        
        # Fallback: if no edges pass the filter, use all silhouette edges
        if not np.any(valid_mask):
            valid_mask = np.ones(len(sil_edges), dtype=bool)
            
        V1 = V1_all[valid_mask]
        V2 = V2_all[valid_mask]
        
        epsilon = 1e-4
        ray_origins = antenna_positions_body + normals_body * epsilon
        
        h, d1 = compute_distances(ray_origins, jammer_dir_body, V1, V2)
        
        # Ray intersections
        ray_hits = mesh.ray.intersects_any(ray_origins, np.tile(jammer_dir_body, (16, 1)))
        hit = ray_hits[idx]
        
        # Closest edge for element 0
        h_idx = h[idx, :]
        d1_idx = d1[idx, :]
        
        # Standard approach: strict min
        min_i = np.argmin(h_idx)
        min_h = h_idx[min_i]
        min_d1 = d1_idx[min_i]
        
        sign = 1.0 if hit else -1.0
        nu = sign * min_h * np.sqrt(2 / (lam * min_d1))
        F = fresnel_diffraction_gain(nu)
        
        h_vals.append(min_h * sign)
        d1_vals.append(min_d1)
        nu_vals.append(nu)
        F_mags.append(np.abs(F))
        
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(rolls, h_vals)
    plt.title("h (sign applied) vs Roll")
    plt.grid()
    
    plt.subplot(2, 2, 2)
    plt.plot(rolls, d1_vals)
    plt.title("d1 vs Roll")
    plt.grid()
    
    plt.subplot(2, 2, 3)
    plt.plot(rolls, nu_vals)
    plt.title("nu vs Roll")
    plt.grid()
    
    plt.subplot(2, 2, 4)
    plt.plot(rolls, 20*np.log10(F_mags))
    plt.title("|F| dB vs Roll")
    plt.grid()
    
    plt.tight_layout()
    plt.savefig('edge_switch_sweep.png')
    print("Saved edge_switch_sweep.png")

if __name__ == "__main__":
    test_edge_switch()
