import numpy as np
from pathlib import Path
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Stage 8 (Mesh Ray-Tracing)')))
from mesh_loader import load_uav_mesh
from conformal_array import get_conformal_array
from attitude import euler_to_quaternion, rotate_vector
from test_edge_extraction import compute_distances

def run_diagnostic():
    mesh = load_uav_mesh(Path(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone_prepared.stl"))
    antenna_positions_body, normals_body = get_conformal_array(mesh)
    
    jammer_dir_inertial = np.array([0, 1, 0])
    idx = 0 # Element 0 (right wing tip)
    
    # We want to check rolls right around the spikes
    # The spikes were around -2 and 9 in the plot. Let's do a fine sweep around those.
    test_rolls = [-3.0, -2.5, -2.0, -1.5, -1.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]
    
    print(f"{'Roll':>5} | {'min_h':>7} | {'d1':>7} | {'Len':>6} | {'V1_x':>7} | {'V1_y':>7} | {'V1_z':>7}")
    print("-" * 70)
    
    for roll in test_rolls:
        phi = np.radians(roll)
        q = euler_to_quaternion(phi, 0, 0)
        q_inv = q.conjugate()
        jammer_dir_body = rotate_vector(jammer_dir_inertial, q_inv)
        
        face_normals = mesh.face_normals
        dot_prod = np.dot(face_normals, jammer_dir_body)
        
        front_facing = dot_prod > 1e-6
        back_facing = dot_prod < -1e-6
        
        adj = mesh.face_adjacency
        edges = mesh.face_adjacency_edges
        
        is_silhouette = (front_facing[adj[:, 0]] & back_facing[adj[:, 1]]) | \
                        (back_facing[adj[:, 0]] & front_facing[adj[:, 1]])
                        
        sil_edges = edges[is_silhouette]
        V1 = mesh.vertices[sil_edges[:, 0]]
        V2 = mesh.vertices[sil_edges[:, 1]]
        
        epsilon = 1e-4
        ray_origins = antenna_positions_body + normals_body * epsilon
        
        h, d1 = compute_distances(ray_origins, jammer_dir_body, V1, V2)
        
        h_idx = h[idx, :]
        d1_idx = d1[idx, :]
        
        min_i = np.argmin(h_idx)
        min_h = h_idx[min_i]
        min_d1 = d1_idx[min_i]
        
        v1_coord = V1[min_i]
        v2_coord = V2[min_i]
        edge_len = np.linalg.norm(v1_coord - v2_coord)
        
        print(f"{roll:5.1f} | {min_h:7.4f} | {min_d1:7.4f} | {edge_len:6.4f} | {v1_coord[0]:7.4f} | {v1_coord[1]:7.4f} | {v1_coord[2]:7.4f}")

if __name__ == "__main__":
    run_diagnostic()
