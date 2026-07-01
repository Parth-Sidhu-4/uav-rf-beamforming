import numpy as np
from pathlib import Path
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Stage 8 (Mesh Ray-Tracing)')))
from mesh_loader import load_uav_mesh
from conformal_array import get_conformal_array
from attitude import euler_to_quaternion, rotate_vector
from test_edge_extraction import compute_distances

def run():
    mesh = load_uav_mesh(Path(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone_prepared.stl"))
    antenna_positions_body, normals_body = get_conformal_array(mesh)
    
    jammer_dir_inertial = np.array([0, 1, 0])
    idx = 0
    
    test_rolls = [-3.0, -2.5, -2.0, -1.5, -1.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]
    
    face_areas = mesh.area_faces
    
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
        V1_all = mesh.vertices[sil_edges[:, 0]]
        V2_all = mesh.vertices[sil_edges[:, 1]]
        
        adj_sil = adj[is_silhouette]
        area0 = face_areas[adj_sil[:, 0]]
        area1 = face_areas[adj_sil[:, 1]]
        min_area = np.minimum(area0, area1)
        
        # Test edge length filter
        edge_lens = np.linalg.norm(V1_all - V2_all, axis=1)
        
        for filter_type in ['none', 'length', 'area']:
            if filter_type == 'none':
                valid = np.ones(len(sil_edges), dtype=bool)
            elif filter_type == 'length':
                valid = edge_lens > 0.015
            else:
                valid = min_area > 1e-5 # 0.1 sq cm
                
            V1 = V1_all[valid]
            V2 = V2_all[valid]
            
            epsilon = 1e-4
            ray_origins = antenna_positions_body + normals_body * epsilon
            h, d1 = compute_distances(ray_origins, jammer_dir_body, V1, V2)
            
            h_idx = h[idx, :]
            d1_idx = d1[idx, :]
            
            min_i = np.argmin(h_idx)
            min_h = h_idx[min_i]
            min_d1 = d1_idx[min_i]
            
            orig_i = np.where(valid)[0][min_i]
            edge_l = edge_lens[orig_i]
            a0 = area0[orig_i]
            a1 = area1[orig_i]
            v1_coord = V1_all[orig_i]
            
            print(f"Roll {roll:5.1f} | {filter_type:>6} | min_h: {min_h:.4f} | d1: {min_d1:.4f} | Len: {edge_l:.4f} | minA: {min(a0, a1):.6f} | V1: {v1_coord[0]:7.4f}, {v1_coord[1]:7.4f}, {v1_coord[2]:7.4f}")
        print("-" * 110)

if __name__ == "__main__":
    run()
