import numpy as np
import os
import sys
import time

sys.path.append(os.path.abspath(r'D:\UAV Internship project\Stage 12 (Operational Analysis)'))
sys.path.append(os.path.abspath(r'D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)'))
sys.path.append(os.path.abspath(r'D:\UAV Internship project\Stage 13 (AI Beamforming)'))

from mesh_loader import load_uav_mesh
from shadow_engine_batched import compute_shadow_mask_batched
from attitude import euler_to_quaternion, rotate_points

def process_chunk(args):
    idx, jam_chunk = args
    import sys, os
    sys.path.append(os.path.abspath(r'D:\UAV Internship project\Stage 12 (Operational Analysis)'))
    sys.path.append(os.path.abspath(r'D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)'))
    sys.path.append(os.path.abspath(r'D:\UAV Internship project\Stage 13 (AI Beamforming)'))
    sys.path.append(os.path.abspath(r'D:\UAV Internship project\Phase 2 Track 2\array_structures'))
    from mesh_loader import load_uav_mesh
    from conformal_array import get_conformal_array_parametric
    from shadow_engine_batched import compute_shadow_mask_batched
    from pathlib import Path
    
    mesh_path_local = Path(r'D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone_prepared.stl')
    mesh_local = load_uav_mesh(mesh_path_local)
    pos_body_local, normals_body_local = get_conformal_array_parametric(mesh_local, N=32)
    
    g_chunk = compute_shadow_mask_batched(mesh_local, pos_body_local, normals_body_local, jam_chunk)
    return idx, g_chunk

def main():
    print("======================================================")
    print(" GENERATING TRUE PHYSICS CACHE (TRIMESH RAY-TRACER)")
    print("======================================================")
    
    mesh_path = r'D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone_prepared.stl'
    print(f"Loading mesh: {mesh_path}")
    from pathlib import Path
    from mesh_loader import load_uav_mesh
    mesh = load_uav_mesh(Path(mesh_path))
    
    print("Loading conformal array...")
    sys.path.append(os.path.abspath(r'D:\UAV Internship project\Phase 2 Track 2\array_structures'))
    from conformal_array import get_conformal_array_parametric
    
    pos_body, normals_body = get_conformal_array_parametric(mesh, N=32)
    
    grid_h = np.arange(165.0, 195.0 + 1e-5, 1.0)
    grid_e = np.arange(-30.0, 30.0 + 1e-5, 1.0)
    
    H_grid, E_grid = np.meshgrid(grid_h, grid_e, indexing='ij')
    flat_h = H_grid.flatten()
    flat_e = E_grid.flatten()
    
    N_points = len(flat_h)
    print(f"Grid size: {len(grid_h)} x {len(grid_e)} = {N_points} points.")
    
    h_rad = np.deg2rad(flat_h)
    e_rad = np.deg2rad(flat_e)
    
    jam_body_x = np.cos(e_rad) * np.cos(h_rad)
    jam_body_y = np.cos(e_rad) * np.sin(h_rad)
    jam_body_z = np.sin(e_rad)
    jam_body = np.stack([jam_body_x, jam_body_y, jam_body_z], axis=1)
    
    start_time = time.perf_counter()
    chunk_size = 50
    g_all = np.zeros((N_points, 32), dtype=np.complex128)
    
    for i in range(0, N_points, chunk_size):
        end_idx = min(i + chunk_size, N_points)
        jam_chunk = jam_body[i:end_idx]
        g_chunk = compute_shadow_mask_batched(mesh, pos_body, normals_body, jam_chunk)
        g_all[i:end_idx] = g_chunk
        progress = end_idx / N_points * 100
        print(f"  Completed chunk to {end_idx}/{N_points} (progress: {progress:.1f}%)")
            
    print(f"Ray-tracing completed in {time.perf_counter() - start_time:.2f} seconds.")
    
    # frequency was 2.4e9. lambda = 0.125
    pos_lambda = pos_body / 0.125
    
    theta_j = np.arccos(jam_body[:, 2] / np.linalg.norm(jam_body, axis=1))
    phi_j = np.arctan2(jam_body[:, 1], jam_body[:, 0])
    
    dx = np.sin(theta_j) * np.cos(phi_j)
    dy = np.sin(theta_j) * np.sin(phi_j)
    dz = np.cos(theta_j)
    d = np.stack([dx, dy, dz], axis=1)
    
    v_ideal = np.exp(1j * 2.0 * np.pi * (d @ pos_lambda.T))
    v_true = g_all * v_ideal
    
    v_true_grid = v_true.reshape(len(grid_h), len(grid_e), 32)
    
    output_path = 'true_raytraced_grid.npz'
    np.savez(output_path, 
             grid_h=grid_h, 
             grid_e=grid_e, 
             v_true_grid=v_true_grid,
             g_grid=g_all.reshape(len(grid_h), len(grid_e), 32))
             
    print(f"Saved true physics cache to {output_path}")

if __name__ == '__main__':
    main()
