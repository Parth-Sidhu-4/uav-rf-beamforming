
import sys, os, numpy as np, time
sys.path.append(os.path.abspath(r'D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)'))
sys.path.append(os.path.abspath(r'D:\UAV Internship project\Phase 2 Track 1\em_realism'))
from conformal_array import get_conformal_array_parametric
from mesh_loader import load_uav_mesh
from pathlib import Path
from attitude import rotate_points, euler_to_quaternion
from shadow_engine_batched import compute_shadow_mask_batched
from multiprocessing import Pool

mesh = None
pos_body = None
normals_body = None

def init_worker():
    global mesh, pos_body, normals_body
    mesh = load_uav_mesh(Path(r'D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone_prepared.stl'))
    pos_body, normals_body = get_conformal_array_parametric(mesh, N=32)

def worker(jam_body_chunk):
    return compute_shadow_mask_batched(mesh, pos_body, normals_body, jam_body_chunk)

if __name__ == '__main__':
    grid_h = np.arange(165.0, 195.0 + 1e-5, 0.5)
    grid_e = np.arange(-30.0, 30.0 + 1e-5, 0.5)
    H_grid, E_grid = np.meshgrid(grid_h, grid_e, indexing='ij')
    flat_h_grid = H_grid.flatten()
    flat_e_grid = E_grid.flatten()
    
    h_rad = np.deg2rad(flat_h_grid)
    e_rad = np.deg2rad(flat_e_grid)
    x = np.cos(e_rad) * np.cos(h_rad)
    y = np.cos(e_rad) * np.sin(h_rad)
    z = np.sin(e_rad)
    jam_world = np.stack([x, y, z], axis=-1)
    
    q_inv = euler_to_quaternion(np.deg2rad(15.0), 0, 0).conjugate()
    jam_body = rotate_points(jam_world, q_inv)
    
    print(f'Running Batched Ray Tracer for {len(jam_body)} points...')
    start = time.time()
    
    chunk_size = 200
    chunks = [jam_body[i:i + chunk_size] for i in range(0, len(jam_body), chunk_size)]
    with Pool(processes=8, initializer=init_worker) as pool:
        results = pool.map(worker, chunks)
        
    g_jams = np.concatenate(results, axis=0)
    print(f'Done in {time.time()-start:.2f} s')
    
    theta_j = np.arccos(jam_body[:, 2] / np.linalg.norm(jam_body, axis=1))
    phi_j = np.arctan2(jam_body[:, 1], jam_body[:, 0])
    
    def get_steering_vector_pt_np(pos_lambda, theta, phi):
        dx = np.sin(theta) * np.cos(phi)
        dy = np.sin(theta) * np.sin(phi)
        dz = np.cos(theta)
        d = np.stack([dx, dy, dz], axis=-1)
        return np.exp(1j * 2.0 * np.pi * (d @ pos_lambda.T))
    
    init_worker()
    pos_lambda = pos_body / 0.15
    v_ideal = get_steering_vector_pt_np(pos_lambda, theta_j, phi_j)
    grid_v_flat = g_jams * v_ideal
    grid_v = grid_v_flat.reshape(len(grid_h), len(grid_e), 32)
    
    np.savez('true_oracle_grid_0_5deg.npz', grid_h=grid_h, grid_e=grid_e, grid_v=grid_v)
    print('Saved true_oracle_grid_0_5deg.npz')
