import os
import sys
import numpy as np
import time
from pathlib import Path
from tqdm import tqdm

sys.path.append(os.path.abspath(r'D:\UAV Internship project\Phase 2 Track 1\em_realism'))
sys.path.append(os.path.abspath(r'D:\UAV Internship project\Phase 2 Track 2\array_structures'))
sys.path.append(os.path.abspath(r'D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)'))

from mesh_loader import load_uav_mesh
from conformal_array import get_conformal_array_parametric
from attitude import euler_to_quaternion, rotate_points
from shadow_engine_batched import compute_shadow_mask_batched
from multiprocessing import Pool

mesh = None
pos_body = None
normals_body = None

def init_worker():
    global mesh, pos_body, normals_body
    MESH_PATH = Path(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone_prepared.stl")
    mesh = load_uav_mesh(MESH_PATH)
    pos_body, normals_body = get_conformal_array_parametric(mesh, N=32)

def process_batch(batch_jams):
    return compute_shadow_mask_batched(mesh, pos_body, normals_body, batch_jams)

def main():
    print("Loading mesh for 3D pilot dataset generation...")
    MESH_PATH = Path(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone_prepared.stl")
    mesh = load_uav_mesh(MESH_PATH)
    pos_body, normals_body = get_conformal_array_parametric(mesh, N=32)

    # 15 degree pitch
    q_inv = euler_to_quaternion(np.deg2rad(15.0), 0, 0).conjugate()

    azimuths = np.linspace(0, 360, 3600, endpoint=False)
    elevations = [-20.0, 0.0, 20.0]
    
    total_points = len(azimuths) * len(elevations)
    
    headings_list = []
    elevations_list = []
    jam_bodies_list = []
    
    print(f"Generating {total_points} positions...")
    
    for el in elevations:
        for az in azimuths:
            jam_world = np.array([
                np.cos(np.deg2rad(az)) * np.cos(np.deg2rad(el)),
                np.sin(np.deg2rad(az)) * np.cos(np.deg2rad(el)),
                np.sin(np.deg2rad(el))
            ])
            jam_body = rotate_points(jam_world.reshape(1, 3), q_inv)[0]
            
            jam_bodies_list.append(jam_body)
            headings_list.append(az)
            elevations_list.append(el)
            
    jam_bodies_np = np.array(jam_bodies_list)
    headings_np = np.array(headings_list)
    elevations_np = np.array(elevations_list)
    
    print("Computing shadow masks...")
    
    BATCH_SIZE = 10
    N_JOBS = 6
    
    batches = [jam_bodies_np[i:min(i + BATCH_SIZE, total_points)] for i in range(0, total_points, BATCH_SIZE)]
    
    t0 = time.time()
    with Pool(processes=N_JOBS, initializer=init_worker) as pool:
        results = list(tqdm(pool.imap(process_batch, batches), total=len(batches), desc="Processing Batches"))
        
    g_exact_np = np.concatenate(results, axis=0)
    
    elapsed = time.time() - t0
    print(f"Mask computation took {elapsed:.1f} seconds.")
    
    save_path = "dataset_3d_pilot_masks.npz"
    np.savez(save_path, 
             g_exact=g_exact_np, 
             jam_bodies=jam_bodies_np, 
             headings=headings_np,
             elevations=elevations_np)
             
    print(f"Saved to {save_path}")

if __name__ == '__main__':
    main()
