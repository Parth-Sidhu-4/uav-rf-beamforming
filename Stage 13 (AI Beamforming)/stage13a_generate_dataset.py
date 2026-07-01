import sys
import os
import time
import time
import numpy as np
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Phase 2 Track 1', 'em_realism')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Phase 2 Track 2', 'array_structures')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Stage 12 (Operational Analysis)')))

from mesh_loader import load_uav_mesh
from conformal_array import get_conformal_array_parametric
from shadow_engine_batched import compute_shadow_mask_batched

mesh = None
pos_body = None
normals_body = None

def init_worker():
    global mesh, pos_body, normals_body
    MESH_PATH = Path(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone_prepared.stl")
    mesh = load_uav_mesh(MESH_PATH)
    pos_body, normals_body = get_conformal_array_parametric(mesh, N=32)

def process_batch(batch_jams):
    g_batch = compute_shadow_mask_batched(mesh, pos_body, normals_body, batch_jams)
    labels = np.zeros((len(batch_jams), 64), dtype=np.float32)
    for b in range(len(batch_jams)):
        labels[b, 0::2] = np.abs(g_batch[b])
        labels[b, 1::2] = np.angle(g_batch[b])
    return labels

def main():
    N_SAMPLES = 100000
    BATCH_SIZE = 10
    N_JOBS = 6
    
    np.random.seed(42)
    jam_body_vectors = np.random.randn(N_SAMPLES, 3)
    jam_body_vectors /= np.linalg.norm(jam_body_vectors, axis=1, keepdims=True)
    
    inputs = np.zeros((N_SAMPLES, 3), dtype=np.float32)
    inputs[:] = jam_body_vectors
    
    print(f"Generating dataset with {N_SAMPLES} samples in parallel batches of {BATCH_SIZE} using {N_JOBS} workers...")
    t0 = time.time()
    
    batches = [jam_body_vectors[i:min(i + BATCH_SIZE, N_SAMPLES)] for i in range(0, N_SAMPLES, BATCH_SIZE)]
    
    with Pool(processes=N_JOBS, initializer=init_worker) as pool:
        # Use imap to get results as they complete, wrapped in tqdm for a progress bar
        results = list(tqdm(pool.imap(process_batch, batches), total=len(batches), desc="Processing Batches"))
        
    labels = np.vstack(results)
    
    elapsed = time.time() - t0
    print(f"Generation completed in {elapsed:.1f} s")
                
    save_path = "dataset_shadow_100k_polar_32el.npz"
    np.savez_compressed(save_path, inputs=inputs, labels=labels)
    print(f"Dataset saved successfully to {save_path}.")

if __name__ == '__main__':
    main()
