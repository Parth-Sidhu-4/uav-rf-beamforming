import sys
import os
import time
import numpy as np
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Phase 2 Track 1', 'em_realism')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Phase 2 Track 2', 'array_structures')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Stage 12 (Operational Analysis)')))

from mesh_loader import load_uav_mesh
from conformal_array import get_conformal_array
from shadow_engine_batched import compute_shadow_mask_batched

mesh = None
pos_body = None
normals_body = None

def init_worker():
    global mesh, pos_body, normals_body
    MESH_PATH = Path(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone_prepared.stl")
    mesh = load_uav_mesh(MESH_PATH)
    pos_body, normals_body = get_conformal_array(mesh)

def process_batch(batch_jams):
    g_batch = compute_shadow_mask_batched(mesh, pos_body, normals_body, batch_jams)
    labels = np.zeros((len(batch_jams), 32), dtype=np.float32)
    for b in range(len(batch_jams)):
        labels[b, 0::2] = np.abs(g_batch[b])
        labels[b, 1::2] = np.angle(g_batch[b])
    return labels

def main():
    print("Loading existing 30k dataset...")
    data = np.load("dataset_shadow_30k.npz")
    inputs_30k = data["inputs"]
    labels_30k_complex = data["labels"] # 30000 x 32
    
    print("Converting 30k to polar...")
    labels_30k_polar = np.zeros_like(labels_30k_complex)
    for i in range(16):
        c_val = labels_30k_complex[:, 2*i] + 1j * labels_30k_complex[:, 2*i+1]
        labels_30k_polar[:, 2*i] = np.abs(c_val)
        labels_30k_polar[:, 2*i+1] = np.angle(c_val)
        
    # Identify transition regions (0.2 < mag < 0.8)
    # We want inputs that cause at least one element to be in transition
    mags = labels_30k_polar[:, 0::2]
    is_transition = (mags > 0.2) & (mags < 0.8)
    transition_mask = np.any(is_transition, axis=1)
    
    transition_inputs = inputs_30k[transition_mask]
    print(f"Found {len(transition_inputs)} directions in the 30k dataset that produce transition regions.")
    
    # Generate 20k oversampled directions by jittering the transition inputs
    np.random.seed(42)
    N_OVERSAMPLE = 20000
    
    # Sample uniformly from the transition directions, then add +/- 5 deg jitter
    idx = np.random.choice(len(transition_inputs), size=N_OVERSAMPLE, replace=True)
    base_vectors = transition_inputs[idx]
    
    # 5 degrees is ~0.087 radians. We can add Gaussian noise with std ~ 0.05
    noise = np.random.randn(N_OVERSAMPLE, 3) * 0.05
    oversample_vectors = base_vectors + noise
    oversample_vectors /= np.linalg.norm(oversample_vectors, axis=1, keepdims=True)
    
    BATCH_SIZE = 10
    N_JOBS = 12
    
    batches = [oversample_vectors[i:min(i + BATCH_SIZE, N_OVERSAMPLE)] for i in range(0, N_OVERSAMPLE, BATCH_SIZE)]
    
    print(f"Running physics engine for {N_OVERSAMPLE} oversampled vectors in batches of {BATCH_SIZE}...")
    t0 = time.time()
    
    with Pool(processes=N_JOBS, initializer=init_worker) as pool:
        results = list(tqdm(pool.imap(process_batch, batches), total=len(batches), desc="Processing Batches"))
        
    labels_oversample = np.vstack(results)
    
    elapsed = time.time() - t0
    print(f"Generation completed in {elapsed:.1f} s")
    
    # Concatenate 30k polar + 20k oversampled polar
    inputs_final = np.vstack([inputs_30k, oversample_vectors])
    labels_final = np.vstack([labels_30k_polar, labels_oversample])
    
    save_path = "dataset_shadow_100k_polar.npz" # Kept same name so downstream scripts don't break
    np.savez_compressed(save_path, inputs=inputs_final, labels=labels_final)
    print(f"Targeted dataset (50k total samples) saved successfully to {save_path}.")

if __name__ == '__main__':
    main()
