import sys
import os
import multiprocessing
import time
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Phase 2 Track 1', 'em_realism')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Phase 2 Track 2', 'array_structures')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Stage 12 (Operational Analysis)')))

from mesh_loader import load_uav_mesh
from conformal_array import get_conformal_array
from shadow_engine import compute_shadow_mask
import numpy as np

_MESH = None
_POS = None
_NORMALS = None

def init_worker(mesh_path):
    global _MESH, _POS, _NORMALS
    print("Worker initializing...")
    _MESH = load_uav_mesh(Path(mesh_path))
    _POS, _NORMALS = get_conformal_array(_MESH)
    print("Worker initialized.")

def worker_process(jam_body):
    g = compute_shadow_mask(_MESH, _POS, _NORMALS, jam_body)
    row = np.zeros(16 * 2, dtype=np.float32)
    row[0::2] = np.real(g)
    row[1::2] = np.imag(g)
    return row

def main():
    MESH_PATH = r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone_prepared.stl"
    print("Starting pool...")
    with multiprocessing.Pool(processes=2, initializer=init_worker, initargs=(MESH_PATH,)) as pool:
        jam_body_vectors = np.random.randn(10, 3)
        jam_body_vectors /= np.linalg.norm(jam_body_vectors, axis=1, keepdims=True)
        results = pool.map(worker_process, jam_body_vectors)
        print(f"Got {len(results)} results")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
