import sys
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Phase 2 Track 1', 'em_realism')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Phase 2 Track 2', 'array_structures')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Stage 12 (Operational Analysis)')))

from mesh_loader import load_uav_mesh
from conformal_array import get_conformal_array
from shadow_engine import compute_shadow_mask
import numpy as np

def main():
    MESH_PATH = Path(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone_prepared.stl")
    mesh = load_uav_mesh(MESH_PATH)
    pos, normals = get_conformal_array(mesh)
    
    jams = np.random.randn(50, 3)
    jams /= np.linalg.norm(jams, axis=1, keepdims=True)
    
    t0 = time.time()
    def process(i):
        return compute_shadow_mask(mesh, pos, normals, jams[i])
        
    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(process, range(50)))
        
    t1 = time.time()
    print(f"50 samples took {t1-t0:.2f}s using 16 threads. ({(t1-t0)/50:.3f}s per sample)")

if __name__ == '__main__':
    main()
