import sys
import os
import time
import numpy as np
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Phase 2 Track 1', 'em_realism')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Phase 2 Track 2', 'array_structures')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Stage 12 (Operational Analysis)')))

from mesh_loader import load_uav_mesh
from conformal_array import get_conformal_array
from shadow_engine_batched import compute_shadow_mask_batched

def main():
    MESH_PATH = Path(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone_prepared.stl")
    
    print("Loading mesh and baseline conformal array...")
    mesh = load_uav_mesh(MESH_PATH)
    pos_body, normals_body = get_conformal_array(mesh)
    
    jams = np.random.randn(100, 3)
    jams /= np.linalg.norm(jams, axis=1, keepdims=True)
    
    print("Running batch of 100...")
    t0 = time.time()
    g_batch = compute_shadow_mask_batched(mesh, pos_body, normals_body, jams)
    t1 = time.time()
    print(f"Batch of 100 took {t1-t0:.2f}s")
    
if __name__ == '__main__':
    main()
