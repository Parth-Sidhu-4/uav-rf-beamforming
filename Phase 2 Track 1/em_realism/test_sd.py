import trimesh
import numpy as np

# Trivial sphere test
sphere = trimesh.creation.icosphere(radius=1.0)
pts = np.array([[0, 0, 0], [2, 0, 0]]) # Origin (inside), Radius 2 (outside)

sd = trimesh.proximity.signed_distance(sphere, pts)
print(f"Sphere Watertight: {sphere.is_watertight}")
print(f"Origin (inside) SD: {sd[0]}")
print(f"Radius 2 (outside) SD: {sd[1]}")

# Load drone mesh
import sys
sys.path.append(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)")
from mesh_loader import load_uav_mesh

from pathlib import Path
mesh = load_uav_mesh(Path(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone_prepared.stl"))
print(f"UAV Mesh Watertight: {mesh.is_watertight}")
