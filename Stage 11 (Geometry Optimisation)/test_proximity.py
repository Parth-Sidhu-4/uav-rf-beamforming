import trimesh
from pathlib import Path
import numpy as np

mesh_path = Path(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone_prepared.stl")
mesh = trimesh.load(mesh_path, force='mesh')

query = trimesh.proximity.ProximityQuery(mesh)

pts = np.array([[0,0,0], [1,1,1]])
print("Calling signed_distance...")
sd = query.signed_distance(pts)
print("Result:", sd)
