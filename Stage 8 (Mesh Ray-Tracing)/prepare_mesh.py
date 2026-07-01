import trimesh
import numpy as np
from pathlib import Path

mesh_path = Path(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone.stl")
print("Loading original mesh...")
mesh = trimesh.load(str(mesh_path), force='mesh')

# 1. Rotate -90 degrees around Z axis so wingspan aligns with Y
print("Rotating mesh -90 degrees around Z axis...")
rot_matrix = trimesh.transformations.rotation_matrix(np.deg2rad(-90), [0, 0, 1])
mesh.apply_transform(rot_matrix)

# Calculate wingspan (which is now along Y after rotation)
y_span = mesh.extents[1]
print(f"Current wingspan (Y-axis): {y_span:.3f}")

# 2. Scale to 3.11 meters wingspan (ScanEagle wingspan)
target_wingspan = 3.11
scale_factor = target_wingspan / y_span
print(f"Scaling by factor: {scale_factor:.6f}")
mesh.apply_scale(scale_factor)

# Save the prepared mesh
out_path = Path(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone_prepared.stl")
mesh.export(str(out_path))
print(f"Prepared mesh saved to {out_path.name}")
