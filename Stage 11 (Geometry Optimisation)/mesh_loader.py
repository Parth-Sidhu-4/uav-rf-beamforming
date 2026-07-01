from pathlib import Path
import trimesh
import numpy as np
import logging

def load_uav_mesh(mesh_path: Path) -> trimesh.Trimesh:
    """
    Loads a 3D mesh for the UAV, validates it, and centers it.
    Expects a pre-scaled and pre-oriented mesh (e.g., from prepare_mesh.py).
    """
    logging.info(f"Loading mesh from {mesh_path.name}")
    mesh = trimesh.load(str(mesh_path), force='mesh')
    
    # Ensure it's watertight
    if not mesh.is_watertight:
        logging.warning("Mesh is not watertight! Attempting to repair (fill_holes)...")
        mesh.fill_holes()
        if not mesh.is_watertight:
            logging.error("Mesh is STILL not watertight after repair. Ray-tracing may leak!")
            # Last resort: convex hull, though it destroys fine details like twin booms.
            # We won't convex hull automatically to preserve the complex shadowing geometry.
    else:
        logging.info("Mesh is watertight.")
        
    # Center the mesh's center of mass at the origin (0, 0, 0)
    # This is crucial so that quaternion rotations apply about the CG.
    mesh.apply_translation(-mesh.center_mass)
    
    # Fix normals to ensure ray-casting works correctly
    mesh.fix_normals()
    
    return mesh

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    p = Path(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone_prepared.stl")
    m = load_uav_mesh(p)
    print("Center of mass:", m.center_mass)
    print("Extents:", m.extents)
