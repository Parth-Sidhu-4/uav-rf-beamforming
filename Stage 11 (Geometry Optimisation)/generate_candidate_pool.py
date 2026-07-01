import numpy as np
import trimesh
from pathlib import Path
from scipy.spatial import cKDTree
import os

from constants import M_CANDIDATES

def generate_symmetric_pool(mesh: trimesh.Trimesh, total_candidates: int = M_CANDIDATES):
    # Sample points on the mesh
    print(f"Sampling {total_candidates*2} points on the surface...")
    points, face_indices = trimesh.sample.sample_surface(mesh, total_candidates * 2)
    
    print("Sampled points shape:", points.shape)
    print("First 5 points:", points[:5])
    print("Y min/max:", points[:, 1].min(), points[:, 1].max())
    
    # Filter for right half (Y > 0)
    # We add a small margin to strictly stay away from the centerline singularity
    right_mask = points[:, 1] > 0.01
    right_points = points[right_mask]
    
    # We want roughly total_candidates // 2 pairs
    target_pairs = total_candidates // 2
    if len(right_points) > target_pairs:
        right_points = right_points[:target_pairs]
    
    print(f"Collected {len(right_points)} points on the right half.")
    
    # Mirror spanwise (Y -> -Y)
    mirrored_points = right_points.copy()
    mirrored_points[:, 1] = -mirrored_points[:, 1]
    
    # Snap the mirrored points exactly to the mesh surface
    print("Snapping mirrored points to the left half of the mesh...")
    closest_points, _, _ = trimesh.proximity.closest_point(mesh, mirrored_points)
    
    # To ensure symmetry, we shouldn't use the original right_points directly, 
    # we should use right_points and closest_points as pairs.
    # However, to be perfectly symmetrical, let's take the closest_points (which are on the mesh),
    # mirror them BACK to the right, and snap again? 
    # Assuming the mesh is reasonably symmetric, closest_points are very close to mirrored_points.
    # We will define our pairs as: Left = closest_points, Right = right_points.
    # But wait, if Left is snapped, it's on the mesh. Is Right on the mesh? Yes, because it was sampled.
    left_points = closest_points
    
    # Deduplicate (if multiple right points snap to the exact same left area)
    # Since we are optimizing over these, we just want unique indices.
    # We can just deduplicate based on rounding coordinates.
    left_rounded = np.round(left_points, 3)
    _, unique_indices = np.unique(left_rounded, axis=0, return_index=True)
    
    right_points = right_points[unique_indices]
    left_points = left_points[unique_indices]
    print(f"After deduplication, we have {len(right_points)} symmetric pairs.")
    
    # --- Locality Sort (Greedy Nearest Neighbor Path) ---
    # We sort the pairs based on the 3D position of the right_points to give DE good spatial locality.
    print("Sorting pairs using a Greedy Nearest Neighbor (TSP) walk for 3D locality...")
    unvisited = set(range(len(right_points)))
    
    # Start at the point with the minimum X (e.g. the nose or tail depending on coordinate system)
    start_idx = np.argmin(right_points[:, 0])
    
    path = [start_idx]
    unvisited.remove(start_idx)
    
    tree = cKDTree(right_points)
    
    current_idx = start_idx
    while unvisited:
        # Query 50 nearest neighbors to find the closest unvisited one
        # (50 is usually enough to find at least one unvisited)
        distances, indices = tree.query(right_points[current_idx], k=min(50, len(right_points)))
        
        found = False
        for neighbor_idx in indices:
            if neighbor_idx in unvisited:
                path.append(neighbor_idx)
                unvisited.remove(neighbor_idx)
                current_idx = neighbor_idx
                found = True
                break
                
        if not found:
            # Fallback if all 50 nearest were visited: just pick an arbitrary unvisited node
            fallback_idx = next(iter(unvisited))
            path.append(fallback_idx)
            unvisited.remove(fallback_idx)
            current_idx = fallback_idx

    path = np.array(path)
    
    # Reorder the arrays according to the path
    right_points = right_points[path]
    left_points = left_points[path]
    
    # Save the arrays
    np.save("candidate_pairs_right.npy", right_points)
    np.save("candidate_pairs_left.npy", left_points)
    print("Candidate pool saved to disk (candidate_pairs_right.npy and candidate_pairs_left.npy).")

from mesh_loader import load_uav_mesh

if __name__ == "__main__":
    mesh_path = Path(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone_prepared.stl")
    print("Loading mesh...")
    mesh = load_uav_mesh(mesh_path)
    generate_symmetric_pool(mesh)
