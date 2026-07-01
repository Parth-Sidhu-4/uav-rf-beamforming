import numpy as np
from pathlib import Path
import trimesh
import sys

sys.path.append(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)")

from mesh_loader import load_uav_mesh
from conformal_array import get_conformal_array
from attitude import euler_to_quaternion, rotate_points
from cognitive_autopilot import generate_lut

def main():
    mesh_path = Path(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone_prepared.stl")
    mesh = load_uav_mesh(mesh_path)
    antennas_body, normals = get_conformal_array(mesh)
    
    intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
    
    print("Generating coarse LUT...")
    lut = generate_lut()
    
    # 10,000 random combinations (uniform random sampling)
    np.random.seed(42)
    N_samples = 10000
    phi_samples = np.random.uniform(-45.0, 45.0, N_samples)
    az_samples = np.random.uniform(0.0, 360.0, N_samples)
    
    # Query LUT
    pts = np.column_stack((phi_samples, az_samples))
    lut_counts = lut(pts)
    
    # Compute Continuous Raycast
    print("Computing continuous raycasts...")
    ray_origins = np.tile(antennas_body, (N_samples, 1)) + np.tile(normals, (N_samples, 1)) * 0.01
    
    all_gcs_body = np.zeros((N_samples, 3))
    for i in range(N_samples):
        q = euler_to_quaternion(np.deg2rad(phi_samples[i]), 0, 0)
        q_inv = q.conjugate()
        az_rad = np.radians(az_samples[i])
        gcs_world = np.array([np.cos(az_rad), np.sin(az_rad), 0.0])
        all_gcs_body[i] = rotate_points(gcs_world.reshape(1,3), q_inv)[0]
        
    N_ant = len(antennas_body)
    total_rays = N_samples * N_ant
    ray_dirs_gcs = np.repeat(all_gcs_body, N_ant, axis=0)
    
    chunk_size = 1000
    hits_list = []
    for k in range(0, total_rays, chunk_size):
        end = min(k + chunk_size, total_rays)
        hits_list.append(intersector.intersects_any(ray_origins[k:end], ray_dirs_gcs[k:end]))
        
    hits = np.concatenate(hits_list)
    mask_all = (~hits).astype(float).reshape(N_samples, N_ant)
    exact_counts = mask_all.sum(axis=1)
    
    # Discrepancy: how many MORE elements the LUT predicted than actually survived
    discrepancy = lut_counts - exact_counts
    
    perc_999 = np.percentile(discrepancy, 99.9)
    max_disc = np.max(discrepancy)
    
    print("\n--- Discrepancy Analysis ---")
    print(f"99.9th Percentile Discrepancy: {perc_999:.1f} elements")
    print(f"Maximum Discrepancy:           {max_disc:.1f} elements")
    
    print("\nOutliers (Discrepancy > 99.9th percentile):")
    outlier_idx = np.where(discrepancy > perc_999)[0]
    for idx in outlier_idx:
        print(f"Phi: {phi_samples[idx]:5.1f} | Az: {az_samples[idx]:5.1f} | LUT: {lut_counts[idx]:4.1f} | Exact: {exact_counts[idx]:4.1f} | Diff: {discrepancy[idx]:4.1f}")
        
    rank_limit = 3
    min_active = rank_limit + int(np.ceil(perc_999))
    print(f"\nRecommended min_active threshold: {rank_limit} (Rank) + {int(np.ceil(perc_999))} (Margin) = {min_active}")

if __name__ == "__main__":
    main()
