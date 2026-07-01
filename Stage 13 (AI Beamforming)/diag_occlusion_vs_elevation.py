"""
Diagnostic: Does the rear-hemisphere occlusion (heading ~180 deg) lift with elevation?

Tests g_exact at heading=180 deg across elevations from 0 to +30 deg.
Also sweeps nearby headings (165-195 deg) at elevation=0 and 20 deg.

Key question: Is g_exact near-zero only in the horizontal plane (-> Option A fixes
the dead zone), or across the full vertical wedge (-> Option A leaves dead zone intact)?
"""
import os
import sys
import numpy as np
from pathlib import Path

sys.path.append(os.path.abspath(r'D:\UAV Internship project\Phase 2 Track 1\em_realism'))
sys.path.append(os.path.abspath(r'D:\UAV Internship project\Phase 2 Track 2\array_structures'))
sys.path.append(os.path.abspath(r'D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)'))

from mesh_loader import load_uav_mesh
from conformal_array import get_conformal_array_parametric
from attitude import euler_to_quaternion, rotate_points
from shadow_engine_batched import compute_shadow_mask_batched

def summarize_g(g, label):
    mag = np.abs(g)
    n_nonzero = np.sum(mag > 0.05)  # elements with meaningful visibility
    mean_mag = mag.mean()
    max_mag = mag.max()
    print(f"  {label:45s} | N_visible={n_nonzero:2d}/32 | mean|g|={mean_mag:.3f} | max|g|={max_mag:.3f}")

def main():
    MESH_PATH = Path(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone_prepared.stl")
    print("Loading mesh...")
    mesh = load_uav_mesh(MESH_PATH)
    pos_body, normals_body = get_conformal_array_parametric(mesh, N=32)

    q_inv = euler_to_quaternion(np.deg2rad(15.0), 0, 0).conjugate()

    # --- Part 1: Heading 180 deg, sweep elevation 0 to +30 deg ---
    print("\n=== Part 1: heading=180°, varying elevation ===")
    print(f"  {'Label':45s} | N_visible  | mean|g|  | max|g|")
    print(f"  {'-'*70}")

    elevations = [0, 5, 10, 15, 20, 25, 30]
    for elev in elevations:
        jam_world = np.array([
            np.cos(np.deg2rad(180)) * np.cos(np.deg2rad(elev)),
            np.sin(np.deg2rad(180)) * np.cos(np.deg2rad(elev)),
            np.sin(np.deg2rad(elev))
        ])
        jam_body = rotate_points(jam_world.reshape(1, 3), q_inv)[0]
        g = compute_shadow_mask_batched(mesh, pos_body, normals_body, jam_body.reshape(1, 3))[0]
        summarize_g(g, f"heading=180°, elev={elev:+.0f}°")

    # --- Part 2: Elevation 0 deg, sweep heading around the dead zone ---
    print("\n=== Part 2: elevation=0°, sweeping heading 160°–200° ===")
    print(f"  {'Label':45s} | N_visible  | mean|g|  | max|g|")
    print(f"  {'-'*70}")

    for hdg in range(160, 205, 5):
        jam_world = np.array([
            np.cos(np.deg2rad(hdg)) * np.cos(np.deg2rad(0)),
            np.sin(np.deg2rad(hdg)) * np.cos(np.deg2rad(0)),
            np.sin(np.deg2rad(0))
        ])
        jam_body = rotate_points(jam_world.reshape(1, 3), q_inv)[0]
        g = compute_shadow_mask_batched(mesh, pos_body, normals_body, jam_body.reshape(1, 3))[0]
        summarize_g(g, f"heading={hdg:3d}°, elev= 0°")

    # --- Part 3: Heading 180 deg +/- 10 deg, elevation 20 deg vs 0 deg ---
    print("\n=== Part 3: Occlusion map near 180° at two elevations ===")
    print(f"  {'Label':45s} | N_visible  | mean|g|  | max|g|")
    print(f"  {'-'*70}")

    for hdg in [170, 175, 180, 185, 190]:
        for elev in [0, 20]:
            jam_world = np.array([
                np.cos(np.deg2rad(hdg)) * np.cos(np.deg2rad(elev)),
                np.sin(np.deg2rad(hdg)) * np.cos(np.deg2rad(elev)),
                np.sin(np.deg2rad(elev))
            ])
            jam_body = rotate_points(jam_world.reshape(1, 3), q_inv)[0]
            g = compute_shadow_mask_batched(mesh, pos_body, normals_body, jam_body.reshape(1, 3))[0]
            summarize_g(g, f"heading={hdg:3d}°, elev={elev:+.0f}°")

    print("\nDone. If N_visible is low at ALL elevations near 180°, the dead zone")
    print("is a full vertical wedge and Option A alone will not fix it.")
    print("If N_visible grows with elevation, Option A should genuinely help.")

if __name__ == '__main__':
    main()
