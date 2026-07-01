"""
Physics Engine Cross-Validation Script
=======================================
Tests BOTH shadow engines on the SAME baseline array at the SAME attitudes.
Old engine: trimesh.proximity.signed_distance (Stage 8 original)
New engine: cross-product projection (Stage 11 fast version)

This tells us definitively which physics engine is correct, and by how much
they differ on element-by-element gain values.
"""
import numpy as np
import trimesh
import sys
from pathlib import Path

sys.path.append(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)")
from mesh_loader import load_uav_mesh
from conformal_array import get_conformal_array
from attitude import euler_to_quaternion, rotate_points
from em_physics import compute_distances, fresnel_diffraction_gain
from constants import NACT_THRESHOLD

MESH_PATH = Path(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone_prepared.stl")

# ------------------------------------------------------------------
# OLD ENGINE: trimesh signed_distance
# ------------------------------------------------------------------
def shadow_old(jam_body, antenna_positions_body, normals_body, mesh):
    from trimesh.proximity import signed_distance
    N = len(antenna_positions_body)
    lam = 0.125
    edges = mesh.face_adjacency_edges
    V1_all = mesh.vertices[edges[:, 0]]
    V2_all = mesh.vertices[edges[:, 1]]
    valid = np.linalg.norm(V1_all - V2_all, axis=1) > 0.015
    V1 = V1_all[valid]; V2 = V2_all[valid]

    ray_origins = antenna_positions_body + normals_body * 1e-4
    h, d1, W = compute_distances(ray_origins, jam_body, V1, V2)

    g = np.ones(N, dtype=complex)
    for i in range(N):
        h_idx = h[i]; d1_idx = d1[i]
        edge1_d1 = d1_idx[np.argmin(h_idx)]
        S_j = 1.0 - np.exp(-(d1_idx - edge1_d1)**2 / (2 * 0.2**2))
        h_pen = h_idx / (S_j + 1e-8)
        idx1 = np.argmin(h_idx); idx2 = np.argmin(h_pen)

        P1 = ray_origins[i] + d1_idx[idx1] * jam_body
        P2 = ray_origins[i] + d1_idx[idx2] * jam_body

        sd1 = signed_distance(mesh, [P1])[0]
        sd2 = signed_distance(mesh, [P2])[0]

        nu1 = sd1 * np.sqrt(2 / (lam * max(d1_idx[idx1], 1e-8)))
        nu2 = sd2 * np.sqrt(2 / (lam * max(d1_idx[idx2], 1e-8)))
        F1 = fresnel_diffraction_gain(nu1)
        F2 = fresnel_diffraction_gain(nu2)
        w2 = S_j[idx2]
        g[i] = F1 * ((1 - w2) * 1.0 + w2 * F2)
    return g

# ------------------------------------------------------------------
# NEW ENGINE: cross-product projection
# ------------------------------------------------------------------
def shadow_new(jam_body, antenna_positions_body, normals_body, mesh):
    N = len(antenna_positions_body)
    lam = 0.125
    edges = mesh.face_adjacency_edges
    V1_all = mesh.vertices[edges[:, 0]]
    V2_all = mesh.vertices[edges[:, 1]]
    valid = np.linalg.norm(V1_all - V2_all, axis=1) > 0.015
    V1 = V1_all[valid]; V2 = V2_all[valid]
    edge_dirs = (V2 - V1) / np.linalg.norm(V2 - V1, axis=1)[:, np.newaxis]

    ray_origins = antenna_positions_body + normals_body * 1e-4
    h, d1, W_all = compute_distances(ray_origins, jam_body, V1, V2)

    g = np.ones(N, dtype=complex)
    for i in range(N):
        h_idx = h[i]; d1_idx = d1[i]; W_i = W_all[i]
        edge1_d1 = d1_idx[np.argmin(h_idx)]
        S_j = 1.0 - np.exp(-(d1_idx - edge1_d1)**2 / (2 * 0.2**2))
        h_pen = h_idx / (S_j + 1e-8)
        idx1 = np.argmin(h_idx); idx2 = np.argmin(h_pen)

        cross1 = np.cross(W_i[idx1], jam_body)
        sd1 = np.sign(np.dot(cross1, edge_dirs[idx1])) * h_idx[idx1]
        cross2 = np.cross(W_i[idx2], jam_body)
        sd2 = np.sign(np.dot(cross2, edge_dirs[idx2])) * h_idx[idx2]

        nu1 = sd1 * np.sqrt(2 / (lam * max(d1_idx[idx1], 1e-8)))
        nu2 = sd2 * np.sqrt(2 / (lam * max(d1_idx[idx2], 1e-8)))
        F1 = fresnel_diffraction_gain(nu1)
        F2 = fresnel_diffraction_gain(nu2)
        w2 = S_j[idx2]
        g[i] = F1 * ((1 - w2) * 1.0 + w2 * F2)
    return g

# ------------------------------------------------------------------
# MAIN: compare both on the baseline array at several attitudes
# ------------------------------------------------------------------
def main():
    print("Loading mesh and baseline array...")
    mesh = load_uav_mesh(MESH_PATH)
    pos_body, normals_body = get_conformal_array(mesh)

    # Test attitudes: (bank_deg, heading_deg, label)
    test_cases = [
        (0,   0,   "Level, jammer ahead (0°)"),
        (0,  90,   "Level, jammer right (90°)"),
        (0, 180,   "Level, jammer behind (180°)"),
        (0, 270,   "Level, jammer left (270°)"),
        (15,  0,   "Bank R 15°, jammer ahead"),
        (15, 120,   "Bank R 15°, jammer at 120°"),
        (-15, 270,  "Bank L 15°, jammer left"),
    ]

    jam_world = np.array([0.0, 1.0, 0.0])  # unit vector, jammer at +Y

    print(f"\nThreshold: {NACT_THRESHOLD:.4f} ({20*np.log10(NACT_THRESHOLD):.1f} dB)\n")
    print(f"{'Attitude':<40} {'OLD N_act':>9} {'NEW N_act':>9} {'OLD gains (abs)':>20} {'NEW gains (abs)':>20}")
    print("-" * 110)

    for bank_deg, h_ang, label in test_cases:
        q     = euler_to_quaternion(np.deg2rad(bank_deg), 0, np.deg2rad(h_ang))
        q_inv = q.conjugate()
        jam_body = rotate_points(jam_world.reshape(1,3), q_inv)[0]

        g_old = shadow_old(jam_body, pos_body, normals_body, mesh)
        g_new = shadow_new(jam_body, pos_body, normals_body, mesh)

        n_old = int(np.sum(np.abs(g_old) >= NACT_THRESHOLD))
        n_new = int(np.sum(np.abs(g_new) >= NACT_THRESHOLD))

        gains_old = " ".join(f"{v:.2f}" for v in np.abs(g_old))
        gains_new = " ".join(f"{v:.2f}" for v in np.abs(g_new))

        print(f"{label:<40} {n_old:>9} {n_new:>9}")

        # Print per-element breakdown
        print(f"  {'Elem':>4}  {'OLD |g|':>8}  {'NEW |g|':>8}  {'OLD nu sign':>12}  Match?")
        for i in range(len(pos_body)):
            o = abs(g_old[i]); n = abs(g_new[i])
            match = "OK" if abs(o - n) < 0.1 else "DIFFER"
            print(f"  {i+1:>4}  {o:>8.4f}  {n:>8.4f}  {match}")
        print()

if __name__ == "__main__":
    main()
