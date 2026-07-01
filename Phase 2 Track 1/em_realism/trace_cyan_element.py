"""
Targeted trace of the cyan element (index 9) gain computation
across heading 28.5-29.5 degrees.

For each heading step, prints:
  h, idx1, idx2, d1[idx1], d1[idx2], S_j[idx2], sd1, sd2,
  nu1, F1, nu2, F2, weight2, F2_blended, total_gain_dB

This lets us see whether the jump comes from:
  (a) S_j[idx2] stepping despite a smooth formula (idx2 changed discretely)
  (b) sd1 or sd2 jumping (SDF discontinuity)
  (c) idx1 changing (primary edge handoff)
"""
import numpy as np
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Stage 8 (Mesh Ray-Tracing)')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path
from mesh_loader import load_uav_mesh
from conformal_array import get_conformal_array
from attitude import euler_to_quaternion, rotate_points
from em_physics import compute_distances, fresnel_diffraction_gain
from trimesh.proximity import signed_distance

ELEMENT = 9           # cyan
HEADING_LO = 28.0
HEADING_HI = 30.5
N_STEPS = 100         # fine resolution across the window
PHI_CMD = 30.0
LAM = 0.125

def run():
    mesh = load_uav_mesh(Path(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone_prepared.stl"))
    antenna_positions_body, normals_body = get_conformal_array(mesh)

    jammer_pos_xy = np.array([0.0, 100000.0])
    jam_world = np.array([jammer_pos_xy[0], jammer_pos_xy[1], 0.0])
    jam_world /= np.linalg.norm(jam_world)

    edges = mesh.face_adjacency_edges
    V1_all = mesh.vertices[edges[:, 0]]
    V2_all = mesh.vertices[edges[:, 1]]
    valid_mask = np.linalg.norm(V1_all - V2_all, axis=1) > 0.015
    if not np.any(valid_mask):
        valid_mask = np.ones(len(edges), dtype=bool)
    V1 = V1_all[valid_mask]
    V2 = V2_all[valid_mask]

    headings = np.linspace(HEADING_LO, HEADING_HI, N_STEPS)

    print(f"{'h':>6} | {'idx1':>6} | {'idx2':>6} | {'d1_1':>8} | {'d1_2':>8} | "
          f"{'S_j@idx2':>10} | {'sd1':>8} | {'sd2':>8} | "
          f"{'F1':>8} | {'F2':>8} | {'w2':>6} | {'gain_dB':>9} | JUMP?")
    print("-" * 115)

    prev_gain_db = None

    for h_ang in headings:
        q = euler_to_quaternion(np.deg2rad(PHI_CMD), 0, np.deg2rad(h_ang))
        q_inv = q.conjugate()
        jam_body = rotate_points(jam_world.reshape(1, 3), q_inv)[0]

        ray_origins = antenna_positions_body + normals_body * 1e-4

        h_dists, d1, W = compute_distances(ray_origins, jam_body, V1, V2)

        i = ELEMENT
        h_idx = h_dists[i, :]
        d1_idx = d1[i, :]

        idx1 = np.argmin(h_idx)
        edge1_d1 = d1_idx[idx1]

        S_j = 1.0 - np.exp(-(d1_idx - edge1_d1)**2 / (2 * 0.2**2))
        h_penalized = h_idx / (S_j + 1e-8)
        idx2 = np.argmin(h_penalized)

        P_ray_1 = ray_origins[i] + d1_idx[idx1] * jam_body
        P_ray_2 = ray_origins[i] + d1_idx[idx2] * jam_body

        sd1 = signed_distance(mesh, [P_ray_1])[0]
        sd2 = signed_distance(mesh, [P_ray_2])[0]

        nu1 = sd1 * np.sqrt(2 / (LAM * d1_idx[idx1]))
        F1 = fresnel_diffraction_gain(nu1)

        d_eff = d1_idx[idx1] + d1_idx[idx2]
        nu2 = sd2 * np.sqrt(2 / (LAM * d_eff))
        F2 = fresnel_diffraction_gain(nu2)

        weight2 = S_j[idx2]
        F2_blended = (1.0 - weight2) * 1.0 + weight2 * F2

        total_gain = F1 * F2_blended
        gain_db = 20 * np.log10(abs(total_gain)) if abs(total_gain) > 1e-12 else -300.0

        jump_str = ""
        if prev_gain_db is not None and abs(gain_db - prev_gain_db) > 0.3:
            jump_str = f"<< JUMP {gain_db - prev_gain_db:+.2f} dB"

        print(f"{h_ang:6.2f} | {idx1:6d} | {idx2:6d} | {d1_idx[idx1]:8.4f} | {d1_idx[idx2]:8.4f} | "
              f"{weight2:10.4f} | {sd1:8.4f} | {sd2:8.4f} | "
              f"{abs(F1):8.4f} | {abs(F2):8.4f} | {weight2:6.4f} | {gain_db:9.4f} | {jump_str}")

        prev_gain_db = gain_db


if __name__ == "__main__":
    run()
