"""
Two measurements to decide whether residual idx2 steps need further fixing:

1. Max per-step gain jump (dB) across all 16 elements, full heading sweep 22-32°
2. Whether those steps show up in the system-level SINR (from the cached npz)
"""
import numpy as np
import sys, os, hashlib
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Stage 8 (Mesh Ray-Tracing)')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mesh_loader import load_uav_mesh
from conformal_array import get_conformal_array
from attitude import euler_to_quaternion, rotate_points
from em_physics import compute_distances, fresnel_diffraction_gain
from trimesh.proximity import signed_distance

def run():
    # ── 1. Re-compute g_all with the current (softmax-primary) code ──────────
    mesh = load_uav_mesh(Path(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone_prepared.stl"))
    antenna_positions_body, normals_body = get_conformal_array(mesh)

    jammer_pos_xy = np.array([0.0, 100000.0])
    jam_world = np.array([jammer_pos_xy[0], jammer_pos_xy[1], 0.0])
    jam_world /= np.linalg.norm(jam_world)

    edges = mesh.face_adjacency_edges
    V1_all = mesh.vertices[edges[:, 0]]
    V2_all = mesh.vertices[edges[:, 1]]
    valid_mask = np.linalg.norm(V1_all - V2_all, axis=1) > 0.015
    V1 = V1_all[valid_mask]; V2 = V2_all[valid_mask]

    headings = np.linspace(22, 32, 200)
    lam = 0.125
    phi_cmd = 30.0
    g_all = np.zeros((len(headings), 16), dtype=complex)

    beta = 500.0

    for idx_h, h_ang in enumerate(headings):
        q = euler_to_quaternion(np.deg2rad(phi_cmd), 0, np.deg2rad(h_ang))
        q_inv = q.conjugate()
        jam_body = rotate_points(jam_world.reshape(1, 3), q_inv)[0]
        ray_origins = antenna_positions_body + normals_body * 1e-4
        h_dists, d1, _ = compute_distances(ray_origins, jam_body, V1, V2)

        for i in range(16):
            h_idx = h_dists[i, :]
            d1_idx = d1[i, :]

            h_min = np.min(h_idx)
            log_w = -beta * (h_idx - h_min); log_w -= np.max(log_w)
            w_soft = np.exp(log_w); w_soft /= w_soft.sum()
            edge1_d1 = np.dot(w_soft, d1_idx)

            S_j = 1.0 - np.exp(-(d1_idx - edge1_d1)**2 / (2 * 0.2**2))
            h_penalized = h_idx / (S_j + 1e-8)
            idx2 = np.argmin(h_penalized)

            top3 = np.argsort(h_idx)[:3]
            w_top3 = w_soft[top3]; w_top3 /= w_top3.sum()
            F1_soft = 0.0
            for k, ek in enumerate(top3):
                P_ray_k = ray_origins[i] + d1_idx[ek] * jam_body
                sd_k = signed_distance(mesh, [P_ray_k])[0]
                nu_k = sd_k * np.sqrt(2 / (lam * d1_idx[ek]))
                F1_soft += w_top3[k] * fresnel_diffraction_gain(nu_k)

            P_ray_2 = ray_origins[i] + d1_idx[idx2] * jam_body
            sd2 = signed_distance(mesh, [P_ray_2])[0]
            d_eff = edge1_d1 + d1_idx[idx2]
            nu2 = sd2 * np.sqrt(2 / (lam * d_eff))
            F2 = fresnel_diffraction_gain(nu2)
            weight2 = S_j[idx2]
            F2_blended = (1.0 - weight2) * 1.0 + weight2 * F2

            g_all[idx_h, i] = F1_soft * F2_blended

    # ── 2. Per-step jump analysis ─────────────────────────────────────────────
    gain_db = 20 * np.log10(np.abs(g_all) + 1e-30)  # shape (200, 16)
    steps = np.abs(np.diff(gain_db, axis=0))          # shape (199, 16)

    max_jump_per_element = steps.max(axis=0)
    max_jump_overall = steps.max()
    worst_element = np.argmax(max_jump_per_element)
    worst_heading_idx = np.unravel_index(np.argmax(steps), steps.shape)[0]
    worst_heading = headings[worst_heading_idx]

    print("=" * 60)
    print("PER-ELEMENT MAX STEP (dB) — all 16 antennas, heading 22-32°")
    print("=" * 60)
    for el in range(16):
        bar = "#" * int(max_jump_per_element[el] / 0.05)
        flag = " *** WORST" if el == worst_element else ""
        print(f"  Ant {el:2d}: {max_jump_per_element[el]:.4f} dB  {bar}{flag}")
    print()
    print(f"Overall max step  : {max_jump_overall:.4f} dB  (ant {worst_element}, heading ≈ {worst_heading:.2f}°)")
    print(f"Steps > 0.50 dB   : {(steps > 0.50).sum()}")
    print(f"Steps > 0.30 dB   : {(steps > 0.30).sum()}")
    print(f"Steps > 0.10 dB   : {(steps > 0.10).sum()}")

    # ── 3. System-level SINR check in 24-30° window ──────────────────────────
    cache = Path(r"D:\UAV Internship project\Phase 2 Track 1\lcmv_results_cache.npz")
    if cache.exists():
        print()
        print("=" * 60)
        print("SYSTEM-LEVEL SINR in 24-30° window (from cached npz)")
        print("=" * 60)
        d = np.load(cache)
        h_cont = d['h_cont']; s_cont = d['s_cont']
        mask = (h_cont >= 24) & (h_cont <= 30)
        h_win = h_cont[mask]; s_win = s_cont[mask]
        sinr_steps = np.abs(np.diff(s_win))
        print(f"  Headings in window : {len(h_win)} points")
        print(f"  SINR range         : {s_win.min():.3f} to {s_win.max():.3f} dB")
        print(f"  Max SINR step/pt   : {sinr_steps.max():.4f} dB")
        print(f"  Steps > 0.50 dB    : {(sinr_steps > 0.50).sum()}")
        print(f"  Steps > 0.20 dB    : {(sinr_steps > 0.20).sum()}")
    else:
        print(f"\n[WARNING] Cache not found at {cache} — run mesh_aware_lcmv.py first")


if __name__ == "__main__":
    run()
