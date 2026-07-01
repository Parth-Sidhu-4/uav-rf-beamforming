"""
Track 3 Diagnostics (no training required):
A. Full-vector ambiguity check at spike headings vs neighbors
   - Pairwise cosine similarities of v_ideal across 32 elements
   - Near-degeneracy (cos_sim ~ 1 between very different headings) 
     would indicate the inverse map is ill-conditioned there
B. U conditioning check from existing K=5 model
   - Shannon entropy effective rank at spike vs. neighbor headings
   - Collapse at spike headings = structural difficulty there
"""
import os, sys
import numpy as np
import torch
from pathlib import Path

sys.path.append(os.path.abspath(r'D:\UAV Internship project\Phase 2 Track 2\array_structures'))
sys.path.append(os.path.abspath(r'D:\UAV Internship project\Stage 13 (AI Beamforming)'))
sys.path.append(os.path.abspath(r'D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)'))

from mesh_loader import load_uav_mesh
from conformal_array import get_conformal_array_parametric
from attitude import euler_to_quaternion, rotate_points
from stage18_d3_covariance_train import (
    SIRENCovariancePredictor, get_steering_vector_pt
)

def eff_rank(S):
    """Shannon entropy effective rank from singular value tensor [K]."""
    s = S / (S.sum() + 1e-12)
    return float(torch.exp(-torch.sum(s * torch.log(s + 1e-12))))

def main():
    device = torch.device('cpu')
    MESH_PATH = Path(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone_prepared.stl")
    mesh = load_uav_mesh(MESH_PATH)
    pos_body, normals_body = get_conformal_array_parametric(mesh, N=32)
    pos_lambda = torch.tensor(pos_body / 0.15, dtype=torch.float64)

    q_inv = euler_to_quaternion(np.deg2rad(15.0), 0, 0).conjugate()

    # Headings of interest:
    # spike headings (from fine-grained eval), their immediate neighbors,
    # a "far" strong heading, and the second bad cluster (235.5 deg)
    headings_of_interest = {
        "spike_A":        179.3,
        "spike_A-1":      178.3,
        "spike_A+1":      180.3,
        "spike_B":        180.8,
        "spike_B+1":      181.8,
        "bowl_center":    180.0,
        "bowl_edge_low":  174.0,
        "bowl_edge_high": 183.0,
        "strong_near":    172.5,  # max SINR in range (28.95 dB)
        "strong_far":     90.0,   # well away from rear
        "second_worst":   235.5,  # second identified bad heading (K=5 global eval)
    }

    # Build unit direction vectors and v_ideal for each
    print("=" * 65)
    print("DIAG A: Full-vector ambiguity check")
    print("=" * 65)

    jam_bodies = {}
    v_ideals   = {}
    for name, h in headings_of_interest.items():
        jw = np.array([np.cos(np.deg2rad(h)), np.sin(np.deg2rad(h)), 0.0])
        jb = rotate_points(jw.reshape(1, 3), q_inv)[0]
        jam_bodies[name] = jb

        jb_t = torch.tensor(jb, dtype=torch.float64).unsqueeze(0)
        theta_j = torch.acos(jb_t[:, 2] / torch.norm(jb_t, dim=1))
        phi_j   = torch.atan2(jb_t[:, 1], jb_t[:, 0])
        v = get_steering_vector_pt(pos_lambda, theta_j, phi_j)[0]  # [32] complex
        v_ideals[name] = v

    # Pairwise cosine similarities
    names = list(v_ideals.keys())
    print(f"\n{'':25} | " + " | ".join([f"{n[:10]:>10}" for n in names]))
    print("-" * (26 + 13 * len(names)))
    for n_i in names:
        vi = v_ideals[n_i]
        vi_norm = vi / (vi.norm() + 1e-12)
        row = f"{n_i:25} |"
        for n_j in names:
            vj = v_ideals[n_j]
            vj_norm = vj / (vj.norm() + 1e-12)
            # Magnitude of complex cosine similarity
            cos_sim = float(torch.abs(torch.dot(torch.conj(vi_norm), vj_norm)))
            if n_i == n_j:
                row += f"  {'1.000':>10} |"
            elif cos_sim > 0.90:
                row += f"  {'*** '+f'{cos_sim:.3f}':>10} |"
            else:
                row += f"  {cos_sim:10.3f} |"
        print(row)

    print("\nNear-degeneracy (cos_sim > 0.9 between DIFFERENT headings)")
    print("would indicate the beamforming inverse problem is ill-conditioned there.\n")

    # Also report: how similar are the spike headings to each other?
    for pair in [("spike_A", "spike_B"), ("spike_A", "bowl_center"),
                 ("spike_A", "strong_near"), ("second_worst", "strong_far")]:
        n_i, n_j = pair
        vi = v_ideals[n_i] / (v_ideals[n_i].norm() + 1e-12)
        vj = v_ideals[n_j] / (v_ideals[n_j].norm() + 1e-12)
        cs = float(torch.abs(torch.dot(torch.conj(vi), vj)))
        print(f"  cos_sim({n_i}, {n_j}) = {cs:.4f}")

    # =========================================================
    # DIAG B: U conditioning at spike vs neighbor headings
    # =========================================================
    print("\n" + "=" * 65)
    print("DIAG B: U conditioning (effective rank) at specific headings")
    print("=" * 65)

    model = SIRENCovariancePredictor(w0=30.0, K_rank=5).to(device)
    model.load_state_dict(torch.load("siren_beamformer_d3_cov_K5.pt", map_location=device))
    model.eval()

    print(f"\n  {'Heading label':25} | {'Heading deg':>11} | {'Eff Rank /5':>11} | {'SVD spectrum':>40}")
    print(f"  {'-'*100}")

    with torch.no_grad():
        for name, h in headings_of_interest.items():
            jb = jam_bodies[name]
            jb_t = torch.tensor(jb, dtype=torch.float32).unsqueeze(0)
            U = model(jb_t).to(torch.complex128)  # [1, 32, 5]
            S = torch.linalg.svdvals(U)[0]         # [5]
            er = eff_rank(S)
            svd_str = " | ".join([f"{s:.3f}" for s in S.tolist()])
            print(f"  {name:25} | {h:>11.1f} | {er:>11.3f} | {svd_str}")

    print("\nIf eff_rank collapses specifically at spike headings (while staying")
    print("high at neighbors), that indicates a real local difficulty in U-space.")
    print("If eff_rank is uniform everywhere, the U is well-conditioned globally")
    print("and the spikes are purely output/SINR-landscape noise.\n")

if __name__ == '__main__':
    main()
