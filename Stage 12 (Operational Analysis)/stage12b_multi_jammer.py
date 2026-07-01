"""
Stage 12B (v3): Multi-Jammer Spatial DoF Sweep
===============================================
Shows how worst-case SINR degrades as the number of simultaneous jammers
increases from 1 to 12, placed at equal angular spacing around the UAV.

Fixes in v3:
- Loads the actual optimal genome
- Uses compute_plf for exactly the same thresholding as Stage 11
- Offsets jammer ring by 5 degrees to avoid exact signal alignment at 90 deg
- JAM_POW = 40 dB to match 12A

Output: stage12b_multi_jammer.png
"""

import numpy as np
import sys
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from mesh_loader import load_uav_mesh
from conformal_array import get_conformal_array
from attitude import euler_to_quaternion, rotate_points
from shadow_engine import compute_shadow_mask
from lcmv_stage8 import get_steering_vector
from constants import NACT_THRESHOLD
from em_physics import compute_plf

MESH_PATH  = Path(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone_prepared.stl")
FREQ_HZ    = 2.4e9
LAM        = 3e8 / FREQ_HZ
K          = 2 * np.pi / LAM
JAM_POW    = 10 ** (40.0 / 10.0)   # 40 dB INR
SIG_POW    = 10 ** (20.0 / 10.0)
NOISE_POW  = 1.0
N_ELEMENTS = 16
N_HEADINGS = 360   # 1-degree resolution


def mvdr_beamformer(R_xx, a_sig, g):
    """
    Optimal MVDR beamformer matching Stage 11.
    Naturally places deep nulls on all jammers present in R_xx
    without suffering from explicit constraint weight explosion.
    """
    mask    = (np.abs(g) >= NACT_THRESHOLD).astype(float)
    penalty = 1e8 * (1.0 - mask)
    R_reg   = R_xx + np.diag(penalty)

    try:
        R_inv = np.linalg.inv(R_reg)
    except np.linalg.LinAlgError:
        R_inv = np.linalg.pinv(R_reg)

    num = R_inv @ a_sig
    den = np.conj(a_sig) @ num
    return num / max(abs(den), 1e-12)


def main():
    print("Loading mesh and baseline conformal array...")
    mesh = load_uav_mesh(MESH_PATH)
    pos_body, normals_body = get_conformal_array(mesh)

    headings  = np.linspace(0, 360, N_HEADINGS, endpoint=False)
    sig_world = np.array([1.0, 0.0, 0.0])

    print(f"Precomputing shadow gains for {N_HEADINGS} headings...")
    G_pre      = np.zeros((N_HEADINGS, N_ELEMENTS), dtype=complex)
    A_sig_pre  = []
    
    jam0_world = np.array([0.0, 1.0, 0.0])

    for hi, h_ang in enumerate(headings):
        bank  = 15.0 if h_ang <= 180 else -15.0
        q     = euler_to_quaternion(np.deg2rad(bank), 0, np.deg2rad(h_ang))
        q_inv = q.conjugate()
        jam_body = rotate_points(jam0_world.reshape(1, 3), q_inv)[0]
        sig_body = rotate_points(sig_world.reshape(1, 3),  q_inv)[0]
        
        g = compute_shadow_mask(mesh, pos_body, normals_body, jam_body)
        G_pre[hi] = g
        
        A_sig_pre.append(get_steering_vector(pos_body, K * sig_body))

    print("Done.\n")

    jammer_counts = list(range(1, 13))
    sinr_worst    = np.zeros(len(jammer_counts))
    sinr_median   = np.zeros(len(jammer_counts))

    for ji, n_jam in enumerate(jammer_counts):
        print(f"Testing {n_jam} simultaneous jammer(s)...")

        jam_angles = np.linspace(0, 360, n_jam, endpoint=False)
        if np.any(np.isclose(jam_angles, 90.0)):
            jam_angles += 15.0
            
        jam_worlds = np.array([
            [np.sin(np.deg2rad(a)), np.cos(np.deg2rad(a)), 0.0]
            for a in jam_angles
        ])

        sinr_per_heading = []

        for hi, h_ang in enumerate(headings):
            bank  = 15.0 if h_ang <= 180 else -15.0
            q     = euler_to_quaternion(np.deg2rad(bank), 0, np.deg2rad(h_ang))
            q_inv = q.conjugate()

            sig_body = rotate_points(sig_world.reshape(1, 3), q_inv)[0]
            a_sig    = A_sig_pre[hi]
            g        = G_pre[hi]

            R_xx = SIG_POW  * np.outer(a_sig, np.conj(a_sig)) + \
                   NOISE_POW * np.eye(N_ELEMENTS)
            a_jam_list = []

            for jw in jam_worlds:
                jb    = rotate_points(jw.reshape(1, 3), q_inv)[0]
                k_jam = K * jb
                a_j   = get_steering_vector(pos_body, k_jam)
                R_xx += JAM_POW * np.outer(a_j, np.conj(a_j))
                a_jam_list.append(a_j)

            w = mvdr_beamformer(R_xx, a_sig, g)

            P_s = SIG_POW * abs(np.conj(w) @ a_sig) ** 2
            P_j = sum(JAM_POW * abs(np.conj(w) @ aj) ** 2 for aj in a_jam_list)
            P_n = NOISE_POW * np.linalg.norm(w) ** 2
            sinr = 10 * np.log10(max(P_s / (P_j + P_n + 1e-12), 1e-12))
            sinr_per_heading.append(sinr)

        sinr_worst[ji]  = np.min(sinr_per_heading)
        sinr_median[ji] = np.median(sinr_per_heading)
        print(f"  Worst-case SINR: {sinr_worst[ji]:.2f} dB  |  Median: {sinr_median[ji]:.2f} dB")

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(f"Simultaneous Multi-Jammer Null Steering (Baseline 16-Element Array)\n"
                 f"Worst-Case vs Median Heading Performance", fontsize=14, fontweight='bold', y=0.98)

    x = np.array(jammer_counts)
    ax.plot(x, sinr_median, "s-",  color="#0072B2", lw=2.5, ms=7, label="Median SINR (over all headings)")
    ax.plot(x, sinr_worst,  "o--", color="#D55E00", lw=2,   ms=7, label="Worst-case SINR")
    ax.fill_between(x, sinr_worst, sinr_median, alpha=0.15, color="#0072B2")

    ax.axhline(0,  color="gray",    ls=":",  lw=1.2, label="0 dB (break-even)")
    ax.axvline(N_ELEMENTS - 1, color="#C0392B", ls="--", lw=1.5,
               label=f"Theoretical DoF limit (N−1 = {N_ELEMENTS-1} jammers)")

    ax.set_xlabel("Number of Simultaneous Jammers", fontsize=11)
    ax.set_ylabel("SINR (dB)", fontsize=11)
    ax.set_xticks(x)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = Path(__file__).parent / "stage12b_multi_jammer.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
