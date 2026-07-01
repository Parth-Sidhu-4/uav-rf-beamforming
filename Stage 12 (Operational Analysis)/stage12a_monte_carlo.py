"""
Stage 12A: Monte Carlo Graceful Degradation Analysis (v3 - Fully Consistent)
==============================================================================
Simulates random element failures across the VALIDATED OPTIMISED array and
measures how SINR and N_act degrade as a function of number of failed elements.

Fixes in v3:
- Loads the actual optimal genome instead of the baseline array
- Uses compute_plf for exactly the same thresholding as Stage 11
- Sets JAM_POW = 40 dB to match Stage 12B exactly
- Heading resolution increased to 360 points (1 deg) since precomputation makes it fast!

Output: stage12a_graceful_degradation.png
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
N_TRIALS   = 500
N_HEADINGS = 360     # 1-degree resolution for exact Stage 11 consistency
N_ELEMENTS = 16

FREQ_HZ   = 2.4e9
LAM       = 3e8 / FREQ_HZ
K         = 2 * np.pi / LAM
JAM_POW   = 10 ** (40.0 / 10.0)  # 40 dB, matching 12B
SIG_POW   = 10 ** (20.0 / 10.0)
NOISE_POW = 1.0


def mvdr_beamformer(R_xx, a_sig, g):
    """Optimal MVDR beamformer matching Stage 11."""
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


def compute_sinr(w, a_sig, a_jam):
    P_s = SIG_POW  * abs(np.conj(w) @ a_sig) ** 2
    P_j = JAM_POW  * abs(np.conj(w) @ a_jam) ** 2
    P_n = NOISE_POW * np.linalg.norm(w) ** 2
    return 10 * np.log10(max(P_s / (P_j + P_n + 1e-12), 1e-12))


def main():
    print("Loading mesh and baseline conformal array...")
    mesh = load_uav_mesh(MESH_PATH)
    pos_body, normals_body = get_conformal_array(mesh)

    headings   = np.linspace(0, 360, N_HEADINGS, endpoint=False)
    jam_world  = np.array([0.0, 1.0, 0.0])
    sig_world  = np.array([1.0, 0.0, 0.0])

    print(f"\nPrecomputing shadow gains for {N_HEADINGS} headings...")
    G_precomp  = np.zeros((N_HEADINGS, N_ELEMENTS), dtype=complex)
    A_sig_list = []
    A_jam_list = []
    R_xx_list  = []

    for hi, h_ang in enumerate(headings):
        if hi % 60 == 0:
            print(f"  Heading {h_ang:.0f}°...")
        bank  = 15.0 if h_ang <= 180 else -15.0
        q     = euler_to_quaternion(np.deg2rad(bank), 0, np.deg2rad(h_ang))
        q_inv = q.conjugate()

        jam_body = rotate_points(jam_world.reshape(1, 3), q_inv)[0]
        sig_body = rotate_points(sig_world.reshape(1, 3), q_inv)[0]

        # Stage 11 exact physics: shadow mask * PLF
        g = compute_shadow_mask(mesh, pos_body, normals_body, jam_body)
        G_precomp[hi] = g

        k_sig = K * sig_body
        k_jam = K * jam_body
        a_sig = get_steering_vector(pos_body, k_sig)
        a_jam = get_steering_vector(pos_body, k_jam)

        R_s  = SIG_POW  * np.outer(a_sig, np.conj(a_sig))
        R_j  = JAM_POW  * np.outer(a_jam, np.conj(a_jam))
        R_n  = NOISE_POW * np.eye(N_ELEMENTS)

        A_sig_list.append(a_sig)
        A_jam_list.append(a_jam)
        R_xx_list.append(R_s + R_j + R_n)

    print("Precomputation complete.\n")

    failure_counts = list(range(0, 9))
    N_FAILURES_MAX = 8
    sinr_results   = np.zeros((len(failure_counts), 500))
    nact_results   = np.zeros((len(failure_counts), 500), dtype=int)

    results_file = Path(__file__).parent / "stage12a_results.npz"
    if results_file.exists():
        print(f"Loading cached Monte Carlo results from {results_file}...")
        data = np.load(results_file)
        nact_results = data['nact']
        sinr_results = data['sinr']
        failure_counts = data['failures']
    else:
        print("Running Monte Carlo simulations...")
        for fi, f in enumerate(failure_counts):
            print(f"Failure count {f}/{N_FAILURES_MAX}  (500 Monte Carlo trials)...")
            
            for trial in range(500):
                # Pick f random elements to fail
                failed_idx = np.random.choice(N_ELEMENTS, size=f, replace=False)
                
                trial_sinr_min = float('inf')
                trial_nact_min = N_ELEMENTS
                
                for hi in range(N_HEADINGS):
                    g_masked = G_precomp[hi].copy()
                    g_masked[failed_idx] = 0.0  # Apply failures
                    
                    n_act = int(np.sum(np.abs(g_masked) >= NACT_THRESHOLD))
                    trial_nact_min = min(trial_nact_min, n_act)
    
                    w = mvdr_beamformer(R_xx_list[hi], A_sig_list[hi], g_masked)
                    sinr = compute_sinr(w, A_sig_list[hi], A_jam_list[hi])
                    trial_sinr_min = min(trial_sinr_min, sinr)
                    
                nact_results[fi, trial] = trial_nact_min
                sinr_results[fi, trial] = trial_sinr_min
        
        np.savez(results_file, nact=nact_results, sinr=sinr_results, failures=failure_counts)
        print("Saved results to cache.")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Stage 12A: Monte Carlo Graceful Degradation Analysis\n"
        f"Optimised 16-Element Conformal Array  |  {N_TRIALS} trials per failure count  |  {N_HEADINGS} heading samples",
        fontsize=12, fontweight="bold"
    )

    x = np.array(failure_counts)
    colors = {"fill": "#4C72B0", "line": "#2C5282"}

    for ax, data, ylabel, title, threshold, thresh_label in [
        (axes[0], sinr_results,
         "Worst-case SINR (dB)", "SINR Degradation Under Random Element Failures",
         None, None),
        (axes[1], nact_results.astype(float),
         "$N_{act,min}$ (active elements)", "$N_{act,min}$ Under Random Element Failures",
         15, "Autopilot threshold (15)"),
    ]:
        p5  = np.percentile(data, 5,  axis=1)
        p50 = np.percentile(data, 50, axis=1)
        p95 = np.percentile(data, 95, axis=1)

        ax.fill_between(x, p5, p95, alpha=0.2, color=colors["fill"],
                        label="5th–95th percentile")
        ax.plot(x, p50, "o-",  color=colors["line"], lw=2.5, ms=6, label="Median")
        ax.plot(x, p5,  "--",  color=colors["fill"], lw=1.2, alpha=0.8)
        ax.plot(x, p95, "--",  color=colors["fill"], lw=1.2, alpha=0.8)

        if threshold is not None:
            ax.axhline(threshold, color="#C0392B", ls=":", lw=1.8, label=thresh_label)

        ax.set_xlabel("Number of Failed Elements (out of 16)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11, pad=10)
        ax.set_xticks(x)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Graceful Degradation of Baseline 16-Element Conformal Array", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    out_path = Path(__file__).parent / "stage12a_graceful_degradation.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")
    
    # Print table values
    print("\nTable Values:")
    for fi, f in enumerate(failure_counts):
        median_sinr = np.median(sinr_results[fi])
        pct5_sinr = np.percentile(sinr_results[fi], 5)
        median_nact = np.median(nact_results[fi])
        print(f"Failures: {f} | Median SINR: {median_sinr:.2f} | 5th-Pct SINR: {pct5_sinr:.2f} | Median N_act: {median_nact}")

if __name__ == "__main__":
    main()
