"""
Two diagnostics for the ~180-degree dead zone:

Diag A: Per-heading SINR curve at fine resolution (K=5 model, 150-210 deg range).
         Narrow spike = topological/phase discontinuity signature.
         Wide smooth degradation = capacity/optimization problem.

Diag B: Target continuity check.
         Compute v_ideal and g_exact * v_ideal at 178-182 deg at fine steps.
         Look for phase jumps, sign flips, or any non-smoothness in the
         TARGET FUNCTION itself - independent of any model.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append(os.path.abspath(r'D:\UAV Internship project\Phase 2 Track 1\em_realism'))
sys.path.append(os.path.abspath(r'D:\UAV Internship project\Phase 2 Track 2\array_structures'))
sys.path.append(os.path.abspath(r'D:\UAV Internship project\Stage 12 (Operational Analysis)'))
sys.path.append(os.path.abspath(r'D:\UAV Internship project\Stage 13 (AI Beamforming)'))
sys.path.append(os.path.abspath(r'D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)'))

import torch
from mesh_loader import load_uav_mesh
from conformal_array import get_conformal_array_parametric
from lcmv_stage8 import get_steering_vector
from attitude import euler_to_quaternion, rotate_points
from shadow_engine_batched import compute_shadow_mask_batched
from stage18_d3_covariance_train import (
    SIRENCovariancePredictor, d3_mvdr_beamformer, compute_sinr,
    get_steering_vector_pt
)

def main():
    device = torch.device('cpu')
    MESH_PATH = Path(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone_prepared.stl")
    mesh = load_uav_mesh(MESH_PATH)
    pos_body, normals_body = get_conformal_array_parametric(mesh, N=32)
    pos_lambda_np = pos_body / 0.15
    pos_lambda = torch.tensor(pos_lambda_np, dtype=torch.float64)

    q_inv = euler_to_quaternion(np.deg2rad(15.0), 0, 0).conjugate()
    K_wave = 2.0 * np.pi / 0.15
    P_S, P_J, sigma2, alpha = 100.0, 10000.0, 1.0, 390.0

    sig_world = np.array([1.0, 0.0, 0.0])
    sig_body = rotate_points(sig_world.reshape(1,3), q_inv)[0]
    g_sig = compute_shadow_mask_batched(mesh, pos_body, normals_body, sig_body.reshape(1,3))[0]
    v_sig_np = g_sig * get_steering_vector(pos_body, K_wave * sig_body)
    v_sig = torch.tensor(v_sig_np, dtype=torch.complex128)

    # =========================================================
    # DIAG B: Target continuity check (NO MODEL NEEDED)
    # =========================================================
    print("\n" + "="*60)
    print("DIAG B: Target continuity check near heading=180°")
    print("="*60)
    print("Checking v_ideal = exp(2*pi*i * pos_lambda . d) and phase of each")
    print("element as heading crosses 180° (the atan2 branch cut).\n")

    fine_headings = np.arange(175.0, 185.1, 0.5)  # 0.5-deg steps around 180°

    # For each heading, compute:
    # 1. phi_j = atan2(dy, dx) in body frame -> check for branch cut
    # 2. v_ideal phase at element 0 (most informative single element)
    # 3. g_exact * v_ideal phase at element 0
    # 4. max phase jump between consecutive headings

    v_ideals = []
    phi_j_vals = []
    jam_bodies_list = []

    for h in fine_headings:
        jam_world = np.array([np.cos(np.deg2rad(h)), np.sin(np.deg2rad(h)), 0.0])
        jam_body = rotate_points(jam_world.reshape(1,3), q_inv)[0]
        jam_bodies_list.append(jam_body)
        phi_j_vals.append(np.arctan2(jam_body[1], jam_body[0]))

    # Get g_exact for all these headings at once
    jam_bodies_arr = np.array(jam_bodies_list)
    g_exact_arr = compute_shadow_mask_batched(mesh, pos_body, normals_body, jam_bodies_arr)

    print(f"  {'Heading':>8} | {'phi_j (body, deg)':>18} | {'|g[0]|':>8} | {'phase(v_ideal[0]) deg':>22} | {'phase(g*v[0]) deg':>18} | {'d_phase(v)':>12}")
    print(f"  {'-'*110}")

    prev_phase_v = None
    prev_phase_gv = None
    for i, h in enumerate(fine_headings):
        jam_body = jam_bodies_list[i]
        jam_body_t = torch.tensor(jam_body, dtype=torch.float64).unsqueeze(0)
        theta_j = torch.acos(jam_body_t[:, 2] / torch.norm(jam_body_t, dim=1))
        phi_j   = torch.atan2(jam_body_t[:, 1], jam_body_t[:, 0])

        v_ideal_t = get_steering_vector_pt(pos_lambda, theta_j, phi_j)
        v_ideal_np = v_ideal_t[0].numpy()

        g = g_exact_arr[i]
        gv = g * v_ideal_np

        phase_v0  = np.angle(v_ideal_np[0], deg=True)
        phase_gv0 = np.angle(gv[0], deg=True)
        phi_body_deg = np.rad2deg(phi_j_vals[i])

        dphase_str = "—"
        if prev_phase_v is not None:
            dp = phase_v0 - prev_phase_v
            # Unwrap manually for display
            if dp > 180: dp -= 360
            if dp < -180: dp += 360
            dphase_str = f"{dp:+.2f}°"
        
        prev_phase_v  = phase_v0
        prev_phase_gv = phase_gv0

        print(f"  {h:>8.1f}° | {phi_body_deg:>18.3f}° | {np.abs(g[0]):>8.4f} | {phase_v0:>22.3f}° | {phase_gv0:>18.3f}° | {dphase_str:>12}")

    # Also check phase continuity across ALL 32 elements at crossing point
    print(f"\n  Phase jump matrix (all 32 elements) at 179.5->180.0->180.5 deg:")
    for step_idx, h_pair in enumerate([(179.5, 180.0), (180.0, 180.5)]):
        h0, h1 = h_pair
        i0 = np.where(np.isclose(fine_headings, h0))[0][0]
        i1 = np.where(np.isclose(fine_headings, h1))[0][0]

        jam0 = torch.tensor(jam_bodies_list[i0], dtype=torch.float64).unsqueeze(0)
        jam1 = torch.tensor(jam_bodies_list[i1], dtype=torch.float64).unsqueeze(0)

        th0 = torch.acos(jam0[:,2]/torch.norm(jam0,dim=1)); ph0 = torch.atan2(jam0[:,1],jam0[:,0])
        th1 = torch.acos(jam1[:,2]/torch.norm(jam1,dim=1)); ph1 = torch.atan2(jam1[:,1],jam1[:,0])

        v0 = get_steering_vector_pt(pos_lambda, th0, ph0)[0].numpy()
        v1 = get_steering_vector_pt(pos_lambda, th1, ph1)[0].numpy()

        phase_diffs = np.rad2deg(np.angle(v1 / v0))  # element-wise phase jump

        print(f"\n  Crossing {h0}->{h1} deg:")
        print(f"    max |Δphase| across elements = {np.max(np.abs(phase_diffs)):.4f}°")
        print(f"    mean|Δphase|                 = {np.mean(np.abs(phase_diffs)):.4f}°")
        print(f"    Any |Δphase| > 90°?          = {np.any(np.abs(phase_diffs) > 90)}")
        print(f"    Full spectrum: {' | '.join([f'{p:+.1f}°' for p in phase_diffs])}")

    # =========================================================
    # DIAG A: Per-heading SINR curve from existing K=5 model
    # =========================================================
    print("\n" + "="*60)
    print("DIAG A: Per-heading SINR from K=5 model (150°–210°)")
    print("="*60)

    model = SIRENCovariancePredictor(w0=30.0, K_rank=5).to(device)
    model.load_state_dict(torch.load("siren_beamformer_d3_cov_K5_seed2.pt", map_location=device))
    model.eval()

    # Use the full 3600-point cached dataset (0.1 deg resolution)
    data = np.load("dataset_3600_masks.npz")
    g_full = data['g_exact']
    headings_full = np.linspace(0, 360, 3600, endpoint=False)  # 0.1 deg steps

    # Select range 150-210 deg
    mask = (headings_full >= 150) & (headings_full <= 210)
    headings_diag = headings_full[mask]
    g_diag = g_full[mask]

    jam_bodies_diag = np.zeros((len(headings_diag), 3))
    for i, h in enumerate(headings_diag):
        jam_world = np.array([np.cos(np.deg2rad(h)), np.sin(np.deg2rad(h)), 0.0])
        jam_bodies_diag[i] = rotate_points(jam_world.reshape(1,3), q_inv)[0]

    jam_bodies_t = torch.tensor(jam_bodies_diag, dtype=torch.float64)
    g_diag_t = torch.tensor(g_diag, dtype=torch.complex128)

    theta_j = torch.acos(jam_bodies_t[:,2] / torch.norm(jam_bodies_t,dim=1))
    phi_j   = torch.atan2(jam_bodies_t[:,1], jam_bodies_t[:,0])

    v_true = g_diag_t * get_steering_vector_pt(pos_lambda, theta_j, phi_j)
    R_j = P_J * torch.einsum('bi,bj->bij', v_true, torch.conj(v_true))
    R_s = P_S * torch.einsum('i,j->ij', v_sig, torch.conj(v_sig)).unsqueeze(0).expand(len(headings_diag),-1,-1)
    R_n = sigma2 * torch.eye(32, dtype=torch.complex128).unsqueeze(0).expand(len(headings_diag),-1,-1)

    with torch.no_grad():
        U = model(jam_bodies_t.float()).to(torch.complex128)
        w = d3_mvdr_beamformer(U, v_sig, P_J, sigma2, alpha)
        sinr = compute_sinr(w, R_s, R_j, R_n).numpy()

    # Print summary stats
    print(f"\n  Heading range 150-210°: {len(headings_diag)} points (0.1° resolution)")
    print(f"  Overall min SINR in range: {sinr.min():.2f} dB at heading {headings_diag[np.argmin(sinr)]:.1f}°")
    print(f"  Overall max SINR in range: {sinr.max():.2f} dB at heading {headings_diag[np.argmax(sinr)]:.1f}°")

    # Check width of the dip below 15 dB
    below_threshold = headings_diag[sinr < 15.0]
    if len(below_threshold) > 0:
        dip_width = below_threshold[-1] - below_threshold[0]
        dip_center = (below_threshold[-1] + below_threshold[0]) / 2
        print(f"\n  Region below 15 dB threshold:")
        print(f"    From {below_threshold[0]:.1f}° to {below_threshold[-1]:.1f}°")
        print(f"    Width: {dip_width:.1f}°  (narrow spike=topological; wide=capacity)")
        print(f"    Center: {dip_center:.1f}°")
    else:
        print("  No region below 15 dB in this range.")

    # Fine printout around the worst region
    worst_idx = np.argmin(sinr)
    print(f"\n  Fine-grained SINR around worst heading ({headings_diag[worst_idx]:.1f}°):")
    window = slice(max(0, worst_idx - 30), min(len(sinr), worst_idx + 31))
    for h, s in zip(headings_diag[window], sinr[window]):
        bar = "█" * max(0, int(s))
        thresh_marker = " <-- BELOW 15dB" if s < 15.0 else ""
        print(f"    {h:6.1f}°: {s:7.2f} dB {thresh_marker}")

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Top: SINR curve
    axes[0].plot(headings_diag, sinr, 'b-', linewidth=1.5)
    axes[0].axhline(15.0, color='red', linestyle='--', label='15 dB threshold')
    axes[0].axvline(180.0, color='orange', linestyle=':', alpha=0.8, label='heading=180°')
    axes[0].fill_between(headings_diag, sinr, 15.0, where=(sinr < 15.0),
                         alpha=0.3, color='red', label='Failure region')
    axes[0].set_xlabel('Jammer Heading (degrees)')
    axes[0].set_ylabel('SINR (dB)')
    axes[0].set_title('Per-Heading SINR: D-3 K=5 Model (150°–210°)\nNarrow spike → topological; Wide degradation → capacity')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(150, 210)

    # Bottom: Phase of v_ideal[0] across the same range
    phases_v0 = []
    for i, h in enumerate(headings_diag):
        jb = torch.tensor(jam_bodies_diag[i], dtype=torch.float64).unsqueeze(0)
        th = torch.acos(jb[:,2]/torch.norm(jb,dim=1)); ph = torch.atan2(jb[:,1],jb[:,0])
        v_i = get_steering_vector_pt(pos_lambda, th, ph)[0].numpy()
        phases_v0.append(np.angle(v_i[0], deg=True))

    phases_v0 = np.unwrap(np.array(phases_v0), period=360)
    axes[1].plot(headings_diag, phases_v0, 'g-', linewidth=1.5)
    axes[1].axvline(180.0, color='orange', linestyle=':', alpha=0.8)
    axes[1].set_xlabel('Jammer Heading (degrees)')
    axes[1].set_ylabel('Phase of v_ideal[0] (degrees, unwrapped)')
    axes[1].set_title('Target Continuity: Phase of v_ideal[element 0] vs Heading\nDiscontinuity here = target-side topological problem')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(150, 210)

    plt.tight_layout()
    plt.savefig('diag_180deg_deadzone_seed2.png', dpi=120)
    plt.close()
    print(f"\n  Plot saved: diag_180deg_deadzone_seed2.png")
    print("\nDone.")

if __name__ == '__main__':
    main()
