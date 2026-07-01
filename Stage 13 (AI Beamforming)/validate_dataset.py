"""
Dataset Validation Script
=========================
Checks whether the stored training labels in dataset_shadow_30k.npz
actually match freshly recomputed exact physics for the same input directions.

Tests:
  1. Input quality  – unit vectors, no NaN/inf, coverage of sphere
  2. Label statistics – magnitude/phase distributions, fraction shadowed
  3. Label vs recompute – pick 200 random samples, rerun shadow engine,
     compare stored labels vs fresh recompute (magnitude MAE, phase MAE)
  4. Steering vector consistency – verify stored labels produce the expected
     MVDR SINR when used as oracle (should match eval script oracle)
"""
import sys, os
import numpy as np
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Phase 2 Track 1', 'em_realism')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Phase 2 Track 2', 'array_structures')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Stage 12 (Operational Analysis)')))

from mesh_loader import load_uav_mesh
from conformal_array import get_conformal_array
from shadow_engine_batched import compute_shadow_mask_batched

DATASET_PATH = "dataset_shadow_30k.npz"
MESH_PATH    = Path(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone_prepared.stl")
LAM = 0.15
K   = 2 * np.pi / LAM
N_VALIDATE = 200   # number of samples to recompute


def get_steering_vector(pos_body, k_vec):
    phases = pos_body @ k_vec
    return np.exp(1j * phases)


def mvdr_sinr_from_gains(pos_body, sig_body, jam_body, g_jam,
                          JAM_POW=10000., NOISE_POW=1., SIG_POW=100.):
    v_sig = get_steering_vector(pos_body, K * sig_body)
    v_jam = g_jam * get_steering_vector(pos_body, K * jam_body)
    R_j   = JAM_POW  * np.outer(v_jam, np.conj(v_jam))
    R_n   = NOISE_POW * np.eye(len(v_sig))
    R_s   = SIG_POW   * np.outer(v_sig, np.conj(v_sig))
    R_xx  = R_j + R_n + 1e-10 * np.eye(len(v_sig))
    try:    R_inv = np.linalg.inv(R_xx)
    except: R_inv = np.linalg.pinv(R_xx)
    num = R_inv @ v_sig
    den = np.conj(v_sig) @ num
    w   = num / max(abs(den), 1e-12)
    S   = np.real(np.conj(w) @ R_s @ w)
    NJ  = np.real(np.conj(w) @ (R_j + R_n) @ w)
    return 10 * np.log10(S / max(NJ, 1e-12))


def main():
    print("=" * 60)
    print("Dataset Validation")
    print("=" * 60)

    # --- Load ---
    print(f"\n[1] Loading dataset: {DATASET_PATH}")
    if not os.path.exists(DATASET_PATH):
        print("  ERROR: dataset file not found.")
        return
    data   = np.load(DATASET_PATH)
    inputs = data['inputs']   # [N, 3]   jammer body directions
    labels = data['labels']   # [N, 32]  interleaved re/im Fresnel gains
    N      = inputs.shape[0]
    print(f"  Samples: {N}")
    print(f"  Input shape:  {inputs.shape}")
    print(f"  Label shape:  {labels.shape}")

    # --- Input quality ---
    print("\n[2] Input quality checks")
    norms = np.linalg.norm(inputs, axis=1)
    print(f"  ||x|| mean={norms.mean():.6f}  min={norms.min():.6f}  max={norms.max():.6f}")
    nan_in = np.isnan(inputs).sum()
    inf_in = np.isinf(inputs).sum()
    print(f"  NaN in inputs: {nan_in}   Inf in inputs: {inf_in}")
    print(f"  x range: [{inputs[:,0].min():.3f}, {inputs[:,0].max():.3f}]")
    print(f"  y range: [{inputs[:,1].min():.3f}, {inputs[:,1].max():.3f}]")
    print(f"  z range: [{inputs[:,2].min():.3f}, {inputs[:,2].max():.3f}]")
    not_unit = np.sum(np.abs(norms - 1.0) > 0.01)
    print(f"  Samples with ||x|| outside [0.99, 1.01]: {not_unit}")

    # --- Label statistics ---
    print("\n[3] Label statistics (Fresnel gain magnitudes & phases)")
    g_complex = labels[:, 0::2] + 1j * labels[:, 1::2]  # [N, 16]
    mags   = np.abs(g_complex)
    phases = np.angle(g_complex)
    nan_lb = np.isnan(labels).sum()
    inf_lb = np.isinf(labels).sum()
    print(f"  NaN in labels: {nan_lb}   Inf in labels: {inf_lb}")
    print(f"  Magnitude: mean={mags.mean():.4f}  std={mags.std():.4f}  "
          f"min={mags.min():.4f}  max={mags.max():.4f}")
    frac_los = np.mean(mags > 0.9)
    frac_shadow = np.mean(mags < 0.1)
    print(f"  Fraction with |g| > 0.9 (near LOS):   {frac_los*100:.1f}%")
    print(f"  Fraction with |g| < 0.1 (deep shadow): {frac_shadow*100:.1f}%")
    print(f"  Phase: mean={phases.mean():.4f} rad  std={phases.std():.4f} rad")
    # Per-element statistics to check for any dead elements
    per_elem_mag_mean = mags.mean(axis=0)
    per_elem_mag_std  = mags.std(axis=0)
    print(f"  Per-element |g| mean range: [{per_elem_mag_mean.min():.3f}, {per_elem_mag_mean.max():.3f}]")
    print(f"  Per-element |g| std  range: [{per_elem_mag_std.min():.3f},  {per_elem_mag_std.max():.3f}]")

    # --- Recompute and compare ---
    print(f"\n[4] Recompute cross-validation ({N_VALIDATE} random samples)")
    print("  Loading mesh...")
    mesh = load_uav_mesh(MESH_PATH)
    pos_body, normals_body = get_conformal_array(mesh)
    print(f"  Array elements: {pos_body.shape[0]}")

    rng  = np.random.default_rng(seed=0)
    idx  = rng.choice(N, size=N_VALIDATE, replace=False)
    sample_dirs = inputs[idx]          # [200, 3]
    sample_labs = labels[idx]          # [200, 32]

    print("  Running compute_shadow_mask_batched on 200 samples...")
    g_recomputed = compute_shadow_mask_batched(mesh, pos_body, normals_body, sample_dirs)
    # g_recomputed: [200, 16] complex

    g_stored = sample_labs[:, 0::2] + 1j * sample_labs[:, 1::2]  # [200, 16]

    mag_recomp  = np.abs(g_recomputed)
    mag_stored  = np.abs(g_stored)
    phase_recomp = np.angle(g_recomputed)
    phase_stored = np.angle(g_stored)

    mag_mae   = np.mean(np.abs(mag_recomp - mag_stored))
    phase_diff = np.abs(phase_recomp - phase_stored)
    phase_diff = np.minimum(phase_diff, 2*np.pi - phase_diff)
    phase_mae = np.mean(phase_diff)

    re_mae = np.mean(np.abs(g_recomputed.real - g_stored.real))
    im_mae = np.mean(np.abs(g_recomputed.imag - g_stored.imag))

    print(f"\n  === Stored vs Recomputed ===")
    print(f"  Magnitude MAE:      {mag_mae:.6f}")
    print(f"  Phase MAE (wrapped): {phase_mae:.6f} rad  ({np.rad2deg(phase_mae):.3f} deg)")
    print(f"  Real part MAE:       {re_mae:.6f}")
    print(f"  Imaginary part MAE:  {im_mae:.6f}")

    # Worst-case sample
    per_sample_mag_err = np.mean(np.abs(mag_recomp - mag_stored), axis=1)
    worst_idx = np.argmax(per_sample_mag_err)
    print(f"\n  Worst sample (idx {idx[worst_idx]}):")
    print(f"    Input direction: {sample_dirs[worst_idx]}")
    print(f"    Magnitude MAE:  {per_sample_mag_err[worst_idx]:.6f}")

    # Correlation between stored and recomputed magnitudes
    corr_mag = np.corrcoef(mag_recomp.flatten(), mag_stored.flatten())[0, 1]
    print(f"\n  Pearson correlation (stored vs recomputed magnitudes): {corr_mag:.6f}")

    # --- SINR sanity check using stored gains as oracle ---
    print(f"\n[5] SINR sanity check: stored gains as oracle (10 samples)")
    sig_body = np.array([1.0, 0.0, 0.0])  # fixed signal direction
    sinrs_stored   = []
    sinrs_recomputed = []
    check_idx = rng.choice(N_VALIDATE, size=10, replace=False)
    for ci in check_idx:
        jb = sample_dirs[ci]
        g_st = g_stored[ci]
        g_rc = g_recomputed[ci]
        s_st = mvdr_sinr_from_gains(pos_body, sig_body, jb, g_st)
        s_rc = mvdr_sinr_from_gains(pos_body, sig_body, jb, g_rc)
        sinrs_stored.append(s_st)
        sinrs_recomputed.append(s_rc)
        print(f"  Sample {ci:3d}: stored SINR={s_st:7.2f} dB  recomputed SINR={s_rc:7.2f} dB  diff={s_st-s_rc:+.2f} dB")

    print(f"\n  Mean SINR  (stored):     {np.mean(sinrs_stored):.2f} dB")
    print(f"  Mean SINR  (recomputed): {np.mean(sinrs_recomputed):.2f} dB")
    print(f"  Mean |SINR diff|:        {np.mean(np.abs(np.array(sinrs_stored)-np.array(sinrs_recomputed))):.2f} dB")

    print("\n[6] Verdict")
    if mag_mae < 0.01 and phase_mae < 0.05 and corr_mag > 0.99:
        print("  PASS – stored labels closely match recomputed physics.")
        print("         Dataset quality is NOT the root cause of performance issues.")
    elif mag_mae < 0.05 and corr_mag > 0.95:
        print("  MARGINAL – minor discrepancy between stored and recomputed labels.")
        print("             Investigate worst-case samples.")
    else:
        print("  FAIL – significant mismatch between stored labels and recomputed physics.")
        print(f"         Magnitude MAE={mag_mae:.4f}, Phase MAE={np.rad2deg(phase_mae):.2f} deg, corr={corr_mag:.4f}")
        print("         The dataset may be corrupted or generated with a different code version.")


if __name__ == '__main__':
    main()
