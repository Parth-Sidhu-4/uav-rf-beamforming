"""
Stage 14D – Four-Way Evaluation: Complete Architectural Progression
====================================================================
Loads four trained models and produces a unified set of comparison figures
showing the contribution of each architectural change:

  Curve 1 – Stage 13 Baseline: ReLU MLP (3→512→256→128→32), Cartesian MSE, no PE
  Curve 2 – Phase A:           ReLU MLP (39→512→256→128→32), Fourier PE + MSE
  Curve 3 – Phase B:           ReLU MLP (39→512→256→128→32), Fourier PE + NullResponseLoss
  Curve 4 – Phase C:           SIREN    (3→512→512→512→32),  NullResponseLoss
  Reference – Exact Physics:   compute_shadow_mask_batched (ground truth)

Output figures (primary first):
  test1_sinr_threeway.png         – SINR vs jammer azimuth (360° sweep)
  test2_sinr_threeway.png         – SINR vs jammer elevation (−60° to +60° sweep)
  diag2_null_placement_threeway.png – Angular null placement error vs azimuth
  diag4_dl_sweep_threeway.png     – Worst-case SINR vs diagonal loading (robustness)

omega_0 Validation (Phase C post-training checks):
  After running this script, verify:
  (a) The ~250° MAE spike in test1 results is reduced below the Phase B level.
  (b) The median null placement error in diag2 is below 5°.
  If either condition fails, retrain Phase C at omega_0=15 and omega_0=60 and
  compare their Test 1 SINR floors to determine the best-performing value.
"""
import sys
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Phase 2 Track 1', 'em_realism')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Phase 2 Track 2', 'array_structures')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Stage 12 (Operational Analysis)')))

from mesh_loader import load_uav_mesh
from conformal_array import get_conformal_array
from shadow_engine_batched import compute_shadow_mask_batched
from attitude import euler_to_quaternion, rotate_points
from lcmv_stage8 import get_steering_vector

# ---------------------------------------------------------------------------
# Model Definitions (inline to avoid import-path conflicts)
# ---------------------------------------------------------------------------
def positional_encoding(x, L=6):
    """x: [batch, 3] -> [batch, 3*(1 + 2L)] = [batch, 39]"""
    encoded = [x]
    for i in range(L):
        encoded.append(torch.sin((2.0 ** i) * torch.pi * x))
        encoded.append(torch.cos((2.0 ** i) * torch.pi * x))
    return torch.cat(encoded, dim=-1)


class ShadowNet_Stage13(nn.Module):
    """Stage 13 Baseline: raw [x,y,z] input, no positional encoding.
    Actual saved checkpoint is 3→128→128→32 (2 hidden layers, 128 units)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 32)
        )
    def forward(self, x):
        return self.net(x)


class ShadowNet_PE(nn.Module):
    """Phase A & B: Fourier PE input (39-dim)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(39, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 32)
        )
    def forward(self, x):
        return self.net(positional_encoding(x))


class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.linear  = nn.Linear(in_features, out_features, bias=bias)
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1.0 / in_features, 1.0 / in_features)
            else:
                bound = np.sqrt(6.0 / in_features) / omega_0
                self.linear.weight.uniform_(-bound, bound)
    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class ShadowNet_SIREN(nn.Module):
    """Phase C: SIREN, raw [x,y,z] input."""
    def __init__(self, hidden=512, omega_0=30):
        super().__init__()
        layers = [SineLayer(3, hidden, is_first=True, omega_0=omega_0)]
        for _ in range(3):
            layers.append(SineLayer(hidden, hidden, is_first=False, omega_0=omega_0))
        final = nn.Linear(hidden, 32)
        with torch.no_grad():
            bound = np.sqrt(6.0 / hidden) / omega_0
            final.weight.uniform_(-bound, bound)
        layers.append(final)
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Physics Helpers
# ---------------------------------------------------------------------------
def get_R_matrices(v_sig, v_jam, JAM_POW=10000.0, NOISE_POW=1.0, SIG_POW=100.0):
    R_j = JAM_POW  * np.outer(v_jam, np.conj(v_jam))
    R_n = NOISE_POW * np.eye(len(v_jam))
    R_s = SIG_POW  * np.outer(v_sig, np.conj(v_sig))
    return R_s, R_j, R_n


def compute_mvdr_sinr(v_sig, v_jam_ai, v_jam_exact, dl_factor=0.0):
    """Build MVDR from AI steering vector, evaluate SINR against exact physics."""
    R_s, R_j, R_n = get_R_matrices(v_sig, v_jam_ai)
    R_xx = R_j + R_n
    trace_val  = np.real(np.trace(R_xx))
    diag_load  = (dl_factor * trace_val / R_xx.shape[0]) if dl_factor > 0 else 1e-12
    R_reg = R_xx + diag_load * np.eye(R_xx.shape[0])
    try:
        R_inv = np.linalg.inv(R_reg)
    except np.linalg.LinAlgError:
        R_inv = np.linalg.pinv(R_reg)
    num = R_inv @ v_sig
    den = np.conj(v_sig) @ num
    w   = num / max(abs(den), 1e-12)

    # Evaluate against exact environment
    _, R_j_ex, R_n_ex = get_R_matrices(v_sig, v_jam_exact)
    R_s_ex, _, _      = get_R_matrices(v_sig, v_jam_exact)
    S   = np.real(np.conj(w) @ R_s_ex @ w)
    NJ  = np.real(np.conj(w) @ (R_j_ex + R_n_ex) @ w)
    return w, 10 * np.log10(S / max(NJ, 1e-12))


def infer(model, jam_bodies_np):
    """Run model inference and return complex gains [N, 16]."""
    with torch.no_grad():
        preds = model(torch.tensor(jam_bodies_np, dtype=torch.float32)).numpy()
    return preds[:, 0::2] + 1j * preds[:, 1::2]


# ---------------------------------------------------------------------------
# Main Evaluation
# ---------------------------------------------------------------------------
MODELS = {
    "Stage 13 Baseline": ("shadow_net.pt",      ShadowNet_Stage13, "#999999"),
    "Phase A (PE+MSE)":  ("shadow_net_pe.pt",   ShadowNet_PE,      "#2196F3"),
    "Phase B (PE+SINR)": ("shadow_net_sinr.pt", ShadowNet_PE,      "#FF9800"),
    "Phase C (SIREN)":   ("shadow_net_siren.pt",ShadowNet_SIREN,   "#4CAF50"),
}
TARGET_SINR = 15.0   # operational minimum


def load_models():
    loaded = {}
    for label, (fname, cls, color) in MODELS.items():
        if not os.path.exists(fname):
            print(f"  WARNING: {fname} not found — skipping '{label}'")
            continue
        m = cls()
        m.load_state_dict(torch.load(fname, map_location="cpu"))
        m.eval()
        loaded[label] = (m, color)
        print(f"  Loaded {fname} as '{label}'")
    return loaded


def main():
    MESH_PATH = Path(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone_prepared.stl")
    LAM = 0.15
    K   = 2 * np.pi / LAM

    print("Loading mesh...")
    mesh = load_uav_mesh(MESH_PATH)
    pos_body, normals_body = get_conformal_array(mesh)

    print("Loading models...")
    loaded = load_models()
    if not loaded:
        print("No models found. Exiting.")
        return

    # ------------------------------------------------------------------
    # Test 1: Azimuth sweep (0–360°, elevation=0, UAV pitched 15°)
    # ------------------------------------------------------------------
    print("\n--- Test 1: Azimuth Sweep (360 points) ---")
    q_inv    = euler_to_quaternion(np.deg2rad(15.0), 0, 0).conjugate()
    headings = np.linspace(0, 360, 360, endpoint=False)

    jam_bodies_az = np.zeros((360, 3))
    for i, h in enumerate(headings):
        jw = np.array([np.cos(np.deg2rad(h)), np.sin(np.deg2rad(h)), 0.0])
        jam_bodies_az[i] = rotate_points(jw.reshape(1, 3), q_inv)[0]

    sig_body = rotate_points(np.array([[1.0, 0.0, 0.0]]), q_inv)[0]
    v_sig    = get_steering_vector(pos_body, K * sig_body)

    print("  Computing exact physics...")
    g_exact_az = compute_shadow_mask_batched(mesh, pos_body, normals_body, jam_bodies_az)

    # Exact physics SINR (oracle)
    sinr_exact_az = np.zeros(360)
    for i in range(360):
        v_jam_exact = g_exact_az[i] * get_steering_vector(pos_body, jam_bodies_az[i] * K)
        _, sinr_exact_az[i] = compute_mvdr_sinr(v_sig, v_jam_exact, v_jam_exact)

    # Per-model SINR curves
    sinr_az = {}
    for label, (model, _) in loaded.items():
        g_ai = infer(model, jam_bodies_az)
        sinrs = np.zeros(360)
        for i in range(360):
            v_jam_ai    = g_ai[i]         * get_steering_vector(pos_body, jam_bodies_az[i] * K)
            v_jam_exact = g_exact_az[i]   * get_steering_vector(pos_body, jam_bodies_az[i] * K)
            _, sinrs[i] = compute_mvdr_sinr(v_sig, v_jam_ai, v_jam_exact)
        sinr_az[label] = sinrs
        worst = sinrs.min()
        print(f"  {label:28s}  worst-case SINR = {worst:.2f} dB  {'PASS' if worst >= TARGET_SINR else 'FAIL'}")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(headings, sinr_exact_az, 'k--', linewidth=1.5, label="Exact Physics (Oracle)")
    ax.axhline(TARGET_SINR, color='red', linestyle=':', linewidth=1.2, label=f"Target ({TARGET_SINR} dB)")
    for label, (_, color) in loaded.items():
        ax.plot(headings, sinr_az[label], color=color, linewidth=1.2, label=label)
    ax.set_xlabel("Jammer Azimuth (deg)")
    ax.set_ylabel("SINR (dB)")
    ax.set_title("Test 1: SINR vs Jammer Azimuth – Four-Model Comparison")
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig("test1_sinr_threeway.png", dpi=150)
    print("  Saved test1_sinr_threeway.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Test 2: Elevation sweep (−60° to +60°, azimuth=0°)
    # ------------------------------------------------------------------
    print("\n--- Test 2: Elevation Sweep (−60° to +60°) ---")
    elevations = np.linspace(-60, 60, 121)

    jam_bodies_el = np.zeros((121, 3))
    for i, el in enumerate(elevations):
        jw = np.array([np.cos(np.deg2rad(el)), 0.0, np.sin(np.deg2rad(el))])
        jam_bodies_el[i] = rotate_points(jw.reshape(1, 3), q_inv)[0]

    print("  Computing exact physics...")
    g_exact_el = compute_shadow_mask_batched(mesh, pos_body, normals_body, jam_bodies_el)

    sinr_exact_el = np.zeros(121)
    for i in range(121):
        v_jam_exact = g_exact_el[i] * get_steering_vector(pos_body, jam_bodies_el[i] * K)
        _, sinr_exact_el[i] = compute_mvdr_sinr(v_sig, v_jam_exact, v_jam_exact)

    sinr_el = {}
    for label, (model, _) in loaded.items():
        g_ai = infer(model, jam_bodies_el)
        sinrs = np.zeros(121)
        for i in range(121):
            v_jam_ai    = g_ai[i]       * get_steering_vector(pos_body, jam_bodies_el[i] * K)
            v_jam_exact = g_exact_el[i] * get_steering_vector(pos_body, jam_bodies_el[i] * K)
            _, sinrs[i] = compute_mvdr_sinr(v_sig, v_jam_ai, v_jam_exact)
        sinr_el[label] = sinrs
        worst = sinrs.min()
        print(f"  {label:28s}  worst-case SINR = {worst:.2f} dB  {'PASS' if worst >= TARGET_SINR else 'FAIL'}")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(elevations, sinr_exact_el, 'k--', linewidth=1.5, label="Exact Physics (Oracle)")
    ax.axhline(TARGET_SINR, color='red', linestyle=':', linewidth=1.2, label=f"Target ({TARGET_SINR} dB)")
    for label, (_, color) in loaded.items():
        ax.plot(elevations, sinr_el[label], color=color, linewidth=1.2, label=label)
    ax.set_xlabel("Jammer Elevation (deg)")
    ax.set_ylabel("SINR (dB)")
    ax.set_title("Test 2: SINR vs Jammer Elevation – Four-Model Comparison")
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig("test2_sinr_threeway.png", dpi=150)
    print("  Saved test2_sinr_threeway.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Diagnostic 2: Null Placement Angular Error (azimuth sweep)
    # ------------------------------------------------------------------
    print("\n--- Diagnostic 2: Null Placement Angular Error ---")
    null_errors = {}
    for label, (model, _) in loaded.items():
        g_ai = infer(model, jam_bodies_az)
        errs = np.zeros(360)
        for i in range(360):
            v_jam_ai = g_ai[i] * get_steering_vector(pos_body, jam_bodies_az[i] * K)
            w_ai, _  = compute_mvdr_sinr(v_sig, v_jam_ai, v_jam_ai)  # use AI null for pattern search

            j_body = jam_bodies_az[i]
            j_az = np.arctan2(j_body[1], j_body[0])
            j_el = np.arcsin(np.clip(j_body[2], -1, 1))

            az_grid = np.linspace(j_az - np.deg2rad(10), j_az + np.deg2rad(10), 30)
            el_grid = np.linspace(j_el - np.deg2rad(10), j_el + np.deg2rad(10), 30)

            min_resp   = float('inf')
            best_err   = 0.0
            for az in az_grid:
                for el in el_grid:
                    td = np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])
                    v_test = g_exact_az[i] * get_steering_vector(pos_body, td * K)
                    resp   = np.abs(np.conj(w_ai) @ v_test)
                    if resp < min_resp:
                        min_resp = resp
                        best_err = np.rad2deg(np.arccos(np.clip(np.dot(j_body, td), -1.0, 1.0)))
            errs[i] = best_err
        null_errors[label] = errs
        median = np.median(errs)
        print(f"  {label:28s}  median null error = {median:.2f}°  {'<5 deg' if median < 5 else 'XX ≥5°'}")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axhline(5.0, color='red', linestyle=':', linewidth=1.2, label="5° criterion")
    for label, (_, color) in loaded.items():
        ax.plot(headings, null_errors[label], color=color, linewidth=1.0, label=label)
    ax.set_xlabel("Jammer Azimuth (deg)")
    ax.set_ylabel("Null Displacement from True Jammer (deg)")
    ax.set_title("Diagnostic 2: Angular Null-Placement Error – Four-Model Comparison")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig("diag2_null_placement_threeway.png", dpi=150)
    print("  Saved diag2_null_placement_threeway.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Diagnostic 4: Diagonal Loading Sweep (worst-case SINR, azimuth sweep)
    # ------------------------------------------------------------------
    print("\n--- Diagnostic 4: Diagonal Loading Sweep ---")
    dl_factors   = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5]
    worst_sinr_dl = {label: [] for label in loaded}

    for dl in dl_factors:
        for label, (model, _) in loaded.items():
            g_ai = infer(model, jam_bodies_az)
            min_sinr = float('inf')
            for i in range(360):
                v_jam_ai    = g_ai[i]       * get_steering_vector(pos_body, jam_bodies_az[i] * K)
                v_jam_exact = g_exact_az[i] * get_steering_vector(pos_body, jam_bodies_az[i] * K)

                R_s, R_j, R_n = get_R_matrices(v_sig, v_jam_ai)
                R_xx   = R_j + R_n
                tval   = np.real(np.trace(R_xx))
                dload  = (dl * tval / R_xx.shape[0]) if dl > 0 else 1e-12
                R_reg  = R_xx + dload * np.eye(R_xx.shape[0])
                try:    R_inv = np.linalg.inv(R_reg)
                except: R_inv = np.linalg.pinv(R_reg)
                num = R_inv @ v_sig
                den = np.conj(v_sig) @ num
                w   = num / max(abs(den), 1e-12)

                _, R_j_ex, R_n_ex = get_R_matrices(v_sig, v_jam_exact)
                R_s_ex, _, _      = get_R_matrices(v_sig, v_jam_exact)
                S  = np.real(np.conj(w) @ R_s_ex @ w)
                NJ = np.real(np.conj(w) @ (R_j_ex + R_n_ex) @ w)
                sinr = 10 * np.log10(S / max(NJ, 1e-12))
                if sinr < min_sinr:
                    min_sinr = sinr
            worst_sinr_dl[label].append(min_sinr)
        print(f"  DL={dl:.3f}  " +
              "  ".join(f"{lbl[:8]}: {worst_sinr_dl[lbl][-1]:.1f}dB" for lbl in loaded))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhline(TARGET_SINR, color='red', linestyle=':', linewidth=1.2, label=f"Target ({TARGET_SINR} dB)")
    for label, (_, color) in loaded.items():
        ax.plot(dl_factors, worst_sinr_dl[label], marker='o', color=color, linewidth=1.2, label=label)
    ax.set_xscale('symlog', linthresh=0.001)
    ax.set_xlabel("Diagonal Loading Factor")
    ax.set_ylabel("Worst-Case SINR over Azimuth Sweep (dB)")
    ax.set_title("Diagnostic 4: DL Robustness – Four-Model Comparison")
    ax.legend()
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig("diag4_dl_sweep_threeway.png", dpi=150)
    print("  Saved diag4_dl_sweep_threeway.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n=== Summary ===")
    print(f"{'Model':30s} {'Test1 Worst (dB)':>18s} {'Test2 Worst (dB)':>18s} {'Null Med (°)':>14s}")
    print("-" * 84)
    for label in loaded:
        t1 = sinr_az[label].min()
        t2 = sinr_el[label].min()
        ne = np.median(null_errors[label])
        flag = "OK" if t1 >= TARGET_SINR and t2 >= TARGET_SINR else "XX"
        print(f"{flag} {label:28s} {t1:>18.2f} {t2:>18.2f} {ne:>14.2f}")


if __name__ == '__main__':
    main()
