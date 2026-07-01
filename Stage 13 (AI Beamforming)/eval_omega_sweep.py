"""
Evaluate multiple Phase C SIREN models with different omega_0 values.
Loads Phase B (as reference) and multiple SIREN models.
"""
import sys, os
import numpy as np
import torch, torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append(os.path.abspath(r'D:\UAV Internship project\Phase 2 Track 1\em_realism'))
sys.path.append(os.path.abspath(r'D:\UAV Internship project\Phase 2 Track 2\array_structures'))
sys.path.append(os.path.abspath(r'D:\UAV Internship project\Stage 12 (Operational Analysis)'))
from mesh_loader import load_uav_mesh
from conformal_array import get_conformal_array
from shadow_engine_batched import compute_shadow_mask_batched
from attitude import euler_to_quaternion, rotate_points
from lcmv_stage8 import get_steering_vector

# --- Models ---
def positional_encoding(x, L=6):
    encoded = [x]
    for i in range(L):
        encoded.append(torch.sin((2.0 ** i) * torch.pi * x))
        encoded.append(torch.cos((2.0 ** i) * torch.pi * x))
    return torch.cat(encoded, dim=-1)

class ShadowNet_PE(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(39, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 32)
        )
    def forward(self, x): return self.net(positional_encoding(x))

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.linear  = nn.Linear(in_features, out_features, bias=bias)
    def forward(self, x): return torch.sin(self.omega_0 * self.linear(x))

class ShadowNet_SIREN(nn.Module):
    def __init__(self, omega_0=30):
        super().__init__()
        layers = [SineLayer(3, 512, is_first=True, omega_0=omega_0)]
        for _ in range(3): layers.append(SineLayer(512, 512, is_first=False, omega_0=omega_0))
        layers.append(nn.Linear(512, 32))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

# --- Physics ---
def get_R_matrices(v_sig, v_jam):
    R_j = 10000. * np.outer(v_jam, np.conj(v_jam))
    R_n = 1.0 * np.eye(len(v_jam))
    R_s = 100. * np.outer(v_sig, np.conj(v_sig))
    return R_s, R_j, R_n

def compute_mvdr_sinr(v_sig, v_jam_ai, v_jam_exact):
    R_s, R_j, R_n = get_R_matrices(v_sig, v_jam_ai)
    R_xx = R_j + R_n + 1e-12 * np.eye(len(v_jam_ai))
    try: R_inv = np.linalg.inv(R_xx)
    except: R_inv = np.linalg.pinv(R_xx)
    num = R_inv @ v_sig; w = num / max(abs(np.conj(v_sig) @ num), 1e-12)
    _, R_j_ex, R_n_ex = get_R_matrices(v_sig, v_jam_exact)
    R_s_ex, _, _      = get_R_matrices(v_sig, v_jam_exact)
    S = np.real(np.conj(w) @ R_s_ex @ w)
    NJ = np.real(np.conj(w) @ (R_j_ex + R_n_ex) @ w)
    return w, 10 * np.log10(S / max(NJ, 1e-12))

# --- Main ---
def main():
    MODELS = {
        "Phase B Reference": ("shadow_net_sinr.pt", ShadowNet_PE()),
        "SIREN w=30": ("shadow_net_siren.pt", ShadowNet_SIREN(omega_0=30)),
        "SIREN w=15": ("shadow_net_siren_w15.pt", ShadowNet_SIREN(omega_0=15)),
        "SIREN w=10": ("shadow_net_siren_w10.pt", ShadowNet_SIREN(omega_0=10)),
        "SIREN w=5":  ("shadow_net_siren_w5.pt", ShadowNet_SIREN(omega_0=5)),
    }
    
    loaded = {}
    for lbl, (fn, m) in MODELS.items():
        if os.path.exists(fn):
            m.load_state_dict(torch.load(fn, map_location="cpu"))
            m.eval()
            loaded[lbl] = m
            print(f"Loaded {lbl}")
            
    mesh = load_uav_mesh(Path(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone_prepared.stl"))
    pos_body, normals_body = get_conformal_array(mesh)
    K = 2 * np.pi / 0.15
    
    print("\nRunning Azimuth Sweep...")
    q_inv = euler_to_quaternion(np.deg2rad(15.0), 0, 0).conjugate()
    headings = np.linspace(0, 360, 360, endpoint=False)
    jam_az = np.array([rotate_points(np.array([[np.cos(np.deg2rad(h)), np.sin(np.deg2rad(h)), 0]]), q_inv)[0] for h in headings])
    v_sig = get_steering_vector(pos_body, K * rotate_points(np.array([[1.0, 0.0, 0.0]]), q_inv)[0])
    
    g_ex = compute_shadow_mask_batched(mesh, pos_body, normals_body, jam_az)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for lbl, m in loaded.items():
        with torch.no_grad():
            preds = m(torch.tensor(jam_az, dtype=torch.float32)).numpy()
        g_ai = preds[:, 0::2] + 1j * preds[:, 1::2]
        
        sinrs = np.zeros(360)
        for i in range(360):
            v_jam_ai = g_ai[i] * get_steering_vector(pos_body, jam_az[i] * K)
            v_jam_ex = g_ex[i] * get_steering_vector(pos_body, jam_az[i] * K)
            _, sinrs[i] = compute_mvdr_sinr(v_sig, v_jam_ai, v_jam_ex)
            
        print(f"{lbl:20s} Worst SINR: {sinrs.min():6.2f} dB")
        ax.plot(headings, sinrs, label=f"{lbl} (Worst: {sinrs.min():.1f} dB)", linewidth=1.5 if "Phase B" in lbl else 1.0)
        
    ax.axhline(15.0, color='r', linestyle=':', label="Target (15 dB)")
    ax.legend(); ax.grid(True); ax.set_xlabel("Azimuth"); ax.set_ylabel("SINR (dB)")
    fig.savefig("omega_sweep_sinr.png", dpi=150)
    print("Saved omega_sweep_sinr.png")

if __name__ == '__main__': main()
