import sys
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

sys.path.append(os.path.abspath(r'D:\UAV Internship project\Phase 2 Track 1\em_realism'))
sys.path.append(os.path.abspath(r'D:\UAV Internship project\Phase 2 Track 2\array_structures'))
sys.path.append(os.path.abspath(r'D:\UAV Internship project\Stage 12 (Operational Analysis)'))
from mesh_loader import load_uav_mesh
from conformal_array import get_conformal_array
from lcmv_stage8 import get_steering_vector
from attitude import euler_to_quaternion, rotate_points

# --- Model Def ---
def positional_encoding(x, L=6):
    encoded = [x]
    for i in range(L):
        encoded.append(torch.sin((2.0 ** i) * torch.pi * x))
        encoded.append(torch.cos((2.0 ** i) * torch.pi * x))
    return torch.cat(encoded, dim=-1)

class ShadowNet(nn.Module):
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

# --- SINR Calculation ---
def get_R_matrices(v_sig, v_jam):
    R_j = 10000. * np.outer(v_jam, np.conj(v_jam))
    R_n = 1.0 * np.eye(len(v_jam))
    R_s = 100. * np.outer(v_sig, np.conj(v_sig))
    return R_s, R_j, R_n

def compute_mvdr_sinr(v_sig, v_jam_ai, v_jam_exact):
    R_s, R_j, R_n = get_R_matrices(v_sig, v_jam_ai)
    R_xx = R_j + R_n + 1e-12 * np.eye(len(v_jam_ai))
    try:
        R_inv = np.linalg.inv(R_xx)
    except np.linalg.LinAlgError:
        R_inv = np.linalg.pinv(R_xx)
    num = R_inv @ v_sig
    w = num / max(abs(np.conj(v_sig) @ num), 1e-12)
    
    R_s_ex, R_j_ex, R_n_ex = get_R_matrices(v_sig, v_jam_exact)
    S = np.real(np.conj(w) @ R_s_ex @ w)
    NJ = np.real(np.conj(w) @ (R_j_ex + R_n_ex) @ w)
    sinr = 10 * np.log10(S / max(NJ, 1e-12))
    return sinr

def main():
    print("Loading model and geometry...")
    model = ShadowNet()
    model.load_state_dict(torch.load("shadow_net_sinr.pt", map_location='cpu'))
    model.eval()

    mesh = load_uav_mesh(Path(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone_prepared.stl"))
    pos_body, _ = get_conformal_array(mesh)
    K = 2 * np.pi / 0.15

    q_inv = euler_to_quaternion(np.deg2rad(15.0), 0, 0).conjugate()
    v_sig = get_steering_vector(pos_body, K * rotate_points(np.array([[1.0, 0.0, 0.0]]), q_inv)[0])

    print("Loading 30k dataset...")
    data = np.load("dataset_shadow_30k.npz")
    inputs = data['inputs']
    labels = data['labels']
    N = inputs.shape[0]
    
    # Convert body vectors back to global Azimuth/Elevation for plotting
    dirs_global = np.zeros_like(inputs)
    q_fwd = euler_to_quaternion(np.deg2rad(15.0), 0, 0)
    for i in range(N):
        dirs_global[i] = rotate_points(inputs[i:i+1], q_fwd)[0]
    
    x, y, z = dirs_global[:, 0], dirs_global[:, 1], dirs_global[:, 2]
    az_rad = np.arctan2(y, x)
    az_deg = np.rad2deg(az_rad) % 360
    el_deg = np.rad2deg(np.arcsin(z))
    
    print("Running AI inference...")
    with torch.no_grad():
        preds = model(torch.tensor(inputs, dtype=torch.float32)).numpy()
    g_ai = preds[:, 0::2] + 1j * preds[:, 1::2]
    g_ex = labels[:, 0::2] + 1j * labels[:, 1::2]
    
    print("Computing SINR for all 30k points...")
    sinrs = np.zeros(N)
    for i in tqdm(range(N), desc="SINR Calculation"):
        k_vec = inputs[i] * K
        v_jam_ai = g_ai[i] * get_steering_vector(pos_body, k_vec)
        v_jam_ex = g_ex[i] * get_steering_vector(pos_body, k_vec)
        sinrs[i] = compute_mvdr_sinr(v_sig, v_jam_ai, v_jam_ex)
    
    print("Plotting scatter heatmap...")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # All points in background
    ax.scatter(az_deg, el_deg, c='lightgrey', alpha=0.2, s=5, label='Pass (>= 15 dB)')
    
    # Failed points on top
    fail_mask = sinrs < 15.0
    sc = ax.scatter(az_deg[fail_mask], el_deg[fail_mask], 
                    c=sinrs[fail_mask], cmap='Reds_r', vmin=0, vmax=15, 
                    s=20, alpha=0.8, edgecolor='none')
    
    fig.colorbar(sc, ax=ax, label='SINR (dB) [Failures Only]')
    ax.set_title("Phase D Prerequisite Diagnosis: Spatial Distribution of SINR < 15 dB Failures")
    ax.set_xlabel("Azimuth (deg)")
    ax.set_ylabel("Elevation (deg)")
    ax.set_xlim(0, 360)
    ax.set_ylim(-90, 90)
    
    plt.tight_layout()
    plt.savefig("phase_d_prereq_diagnosis.png", dpi=150)
    print("Saved Phase D diagnosis to phase_d_prereq_diagnosis.png")

if __name__ == '__main__':
    main()
