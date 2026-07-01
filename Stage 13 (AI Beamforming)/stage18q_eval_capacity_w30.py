import torch.nn as nn
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path

sys.path.append(os.path.abspath(r'D:\UAV Internship project\Phase 2 Track 2\array_structures'))
sys.path.append(os.path.abspath(r'D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)'))

from mesh_loader import load_uav_mesh
from conformal_array import get_conformal_array_parametric
from attitude import euler_to_quaternion, rotate_points

from stage15d_train_cartesian_32 import CartesianShadowNet32


def get_steering_vector_pt(pos_lambda, theta, phi):
    dx = torch.sin(theta) * torch.cos(phi)
    dy = torch.sin(theta) * torch.sin(phi)
    dz = torch.cos(theta)
    d = torch.stack([dx, dy, dz], dim=-1)
    phases = 2.0 * torch.pi * (d @ pos_lambda.T)
    return torch.exp(1j * phases)

class Sine(nn.Module):
    def __init__(self, w0=30.0):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)

class SirenLayer(nn.Module):
    def __init__(self, in_features, out_features, w0=30.0, is_first=False):
        super().__init__()
        self.in_features = in_features
        self.is_first = is_first
        self.w0 = w0
        self.linear = nn.Linear(in_features, out_features)
        self.init_weights()
        self.activation = Sine(w0)

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                b = 1.0 / self.in_features
            else:
                b = np.sqrt(6.0 / self.in_features) / self.w0
            self.linear.weight.uniform_(-b, b)
            self.linear.bias.uniform_(-b, b)

    def forward(self, x):
        return self.activation(self.linear(x))

class FourierEncoding(nn.Module):
    def __init__(self, in_features=3, out_features=128, sigma=1.0):
        super().__init__()
        self.B = nn.Parameter(torch.randn(in_features, out_features // 2) * sigma, requires_grad=False)
        
    def forward(self, x):
        x_proj = 2.0 * np.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class SIRENCovariancePredictor(nn.Module):
    def __init__(self, w0=30.0, K_rank=3, hidden_dim=256):
        super().__init__()
        self.K = K_rank
        self.fourier = FourierEncoding(in_features=3, out_features=128, sigma=1.0)
        self.net = nn.Sequential(
            SirenLayer(128, hidden_dim, w0=w0, is_first=True),
            SirenLayer(hidden_dim, hidden_dim, w0=w0, is_first=False),
            SirenLayer(hidden_dim, 128, w0=w0, is_first=False)
        )
        self.u_head = nn.Linear(128, 32 * K_rank * 2)
        
        with torch.no_grad():
            b = np.sqrt(6.0 / 128) / w0
            self.u_head.weight.uniform_(-b, b)
            self.u_head.bias.uniform_(-b, b)

    def forward(self, x):
        # x is unit direction vector (jammer XYZ)
        x = self.fourier(x)
        features = self.net(x)
        
        u_raw = self.u_head(features).view(-1, 32, self.K, 2)
        
        # We don't normalize the amplitude of U because we want the model to learn 
        # how much power to distribute to each mode.
        U = torch.complex(u_raw[..., 0], u_raw[..., 1])
        
        return U

# --- Main Script ---


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Loading mesh and physics...")
    MESH_PATH = Path(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone_prepared.stl")
    mesh = load_uav_mesh(MESH_PATH)
    pos_body, normals_body = get_conformal_array_parametric(mesh, N=32)
    pos_lambda = torch.tensor(pos_body / 0.15, dtype=torch.float64, device=device)
    
    # Fast shadow masks
    shadow_net = CartesianShadowNet32().to(device)
    shadow_net.load_state_dict(torch.load("shadow_net_cartesian_32.pt", map_location=device))
    shadow_net.eval()
    
    # 3D Pilot Model
    model = SIRENCovariancePredictor(w0=30.0, K_rank=5, hidden_dim=512).to(device)
    model.load_state_dict(torch.load("siren_beamformer_d3_cov_K5_3D_w30_h512.pt", map_location=device))
    model.eval()
    
    q_inv = euler_to_quaternion(np.deg2rad(15.0), 0, 0).conjugate()
    
    # Signal
    K_wave = 2.0 * np.pi / 0.15
    sig_world = np.array([1.0, 0.0, 0.0])
    sig_body = rotate_points(sig_world.reshape(1, 3), q_inv)[0]
    
    sig_t = torch.tensor(sig_body, dtype=torch.float32, device=device).unsqueeze(0)
    out_sig = shadow_net(sig_t)
    g_sig_raw = (out_sig[:, 0::2] + 1j * out_sig[:, 1::2]).to(torch.complex128)
    g_sig = g_sig_raw / torch.clamp(torch.abs(g_sig_raw), min=1.0)
    
    theta_s = torch.acos(sig_t[:, 2] / torch.norm(sig_t, dim=1)).to(torch.float64)
    phi_s = torch.atan2(sig_t[:, 1], sig_t[:, 0]).to(torch.float64)
    v_sig_ideal = get_steering_vector_pt(pos_lambda, theta_s, phi_s)
    v_sig_masked = g_sig * v_sig_ideal
    
    elevations = [20.0, 10.0, 0.0, -10.0, -20.0]
    fig, axes = plt.subplots(5, 1, figsize=(10, 15), sharex=True)
    
    headings = np.linspace(150, 210, 601) # 0.1 deg
    
    for idx, el in enumerate(elevations):
        jam_bodies_list = []
        for az in headings:
            jam_world = np.array([
                np.cos(np.deg2rad(az)) * np.cos(np.deg2rad(el)),
                np.sin(np.deg2rad(az)) * np.cos(np.deg2rad(el)),
                np.sin(np.deg2rad(el))
            ])
            jam_body = rotate_points(jam_world.reshape(1, 3), q_inv)[0]
            jam_bodies_list.append(jam_body)
            
        jam_bodies_np = np.array(jam_bodies_list, dtype=np.float32)
        
        with torch.no_grad():
            inputs_t = torch.tensor(jam_bodies_np, device=device)
            out = shadow_net(inputs_t)
            g_raw = (out[:, 0::2] + 1j * out[:, 1::2]).to(torch.complex128)
            g_exact = g_raw / torch.clamp(torch.abs(g_raw), min=1.0)
            
            theta_j = torch.acos(inputs_t[:, 2].to(torch.float64) / torch.norm(inputs_t.to(torch.float64), dim=1))
            phi_j = torch.atan2(inputs_t[:, 1].to(torch.float64), inputs_t[:, 0].to(torch.float64))
            v_ideal = get_steering_vector_pt(pos_lambda, theta_j, phi_j)
            v_true = g_exact * v_ideal
            
            U = model(inputs_t).to(torch.complex128)
            R_s = 100.0 * torch.einsum('bi,bj->bij', v_sig_masked, torch.conj(v_sig_masked)).expand(len(headings), -1, -1)
            R_j_true = 10000.0 * torch.einsum('bi,bj->bij', v_true, torch.conj(v_true))
            R_n = torch.eye(32, dtype=torch.complex128, device=device).unsqueeze(0).expand(len(headings), -1, -1)
            R_true = R_s + R_j_true + R_n
            
            sinrs = []
            for i in range(len(headings)):
                Ui = U[i]
                R_in = 10000.0 * (Ui @ torch.conj(Ui.T)) + 390.0 * torch.eye(32, dtype=torch.complex128, device=device)
                try:
                    R_in_inv = torch.linalg.inv(R_in)
                    w = R_in_inv @ v_sig_masked[0]
                    w = w / (torch.conj(v_sig_masked[0]) @ w)
                    
                    P_sig_out = 100.0 * torch.abs(torch.conj(w) @ v_sig_masked[0])**2
                    P_jam_out = 10000.0 * torch.abs(torch.conj(w) @ v_true[i])**2
                    P_n_out = torch.real(torch.conj(w) @ w)
                    
                    sinr = 10 * np.log10(float(P_sig_out / (P_jam_out + P_n_out)))
                    sinrs.append(sinr)
                except Exception as e:
                    sinrs.append(np.nan)
                    
        sinrs = np.array(sinrs)
        ax = axes[idx]
        ax.plot(headings, sinrs, 'b-', label='SINR')
        ax.axhline(15, color='r', linestyle='--', label='15 dB Threshold')
        ax.set_title(f"Elevation {el}°")
        ax.set_ylabel("SINR (dB)")
        ax.grid(True)
        
        min_sinr = np.nanmin(sinrs)
        argmin_h = headings[np.nanargmin(sinrs)]
        print(f"El {el:>5.1f} | Min SINR: {min_sinr:>6.2f} dB at {argmin_h:.1f}°")
        
    axes[-1].set_xlabel("Heading (deg)")
    plt.tight_layout()
    plt.savefig('diag_180deg_pilot_3D_w30_h512.png', dpi=120)
    plt.close()
    print("Saved plot to diag_180deg_pilot_3D.png")
    
if __name__ == '__main__':
    main()
