import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

sys.path.append(os.path.abspath(r'D:\UAV Internship project\Phase 2 Track 2\array_structures'))
sys.path.append(os.path.abspath(r'D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)'))

from conformal_array import get_conformal_array_parametric
from mesh_loader import load_uav_mesh
from attitude import euler_to_quaternion, rotate_points
from stage15d_train_cartesian_32 import CartesianShadowNet32

def get_steering_vector_pt(pos_lambda, theta, phi):
    dx = torch.sin(theta) * torch.cos(phi)
    dy = torch.sin(theta) * torch.sin(phi)
    dz = torch.cos(theta)
    k_dir = torch.stack([-dx, -dy, -dz], dim=-1) # [B, 3]
    phase = 2.0 * np.pi * torch.matmul(k_dir, pos_lambda.T) # [B, N]
    return torch.exp(1j * phase)

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
            
            R_s = 100.0 * torch.einsum('bi,bj->bij', v_sig_masked, torch.conj(v_sig_masked)).expand(len(headings), -1, -1)
            R_j_true = 10000.0 * torch.einsum('bi,bj->bij', v_true, torch.conj(v_true))
            R_n = torch.eye(32, dtype=torch.complex128, device=device).unsqueeze(0).expand(len(headings), -1, -1)
            R_true = R_s + R_j_true + R_n
            
            sinrs = []
            for i in range(len(headings)):
                # Oracle covariance (Exact R_j_true + diagonal loading)
                R_in = R_j_true[i] + 390.0 * torch.eye(32, dtype=torch.complex128, device=device)
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
        ax.plot(headings, sinrs, 'b-', label='Oracle SINR')
        ax.axhline(15, color='r', linestyle='--', label='15 dB Threshold')
        ax.set_title(f"Elevation {el}°")
        ax.set_ylabel("SINR (dB)")
        ax.grid(True)
        
        min_sinr = np.nanmin(sinrs)
        argmin_h = headings[np.nanargmin(sinrs)]
        print(f"El {el:>5.1f} | Min Oracle SINR: {min_sinr:>6.2f} dB at {argmin_h:.1f}°")
        
    axes[-1].set_xlabel("Heading (deg)")
    plt.tight_layout()
    plt.savefig('diag_oracle_mvdr_3D.png', dpi=120)
    plt.close()

if __name__ == "__main__":
    main()
