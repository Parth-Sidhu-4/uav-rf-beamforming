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
    v_sig_masked = (g_sig * v_sig_ideal)[0]
    
    # Test headings at elevation 0
    test_headings = [90.0, 135.0, 160.0, 175.0, 180.0, 185.0]
    el = 0.0
    
    # Noise standard deviations (relative to the norm of U)
    # The true U column has norm sqrt(1) = 1. Wait, U is scaled to produce R_j.
    # The SIREN produces normalized U, which is scaled by 10000.0 later.
    # So U_true should have norm 1. 
    epsilons = np.logspace(-4, 0, 30) # from 0.0001 to 1.0
    N_trials = 50
    
    plt.figure(figsize=(10, 6))
    
    for az in test_headings:
        print(f"Testing Heading {az}°...")
        jam_world = np.array([
            np.cos(np.deg2rad(az)) * np.cos(np.deg2rad(el)),
            np.sin(np.deg2rad(az)) * np.cos(np.deg2rad(el)),
            np.sin(np.deg2rad(el))
        ])
        jam_body = rotate_points(jam_world.reshape(1, 3), q_inv)[0]
        
        inputs_t = torch.tensor(jam_body, dtype=torch.float32, device=device).unsqueeze(0)
        out = shadow_net(inputs_t)
        g_raw = (out[:, 0::2] + 1j * out[:, 1::2]).to(torch.complex128)
        g_exact = g_raw / torch.clamp(torch.abs(g_raw), min=1.0)
        
        theta_j = torch.acos(inputs_t[:, 2].to(torch.float64) / torch.norm(inputs_t.to(torch.float64), dim=1))
        phi_j = torch.atan2(inputs_t[:, 1].to(torch.float64), inputs_t[:, 0].to(torch.float64))
        v_ideal = get_steering_vector_pt(pos_lambda, theta_j, phi_j)
        v_true = (g_exact * v_ideal)[0]
        
        # Construct Oracle U (Rank 5, but only first column is non-zero)
        # U @ U.T should equal v_true @ v_true.T. 
        # So U[:, 0] = v_true. U has shape (32, 5)
        U_true = torch.zeros((32, 5), dtype=torch.complex128, device=device)
        U_true[:, 0] = v_true
        
        avg_sinrs = []
        
        for eps in epsilons:
            trial_sinrs = []
            for t in range(N_trials):
                # Complex Gaussian noise: variance eps^2 per element
                noise_re = torch.randn_like(U_true.real) * (eps / np.sqrt(2))
                noise_im = torch.randn_like(U_true.imag) * (eps / np.sqrt(2))
                noise = noise_re + 1j * noise_im
                
                U_pert = U_true + noise
                
                # Compute MVDR
                R_in = 10000.0 * (U_pert @ torch.conj(U_pert.T)) + 390.0 * torch.eye(32, dtype=torch.complex128, device=device)
                try:
                    R_in_inv = torch.linalg.inv(R_in)
                    w = R_in_inv @ v_sig_masked
                    w = w / (torch.conj(v_sig_masked) @ w)
                    
                    P_sig_out = 100.0 * torch.abs(torch.conj(w) @ v_sig_masked)**2
                    P_jam_out = 10000.0 * torch.abs(torch.conj(w) @ v_true)**2
                    P_n_out = torch.real(torch.conj(w) @ w)
                    
                    sinr = 10 * np.log10(float(P_sig_out / (P_jam_out + P_n_out)))
                    trial_sinrs.append(sinr)
                except Exception as e:
                    trial_sinrs.append(np.nan)
                    
            avg_sinrs.append(np.nanmean(trial_sinrs))
            
        plt.plot(epsilons, avg_sinrs, marker='.', label=f'{az}°')

    plt.xscale('log')
    plt.axhline(15, color='r', linestyle='--', label='15 dB Threshold')
    plt.xlabel('Perturbation Magnitude ($\epsilon$)')
    plt.ylabel('Average SINR (dB)')
    plt.title('SINR Sensitivity to Output Errors vs Heading')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig('diag_sensitivity_sweep.png', dpi=120)
    plt.close()
    print("Saved plot to diag_sensitivity_sweep.png")

if __name__ == '__main__':
    main()
