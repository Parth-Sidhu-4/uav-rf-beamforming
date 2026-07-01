import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import os
import sys

from stage18w_train_100k_relu_ablation import ReluCovariancePredictor

def get_steering_vector_pt(pos_lambda, theta, phi):
    dx = torch.sin(theta) * torch.cos(phi)
    dy = torch.sin(theta) * torch.sin(phi)
    dz = torch.cos(theta)
    d = torch.stack([dx, dy, dz], dim=-1)
    return torch.exp(1j * 2.0 * torch.pi * (d @ pos_lambda.T))

def d3_mvdr_beamformer(U, v_sig, P_J, sigma2, alpha):
    K = U.shape[2]
    R_pred = P_J * (U @ torch.conj(U.transpose(1, 2))) + sigma2 * torch.eye(32, dtype=torch.complex128, device=U.device).unsqueeze(0)
    R_inv = torch.linalg.inv(R_pred + alpha * torch.eye(32, dtype=torch.complex128, device=U.device).unsqueeze(0))
    w_unnorm = torch.einsum('bij,j->bi', R_inv, v_sig)
    norm_factor = torch.einsum('bi,i->b', w_unnorm, torch.conj(v_sig))
    w = w_unnorm / norm_factor.unsqueeze(1)
    return w

def evaluate_model(model, inputs_t, v_true_t, v_sig_masked, device):
    model.eval()
    sinrs = []
    with torch.no_grad():
        U = model(inputs_t).to(torch.complex128)
        # Process in chunks
        chunk_size = 1000
        for i in range(0, len(U), chunk_size):
            U_chunk = U[i:i+chunk_size]
            v_true_chunk = v_true_t[i:i+chunk_size]
            
            w = d3_mvdr_beamformer(U_chunk, v_sig_masked[0], 10000.0, 390.0, 1e-4)
            Ps = 100.0 * torch.abs(torch.sum(torch.conj(w) * v_sig_masked[0], dim=1))**2
            Pj = 10000.0 * torch.abs(torch.sum(torch.conj(w) * v_true_chunk, dim=1))**2
            Pn = torch.real(torch.sum(torch.conj(w) * w, dim=1))
            
            sinr = 10 * torch.log10(Ps / (Pj + Pn))
            sinrs.append(sinr.cpu().numpy())
            
    sinrs = np.concatenate(sinrs)
    return np.min(sinrs), np.sum(sinrs < 15.0)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Loading dataset...')
    data = np.load('dataset_shadow_100k_polar_32el.npz')
    inputs_all = data['inputs']
    labels_polar = data['labels']
    
    mag = labels_polar[:, 0::2]
    phase = labels_polar[:, 1::2]
    
    theta_j = np.arccos(inputs_all[:, 2] / np.linalg.norm(inputs_all, axis=1))
    phi_j = np.arctan2(inputs_all[:, 1], inputs_all[:, 0])
    elevs = 90.0 - np.rad2deg(theta_j)
    heads = np.rad2deg(phi_j)
    heads[heads < 0] += 360.0
    
    idx_rear = (heads >= 150) & (heads <= 210) & (elevs >= -25) & (elevs <= 25)
    inputs_rear = inputs_all[idx_rear]
    mag_rear = mag[idx_rear]
    phase_rear = phase[idx_rear]
    g_exact_rear = mag_rear * np.exp(1j * phase_rear)
    
    inputs_t = torch.tensor(inputs_rear, dtype=torch.float32, device=device)
    v_true_rear = g_exact_rear * np.exp(1j * 2.0 * np.pi * 0.0) # wait, I need proper v_ideal
    
    # Physics setup
    sys.path.append(os.path.abspath(r"D:\UAV Internship project\Stage 12 (Operational Analysis)"))
    sys.path.append(os.path.abspath(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)"))
    sys.path.append(os.path.abspath(r"D:\UAV Internship project\Phase 2 Track 1\em_realism"))
    
    from conformal_array import get_conformal_array_parametric
    from mesh_loader import load_uav_mesh
    from pathlib import Path
    from attitude import rotate_points, euler_to_quaternion
    
    mesh = load_uav_mesh(Path(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone_prepared.stl"))
    pos_body, _ = get_conformal_array_parametric(mesh, N=32)
    pos_lambda_np = pos_body / 0.15
    q_inv = euler_to_quaternion(np.deg2rad(15.0), 0, 0).conjugate()
    pos_lambda = torch.tensor(pos_lambda_np, dtype=torch.float64, device=device)
    
    sig_world = np.array([1.0, 0.0, 0.0])
    sig_body = rotate_points(sig_world.reshape(1, 3), q_inv)[0]
    
    from stage15d_train_cartesian_32 import CartesianShadowNet32
    shadow_net = CartesianShadowNet32().to(device)
    shadow_net.load_state_dict(torch.load('shadow_net_cartesian_32.pt', map_location=device))
    shadow_net.eval()
    
    sig_t = torch.tensor(sig_body, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        out_sig = shadow_net(sig_t)
        g_sig_raw = (out_sig[:, 0::2] + 1j * out_sig[:, 1::2]).to(torch.complex128)
        g_sig = g_sig_raw / torch.clamp(torch.abs(g_sig_raw), min=1.0)
    
    theta_s = torch.acos(sig_t[:, 2] / torch.norm(sig_t, dim=1)).to(torch.float64)
    phi_s = torch.atan2(sig_t[:, 1], sig_t[:, 0]).to(torch.float64)
    v_sig_masked = g_sig * get_steering_vector_pt(pos_lambda, theta_s, phi_s)
    
    theta_j_t = torch.acos(inputs_t[:, 2].to(torch.float64) / torch.norm(inputs_t.to(torch.float64), dim=1))
    phi_j_t = torch.atan2(inputs_t[:, 1].to(torch.float64), inputs_t[:, 0].to(torch.float64))
    v_ideal = get_steering_vector_pt(pos_lambda, theta_j_t, phi_j_t)
    g_exact_t = torch.tensor(g_exact_rear, dtype=torch.complex128, device=device)
    v_true_t = g_exact_t * v_ideal
    
    dataset = TensorDataset(inputs_t, v_true_t)
    loader = DataLoader(dataset, batch_size=1024, shuffle=True)
    
    widths = [384, 768, 1024]
    sigmas = [8.0, 12.0, 15.0]
    results = {}
    
    P_S = 100.0
    P_J = 10000.0
    sigma2 = 390.0
    alpha = 1e-4
    
    R_sig = P_S * (v_sig_masked.T @ torch.conj(v_sig_masked))
    R_n = sigma2 * torch.eye(32, dtype=torch.complex128, device=device)
    R_sn = R_sig + R_n
    
    for w in widths:
        for s in sigmas:
            print(f"\n--- Training Width: {w}, Sigma: {s} ---")
            model = ReluCovariancePredictor(K_rank=5, hidden_dim=w, sigma=s).to(device)
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, eta_min=1e-5)
            
            for epoch in range(30):
                model.train()
                total_loss = 0.0
                for X, v_t in loader:
                    optimizer.zero_grad()
                    U = model(X).to(torch.complex128)
                    
                    w_b = d3_mvdr_beamformer(U, v_sig_masked[0], P_J, sigma2, alpha)
                    
                    S = torch.real(torch.sum(torch.conj(w_b) * (R_sn @ w_b.T).T, dim=1))
                    Pj_true = P_J * torch.abs(torch.sum(torch.conj(w_b) * v_t, dim=1))**2
                    NJ = Pj_true + sigma2 * torch.real(torch.sum(torch.conj(w_b) * w_b, dim=1))
                    
                    sinr = 10 * torch.log10(S / torch.clamp(NJ, min=1e-12))
                    
                    loss = torch.mean(-sinr)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                scheduler.step()
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/30, Loss: {total_loss / len(loader):.4f}")
            
            min_sinr, num_below_15 = evaluate_model(model, inputs_t, v_true_t, v_sig_masked, device)
            print(f"=> Min SINR: {min_sinr:.2f} dB, Below 15dB: {num_below_15}")
            results[f"{w}_{s}"] = min_sinr
            torch.save(model.state_dict(), f'relu_beamformer_w{w}_s{s}.pt')
            
    print("\n--- ABLATION RESULTS (Min SINR) ---")
    for k, v in results.items():
        print(f"{k}: {v:.2f} dB")

if __name__ == '__main__':
    main()
