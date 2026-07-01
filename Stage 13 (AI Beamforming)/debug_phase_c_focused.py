import os
import sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

sys.path.append(os.path.abspath(r'D:\UAV Internship project\Phase 2 Track 1\em_realism'))
sys.path.append(os.path.abspath(r'D:\UAV Internship project\Phase 2 Track 2\array_structures'))
sys.path.append(os.path.abspath(r'D:\UAV Internship project\Stage 12 (Operational Analysis)'))

from mesh_loader import load_uav_mesh
from conformal_array import get_conformal_array_parametric
from lcmv_stage8 import get_steering_vector
from attitude import euler_to_quaternion, rotate_points
from shadow_engine_batched import compute_shadow_mask_batched

class PositionalEncoding(nn.Module):
    def __init__(self, L=10):
        super().__init__()
        self.L = L
    def forward(self, x):
        features = [x]
        for i in range(self.L):
            freq = 2.0**i * torch.pi
            features.append(torch.sin(freq * x))
            features.append(torch.cos(freq * x))
        return torch.cat(features, dim=-1)

class DirectWeightNet32(nn.Module):
    def __init__(self):
        super().__init__()
        self.pe = PositionalEncoding(L=10)
        self.net = nn.Sequential(
            nn.Linear(63, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64)
        )
    def forward(self, x):
        out = self.net(self.pe(x))
        return (out[:, 0::2] + 1j * out[:, 1::2]).to(torch.complex64)

def main():
    MESH_PATH = Path(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone_prepared.stl")
    mesh = load_uav_mesh(MESH_PATH)
    pos_body, normals_body = get_conformal_array_parametric(mesh, N=32)
    device = torch.device("cpu")
    pos_body_t = torch.tensor(pos_body, dtype=torch.float32).to(device)
    
    q_inv = euler_to_quaternion(np.deg2rad(15.0), 0, 0).conjugate()
    sig_world = np.array([1.0, 0.0, 0.0])
    sig_body = rotate_points(sig_world.reshape(1, 3), q_inv)[0]
    K = 2.0 * np.pi / 0.15
    v_sig = get_steering_vector(pos_body, K * sig_body)
    v_sig_t = torch.tensor(v_sig, dtype=torch.complex64).to(device)
    
    # Train a model for exactly 1 epoch on 1 point to see what the loss is and what the eval is
    jam_world = np.array([1.0, 0.0, 0.0]) # Azimuth 0
    jam_body = rotate_points(jam_world.reshape(1, 3), q_inv)[0]
    
    g_exact = compute_shadow_mask_batched(mesh, pos_body, normals_body, jam_body.reshape(1,3))[0]
    
    model = DirectWeightNet32().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    cart_inputs = torch.tensor(jam_body.reshape(1,3), dtype=torch.float32)
    g_exact_complex = torch.tensor(g_exact.reshape(1,32), dtype=torch.complex64)
    
    for _ in range(50):
        w_raw = model(cart_inputs)
        v_sig_batch = v_sig_t.unsqueeze(0)
        w_H_vsig = torch.sum(torch.conj(w_raw) * v_sig_batch, dim=1, keepdim=True)
        w_norm = w_raw / (w_H_vsig + 1e-8)
        
        v_jam_exact = g_exact_complex * torch.exp(1j * K * (cart_inputs @ pos_body_t.T))
        leakage = torch.abs(torch.sum(torch.conj(w_norm) * v_jam_exact, dim=1))**2
        noise_gain = torch.sum(torch.abs(w_norm)**2, dim=1)
        
        loss = torch.log10(leakage + 1e-12).mean() + 0.1 * torch.log10(noise_gain + 1e-12).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f"Final training leakage: {leakage.item()}")
    print(f"Final training loss: {loss.item()}")
    
    # Eval
    model.eval()
    w_raw_eval = model(cart_inputs)
    w_H_vsig_eval = torch.sum(torch.conj(w_raw_eval) * v_sig_batch, dim=1, keepdim=True)
    w_norm_eval = w_raw_eval / (w_H_vsig_eval + 1e-8)
    
    w = w_norm_eval.detach().numpy()[0]
    v_jam_ex = g_exact * get_steering_vector(pos_body, jam_body * K)
    
    R_s = 100.0 * np.outer(v_sig, np.conj(v_sig))
    R_j = 10000.0 * np.outer(v_jam_ex, np.conj(v_jam_ex))
    R_n = 1.0 * np.eye(32)
    
    S = np.real(np.conj(w) @ R_s @ w)
    NJ = np.real(np.conj(w) @ (R_j + R_n) @ w)
    sinr = 10 * np.log10(S / max(NJ, 1e-12))
    
    print(f"Eval S: {S}")
    print(f"Eval NJ: {NJ}")
    print(f"Eval SINR: {sinr}")
    
if __name__ == '__main__':
    main()
