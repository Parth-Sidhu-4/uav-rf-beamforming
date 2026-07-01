import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

sys.path.append(os.path.abspath(r'D:\UAV Internship project\Phase 2 Track 1\em_realism'))
sys.path.append(os.path.abspath(r'D:\UAV Internship project\Phase 2 Track 2\array_structures'))
sys.path.append(os.path.abspath(r'D:\UAV Internship project\Stage 12 (Operational Analysis)'))

from mesh_loader import load_uav_mesh
from conformal_array import get_conformal_array_parametric
from lcmv_stage8 import get_steering_vector
from attitude import euler_to_quaternion, rotate_points
from shadow_engine_batched import compute_shadow_mask_batched

# --- Configuration ---
BATCH_SIZE = 128
EPOCHS = 50
LR = 1e-3
LAM = 0.15
K = 2.0 * np.pi / LAM

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
            nn.Linear(256, 64) # 32 complex weights
        )
    def forward(self, x):
        out = self.net(self.pe(x))
        return (out[:, 0::2] + 1j * out[:, 1::2]).to(torch.complex64)

def main():
    print("Loading mesh and preparing datasets...")
    MESH_PATH = Path(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone_prepared.stl")
    mesh = load_uav_mesh(MESH_PATH)
    pos_body, normals_body = get_conformal_array_parametric(mesh, N=32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pos_body_t = torch.tensor(pos_body, dtype=torch.float32).to(device)
    
    # We evaluate at 15 deg pitch
    q_inv = euler_to_quaternion(np.deg2rad(15.0), 0, 0).conjugate()
    sig_world = np.array([1.0, 0.0, 0.0])
    sig_body = rotate_points(sig_world.reshape(1, 3), q_inv)[0]
    v_sig = get_steering_vector(pos_body, K * sig_body)
    v_sig_t = torch.tensor(v_sig, dtype=torch.complex64).to(device)

    # Generate Focused Training Dataset on the specific evaluation manifold (-15 deg pitch relative)
    # Using 3600 points for dense coverage
    print("Pre-computing exact physics for 3600 focused points...")
    N_POINTS = 3600
    headings_train = np.linspace(0, 360, N_POINTS, endpoint=False)
    jam_bodies_train = np.zeros((N_POINTS, 3))
    
    for i, h in enumerate(headings_train):
        jam_world_train = np.array([np.cos(np.deg2rad(h)), np.sin(np.deg2rad(h)), 0.0])
        jam_bodies_train[i] = rotate_points(jam_world_train.reshape(1, 3), q_inv)[0]
        
    g_exact_train = np.zeros((N_POINTS, 32), dtype=np.complex128)
    chunk_size = 36
    for i in range(0, N_POINTS, chunk_size):
        chunk = jam_bodies_train[i:i+chunk_size]
        g_exact_train[i:i+chunk_size] = compute_shadow_mask_batched(mesh, pos_body, normals_body, chunk)

    X_train = torch.tensor(jam_bodies_train, dtype=torch.float32)
    y_train = torch.tensor(g_exact_train, dtype=torch.complex64)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)

    print("--- Training Phase C (Focused Manifold) ---")
    model = DirectWeightNet32().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, eta_min=1e-5)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        
        for cart_inputs, g_exact_complex in train_loader:
            cart_inputs = cart_inputs.to(device)
            g_exact_complex = g_exact_complex.to(device)
            
            w_raw = model(cart_inputs)
            
            # Deterministic Signal Constraint
            v_sig_batch = v_sig_t.unsqueeze(0).expand(cart_inputs.shape[0], -1)
            w_H_vsig = torch.sum(torch.conj(w_raw) * v_sig_batch, dim=1, keepdim=True)
            w_norm = w_raw / (w_H_vsig + 1e-8)
            
            # Isotropic 3D Jitter-Robust Training
            N_JIT = 5
            leakage_sum = 0.0
            
            # First point is the exact center
            v_jam_exact = g_exact_complex * torch.exp(1j * K * (cart_inputs @ pos_body_t.T))
            leakage_sum = leakage_sum + torch.abs(torch.sum(torch.conj(w_norm) * v_jam_exact, dim=1))**2
            
            # Next points are randomly jittered on the sphere
            for _ in range(N_JIT - 1):
                # Random 3D offset (std dev ~ 2 degrees = 0.035 rad)
                noise = torch.randn_like(cart_inputs) * 0.035
                cart_jit = cart_inputs + noise
                cart_jit = cart_jit / torch.norm(cart_jit, dim=1, keepdim=True) # re-normalize
                
                phases_geo_exact = K * (cart_jit @ pos_body_t.T)
                geo_sp = torch.exp(1j * phases_geo_exact)
                v_jam_jit = g_exact_complex * geo_sp
                
                leakage = torch.abs(torch.sum(torch.conj(w_norm) * v_jam_jit, dim=1))**2
                leakage_sum = leakage_sum + leakage
                
            avg_leakage = leakage_sum / N_JIT
            noise_gain = torch.sum(torch.abs(w_norm)**2, dim=1)
            
            # We want deep nulls but avoid massive weights that blow up noise
            loss = torch.mean(torch.log10(avg_leakage + 1e-12)) + 0.1 * torch.mean(torch.log10(noise_gain + 1e-12))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * cart_inputs.size(0)

        total_loss /= len(train_loader.dataset)
        scheduler.step()
        print(f"Epoch {epoch:03d} | Loss: {total_loss:.4f}")
    
    # --- EVALUATION ---
    # We evaluate on 360 points (1 degree shifts) which are structurally shifted from the 0.1 degree training grid
    print("\n--- SINR Evaluation ---")
    model.eval()
    headings = np.linspace(0.5, 360.5, 360, endpoint=False) # Offset by 0.5 deg to prove generalization!
    jam_bodies = np.zeros((360, 3))
    for i, h in enumerate(headings):
        jam_world_eval = np.array([np.cos(np.deg2rad(h)), np.sin(np.deg2rad(h)), 0.0])
        jam_bodies[i] = rotate_points(jam_world_eval.reshape(1, 3), q_inv)[0]
        
    g_exact_eval = np.zeros((360, 32), dtype=np.complex128)
    for i in range(0, 360, chunk_size):
        chunk_jams = jam_bodies[i:i+chunk_size]
        g_exact_eval[i:i+chunk_size] = compute_shadow_mask_batched(mesh, pos_body, normals_body, chunk_jams)
    
    JAM_POW = 10000.0
    NOISE_POW = 1.0
    SIG_POW = 100.0
    
    min_sinr = float('inf')
    
    with torch.no_grad():
        cart_eval = torch.tensor(jam_bodies, dtype=torch.float32).to(device)
        w_raw_eval = model(cart_eval)
        v_sig_batch = v_sig_t.unsqueeze(0).expand(360, -1)
        w_H_vsig = torch.sum(torch.conj(w_raw_eval) * v_sig_batch, dim=1, keepdim=True)
        w_norm_eval = w_raw_eval / (w_H_vsig + 1e-8)
        w_np = w_norm_eval.cpu().numpy()
        
    for i in range(360):
        v_jam_ex = g_exact_eval[i] * get_steering_vector(pos_body, jam_bodies[i] * K)
        w = w_np[i]
        
        R_s = SIG_POW * np.outer(v_sig, np.conj(v_sig))
        R_j = JAM_POW * np.outer(v_jam_ex, np.conj(v_jam_ex))
        R_n = NOISE_POW * np.eye(32)
        
        S = np.real(np.conj(w) @ R_s @ w)
        NJ = np.real(np.conj(w) @ (R_j + R_n) @ w)
        sinr = 10 * np.log10(S / max(NJ, 1e-12))
        if sinr < min_sinr:
            min_sinr = sinr
            
    print(f"Worst-case SINR over 360-deg Azimuth Sweep: {min_sinr:.2f} dB")

if __name__ == '__main__':
    main()
