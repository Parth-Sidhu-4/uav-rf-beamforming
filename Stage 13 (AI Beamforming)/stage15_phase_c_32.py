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

# --- Configuration ---
BATCH_SIZE = 1024
EPOCHS = 30
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
    
    # We evaluate at 15 deg pitch, so we TRAIN with 15 deg pitch v_sig to avoid reward hacking!
    q_inv = euler_to_quaternion(np.deg2rad(15.0), 0, 0).conjugate()
    sig_world = np.array([1.0, 0.0, 0.0])
    sig_body = rotate_points(sig_world.reshape(1, 3), q_inv)[0]
    v_sig = get_steering_vector(pos_body, K * sig_body)
    v_sig_t = torch.tensor(v_sig, dtype=torch.complex64).to(device)

    data = np.load("dataset_shadow_100k_polar_32el.npz")
    inputs = data['inputs']
    labels_polar = data['labels']

    X_train = torch.tensor(inputs, dtype=torch.float32)
    y_polar = torch.tensor(labels_polar, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train, y_polar), batch_size=BATCH_SIZE, shuffle=True)

    print("--- Training Phase C (Direct Weight Prediction) on 32 Elements ---")
    model = DirectWeightNet32().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, eta_min=1e-5)

    if os.path.exists("phase_c_weights_32.pt"):
        print("Found existing weights, skipping training...")
        model.load_state_dict(torch.load("phase_c_weights_32.pt"))
    else:
        for epoch in range(1, EPOCHS + 1):
            model.train()
            total_loss = 0.0
            
            for cart_inputs, g_exact_polar in train_loader:
                cart_inputs = cart_inputs.to(device)
                g_exact_polar = g_exact_polar.to(device)
                
                w_raw = model(cart_inputs)
                
                # Deterministic Signal Constraint
                v_sig_batch = v_sig_t.unsqueeze(0).expand(cart_inputs.shape[0], -1)
                w_H_vsig = torch.sum(torch.conj(w_raw) * v_sig_batch, dim=1, keepdim=True)
                w_norm = w_raw / (w_H_vsig + 1e-8)
                
                # Construct exact jammer vector
                mag_exact = g_exact_polar[:, 0::2]
                phase_exact = g_exact_polar[:, 1::2]
                g_exact_complex = (mag_exact * torch.exp(1j * phase_exact)).to(torch.complex64)
                
                # Jitter-Robust Training
                # We apply small angular jitters to the geometric phase
                N_JIT = 5
                jitter_angles = torch.linspace(-2.0, 2.0, N_JIT).to(device) # degrees
                leakage_sum = 0.0
                
                for j_ang in jitter_angles:
                    # In this simplified dataset, cart_inputs is just the unit vector of the jammer
                    # We can approximate jitter by rotating the cart_inputs in the xy plane
                    theta = j_ang * np.pi / 180.0
                    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
                    rot_mat = torch.tensor([[cos_t, -sin_t, 0], [sin_t, cos_t, 0], [0, 0, 1]], dtype=torch.float32).to(device)
                    cart_jit = cart_inputs @ rot_mat.T
                    
                    phases_geo_exact = K * (cart_jit @ pos_body_t.T)
                    geo_sp = torch.exp(1j * phases_geo_exact)
                    v_jam_exact = g_exact_complex * geo_sp
                    
                    leakage = torch.abs(torch.sum(torch.conj(w_norm) * v_jam_exact, dim=1))**2
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
    
        torch.save(model.state_dict(), "phase_c_weights_32.pt")
    
    # --- EVALUATION ---
    print("\n--- SINR Evaluation ---")
    model.eval()
    headings = np.linspace(0, 360, 360, endpoint=False)
    jam_bodies = np.zeros((360, 3))
    for i, h in enumerate(headings):
        jam_world_eval = np.array([np.cos(np.deg2rad(h)), np.sin(np.deg2rad(h)), 0.0])
        jam_bodies[i] = rotate_points(jam_world_eval.reshape(1, 3), q_inv)[0]
        
    from shadow_engine_batched import compute_shadow_mask_batched
    
    g_exact_all = np.zeros((360, 32), dtype=np.complex128)
    chunk_size = 36
    for i in range(0, 360, chunk_size):
        chunk_jams = jam_bodies[i:i+chunk_size]
        g_exact_all[i:i+chunk_size] = compute_shadow_mask_batched(mesh, pos_body, normals_body, chunk_jams)
    
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
        v_jam_ex = g_exact_all[i] * get_steering_vector(pos_body, jam_bodies[i] * K)
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
