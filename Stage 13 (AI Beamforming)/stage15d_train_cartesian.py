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
from conformal_array import get_conformal_array
from lcmv_stage8 import get_steering_vector

# --- Configuration ---
BATCH_SIZE = 512
EPOCHS = 100
LR = 5e-4
LAM = 0.15
K = 2.0 * np.pi / LAM
JAM_POW = 10000.0
NOISE_POW = 1.0
DL_ALPHA = 0.0
DELTA_JITTER_RAD = np.deg2rad(2.0)
MSE_WEIGHT = 5.0  # Strong tether to Cartesian truth

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

class CartesianShadowNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pe = PositionalEncoding(L=10)
        self.net = nn.Sequential(
            nn.Linear(63, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 32)
        )
    def forward(self, x):
        return self.net(self.pe(x))

class CartesianRobustLoss(nn.Module):
    def __init__(self, pos_body, v_sig):
        super().__init__()
        self.pos_body = torch.tensor(pos_body, dtype=torch.float32)
        self.v_sig = torch.tensor(v_sig, dtype=torch.complex64)
        self.mse = nn.MSELoss()

    def forward(self, cart_inputs, g_exact_polar, model):
        device = cart_inputs.device
        self.pos_body = self.pos_body.to(device)
        self.v_sig = self.v_sig.to(device)
        
        out = model(cart_inputs)
        g_raw = (out[:, 0::2] + 1j * out[:, 1::2]).to(torch.complex64)
        
        # Structural Magnitude Bounding: |g| <= 1.0
        mag_raw = torch.abs(g_raw)
        scale = torch.clamp(mag_raw, min=1.0)
        g_bounded = g_raw / scale

        # Exact ground truth is the pure diffraction shadow
        mag_exact = g_exact_polar[:, 0::2]
        phase_exact = g_exact_polar[:, 1::2]
        g_exact_complex = (mag_exact * torch.exp(1j * phase_exact)).to(torch.complex64)

        # Cartesian MSE
        loss_mse = torch.mean(torch.abs(g_bounded - g_exact_complex)**2)

        # Physics Decomposition: add geometric phase
        phases_geo_exact = K * (cart_inputs @ self.pos_body.T)
        geo_sp = torch.exp(1j * phases_geo_exact)
        v_jam_ai = geo_sp * g_bounded

        # SM-MVDR Weights
        norm_v2 = torch.sum(torch.abs(v_jam_ai)**2, dim=1)
        c_B = NOISE_POW + DL_ALPHA * (NOISE_POW + JAM_POW * norm_v2 / 16.0)
        v_H_vsig = torch.sum(torch.conj(v_jam_ai) * self.v_sig.unsqueeze(0), dim=1)
        term2_scalar = JAM_POW / (c_B * (c_B + JAM_POW * norm_v2))
        u = (1.0 / c_B.unsqueeze(1)) * self.v_sig.unsqueeze(0) - (term2_scalar * v_H_vsig).unsqueeze(1) * v_jam_ai
        vsig_H_u = torch.sum(torch.conj(self.v_sig).unsqueeze(0) * u, dim=1)
        w = u / vsig_H_u.unsqueeze(1)

        # Jitter Robust Leakage
        total_leakage = 0.0
        NUM_JITTERS = 3
        for _ in range(NUM_JITTERS):
            noise_scale = np.tan(DELTA_JITTER_RAD)
            noise = torch.rand_like(cart_inputs) * 2 * noise_scale - noise_scale
            cart_jit = cart_inputs + noise
            cart_jit = cart_jit / torch.norm(cart_jit, dim=1, keepdim=True)
            
            phases_geo_jit = K * (cart_jit @ self.pos_body.T)
            geo_sp_jit = torch.exp(1j * phases_geo_jit)
            v_jam_jit = g_exact_complex * geo_sp_jit
            
            leakage = torch.abs(torch.sum(torch.conj(w) * v_jam_jit, dim=1))**2
            total_leakage = total_leakage + leakage
            
        avg_leakage = total_leakage / NUM_JITTERS
        loss_null = torch.mean(torch.log10(avg_leakage + 1e-12))

        return MSE_WEIGHT * loss_mse + loss_null, loss_mse, loss_null

def train_model(model, criterion, train_loader, val_loader, epochs, save_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, eta_min=1e-6)

    best_val_loss = float('inf')

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_mse = 0.0
        train_null = 0.0
        
        for cart_inputs, g_exact_polar in train_loader:
            cart_inputs = cart_inputs.to(device)
            g_exact_polar = g_exact_polar.to(device)

            optimizer.zero_grad()
            total, mse, null = criterion(cart_inputs, g_exact_polar, model)
            total.backward()
            optimizer.step()
            
            train_loss += total.item() * cart_inputs.size(0)
            train_mse += mse.item() * cart_inputs.size(0)
            train_null += null.item() * cart_inputs.size(0)

        train_loss /= len(train_loader.dataset)
        train_mse /= len(train_loader.dataset)
        train_null /= len(train_loader.dataset)

        scheduler.step()
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}/{epochs} | Tot: {train_loss:.4f} | MSE: {train_mse:.6f} | Null: {train_null:.4f}")

        if train_loss < best_val_loss:
            best_val_loss = train_loss
            if save_path:
                torch.save(model.state_dict(), save_path)

def main():
    dataset_path = "dataset_shadow_100k_polar.npz"

    print("Loading mesh and preparing datasets...")
    MESH_PATH = Path(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone_prepared.stl")
    mesh = load_uav_mesh(MESH_PATH)
    pos_body, _ = get_conformal_array(mesh)
    
    sig_body = np.array([1.0, 0.0, 0.0])
    v_sig = get_steering_vector(pos_body, K * sig_body)

    data = np.load(dataset_path)
    inputs = data['inputs']
    labels_polar = data['labels']

    np.random.seed(42)
    torch.manual_seed(42)

    N = inputs.shape[0]
    indices = np.random.permutation(N)
    split_idx = int(N * 0.8)

    X_cart_train = torch.tensor(inputs[indices[:split_idx]], dtype=torch.float32)
    y_train = torch.tensor(labels_polar[indices[:split_idx]], dtype=torch.float32)
    
    X_cart_val = torch.tensor(inputs[indices[split_idx:]], dtype=torch.float32)
    y_val = torch.tensor(labels_polar[indices[split_idx:]], dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_cart_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_cart_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    print(f"--- Running Cartesian Robust Training ---")
    model = CartesianShadowNet()
    criterion = CartesianRobustLoss(pos_body, v_sig)
    
    train_model(model, criterion, train_loader, val_loader, epochs=EPOCHS, save_path="shadow_net_cartesian.pt")
    print("Training complete. Saved to shadow_net_cartesian.pt")

if __name__ == '__main__':
    main()
