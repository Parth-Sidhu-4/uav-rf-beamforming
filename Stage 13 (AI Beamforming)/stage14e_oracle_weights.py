"""
Stage 14E – Oracle Weight Prediction (Fix 2)
=============================================
Architecture: Phase D-B (L=8, 51->512(LN)->512(LN)->256(LN)->32)
Data: 100k uniform
Task: Direct MSE regression on optimal MVDR weights
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Phase 2 Track 1', 'em_realism')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Phase 2 Track 2', 'array_structures')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Stage 12 (Operational Analysis)')))

from mesh_loader import load_uav_mesh
from conformal_array import get_conformal_array

LR = 1e-3
BATCH_SIZE = 512

def positional_encoding(x, L=8):
    encoded = [x]
    for i in range(L):
        encoded.append(torch.sin((2.0 ** i) * torch.pi * x))
        encoded.append(torch.cos((2.0 ** i) * torch.pi * x))
    return torch.cat(encoded, dim=-1)

class ShadowNetDB(nn.Module):
    def __init__(self):
        super(ShadowNetDB, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(51, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 32)
        )

    def forward(self, x):
        x_enc = positional_encoding(x, L=8)
        return self.net(x_enc)

def compute_optimal_weights(batch_X, batch_g, pos_body, lam=0.15):
    """
    batch_X: (N, 3) jammer direction
    batch_g: (N, 16) complex gains
    Returns w_exact: (N, 32) real/imag interleaved
    """
    N = batch_X.shape[0]
    K = 2.0 * np.pi / lam
    JAM_POW = 10000.0
    NOISE_POW = 1.0

    sig_body = np.array([1.0, 0.0, 0.0])
    phases_sig = K * pos_body @ sig_body
    v_sig = np.exp(1j * phases_sig)

    phases_jam = K * batch_X @ pos_body.T
    sp = np.exp(1j * phases_jam)
    v_jam = batch_g * sp

    w_exact = np.zeros((N, 32), dtype=np.float32)

    for i in range(N):
        v_j = v_jam[i]
        R_j = JAM_POW * np.outer(v_j, np.conj(v_j))
        R_n = NOISE_POW * np.eye(16)
        dl = 1e-6 * np.eye(16)
        R_xx = R_j + R_n + dl
        
        try:
            R_inv = np.linalg.inv(R_xx)
        except np.linalg.LinAlgError:
            R_inv = np.linalg.pinv(R_xx)
            
        num = R_inv @ v_sig
        den = np.conj(v_sig) @ num
        w = num / max(abs(den), 1e-12)
        
        w_exact[i, 0::2] = np.real(w)
        w_exact[i, 1::2] = np.imag(w)

    return w_exact

def train_model(model, train_loader, val_loader, epochs, save_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, eta_min=1e-6)

    best_val_loss = float('inf')

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_X.size(0)
        val_loss /= len(val_loader.dataset)

        scheduler.step()
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}/{epochs} | LR: {scheduler.get_last_lr()[0]:.2e} | "
                  f"Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if save_path:
                torch.save(model.state_dict(), save_path)

    return best_val_loss

def main():
    dataset_path = "dataset_shadow_100k_polar.npz"
    if not os.path.exists(dataset_path):
        print(f"Dataset {dataset_path} not found.")
        return

    print("Loading mesh and preparing dataset...")
    MESH_PATH = Path(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone_prepared.stl")
    mesh = load_uav_mesh(MESH_PATH)
    pos_body, _ = get_conformal_array(mesh)

    data = np.load(dataset_path)
    inputs = data['inputs']
    labels_polar = data['labels']
    
    # Convert polar to complex gains
    mag = labels_polar[:, 0::2]
    phase = labels_polar[:, 1::2]
    g_exact = mag * np.exp(1j * phase)

    print("Precomputing oracle weights (this takes ~10 seconds)...")
    w_exact = compute_optimal_weights(inputs, g_exact, pos_body)

    np.random.seed(42)
    torch.manual_seed(42)

    N = inputs.shape[0]
    indices = np.random.permutation(N)
    split_idx = int(N * 0.8)

    X_train = torch.tensor(inputs[indices[:split_idx]], dtype=torch.float32)
    y_train = torch.tensor(w_exact[indices[:split_idx]], dtype=torch.float32)
    X_val = torch.tensor(inputs[indices[split_idx:]], dtype=torch.float32)
    y_val = torch.tensor(w_exact[indices[split_idx:]], dtype=torch.float32)

    print(f"--- Running Oracle Weight Regression (100k samples, L=8, Phase D-B arch, 500 epochs) ---")
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    model = ShadowNetDB()
    train_model(model, train_loader, val_loader, epochs=500, save_path="shadow_net_oracle.pt")
    print("Oracle training complete. Saved to shadow_net_oracle.pt")

if __name__ == '__main__':
    main()
