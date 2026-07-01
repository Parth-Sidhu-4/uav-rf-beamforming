"""
Stage 14D – Run D-3: Architecture and Data Scaling
===================================================
Architecture: Phase D-B (L=8, 51->512(LN)->512(LN)->256(LN)->32)
Data: 100k uniform (converted from polar dataset_shadow_100k_polar.npz)
Curriculum: Dynamic MSE_WEIGHT + CosineAnnealingWarmRestarts + Gradient Clipping
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

# Add paths for conformal_array and mesh_loader
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Phase 2 Track 1', 'em_realism')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Phase 2 Track 2', 'array_structures')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Stage 12 (Operational Analysis)')))

from mesh_loader import load_uav_mesh
from conformal_array import get_conformal_array

LR = 3e-4
BATCH_SIZE = 512
NULL_WEIGHT = 1.0


def positional_encoding(x, L=8):
    # L=8 gives 3*(1 + 16) = 51 dimensions
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


class NullResponseLoss(nn.Module):
    def __init__(self, pos_body, lam=0.15):
        super().__init__()
        self.register_buffer('pos_body', torch.tensor(pos_body, dtype=torch.float32))
        self.K = 2.0 * np.pi / lam
        self.JAM_POW = 10000.0
        self.NOISE_POW = 1.0

        sig_body = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
        phases_sig = self.K * torch.matmul(self.pos_body, sig_body)
        v_sig = torch.exp(1j * phases_sig)
        self.register_buffer('v_sig', v_sig)

    def _build_steering(self, jam_dirs):
        phases = self.K * torch.matmul(jam_dirs, self.pos_body.T)
        return torch.exp(1j * phases.to(torch.float32))

    def forward(self, pred, batch_X, batch_y, mse_weight=0.1):
        N = pred.shape[0]
        device = pred.device

        g_ai = (pred[:, 0::2] + 1j * pred[:, 1::2]).to(torch.complex64)
        g_exact = (batch_y[:, 0::2] + 1j * batch_y[:, 1::2]).to(torch.complex64)
        sp = self._build_steering(batch_X.to(torch.float32)).to(torch.complex64)
        v_jam_ai = g_ai * sp

        with torch.no_grad():
            v_jam_exact = (g_exact * sp).detach()

        v_ai_col = v_jam_ai.unsqueeze(2)
        R_j = self.JAM_POW * torch.matmul(v_ai_col, v_ai_col.conj().transpose(1, 2))
        R_n = self.NOISE_POW * torch.eye(16, device=device, dtype=torch.complex64).unsqueeze(0).expand(N, 16, 16)
        dl = 1e-6 * torch.eye(16, device=device, dtype=torch.complex64).unsqueeze(0).expand(N, 16, 16)
        R_xx = R_j + R_n + dl

        v_sig_batch = self.v_sig.unsqueeze(0).unsqueeze(2).expand(N, 16, 1).to(torch.complex64)
        num = torch.linalg.solve(R_xx, v_sig_batch)
        den = torch.matmul(v_sig_batch.conj().transpose(1, 2), num)
        w = num / (den + 1e-12)

        v_exact_col = v_jam_exact.unsqueeze(2)
        leakage = torch.abs(torch.matmul(w.conj().transpose(1, 2), v_exact_col)).squeeze() ** 2

        null_loss = torch.mean(torch.log10(leakage + 1e-12))
        mse_loss = torch.mean((pred - batch_y) ** 2)

        total = NULL_WEIGHT * null_loss + mse_weight * mse_loss
        return total, null_loss.detach(), mse_loss.detach()


def get_mse_weight(epoch):
    if epoch <= 80:
        return 0.50
    elif epoch <= 200:
        return 0.10
    else:
        return 0.02


def train_model(model, criterion, train_loader, val_loader, epochs, save_path=None):
    # Using CUDA if available (CUDA makes large batch 100k training much faster)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    model.to(device)
    criterion.to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, eta_min=1e-6)

    best_val_loss = float('inf')

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = train_null = train_mse = 0.0
        
        current_mse_weight = get_mse_weight(epoch)
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss, nl, ml = criterion(outputs, batch_X, batch_y, mse_weight=current_mse_weight)
            loss.backward()
            
            # User request: max_norm = 1.0 for first 50 epochs to stabilise wider architecture
            if epoch <= 50:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
            optimizer.step()
            sz = batch_X.size(0)
            train_loss += loss.item() * sz
            train_null += nl.item() * sz
            train_mse += ml.item() * sz

        n_train = len(train_loader.dataset)
        train_loss /= n_train
        train_null /= n_train
        train_mse /= n_train

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                outputs = model(batch_X)
                loss, _, _ = criterion(outputs, batch_X, batch_y, mse_weight=current_mse_weight)
                val_loss += loss.item() * batch_X.size(0)
        val_loss /= len(val_loader.dataset)

        scheduler.step()
        
        current_lr = scheduler.get_last_lr()[0]

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}/{epochs} | LR: {current_lr:.2e} | MSE_WT: {current_mse_weight:.3f} | "
                  f"Total: {train_loss:.4f} | Null: {train_null:.4f} | MSE: {train_mse:.6f} | Val: {val_loss:.4f}")

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

    print("Loading mesh and dataset...")
    MESH_PATH = Path(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone_prepared.stl")
    mesh = load_uav_mesh(MESH_PATH)
    pos_body, _ = get_conformal_array(mesh)

    data = np.load(dataset_path)
    inputs = data['inputs']
    labels_polar = data['labels'] # interleaved mag, phase
    
    # Convert polar to Cartesian to match what the network output expects
    labels_cartesian = np.zeros_like(labels_polar)
    mag = labels_polar[:, 0::2]
    phase = labels_polar[:, 1::2]
    labels_cartesian[:, 0::2] = mag * np.cos(phase)
    labels_cartesian[:, 1::2] = mag * np.sin(phase)

    np.random.seed(42)
    torch.manual_seed(42)

    N = inputs.shape[0]
    indices = np.random.permutation(N)
    split_idx = int(N * 0.8)

    X_train = torch.tensor(inputs[indices[:split_idx]], dtype=torch.float32)
    y_train = torch.tensor(labels_cartesian[indices[:split_idx]], dtype=torch.float32)
    X_val = torch.tensor(inputs[indices[split_idx:]], dtype=torch.float32)
    y_val = torch.tensor(labels_cartesian[indices[split_idx:]], dtype=torch.float32)

    print(f"--- Running Phase D-3 (100k samples, L=8, Phase D-B arch, 500 epochs) ---")
    # Increase num_workers for faster loading if cpu
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    model = ShadowNetDB()
    # Since architecture changed, we MUST train from scratch
    print("Training Phase D-B architecture from scratch...")

    criterion = NullResponseLoss(pos_body)

    train_model(model, criterion, train_loader, val_loader,
                epochs=500, save_path="shadow_net_d3.pt")
    print("Phase D-3 complete. Saved to shadow_net_d3.pt")


if __name__ == '__main__':
    main()
