"""
Stage 14B – Phase B: Null-Response SINR-Aware Fine-Tuning
==========================================================
Loss: NullResponseLoss
  For each sample the loss penalises the jammer power leakage at the 
  *exact physics* jammer steering vector through the AI-derived MVDR weights.

  The gradient path is:
      pred (g_ai) -> v_jam_ai -> R_xx_ai -> w_ai (via linalg.solve) 
                  -> |w_ai^H v_jam_exact|^2   ← minimise this

  This is never saturated: as long as the AI null is misplaced, the exact
  jammer leaks through the weights, providing a gradient.

  Combined with an MSE regulariser on the labels (exact Fresnel gains) to 
  prevent the model from collapsing the gains to zero.
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
BATCH_SIZE = 256
# Loss weighting: null_weight * null_loss + mse_weight * mse_loss
NULL_WEIGHT = 1.0
MSE_WEIGHT = 0.1  # Regulariser to keep predictions anchored to true gains


def positional_encoding(x, L=6):
    """x: [batch, 3] -> [batch, 3*(1 + 2L)] = [batch, 39]"""
    encoded = [x]
    for i in range(L):
        encoded.append(torch.sin((2.0 ** i) * torch.pi * x))
        encoded.append(torch.cos((2.0 ** i) * torch.pi * x))
    return torch.cat(encoded, dim=-1)


class ShadowNet(nn.Module):
    def __init__(self):
        super(ShadowNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(39, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 32)
        )

    def forward(self, x):
        x_enc = positional_encoding(x)
        return self.net(x_enc)


class NullResponseLoss(nn.Module):
    """
    Null-response loss:
      1. Build v_jam_ai from AI predictions (differentiable path)
      2. Compute MVDR weights w_ai using R_xx = JAM_POW * v_jam_ai @ v_jam_ai^H + R_n
      3. Measure jammer leakage = |w_ai^H * v_jam_exact|^2 (exact physics jammer vector)
      4. Minimise leakage  =>  deepens the null at the correct direction
      + MSE regulariser to stop gain collapse to zero
    """
    def __init__(self, pos_body, lam=0.15):
        super().__init__()
        self.register_buffer('pos_body', torch.tensor(pos_body, dtype=torch.float32))
        self.K = 2.0 * np.pi / lam
        JAM_POW = 10000.0
        NOISE_POW = 1.0
        self.JAM_POW = JAM_POW
        self.NOISE_POW = NOISE_POW

        # Pinned signal direction in body frame: [1, 0, 0]
        sig_body = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
        phases_sig = self.K * torch.matmul(self.pos_body, sig_body)
        v_sig = torch.exp(1j * phases_sig)          # [16]
        self.register_buffer('v_sig', v_sig)

    def _build_steering(self, jam_dirs):
        """jam_dirs: [batch, 3] -> spatial phase vectors [batch, 16]"""
        phases = self.K * torch.matmul(jam_dirs, self.pos_body.T)   # [batch, 16]
        return torch.exp(1j * phases.to(torch.float32))               # [batch, 16]

    def forward(self, pred, batch_X, batch_y):
        """
        pred:    [batch, 32]  – AI Cartesian predictions (re0,im0, re1,im1, ...)
        batch_X: [batch, 3]   – jammer direction in body frame
        batch_y: [batch, 32]  – exact Cartesian labels (same interleave)
        """
        N = pred.shape[0]
        device = pred.device

        # 1. AI Fresnel gains [batch, 16] complex
        g_ai = (pred[:, 0::2] + 1j * pred[:, 1::2]).to(torch.complex64)

        # 2. Exact Fresnel gains from labels [batch, 16] complex
        g_exact = (batch_y[:, 0::2] + 1j * batch_y[:, 1::2]).to(torch.complex64)

        # 3. Spatial steering phase for jammer direction [batch, 16] complex
        sp = self._build_steering(batch_X.to(torch.float32)).to(torch.complex64)  # [batch,16]

        # 4. AI jammer steering vector (differentiable)
        v_jam_ai = g_ai * sp          # [batch, 16]

        # 5. Exact jammer steering vector (no gradient)
        with torch.no_grad():
            v_jam_exact = (g_exact * sp).detach()  # [batch, 16]

        # 6. Covariance: R_xx = JAM_POW * v_ai @ v_ai^H + NOISE * I
        v_ai_col = v_jam_ai.unsqueeze(2)   # [batch, 16, 1]
        R_j = self.JAM_POW * torch.matmul(v_ai_col, v_ai_col.conj().transpose(1, 2))  # [batch, 16, 16]
        R_n = self.NOISE_POW * torch.eye(16, device=device, dtype=torch.complex64
                                         ).unsqueeze(0).expand(N, 16, 16)
        # Small diagonal load for numerical stability
        dl = 1e-6 * torch.eye(16, device=device, dtype=torch.complex64
                              ).unsqueeze(0).expand(N, 16, 16)
        R_xx = R_j + R_n + dl            # [batch, 16, 16]

        # 7. MVDR: w = R_xx^-1 v_sig / (v_sig^H R_xx^-1 v_sig)
        v_sig_batch = self.v_sig.unsqueeze(0).unsqueeze(2).expand(N, 16, 1).to(torch.complex64)
        num = torch.linalg.solve(R_xx, v_sig_batch)           # [batch, 16, 1]
        den = torch.matmul(v_sig_batch.conj().transpose(1, 2), num)  # [batch, 1, 1]
        w = num / (den + 1e-12)                               # [batch, 16, 1]

        # 8. Jammer leakage at the EXACT jammer direction through AI weights
        v_exact_col = v_jam_exact.unsqueeze(2)                # [batch, 16, 1]
        leakage = torch.abs(torch.matmul(
            w.conj().transpose(1, 2), v_exact_col
        )).squeeze() ** 2                                      # [batch]

        null_loss = torch.mean(torch.log10(leakage + 1e-12))  # minimise (less negative = worse)

        # 9. MSE regulariser on Fresnel gain predictions (uses the exact labels)
        mse_loss = torch.mean((pred - batch_y) ** 2)

        total = NULL_WEIGHT * null_loss + MSE_WEIGHT * mse_loss
        return total, null_loss.detach(), mse_loss.detach()


def train_model(model, criterion, train_loader, val_loader, epochs, patience, save_path=None):
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=20)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = train_null = train_mse = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss, nl, ml = criterion(outputs, batch_X, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
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
                outputs = model(batch_X)
                loss, _, _ = criterion(outputs, batch_X, batch_y)
                val_loss += loss.item() * batch_X.size(0)
        val_loss /= len(val_loader.dataset)

        scheduler.step(val_loss)

        print(f"Epoch {epoch+1:03d}/{epochs} | "
              f"Total: {train_loss:.4f} | Null: {train_null:.4f} | MSE: {train_mse:.6f} | Val: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            if save_path:
                torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1} (Best Val Loss: {best_val_loss:.4f})")
                break

    return best_val_loss


def main():
    dataset_path = "dataset_shadow_30k.npz"
    if not os.path.exists(dataset_path):
        print(f"Dataset {dataset_path} not found.")
        return

    print("Loading mesh and dataset...")
    MESH_PATH = Path(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone_prepared.stl")
    mesh = load_uav_mesh(MESH_PATH)
    pos_body, _ = get_conformal_array(mesh)

    data = np.load(dataset_path)
    inputs = data['inputs']
    labels = data['labels']

    np.random.seed(42)
    torch.manual_seed(42)

    N = inputs.shape[0]
    indices = np.random.permutation(N)
    split_idx = int(N * 0.8)

    X_train = torch.tensor(inputs[indices[:split_idx]], dtype=torch.float32)
    y_train = torch.tensor(labels[indices[:split_idx]], dtype=torch.float32)
    X_val = torch.tensor(inputs[indices[split_idx:]], dtype=torch.float32)
    y_val = torch.tensor(labels[indices[split_idx:]], dtype=torch.float32)

    print(f"--- Running Phase B Null-Response Training (30k samples, 300 epochs) ---")
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    # Load Phase A weights
    model = ShadowNet()
    if os.path.exists("shadow_net_pe.pt"):
        model.load_state_dict(torch.load("shadow_net_pe.pt"))
        print("Loaded Phase A weights (shadow_net_pe.pt) for Phase B initialisation.")
    else:
        print("Warning: shadow_net_pe.pt not found. Training from scratch.")

    criterion = NullResponseLoss(pos_body)

    train_model(model, criterion, train_loader, val_loader,
                epochs=300, patience=50, save_path="shadow_net_sinr.pt")
    print("Phase B complete. Saved to shadow_net_sinr.pt")


if __name__ == '__main__':
    main()
