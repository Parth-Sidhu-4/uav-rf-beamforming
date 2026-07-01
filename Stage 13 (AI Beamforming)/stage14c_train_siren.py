"""
Stage 14C – Phase C: SIREN Architecture for High-Fidelity Beamforming
=======================================================================
Architecture: ShadowNet_SIREN
  - Replaces ReLU activations with sin(omega_0 * x)
  - Input: raw [x, y, z] unit-vector direction (no positional encoding)
  - SIREN inherently represents superpositions of plane waves — the exact
    function class of Fresnel diffraction physics — rather than approximating
    them with piecewise-linear segments.
  - omega_0=30 (Sitzmann default). Validated post-training via:
      (a) 250° MAE spike reduced below Phase B level, AND
      (b) Median null placement < 5°.
    If either fails, rerun at omega_0=15 and omega_0=60 to pick the best.

Loss: NullResponseLoss (identical to Phase B)
Learning Rate: 1e-4 (Phase B used 3e-4; SIREN's sinusoidal activations
  create higher-curvature loss landscape requiring a lower LR to avoid
  instability during the initial 50-100 epoch frequency-discovery phase.)
Epochs: 500 max with patience=50 (SIREN starts from a colder initialisation
  and may use more epochs before frequency fitting completes.)
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

# --- Hyperparameters ---
LR = 1e-4           # Lower than Phase B (3e-4): required for stable SIREN training
BATCH_SIZE = 256
NULL_WEIGHT = 1.0
MSE_WEIGHT = 0.1
OMEGA_0 = 30        # Sitzmann default; validated empirically post-training
MAX_EPOCHS = 500
PATIENCE = 50


# ---------------------------------------------------------------------------
# SIREN Architecture
# ---------------------------------------------------------------------------
class SineLayer(nn.Module):
    """
    One layer of a SIREN: linear projection followed by a sinusoidal activation.
    Weight initialisation follows Sitzmann et al. (2020) to preserve the
    distribution of activations across layers.
    """
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            if self.is_first:
                # First layer: uniform in [-1/fan_in, 1/fan_in]
                self.linear.weight.uniform_(
                    -1.0 / self.linear.weight.size(1),
                     1.0 / self.linear.weight.size(1)
                )
            else:
                # Hidden layers: scaled by 1/omega_0 to preserve activation distribution
                bound = np.sqrt(6.0 / self.linear.weight.size(1)) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class ShadowNet_SIREN(nn.Module):
    """
    SIREN-based shadow mask predictor.
    Input:  [batch, 3]  – raw (x, y, z) jammer direction in body frame
    Output: [batch, 32] – interleaved (re0,im0, re1,im1, ...) Fresnel gains
    Width 512 matches Phase B's first hidden layer for a fair capacity comparison.
    """
    def __init__(self, in_features=3, hidden_features=512, hidden_layers=3,
                 out_features=32, omega_0=OMEGA_0):
        super().__init__()
        layers = []
        layers.append(SineLayer(in_features, hidden_features, is_first=True, omega_0=omega_0))
        for _ in range(hidden_layers):
            layers.append(SineLayer(hidden_features, hidden_features, is_first=False, omega_0=omega_0))
        # Final linear layer (no activation) following Sitzmann et al.
        final = nn.Linear(hidden_features, out_features)
        with torch.no_grad():
            bound = np.sqrt(6.0 / hidden_features) / omega_0
            final.weight.uniform_(-bound, bound)
        layers.append(final)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # Raw [x, y, z] input — SIREN's first sine layer acts as adaptive PE
        return self.net(x)


# ---------------------------------------------------------------------------
# Null-Response Loss (identical to Phase B)
# ---------------------------------------------------------------------------
class NullResponseLoss(nn.Module):
    """
    Penalises jammer power leakage at the exact-physics jammer steering vector
    through the AI-derived MVDR weights. Gradient path:
      pred (g_ai) -> v_jam_ai -> R_xx_ai -> w_ai (linalg.solve)
                  -> |w_ai^H v_jam_exact|^2  <- minimise this
    """
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

    def forward(self, pred, batch_X, batch_y):
        N = pred.shape[0]
        device = pred.device

        g_ai    = (pred[:, 0::2]    + 1j * pred[:, 1::2]).to(torch.complex64)
        g_exact = (batch_y[:, 0::2] + 1j * batch_y[:, 1::2]).to(torch.complex64)

        sp = self._build_steering(batch_X.to(torch.float32)).to(torch.complex64)

        v_jam_ai = g_ai * sp
        with torch.no_grad():
            v_jam_exact = (g_exact * sp).detach()

        v_ai_col = v_jam_ai.unsqueeze(2)
        R_j = self.JAM_POW * torch.matmul(v_ai_col, v_ai_col.conj().transpose(1, 2))
        R_n = self.NOISE_POW * torch.eye(16, device=device, dtype=torch.complex64).unsqueeze(0).expand(N, 16, 16)
        dl  = 1e-6  * torch.eye(16, device=device, dtype=torch.complex64).unsqueeze(0).expand(N, 16, 16)
        R_xx = R_j + R_n + dl

        v_sig_batch = self.v_sig.unsqueeze(0).unsqueeze(2).expand(N, 16, 1).to(torch.complex64)
        num = torch.linalg.solve(R_xx, v_sig_batch)
        den = torch.matmul(v_sig_batch.conj().transpose(1, 2), num)
        w   = num / (den + 1e-12)

        v_exact_col = v_jam_exact.unsqueeze(2)
        leakage = torch.abs(torch.matmul(w.conj().transpose(1, 2), v_exact_col)).squeeze() ** 2
        null_loss = torch.mean(torch.log10(leakage + 1e-12))

        mse_loss = torch.mean((pred - batch_y) ** 2)
        total = NULL_WEIGHT * null_loss + MSE_WEIGHT * mse_loss
        return total, null_loss.detach(), mse_loss.detach()


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------
def train_model(model, criterion, train_loader, val_loader, epochs, patience, save_path):
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
            train_mse  += ml.item() * sz

        n_train    = len(train_loader.dataset)
        train_loss /= n_train
        train_null /= n_train
        train_mse  /= n_train

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                loss, _, _ = criterion(model(batch_X), batch_X, batch_y)
                val_loss += loss.item() * batch_X.size(0)
        val_loss /= len(val_loader.dataset)

        scheduler.step(val_loss)
        print(f"Epoch {epoch+1:03d}/{epochs} | "
              f"Total: {train_loss:.4f} | Null: {train_null:.4f} | MSE: {train_mse:.6f} | Val: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1} (Best Val Loss: {best_val_loss:.4f})")
                break

    return best_val_loss


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    dataset_path = "dataset_shadow_30k.npz"
    if not os.path.exists(dataset_path):
        print(f"Dataset {dataset_path} not found.")
        return

    print("Loading mesh and dataset...")
    MESH_PATH = Path(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone_prepared.stl")
    mesh = load_uav_mesh(MESH_PATH)
    pos_body, _ = get_conformal_array(mesh)

    data   = np.load(dataset_path)
    inputs = data['inputs']
    labels = data['labels']

    np.random.seed(42)
    torch.manual_seed(42)

    N = inputs.shape[0]
    indices   = np.random.permutation(N)
    split_idx = int(N * 0.8)

    X_train = torch.tensor(inputs[indices[:split_idx]], dtype=torch.float32)
    y_train = torch.tensor(labels[indices[:split_idx]], dtype=torch.float32)
    X_val   = torch.tensor(inputs[indices[split_idx:]], dtype=torch.float32)
    y_val   = torch.tensor(labels[indices[split_idx:]], dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val,   y_val),   batch_size=BATCH_SIZE, shuffle=False)

    print(f"--- Phase C: SIREN Training (omega_0={OMEGA_0}, lr={LR}, max_epochs={MAX_EPOCHS}) ---")
    print(f"    Architecture: 3 -> 512 -> 512 -> 512 -> 512 -> 32 (all sine, linear output)")
    print(f"    Training from scratch with Sitzmann weight initialisation.")

    model = ShadowNet_SIREN()
    criterion = NullResponseLoss(pos_body)

    train_model(model, criterion, train_loader, val_loader,
                epochs=MAX_EPOCHS, patience=PATIENCE, save_path="shadow_net_siren.pt")
    print("Phase C complete. Saved to shadow_net_siren.pt")


if __name__ == '__main__':
    main()
