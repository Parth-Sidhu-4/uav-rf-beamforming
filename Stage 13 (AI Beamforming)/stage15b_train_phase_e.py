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
EPOCHS = 400
LR = 5e-4
LAM = 0.15
K = 2.0 * np.pi / LAM
JAM_POW = 10000.0
NOISE_POW = 1.0
DL_ALPHA = 0.05
DELTA_JITTER_RAD = np.deg2rad(2.0)

# --- SIREN Network ---
class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / in_features, 1 / in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / in_features) / omega_0, np.sqrt(6 / in_features) / omega_0)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

class SirenResidualNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            SineLayer(25, 256, is_first=True),
            SineLayer(256, 256),
            SineLayer(256, 256),
            SineLayer(256, 256),
            nn.Linear(256, 32)
        )
        # Initialize final layer near zero to start with ~0 residual
        with torch.no_grad():
            self.net[-1].weight.uniform_(-1e-4, 1e-4)
            self.net[-1].bias.uniform_(-1e-4, 1e-4)

    def forward(self, x):
        return self.net(x)

# --- Training Loss ---
class JitterRobustNullLoss(nn.Module):
    def __init__(self, pos_body, v_sig):
        super().__init__()
        self.pos_body = torch.tensor(pos_body, dtype=torch.float32)
        self.v_sig = torch.tensor(v_sig, dtype=torch.complex64)

    def forward(self, sh_features, cart_inputs, g_exact_polar, model):
        device = sh_features.device
        self.pos_body = self.pos_body.to(device)
        self.v_sig = self.v_sig.to(device)
        B = sh_features.shape[0]

        # 1. Predict Residual
        out = model(sh_features) # [B, 32]
        mag_ai = torch.sigmoid(out[:, 0::2]) # [B, 16]
        # Bounded phase perturbation: [-pi, pi] based on our diagnostic
        phase_pert = torch.pi * torch.tanh(out[:, 1::2]) # [B, 16]

        # 2. Geometric Phase for central angle
        phases_geo = K * (cart_inputs @ self.pos_body.T) # [B, 16]
        geo_sp = torch.exp(1j * phases_geo) # [B, 16]

        # 3. AI steering vector (central angle)
        g_ai = mag_ai * torch.exp(1j * phase_pert)
        v_jam_ai = geo_sp * g_ai # [B, 16]

        # 4. Sherman-Morrison MVDR
        norm_v2 = torch.sum(torch.abs(v_jam_ai)**2, dim=1) # [B]
        c_B = NOISE_POW + DL_ALPHA * (NOISE_POW + JAM_POW * norm_v2 / 16.0) # [B]
        
        v_H_vsig = torch.sum(torch.conj(v_jam_ai) * self.v_sig.unsqueeze(0), dim=1) # [B]
        term2_scalar = JAM_POW / (c_B * (c_B + JAM_POW * norm_v2)) # [B]
        
        u = (1.0 / c_B.unsqueeze(1)) * self.v_sig.unsqueeze(0) - (term2_scalar * v_H_vsig).unsqueeze(1) * v_jam_ai # [B, 16]
        vsig_H_u = torch.sum(torch.conj(self.v_sig).unsqueeze(0) * u, dim=1) # [B]
        w = u / vsig_H_u.unsqueeze(1) # [B, 16]

        # 5. Jitter-Robust Loss
        total_leakage = 0.0
        NUM_JITTERS = 1
        DELTA_JITTER_RAD = 0.0
        
        mag_exact = g_exact_polar[:, 0::2]
        phase_exact = g_exact_polar[:, 1::2]
        g_exact_complex = mag_exact * torch.exp(1j * phase_exact)

        for _ in range(NUM_JITTERS):
            # Jitter Cartesian inputs
            noise_scale = 0.0 # np.tan(DELTA_JITTER_RAD)
            noise = torch.zeros_like(cart_inputs)
            cart_jit = cart_inputs + noise
            cart_jit = cart_jit / torch.norm(cart_jit, dim=1, keepdim=True)
            
            # Geometric phase for jittered angle
            phases_geo_jit = K * (cart_jit @ self.pos_body.T)
            geo_sp_jit = torch.exp(1j * phases_geo_jit)
            
            # Physical Approximation
            v_jam_jit = g_exact_complex * geo_sp_jit # [B, 16]
            
            # Leakage = |w^H v_jam_jit|^2
            leakage = torch.abs(torch.sum(torch.conj(w) * v_jam_jit, dim=1))**2 # [B]
            total_leakage = total_leakage + leakage
            
        avg_leakage = total_leakage / NUM_JITTERS
        
        # Pure leakage loss! No MSE tether needed due to structural bounds.
        loss = torch.mean(torch.log10(avg_leakage + 1e-12))
        return loss

def train_model(model, criterion, train_loader, val_loader, epochs, save_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, eta_min=1e-6)

    best_val_loss = float('inf')

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        
        for sh_features, cart_inputs, g_exact_polar in train_loader:
            sh_features = sh_features.to(device)
            cart_inputs = cart_inputs.to(device)
            g_exact_polar = g_exact_polar.to(device)

            optimizer.zero_grad()
            loss = criterion(sh_features, cart_inputs, g_exact_polar, model)
            loss.backward()
            
            # Gradient clipping is very important for SIRENs
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item() * sh_features.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sh_features, cart_inputs, g_exact_polar in val_loader:
                sh_features = sh_features.to(device)
                cart_inputs = cart_inputs.to(device)
                g_exact_polar = g_exact_polar.to(device)
                
                loss = criterion(sh_features, cart_inputs, g_exact_polar, model)
                val_loss += loss.item() * sh_features.size(0)
                
        val_loss /= len(val_loader.dataset)

        scheduler.step()
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}/{epochs} | LR: {scheduler.get_last_lr()[0]:.2e} | "
                  f"Train Null Loss: {train_loss:.4f} | Val Null Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if save_path:
                torch.save(model.state_dict(), save_path)

    return best_val_loss

def main():
    dataset_path = "dataset_shadow_100k_sh.npz"
    if not os.path.exists(dataset_path):
        print(f"Dataset {dataset_path} not found. Run prep script first.")
        return

    print("Loading mesh and preparing datasets...")
    MESH_PATH = Path(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone_prepared.stl")
    mesh = load_uav_mesh(MESH_PATH)
    pos_body, _ = get_conformal_array(mesh)
    
    sig_body = np.array([1.0, 0.0, 0.0])
    v_sig = get_steering_vector(pos_body, K * sig_body)

    data = np.load(dataset_path)
    inputs = data['inputs']
    sh_features = data['sh_features']
    labels_polar = data['labels']

    np.random.seed(42)
    torch.manual_seed(42)

    N = inputs.shape[0]
    indices = np.random.permutation(N)
    split_idx = int(N * 0.8)

    X_cart_train = torch.tensor(inputs[indices[:split_idx]], dtype=torch.float32)
    X_sh_train = torch.tensor(sh_features[indices[:split_idx]], dtype=torch.float32)
    y_train = torch.tensor(labels_polar[indices[:split_idx]], dtype=torch.float32)
    
    X_cart_val = torch.tensor(inputs[indices[split_idx:]], dtype=torch.float32)
    X_sh_val = torch.tensor(sh_features[indices[split_idx:]], dtype=torch.float32)
    y_val = torch.tensor(labels_polar[indices[split_idx:]], dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_sh_train, X_cart_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_sh_val, X_cart_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    print(f"--- Running Phase E Training (Combined Architecture) ---")
    model = SirenResidualNet()
    criterion = JitterRobustNullLoss(pos_body, v_sig)
    
    train_model(model, criterion, train_loader, val_loader, epochs=EPOCHS, save_path="shadow_net_phase_e.pt")
    print("Phase E training complete. Saved to shadow_net_phase_e.pt")

if __name__ == '__main__':
    main()
