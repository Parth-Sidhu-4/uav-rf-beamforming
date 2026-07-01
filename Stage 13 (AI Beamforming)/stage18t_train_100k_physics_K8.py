import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

sys.path.append(os.path.abspath(r'D:\UAV Internship project\Phase 2 Track 1\em_realism'))
sys.path.append(os.path.abspath(r'D:\UAV Internship project\Phase 2 Track 2\array_structures'))
sys.path.append(os.path.abspath(r'D:\UAV Internship project\Stage 12 (Operational Analysis)'))
sys.path.append(os.path.abspath(r'D:\UAV Internship project\Stage 13 (AI Beamforming)'))
sys.path.append(os.path.abspath(r'D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)'))

from mesh_loader import load_uav_mesh
from conformal_array import get_conformal_array_parametric
from lcmv_stage8 import get_steering_vector
from attitude import euler_to_quaternion, rotate_points
from shadow_engine_batched import compute_shadow_mask_batched

# --- PyTorch DCMVDR ---
def get_steering_vector_pt(pos_lambda, theta, phi):
    B = theta.shape[0]
    u = torch.stack([
        torch.sin(theta)*torch.cos(phi),
        torch.sin(theta)*torch.sin(phi),
        torch.cos(theta)
    ], dim=1)
    phase = 2.0 * np.pi * torch.matmul(pos_lambda, u.unsqueeze(-1)).squeeze(-1)
    return torch.exp(1j * phase)

def d3_mvdr_beamformer(U, v_sig, P_J, sigma2, alpha):
    """
    U: [B, 32, K] predicted interference modes
    v_sig: [32] signal steering vector
    """
    B, N, K = U.shape
    
    # R_hat = P_J * (U U^H) + (sigma2 + alpha) * I
    # Optimize by using matrix multiplication
    R_j = P_J * torch.bmm(U, torch.conj(U.transpose(1, 2)))
    R_in = R_j + (sigma2 + alpha) * torch.eye(N, dtype=torch.complex128, device=U.device).unsqueeze(0)
    
    R_in_inv = torch.linalg.inv(R_in)
    
    v_sig_b = v_sig.unsqueeze(0).unsqueeze(-1).expand(B, N, 1)
    
    # w_raw = R_in_inv @ v_sig
    w_raw = torch.bmm(R_in_inv, v_sig_b)
    
    # Unit gain normalization: w = w_raw / (v_sig^H @ w_raw)
    v_sig_H = torch.conj(v_sig).unsqueeze(0).unsqueeze(0).expand(B, 1, N)
    denominator = torch.bmm(v_sig_H, w_raw)
    
    w = (w_raw / denominator).squeeze(-1)
    
    return w

def compute_sinr(w, R_s, R_j, R_n):
    S = torch.real(torch.einsum('bi,bij,bj->b', torch.conj(w), R_s, w))
    NJ = torch.real(torch.einsum('bi,bij,bj->b', torch.conj(w), R_j + R_n, w))
    return 10 * torch.log10(S / torch.clamp(NJ, min=1e-12))

# --- SIREN Model ---
class Sine(nn.Module):
    def __init__(self, w0=30.0):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)

class SirenLayer(nn.Module):
    def __init__(self, in_features, out_features, w0=30.0, is_first=False):
        super().__init__()
        self.in_features = in_features
        self.is_first = is_first
        self.w0 = w0
        self.linear = nn.Linear(in_features, out_features)
        self.init_weights()
        self.activation = Sine(w0)

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                b = 1.0 / self.in_features
            else:
                b = np.sqrt(6.0 / self.in_features) / self.w0
            self.linear.weight.uniform_(-b, b)
            self.linear.bias.uniform_(-b, b)

    def forward(self, x):
        return self.activation(self.linear(x))

class FourierEncoding(nn.Module):
    def __init__(self, in_features=3, out_features=128, sigma=1.0):
        super().__init__()
        self.B = nn.Parameter(torch.randn(in_features, out_features // 2) * sigma, requires_grad=False)
        
    def forward(self, x):
        x_proj = 2.0 * np.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class SIRENCovariancePredictor(nn.Module):
    def __init__(self, w0=30.0, K_rank=3):
        super().__init__()
        self.K = K_rank
        self.fourier = FourierEncoding(in_features=3, out_features=128, sigma=1.0)
        self.net = nn.Sequential(
            SirenLayer(128, 256, w0=w0, is_first=True),
            SirenLayer(256, 256, w0=w0, is_first=False),
            SirenLayer(256, 128, w0=w0, is_first=False)
        )
        self.u_head = nn.Linear(128, 32 * K_rank * 2)
        
        with torch.no_grad():
            b = np.sqrt(6.0 / 128) / w0
            self.u_head.weight.uniform_(-b, b)
            self.u_head.bias.uniform_(-b, b)

    def forward(self, x):
        # x is unit direction vector (jammer XYZ)
        x = self.fourier(x)
        features = self.net(x)
        
        u_raw = self.u_head(features).view(-1, 32, self.K, 2)
        
        # We don't normalize the amplitude of U because we want the model to learn 
        # how much power to distribute to each mode.
        U = torch.complex(u_raw[..., 0], u_raw[..., 1])
        
        return U

# --- Main Script ---
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Loading mesh...")
    MESH_PATH = Path(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone_prepared.stl")
    mesh = load_uav_mesh(MESH_PATH)
    pos_body, normals_body = get_conformal_array_parametric(mesh, N=32)
    pos_lambda_np = pos_body / 0.15
    q_inv = euler_to_quaternion(np.deg2rad(15.0), 0, 0).conjugate()
    
    print("Loading cached 3D pilot dataset...")
    data = np.load("dataset_shadow_100k_polar_32el.npz")
    inputs = data['inputs']
    labels_polar = data['labels']
    mag = labels_polar[:, 0::2]
    phase = labels_polar[:, 1::2]
    g_exact_full = mag * np.exp(1j * phase)
    jam_bodies_full = inputs
    headings_full = np.rad2deg(np.arctan2(inputs[:, 1], inputs[:, 0]))
    headings_full = np.where(headings_full < 0, headings_full + 360.0, headings_full)
    N_POINTS = len(headings_full)
    P_S = 100.0
    P_J = 10000.0
    sigma2 = 1.0
    alpha = 390.0
    K_wave = 2.0 * np.pi / 0.15
    
    sig_world = np.array([1.0, 0.0, 0.0])
    sig_body = rotate_points(sig_world.reshape(1, 3), q_inv)[0]
    
    # NEW PHYSICS: Compute mask for signal
    g_sig = compute_shadow_mask_batched(mesh, pos_body, normals_body, sig_body.reshape(1, 3))[0]
    v_sig_ideal_np = get_steering_vector(pos_body, K_wave * sig_body)
    v_sig_masked_np = g_sig * v_sig_ideal_np
    
    # Filter dataset for Exclusion Zone
    valid_indices = []
    margin_weights_list = []
    
    for i, h in enumerate(headings_full):
        if h <= 10.0 or h >= 350.0: continue
        valid_indices.append(i)
        if h <= 30.0 or h >= 330.0: margin_weights_list.append(0.1)
        else: margin_weights_list.append(1.0)
            
    valid_indices = np.array(valid_indices)
    margin_weights_np = np.array(margin_weights_list)
    
    N_VALID = len(valid_indices)
    print(f"Filtered dataset from {N_POINTS} to {N_VALID} points outside exclusion zone.")
    
    jam_bodies = jam_bodies_full[valid_indices]
    g_exact = g_exact_full[valid_indices]
    
    # 20% validation split, evenly spaced
    val_indices = np.arange(0, N_VALID, 5) 
    train_indices = np.setdiff1d(np.arange(N_VALID), val_indices)
    
    pos_lambda = torch.tensor(pos_lambda_np, dtype=torch.float64, device=device)
    v_sig = torch.tensor(v_sig_masked_np, dtype=torch.complex128, device=device)
    jam_bodies_t = torch.tensor(jam_bodies, dtype=torch.float64, device=device)
    g_exact_t = torch.tensor(g_exact, dtype=torch.complex128, device=device)
    margin_weights_t = torch.tensor(margin_weights_np, dtype=torch.float64, device=device)
    
    theta_j_all = torch.acos(jam_bodies_t[:, 2] / torch.norm(jam_bodies_t, dim=1))
    phi_j_all = torch.atan2(jam_bodies_t[:, 1], jam_bodies_t[:, 0])
    v_true_all = g_exact_t * get_steering_vector_pt(pos_lambda, theta_j_all, phi_j_all)
    
    R_j_all = P_J * torch.einsum('bi,bj->bij', v_true_all, torch.conj(v_true_all))
    R_s_all = P_S * torch.einsum('i,j->ij', v_sig, torch.conj(v_sig)).unsqueeze(0).expand(N_VALID, -1, -1)
    R_n_all = sigma2 * torch.eye(32, dtype=torch.complex128, device=device).unsqueeze(0).expand(N_VALID, -1, -1)
    
    K_RANK = 8
    print(f"Instantiating SIRENCovariancePredictor with Rank-K = {K_RANK}")
    model = SIRENCovariancePredictor(w0=30.0, K_rank=K_RANK).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    EPOCHS = 75
    BATCH_SIZE = 360
    
    print("\n--- Starting Phase D-3 Rank-K Training ---")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        np.random.shuffle(train_indices)
        
        epoch_loss = 0.0
        
        for b in range(0, len(train_indices), BATCH_SIZE):
            idx = train_indices[b:b+BATCH_SIZE]
            
            xyz = jam_bodies_t[idx].float()
            
            # Continuous spatial jitter to prevent grid-collapse
            # Jitter by a small random rotation equivalent to ~0.05 degrees
            jitter = torch.randn_like(xyz) * 0.001
            xyz = F.normalize(xyz + jitter, p=2, dim=1)
            
            theta_j = theta_j_all[idx]
            phi_j = phi_j_all[idx]
            g_true = g_exact_t[idx]
            
            R_s = R_s_all[idx]
            R_j = R_j_all[idx]
            R_n = R_n_all[idx]
            m_weights = margin_weights_t[idx]
            
            optimizer.zero_grad()
            
            U = model(xyz).to(torch.complex128)
            
            w = d3_mvdr_beamformer(U, v_sig, P_J, sigma2, alpha)
            
            sinr = compute_sinr(w, R_s, R_j, R_n)
            
            # Hard-Negative Mining
            k = max(1, sinr.shape[0] // 10)
            worst_sinr_vals, worst_idx = torch.topk(-sinr, k)
            
            weights = torch.ones_like(sinr)
            weights[worst_idx] = 5.0
            
            # Apply soft margin weights
            weights = weights * m_weights
            
            loss_sinr = -(sinr * weights).mean()
            
            # No calibration loss for covariance matrix directly.
            loss = loss_sinr
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            
        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                xyz_val = jam_bodies_t[val_indices].float()
                U_val = model(xyz_val).to(torch.complex128)
                
                w_val = d3_mvdr_beamformer(U_val, v_sig, P_J, sigma2, alpha)
                sinr_val = compute_sinr(w_val, R_s_all[val_indices], R_j_all[val_indices], R_n_all[val_indices])
                
                min_sinr = sinr_val.min().item()
                avg_sinr = sinr_val.mean().item()
                
                # SVD Diagnostic for Effective Rank
                S = torch.linalg.svdvals(U_val)
                # S is [B, K] containing singular values sorted descending
                s_mean = S.mean(dim=0)
                # Calculate Condition Number (s_max / s_min)
                cond = (S[:, 0] / torch.clamp(S[:, -1], min=1e-8)).mean().item()
                
            svd_str = " | ".join([f"{s:.2f}" for s in s_mean.tolist()])
            print(f"Epoch {epoch:3d}/{EPOCHS} | Loss: {epoch_loss:.2f} | Val Min SINR: {min_sinr:.2f} dB | Avg SVD: [{svd_str}] | Mean Cond: {cond:.1f}")
            
    print("\nTraining Complete! Saving model...")
    torch.save(model.state_dict(), "siren_beamformer_d3_cov_K8_100k_physics.pt")

if __name__ == '__main__':
    main()
