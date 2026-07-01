import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

sys.path.append(os.path.abspath(r'D:\UAV Internship project\Phase 2 Track 1\em_realism'))
sys.path.append(os.path.abspath(r'D:\UAV Internship project\Phase 2 Track 2\array_structures'))
sys.path.append(os.path.abspath(r'D:\UAV Internship project\Stage 12 (Operational Analysis)'))
sys.path.append(os.path.abspath(r'D:\UAV Internship project\Stage 13 (AI Beamforming)'))

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

def dcmvdr_beamformer_pytorch(theta_j, phi_j, pos_lambda, v_hat, v_sig, P_J, P_S, sigma2, alpha, K=7, delta_max=0.035):
    B = theta_j.shape[0]
    N = 32
    
    u_center = torch.stack([
        torch.sin(theta_j)*torch.cos(phi_j),
        torch.sin(theta_j)*torch.sin(phi_j),
        torch.cos(theta_j)
    ], dim=1)
    
    psi = torch.arange(K-1, dtype=theta_j.dtype, device=theta_j.device) * (2 * np.pi / (K-1))
    delta_vals = torch.zeros(K, dtype=theta_j.dtype, device=theta_j.device)
    delta_vals[1:] = delta_max
    psi = torch.cat((torch.tensor([0.0], dtype=theta_j.dtype, device=theta_j.device), psi))
    
    R_interf = torch.zeros((B, N, N), dtype=torch.complex128, device=theta_j.device)
    
    for k in range(K):
        th_k = theta_j + delta_vals[k] * torch.cos(psi[k])
        ph_k = phi_j + delta_vals[k] * torch.sin(psi[k])
        
        u_k = torch.stack([
            torch.sin(th_k)*torch.cos(ph_k),
            torch.sin(th_k)*torch.sin(ph_k),
            torch.cos(th_k)
        ], dim=1)
        
        phase_diff = 2j * np.pi * torch.matmul(pos_lambda, (u_k - u_center).unsqueeze(-1)).squeeze(-1)
        v_k = v_hat * torch.exp(phase_diff)
        
        R_interf += (P_J / K) * torch.einsum('bi,bj->bij', v_k, torch.conj(v_k))
        
    R_hat = R_interf + (sigma2 + alpha) * torch.eye(N, dtype=torch.complex128, device=theta_j.device).unsqueeze(0)
    R_hat_inv = torch.linalg.inv(R_hat)
    
    du_dth = torch.stack([
        torch.cos(theta_j)*torch.cos(phi_j),
        torch.cos(theta_j)*torch.sin(phi_j),
        -torch.sin(theta_j)
    ], dim=1)
    
    du_dph = torch.stack([
        -torch.sin(theta_j)*torch.sin(phi_j),
        torch.sin(theta_j)*torch.cos(phi_j),
        torch.zeros_like(theta_j)
    ], dim=1)
    
    dv_j_dth = 2j * np.pi * torch.matmul(pos_lambda, du_dth.unsqueeze(-1)).squeeze(-1) * v_hat
    dv_j_dph = 2j * np.pi * torch.matmul(pos_lambda, du_dph.unsqueeze(-1)).squeeze(-1) * v_hat
    
    v_sig_b = v_sig.unsqueeze(0).expand(B, -1)
    C = torch.stack([v_sig_b, v_hat, dv_j_dth, dv_j_dph], dim=2)
    f = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.complex128, device=theta_j.device).unsqueeze(0).expand(B, -1).unsqueeze(-1)
    
    R_inv_C = torch.bmm(R_hat_inv, C)
    CRC = torch.bmm(torch.conj(C.transpose(1,2)), R_inv_C)
    CRC += 1e-10 * torch.eye(4, dtype=torch.complex128, device=theta_j.device).unsqueeze(0)
    
    CRC_inv = torch.linalg.inv(CRC)
    w = torch.bmm(R_inv_C, torch.bmm(CRC_inv, f)).squeeze(-1)
    
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

class SIRENShadowMaskPredictor(nn.Module):
    def __init__(self, w0=30.0):
        super().__init__()
        self.fourier = FourierEncoding(in_features=3, out_features=128, sigma=1.0)
        self.net = nn.Sequential(
            SirenLayer(128, 256, w0=w0, is_first=True),
            SirenLayer(256, 256, w0=w0, is_first=False),
            SirenLayer(256, 128, w0=w0, is_first=False)
        )
        self.amp_head = nn.Linear(128, 32)
        self.phase_head = nn.Linear(128, 64)
        self.unc_head = nn.Linear(128, 1)
        
        with torch.no_grad():
            b = np.sqrt(6.0 / 128) / w0
            self.amp_head.weight.uniform_(-b, b)
            self.amp_head.bias.uniform_(-b, b)
            self.phase_head.weight.uniform_(-b, b)
            self.phase_head.bias.uniform_(-b, b)
            self.unc_head.weight.uniform_(-b, b)
            self.unc_head.bias.uniform_(-b, b)

    def forward(self, x):
        x = self.fourier(x)
        features = self.net(x)
        
        amp = F.softplus(self.amp_head(features))
        
        phase_raw = self.phase_head(features).view(-1, 32, 2)
        phase_norm = F.normalize(phase_raw, p=2, dim=-1)
        
        g_hat = torch.complex(amp * phase_norm[..., 0], amp * phase_norm[..., 1])
        sigma_v = F.softplus(self.unc_head(features)).squeeze(-1)
        
        return g_hat, sigma_v

# --- Main Script ---
def main():
    device = torch.device('cpu')
    
    print("Loading mesh...")
    MESH_PATH = Path(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone_prepared.stl")
    mesh = load_uav_mesh(MESH_PATH)
    pos_body, normals_body = get_conformal_array_parametric(mesh, N=32)
    pos_lambda_np = pos_body / 0.15
    
    dataset_path = "dataset_3600_masks.npz"
    N_POINTS = 3600
    headings = np.linspace(0, 360, N_POINTS, endpoint=False)
    jam_bodies = np.zeros((N_POINTS, 3))
    q_inv = euler_to_quaternion(np.deg2rad(15.0), 0, 0).conjugate()
    for i, h in enumerate(headings):
        jam_world = np.array([np.cos(np.deg2rad(h)), np.sin(np.deg2rad(h)), 0.0])
        jam_bodies[i] = rotate_points(jam_world.reshape(1, 3), q_inv)[0]
        
    if os.path.exists(dataset_path):
        print("Loading cached 3600-point dataset...")
        data = np.load(dataset_path)
        g_exact = data['g_exact']
    else:
        print(f"Pre-computing exact physics for {N_POINTS} points. This will take ~5-10 minutes...")
        g_exact = np.zeros((N_POINTS, 32), dtype=np.complex128)
        chunk_size = 36
        for i in range(0, N_POINTS, chunk_size):
            chunk = jam_bodies[i:i+chunk_size]
            g_exact[i:i+chunk_size] = compute_shadow_mask_batched(mesh, pos_body, normals_body, chunk)
            if (i+chunk_size) % 360 == 0:
                print(f"  {i+chunk_size}/{N_POINTS} done")
        np.savez(dataset_path, g_exact=g_exact)
        print("Dataset cached.")
        
    val_indices = np.arange(0, N_POINTS, 5) # 720 points, 0.5 deg resolution
    train_indices = np.setdiff1d(np.arange(N_POINTS), val_indices) # 2880 points
    
    P_S = 100.0
    P_J = 10000.0
    sigma2 = 1.0
    alpha = 390.0
    K_wave = 2.0 * np.pi / 0.15
    
    sig_world = np.array([1.0, 0.0, 0.0])
    sig_body = rotate_points(sig_world.reshape(1, 3), q_inv)[0]
    v_sig_np = get_steering_vector(pos_body, K_wave * sig_body)
    
    pos_lambda = torch.tensor(pos_lambda_np, dtype=torch.float64, device=device)
    v_sig = torch.tensor(v_sig_np, dtype=torch.complex128, device=device)
    jam_bodies_t = torch.tensor(jam_bodies, dtype=torch.float64, device=device)
    g_exact_t = torch.tensor(g_exact, dtype=torch.complex128, device=device)
    
    theta_j_all = torch.acos(jam_bodies_t[:, 2] / torch.norm(jam_bodies_t, dim=1))
    phi_j_all = torch.atan2(jam_bodies_t[:, 1], jam_bodies_t[:, 0])
    v_true_all = g_exact_t * get_steering_vector_pt(pos_lambda, theta_j_all, phi_j_all)
    
    R_j_all = P_J * torch.einsum('bi,bj->bij', v_true_all, torch.conj(v_true_all))
    R_s_all = P_S * torch.einsum('i,j->ij', v_sig, torch.conj(v_sig)).unsqueeze(0).expand(N_POINTS, -1, -1)
    R_n_all = sigma2 * torch.eye(32, dtype=torch.complex128, device=device).unsqueeze(0).expand(N_POINTS, -1, -1)
    
    model = SIRENShadowMaskPredictor(w0=30.0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    EPOCHS = 300
    BATCH_SIZE = 360
    
    print("\n--- Starting Checkpoint 3 Training ---")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        np.random.shuffle(train_indices)
        
        epoch_loss = 0.0
        
        for b in range(0, len(train_indices), BATCH_SIZE):
            idx = train_indices[b:b+BATCH_SIZE]
            
            xyz = jam_bodies_t[idx].float()
            theta_j = theta_j_all[idx]
            phi_j = phi_j_all[idx]
            g_true = g_exact_t[idx]
            
            R_s = R_s_all[idx]
            R_j = R_j_all[idx]
            R_n = R_n_all[idx]
            
            optimizer.zero_grad()
            
            g_hat, sigma_v_pred = model(xyz)
            g_hat = g_hat.to(torch.complex128)
            
            v_hat = g_hat * get_steering_vector_pt(pos_lambda, theta_j, phi_j)
            
            # Physics Engine in loop
            w = dcmvdr_beamformer_pytorch(theta_j, phi_j, pos_lambda, v_hat, v_sig, P_J, P_S, sigma2, alpha)
            
            sinr = compute_sinr(w, R_s, R_j, R_n)
            
            # Hard-Negative Mining
            k = max(1, sinr.shape[0] // 10)
            worst_sinr_vals, worst_idx = torch.topk(-sinr, k)
            
            weights = torch.ones_like(sinr)
            weights[worst_idx] = 5.0
            
            loss_sinr = -(sinr * weights).mean()
            
            # Calibration loss
            g_err_norm = torch.norm(g_hat - g_true, p=2, dim=1).float()
            loss_calib = torch.mean((sigma_v_pred - g_err_norm)**2)
            
            loss = loss_sinr + loss_calib
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            
        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                xyz_val = jam_bodies_t[val_indices].float()
                g_hat_val, _ = model(xyz_val)
                g_hat_val = g_hat_val.to(torch.complex128)
                
                theta_j_val = theta_j_all[val_indices]
                phi_j_val = phi_j_all[val_indices]
                v_hat_val = g_hat_val * get_steering_vector_pt(pos_lambda, theta_j_val, phi_j_val)
                
                w_val = dcmvdr_beamformer_pytorch(theta_j_val, phi_j_val, pos_lambda, v_hat_val, v_sig, P_J, P_S, sigma2, alpha)
                sinr_val = compute_sinr(w_val, R_s_all[val_indices], R_j_all[val_indices], R_n_all[val_indices])
                
                min_sinr = sinr_val.min().item()
                avg_sinr = sinr_val.mean().item()
                
            print(f"Epoch {epoch:3d}/{EPOCHS} | Loss: {epoch_loss:.2f} | Val Min SINR: {min_sinr:.2f} dB | Val Avg SINR: {avg_sinr:.2f} dB")
            
    print("\nTraining Complete!")

if __name__ == '__main__':
    main()
