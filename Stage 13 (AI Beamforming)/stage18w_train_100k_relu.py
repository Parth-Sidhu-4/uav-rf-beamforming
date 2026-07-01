import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time



def get_steering_vector_pt(pos_lambda, theta, phi):
    dx = torch.sin(theta) * torch.cos(phi)
    dy = torch.sin(theta) * torch.sin(phi)
    dz = torch.cos(theta)
    d = torch.stack([dx, dy, dz], dim=-1)
    return torch.exp(1j * 2.0 * torch.pi * (d @ pos_lambda.T))

class FourierEncoding(nn.Module):
    def __init__(self, in_features=3, out_features=256, sigma=10.0):
        super().__init__()
        self.B = nn.Parameter(torch.randn(in_features, out_features // 2) * sigma, requires_grad=False)
    def forward(self, x):
        x_proj = 2.0 * np.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class ReluCovariancePredictor(nn.Module):
    def __init__(self, K_rank=5, hidden_dim=256, sigma=10.0):
        super().__init__()
        self.K = K_rank
        self.fourier = FourierEncoding(in_features=3, out_features=256, sigma=sigma)
        self.net = nn.Sequential(
            nn.Linear(256, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.u_head = nn.Linear(hidden_dim, 32 * K_rank * 2)
        
        with torch.no_grad():
            self.u_head.weight.uniform_(-0.01, 0.01)
            self.u_head.bias.uniform_(-0.01, 0.01)

    def forward(self, x):
        x = self.fourier(x)
        features = self.net(x)
        u_raw = self.u_head(features).view(-1, 32, self.K, 2)
        U = torch.complex(u_raw[..., 0], u_raw[..., 1])
        return U


def d3_mvdr_beamformer(U, v_sig, P_J, sigma2, alpha):
    B, N, K = U.shape
    R_j = P_J * torch.bmm(U, torch.conj(U.transpose(1, 2)))
    R_in = R_j + (sigma2 + alpha) * torch.eye(N, dtype=torch.complex128, device=U.device).unsqueeze(0)
    R_in_inv = torch.linalg.inv(R_in)
    v_sig_b = v_sig.unsqueeze(0).unsqueeze(-1).expand(B, N, 1)
    w_raw = torch.bmm(R_in_inv, v_sig_b)
    v_sig_H = torch.conj(v_sig).unsqueeze(0).unsqueeze(0).expand(B, 1, N)
    denominator = torch.bmm(v_sig_H, w_raw)
    return (w_raw / denominator).squeeze(-1)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Loading mesh physics...")
    import sys
    sys_path_save = sys.path.copy()
    import sys, os
    sys.path.append(os.path.abspath(r'D:\UAV Internship project\Stage 12 (Operational Analysis)'))
    sys.path.append(os.path.abspath(r'D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)'))
    from conformal_array import get_conformal_array_parametric
    from mesh_loader import load_uav_mesh
    from pathlib import Path
    from attitude import rotate_points, euler_to_quaternion
    mesh = load_uav_mesh(Path(r'D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone_prepared.stl'))
    pos_body, _ = get_conformal_array_parametric(mesh, N=32)
    pos_lambda_np = pos_body / 0.15
    q_inv = euler_to_quaternion(np.deg2rad(15.0), 0, 0).conjugate()
    sys.path = sys_path_save
    
    print("Loading true physics 100k dataset...")
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

    valid_indices = []
    margin_weights_list = []
    for i, h in enumerate(headings_full):
        if h <= 10.0 or h >= 350.0: continue
        valid_indices.append(i)
        if h <= 30.0 or h >= 330.0: margin_weights_list.append(0.1)
        else: margin_weights_list.append(1.0)
        
    valid_indices = np.array(valid_indices)
    margin_weights_np = np.array(margin_weights_list)
    print(f"Filtered dataset from {N_POINTS} to {len(valid_indices)} points outside exclusion zone.")
    
    g_exact_subset = g_exact_full[valid_indices]
    jam_bodies_subset = jam_bodies_full[valid_indices]
    
    pos_lambda = torch.tensor(pos_lambda_np, dtype=torch.float64, device=device)
    sig_world = np.array([1.0, 0.0, 0.0])
    sig_body = rotate_points(sig_world.reshape(1, 3), q_inv)[0]
    
    from stage15d_train_cartesian_32 import CartesianShadowNet32
    shadow_net = CartesianShadowNet32().to(device)
    shadow_net.load_state_dict(torch.load('shadow_net_cartesian_32.pt', map_location=device))
    shadow_net.eval()
    sig_t = torch.tensor(sig_body, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        out_sig = shadow_net(sig_t)
        g_sig_raw = (out_sig[:, 0::2] + 1j * out_sig[:, 1::2]).to(torch.complex128)
        g_sig = g_sig_raw / torch.clamp(torch.abs(g_sig_raw), min=1.0)
    theta_s = torch.acos(sig_t[:, 2] / torch.norm(sig_t, dim=1)).to(torch.float64)
    phi_s = torch.atan2(sig_t[:, 1], sig_t[:, 0]).to(torch.float64)
    v_sig_masked = g_sig * get_steering_vector_pt(pos_lambda, theta_s, phi_s)
    
    jam_t = torch.tensor(jam_bodies_subset, dtype=torch.float64, device=device)
    theta_j = torch.acos(jam_t[:, 2] / torch.norm(jam_t, dim=1))
    phi_j = torch.atan2(jam_t[:, 1], jam_t[:, 0])
    v_ideal = get_steering_vector_pt(pos_lambda, theta_j, phi_j)
    g_exact_t = torch.tensor(g_exact_subset, dtype=torch.complex128, device=device)
    v_true_all = g_exact_t * v_ideal
    
    R_sig = P_S * (v_sig_masked.T @ torch.conj(v_sig_masked))
    R_n = sigma2 * torch.eye(32, dtype=torch.complex128, device=device)
    R_sn = R_sig + R_n

    X_train_tensor = torch.tensor(jam_bodies_subset, dtype=torch.float32)
    v_true_cpu = v_true_all.cpu()
    margin_weights_tensor = torch.tensor(margin_weights_np, dtype=torch.float32)

    dataset = TensorDataset(X_train_tensor, v_true_cpu, margin_weights_tensor)
    
    torch.manual_seed(42)
    np.random.seed(42)
    N_valid = len(valid_indices)
    indices = np.random.permutation(N_valid)
    split = int(0.8 * N_valid)
    train_idx, val_idx = indices[:split], indices[split:]
    
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    
    EPOCHS = 75
    BATCH_SIZE = 1024
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    K_RANK = 5
    model = ReluCovariancePredictor(K_rank=K_RANK, hidden_dim=256).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=25, eta_min=1e-5)

    print(f"\n--- Starting ReLU Fourier Training ---")
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        
        for X_b, v_true_b, m_b in train_loader:
            X_b = X_b.to(device)
            v_true_b = v_true_b.to(device)
            m_b = m_b.to(device)
            
            optimizer.zero_grad()
            U = model(X_b)
            
            w = d3_mvdr_beamformer(U, v_sig_masked[0], P_J, sigma2, alpha)
            
            S = torch.real(torch.sum(torch.conj(w) * (R_sn @ w.T).T, dim=1))
            Pj_true = P_J * torch.abs(torch.sum(torch.conj(w) * v_true_b, dim=1))**2
            NJ = Pj_true + sigma2 * torch.real(torch.sum(torch.conj(w) * w, dim=1))
            
            sinr = 10 * torch.log10(S / torch.clamp(NJ, min=1e-12))
            
            loss = torch.mean(m_b * (-sinr))
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X_b.size(0)
            
        scheduler.step()
        
        if epoch % 10 == 1 or epoch == EPOCHS:
            model.eval()
            val_sinrs = []
            with torch.no_grad():
                for X_b, v_true_b, m_b in val_loader:
                    X_b = X_b.to(device)
                    v_true_b = v_true_b.to(device)
                    U = model(X_b)
                    
                    w = d3_mvdr_beamformer(U, v_sig_masked[0], P_J, sigma2, alpha)
                    
                    S = torch.real(torch.sum(torch.conj(w) * (R_sn @ w.T).T, dim=1))
                    Pj_true = P_J * torch.abs(torch.sum(torch.conj(w) * v_true_b, dim=1))**2
                    NJ = Pj_true + sigma2 * torch.real(torch.sum(torch.conj(w) * w, dim=1))
                    sinr = 10 * torch.log10(S / torch.clamp(NJ, min=1e-12))
                    val_sinrs.extend(sinr.cpu().numpy())
            
            print(f"Epoch {epoch:3d}/{EPOCHS} | Loss: {train_loss/len(train_dataset):.2f} | Val Min SINR: {np.min(val_sinrs):.2f} dB | Mean SINR: {np.mean(val_sinrs):.2f} dB")
            
    torch.save(model.state_dict(), 'relu_beamformer_d3_cov_K5_100k_sigma10.pt')
    print("Training Complete! Saved.")

if __name__ == '__main__':
    import sys
    main()
