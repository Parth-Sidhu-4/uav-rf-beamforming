import torch, numpy as np, sys, os
from pathlib import Path
sys.path.append(os.path.abspath(r'D:\UAV Internship project\Phase 2 Track 2\array_structures'))
sys.path.append(os.path.abspath(r'D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)'))
from stage18w_train_100k_relu import ReluCovariancePredictor
from conformal_array import get_conformal_array_parametric
from mesh_loader import load_uav_mesh
from attitude import euler_to_quaternion, rotate_points

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ReluCovariancePredictor(K_rank=5, hidden_dim=512).to(device)
model.load_state_dict(torch.load('relu_beamformer_d3_cov_K5_100k_w512.pt', map_location=device))
model.eval()

mesh = load_uav_mesh(Path(r'D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone_prepared.stl'))
pos_body, normals_body = get_conformal_array_parametric(mesh, N=32)
pos_lambda = torch.tensor(pos_body / 0.15, dtype=torch.float64, device=device)

data = np.load('dataset_shadow_100k_polar_32el.npz')
inputs = data['inputs']
labels_polar = data['labels']
mag = labels_polar[:, 0::2]
phase = labels_polar[:, 1::2]
g_exact = mag * np.exp(1j * phase)

headings = np.rad2deg(np.arctan2(inputs[:, 1], inputs[:, 0]))
headings = np.where(headings < 0, headings + 360.0, headings)
elevations = np.rad2deg(np.arcsin(inputs[:, 2]))

mask = (headings >= 150) & (headings <= 210) & (elevations >= -25) & (elevations <= 25)
inputs_rear = inputs[mask]
g_exact_rear = g_exact[mask]
headings_rear = headings[mask]
elevations_rear = elevations[mask]

inputs_t = torch.tensor(inputs_rear, dtype=torch.float32, device=device)
with torch.no_grad():
    U = model(inputs_t).to(torch.complex128)

def get_steering_vector_pt(pos_lambda, theta, phi):
    dx = torch.sin(theta) * torch.cos(phi)
    dy = torch.sin(theta) * torch.sin(phi)
    dz = torch.cos(theta)
    d = torch.stack([dx, dy, dz], dim=-1)
    return torch.exp(1j * 2.0 * torch.pi * (d @ pos_lambda.T))

q_inv = euler_to_quaternion(np.deg2rad(15.0), 0, 0).conjugate()
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

theta_j = torch.acos(inputs_t[:, 2].to(torch.float64) / torch.norm(inputs_t.to(torch.float64), dim=1))
phi_j = torch.atan2(inputs_t[:, 1].to(torch.float64), inputs_t[:, 0].to(torch.float64))
v_ideal = get_steering_vector_pt(pos_lambda, theta_j, phi_j)
g_exact_t = torch.tensor(g_exact_rear, dtype=torch.complex128, device=device)
v_true = g_exact_t * v_ideal

sinrs = []
for i in range(len(inputs_rear)):
    Ui = U[i]
    R_in = 10000.0 * (Ui @ torch.conj(Ui.T)) + 390.0 * torch.eye(32, dtype=torch.complex128, device=device)
    R_in_inv = torch.linalg.inv(R_in)
    w = R_in_inv @ v_sig_masked[0]
    w = w / (torch.conj(v_sig_masked[0]) @ w)
    Ps = 100.0 * torch.abs(torch.conj(w) @ v_sig_masked[0])**2
    Pj = 10000.0 * torch.abs(torch.conj(w) @ v_true[i])**2
    Pn = torch.real(torch.conj(w) @ w)
    sinrs.append(10 * np.log10(float(Ps / (Pj + Pn))))

sinrs = np.array(sinrs)
below_15 = np.sum(sinrs < 15.0)
print(f'Total points in rear sector: {len(sinrs)}')
print(f'Below 15dB: {below_15} / {len(sinrs)} ({below_15/len(sinrs)*100:.1f}%)')
print(f'Min SINR: {np.min(sinrs):.2f} dB')
print(f'Mean SINR: {np.mean(sinrs):.2f} dB')

sinrs = np.array(sinrs)
below_15 = sinrs < 15.0

fail_headings = headings_rear[below_15]
fail_elevations = elevations_rear[below_15]
fail_sinrs = sinrs[below_15]

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
sc = plt.scatter(fail_headings, fail_elevations, c=fail_sinrs, cmap='Reds_r', s=10, alpha=0.7)
plt.colorbar(sc, label='SINR (dB)')
plt.xlim(150, 210)
plt.ylim(-25, 25)
plt.title('Distribution of <15dB SINR Failures (Sigma=10.0, Width=512)')
plt.xlabel('Heading (deg)')
plt.ylabel('Elevation (deg)')
plt.grid(True, alpha=0.3)
plt.savefig('relu_failures_distribution_w512.png', dpi=300)
print(f'Saved failure distribution to relu_failures_distribution_w512.png')

# Also print 2D histogram
hist, xedges, yedges = np.histogram2d(fail_headings, fail_elevations, bins=[6, 5], range=[[150, 210], [-25, 25]])
print('Failure Histogram (Headings 150-210, Elev -25 to 25):')
print(hist.T[::-1]) # T and [::-1] to match visual orientation (high elev at top)

