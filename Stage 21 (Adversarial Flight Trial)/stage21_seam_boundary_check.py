import numpy as np
import torch
import time
import matplotlib.pyplot as plt
import os
import sys

# Paths setup
sys.path.append(os.path.abspath(r'D:\UAV Internship project\Stage 12 (Operational Analysis)'))
sys.path.append(os.path.abspath(r'D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)'))
sys.path.append(os.path.abspath(r'D:\UAV Internship project\Stage 13 (AI Beamforming)'))
sys.path.append(os.path.abspath(r'D:\UAV Internship project\Phase 2 Track 2\array_structures'))

from stage18w_train_100k_relu import ReluCovariancePredictor
from stage21_interpolate_test import phase_aware_bilinear_interpolate
from conformal_array import get_conformal_array_parametric
from mesh_loader import load_uav_mesh
from pathlib import Path
from attitude import euler_to_quaternion, rotate_points
from shadow_engine_batched import compute_shadow_mask_batched

def get_blend_weights(az_deg, el_deg):
    w_az = 0.0
    if 170.0 <= az_deg <= 190.0: w_az = 1.0
    elif 165.0 <= az_deg < 170.0: w_az = (az_deg - 165.0) / 5.0
    elif 190.0 < az_deg <= 195.0: w_az = (195.0 - az_deg) / 5.0
    
    w_el = 0.0
    if -25.0 <= el_deg <= 25.0: w_el = 1.0
    elif -30.0 <= el_deg < -25.0: w_el = (el_deg + 30.0) / 5.0
    elif 25.0 < el_deg <= 30.0: w_el = (30.0 - el_deg) / 5.0
        
    return w_az * w_el

def calc_sinr(w, v_sig, v_jam, P_S=100.0, P_J=10000.0, sigma2=1.0):
    sig_power = P_S * np.abs(np.vdot(w, v_sig))**2
    jam_power = P_J * np.abs(np.vdot(w, v_jam))**2
    noise_power = sigma2 * np.linalg.norm(w)**2
    return 10 * np.log10(sig_power / (jam_power + noise_power))

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Loading AI model...")
    ai_model = ReluCovariancePredictor(K_rank=5, hidden_dim=512, sigma=10.0).to(device)
    ai_model.load_state_dict(torch.load(r'D:\UAV Internship project\Stage 13 (AI Beamforming)\relu_beamformer_d3_cov_K5_100k_w512.pt', map_location=device))
    ai_model.eval()
    
    print("Loading physics...")
    mesh = load_uav_mesh(Path(r'D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone_prepared.stl'))
    pos_body, normals_body = get_conformal_array_parametric(mesh, N=32)
    pos_lambda = pos_body / 0.15
    
    print("Loading fallback table...")
    cache_path = Path(r"D:\UAV Internship project\Stage 21 (Adversarial Flight Trial)\true_raytraced_grid.npz")
    cache_data = np.load(cache_path)
    grid_az = cache_data['grid_h']
    grid_el = cache_data['grid_e']
    grid_weights = cache_data['v_true_grid']
    
    # 10 second trial to capture 2 full seam-hunting oscillations
    T_START = 0.0
    T_END = 10.0
    FS = 100
    t = np.arange(T_START, T_END, 1/FS)
    N_STEPS = len(t)
    
    np.random.seed(42) # Fixed seed for replicability
    
    print(f"Running {T_END}s Seam-Hunting trial...")
    
    # Randomize maneuvers
    roll_freq = np.random.uniform(0.05, 0.15)
    pitch_freq = np.random.uniform(0.02, 0.08)
    roll_amp = np.radians(np.random.uniform(20, 45))
    pitch_amp = np.radians(np.random.uniform(10, 25))
    
    drone_roll = np.sin(t * 2 * np.pi * roll_freq) * roll_amp
    drone_pitch = np.sin(t * 2 * np.pi * pitch_freq) * pitch_amp
    drone_yaw = np.zeros_like(t)
    
    rel_jam_az = np.zeros(N_STEPS)
    rel_jam_el = np.zeros(N_STEPS)
    
    for i in range(N_STEPS):
        target_az = 180.0 + 10.0 * np.sign(np.sin(t[i] * 2 * np.pi * 0.2)) # bang-bang between 170 and 190
        target_el = 25.0 * np.sign(np.cos(t[i] * 2 * np.pi * 0.15)) # +/- 25
        rel_jam_az[i] = np.radians(target_az + np.random.normal(0, 1.0))
        rel_jam_el[i] = np.radians(target_el + np.random.normal(0, 1.0))
        
    print("Generating continuous true ground truth (batched ray-tracing)...")
    true_jam_v = np.zeros((N_STEPS, 32), dtype=np.complex128)
    jam_body_all = np.zeros((N_STEPS, 3))
    
    for i in range(N_STEPS):
        dx = np.sin(np.pi/2 - rel_jam_el[i]) * np.cos(rel_jam_az[i])
        dy = np.sin(np.pi/2 - rel_jam_el[i]) * np.sin(rel_jam_az[i])
        dz = np.cos(np.pi/2 - rel_jam_el[i])
        jam_body_all[i] = [dx, dy, dz]
        
    chunk_size = 25
    t0 = time.time()
    for chunk_idx in range(0, N_STEPS, chunk_size):
        end_idx = min(chunk_idx + chunk_size, N_STEPS)
        jam_body_chunk = jam_body_all[chunk_idx:end_idx]
        g_batch = compute_shadow_mask_batched(mesh, pos_body, normals_body, jam_body_chunk)
        for j in range(end_idx - chunk_idx):
            idx = chunk_idx + j
            v_ideal = np.exp(1j * 2.0 * np.pi * (jam_body_chunk[j] @ pos_lambda.T))
            true_jam_v[idx] = g_batch[j] * v_ideal
            
    print(f"Finished continuous judge generation in {(time.time()-t0):.1f}s.")
    
    true_sig_v = np.zeros((N_STEPS, 32), dtype=np.complex128)
    sig_world = np.array([1.0, 0.0, 0.0])
    for i in range(N_STEPS):
        q_inv = euler_to_quaternion(drone_roll[i], drone_pitch[i], drone_yaw[i]).conjugate()
        sig_body = rotate_points(sig_world.reshape(1, 3), q_inv)[0]
        theta_s = np.arccos(np.clip(sig_body[2] / np.linalg.norm(sig_body), -1.0, 1.0))
        phi_s = np.arctan2(sig_body[1], sig_body[0])
        dx = np.sin(theta_s) * np.cos(phi_s)
        dy = np.sin(theta_s) * np.sin(phi_s)
        dz = np.cos(theta_s)
        d_s = np.array([dx, dy, dz])
        true_sig_v[i] = np.exp(1j * 2.0 * np.pi * (d_s @ pos_lambda.T))
        
    inputs = np.stack([
        np.cos(rel_jam_el) * np.cos(rel_jam_az),
        np.cos(rel_jam_el) * np.sin(rel_jam_az),
        np.sin(rel_jam_el)
    ], axis=1)
    X = torch.tensor(inputs, dtype=torch.float32, device=device)
    with torch.no_grad():
        U_ai = ai_model(X).to(torch.complex128).cpu().numpy()
        
    sinr_c = np.zeros(N_STEPS)
    w_blend_history = np.zeros(N_STEPS)
    
    P_S = 100.0
    P_J = 10000.0
    sigma2 = 1.0
    
    for i in range(N_STEPS):
        az_deg = np.degrees(rel_jam_az[i])
        el_deg = np.degrees(rel_jam_el[i])
        
        az_deg = az_deg % 360.0
        if az_deg < 0: az_deg += 360.0
        
        w_blend = get_blend_weights(az_deg, el_deg)
        w_blend_history[i] = w_blend
        
        # Covariance Blending (correct logic)
        Ui = U_ai[i]
        R_C_ai = P_J * (Ui @ Ui.conj().T)
        
        if w_blend > 0.0:
            v_fallback = phase_aware_bilinear_interpolate(np.array([az_deg]), np.array([el_deg]), grid_az, grid_el, grid_weights)[0]
            R_C_oracle = P_J * np.outer(v_fallback, v_fallback.conj())
            R_C_blend = (1 - w_blend) * R_C_ai + w_blend * R_C_oracle
        else:
            R_C_blend = R_C_ai
            
        R_C_final = R_C_blend + (sigma2 + 390.0) * np.eye(32)
        w_C = np.linalg.solve(R_C_final, true_sig_v[i])
        w_C = w_C / (true_sig_v[i].conj() @ w_C)
            
        sinr_c[i] = calc_sinr(w_C, true_sig_v[i], true_jam_v[i])
        
    # Save raw arrays for inspection
    np.savez('seam_boundary_data.npz', t=t, sinr_c=sinr_c, w_blend=w_blend_history)
    
    # Plotting
    plt.figure(figsize=(12, 6))
    
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    # Plot SINR
    ax1.plot(t, sinr_c, color='blue', label='Arm C SINR')
    ax1.axhline(y=15.0, color='red', linestyle='--', label='15 dB Threshold')
    
    # Highlight failures
    failures = sinr_c < 15.0
    # Use red scatter dots at y=15 for failures so it cannot possibly shade the whole plot incorrectly
    ax1.scatter(t[failures], [15.0]*np.sum(failures), color='red', zorder=5, label='Link Down (Points)')
    
    # Plot Blend Weight
    ax2.plot(t, w_blend_history, color='green', alpha=0.7, label='Blend Weight (w_blend)')
    ax2.set_ylabel('Blend Weight', color='green')
    ax2.set_ylim(-0.1, 1.1)
    
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('SINR (dB)', color='blue')
    ax1.set_ylim(0, 40)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.title('Arm C Seam-Hunting Temporal Analysis (SINR vs. Blend Weight)')
    plt.tight_layout()
    plt.savefig('seam_boundary_check.png', dpi=300)
    print("Plot saved to seam_boundary_check.png")

if __name__ == '__main__':
    main()
