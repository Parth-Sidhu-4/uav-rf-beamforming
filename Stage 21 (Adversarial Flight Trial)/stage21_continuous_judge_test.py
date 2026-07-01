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

from stage18w_train_100k_relu import ReluCovariancePredictor
from stage21_interpolate_test import phase_aware_bilinear_interpolate

# Global configuration
T_START_SCORE = 1.5
T_END_SCORE = 60.0
FS = 10
N_STEPS = int(T_END_SCORE * FS)
SCORE_START_STEP = int(T_START_SCORE * FS)

class FallbackCache:
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.grid_h = data['grid_h']
        self.grid_e = data['grid_e']
        self.v_grid = data['v_true_grid']
        self.v_grid[np.isnan(self.v_grid)] = 1.0 + 0j

    def get_ground_truth(self, h_rad, e_rad):
        # Using nearest neighbor for ground truth evaluation (since 0.5 deg is dense enough)
        h_deg = np.rad2deg(h_rad)
        e_deg = np.rad2deg(e_rad)
        
        idx_h = np.abs(self.grid_h[:, None] - h_deg[None, :]).argmin(axis=0)
        idx_e = np.abs(self.grid_e[:, None] - e_deg[None, :]).argmin(axis=0)
        
        return self.v_grid[idx_h, idx_e]
        
    def interpolate(self, h_deg, e_deg):
        # phase-aware fallback for AI
        return phase_aware_bilinear_interpolate(h_deg, e_deg, self.grid_h, self.grid_e, self.v_grid)

def get_blend_weights(h, e):
    # h, e in degrees
    d_h = np.maximum(0, np.maximum(170 - h, h - 190))
    d_e = np.maximum(0, np.maximum(-25 - e, e - 25))
    d = np.maximum(d_h, d_e)
    w_ai = np.clip(d / 2.0, 0.0, 1.0)
    return 1.0 - w_ai

def main():
    print("======================================================")
    print(" STAGE 21: ADVERSARIAL CLOSED-LOOP FLIGHT TRIAL")
    print("======================================================")
    
    physics = FallbackCache('true_raytraced_grid.npz')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ai_model = ReluCovariancePredictor(K_rank=5, hidden_dim=512).to(device)
    ai_model.load_state_dict(torch.load(r'D:\UAV Internship project\Stage 13 (AI Beamforming)\relu_beamformer_d3_cov_K5_100k_w512.pt', map_location=device))
    ai_model.eval()
    
    # 32 elements array position setup
    sys.path.append(os.path.abspath(r'D:\UAV Internship project\Phase 2 Track 2\array_structures'))
    from conformal_array import get_conformal_array_parametric
    from mesh_loader import load_uav_mesh
    from pathlib import Path
    mesh = load_uav_mesh(Path(r'D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone_prepared.stl'))
    pos_body, normals_body = get_conformal_array_parametric(mesh, N=32)
    pos_lambda = pos_body / 0.125
    
    from shadow_engine_batched import compute_shadow_mask_batched
    
    results = {'A': [], 'B': [], 'C': [], 'D': [], 'C_in_seam': [], 'C_out_seam': [], 'time_in_seam': [], 'mode': [], 'C_min_sinr': []}
    
    t = np.linspace(0, T_END_SCORE, N_STEPS)
    
    seeds_to_run = [0, 5]
    for seed in seeds_to_run:
        np.random.seed(seed)
        
        if seed == 0:
            mode = "Center-Seeking (180, 0)"
            target_az_base = 180.0
            target_el_base = 0.0
        else:
            mode = "Off-Axis Pocket (175, 5)"
            target_az_base = 175.0
            target_el_base = 5.0
            
        results['mode'].append(mode)
        print(f"\n--- Running Seed {seed} ({mode}) with Continuous Judge ---")
        
        # Randomize maneuvers
        roll_freq = np.random.uniform(0.05, 0.15)
        pitch_freq = np.random.uniform(0.02, 0.08)
        roll_amp = np.radians(np.random.uniform(20, 45))
        pitch_amp = np.radians(np.random.uniform(10, 25))
        
        drone_roll = np.sin(t * 2 * np.pi * roll_freq) * roll_amp
        drone_pitch = np.sin(t * 2 * np.pi * pitch_freq) * pitch_amp
        drone_yaw = np.zeros_like(t) # Yaw doesn't matter for relative angles
        
        # Jammer trajectory
        rel_jam_az = np.zeros(N_STEPS)
        rel_jam_el = np.zeros(N_STEPS)
        
        for i in range(N_STEPS):
            rel_jam_az[i] = np.radians(target_az_base + np.random.normal(0, 1.0))
            rel_jam_el[i] = np.radians(target_el_base + np.random.normal(0, 1.0))
            
        # Ground Truth Evaluation for Jammer (CONTINUOUS RAY-TRACING)
        print("  Generating continuous true ground truth (batched ray-tracing)...")
        true_jam_v = np.zeros((N_STEPS, 32), dtype=np.complex128)
        
        # Calculate jam_body batch
        jam_body_all = np.zeros((N_STEPS, 3))
        for i in range(N_STEPS):
            az = rel_jam_az[i]
            el = rel_jam_el[i]
            dx = np.sin(np.pi/2 - el) * np.cos(az)
            dy = np.sin(np.pi/2 - el) * np.sin(az)
            dz = np.cos(np.pi/2 - el)
            jam_body_all[i] = [dx, dy, dz]
            
        # Chunked ray-tracing
        chunk_size = 25
        import time
        t0 = time.time()
        for chunk_idx in range(0, N_STEPS, chunk_size):
            jam_body_chunk = jam_body_all[chunk_idx:chunk_idx+chunk_size]
            g_batch = compute_shadow_mask_batched(mesh, pos_body, normals_body, jam_body_chunk)
            
            # Multiply by ideal phase
            for j in range(len(jam_body_chunk)):
                idx = chunk_idx + j
                v_ideal = np.exp(1j * 2.0 * np.pi * (jam_body_chunk[j] @ pos_lambda.T))
                true_jam_v[idx] = g_batch[j] * v_ideal
                
            if (chunk_idx % 1000) == 0:
                print(f"    Completed {chunk_idx}/{N_STEPS} frames ({(time.time()-t0):.1f}s)")
        print(f"  Finished continuous judge generation in {(time.time()-t0):.1f}s.")
        
        # Ground Truth Evaluation for Signal
        true_sig_v = np.zeros((N_STEPS, 32), dtype=np.complex128)
        sig_world = np.array([1.0, 0.0, 0.0])
        from attitude import euler_to_quaternion, rotate_points
        for i in range(N_STEPS):
            # Compute signal body vector
            q_inv = euler_to_quaternion(drone_roll[i], drone_pitch[i], drone_yaw[i]).conjugate()
            sig_body = rotate_points(sig_world.reshape(1, 3), q_inv)[0]
            
            theta_s = np.arccos(np.clip(sig_body[2] / np.linalg.norm(sig_body), -1.0, 1.0))
            phi_s = np.arctan2(sig_body[1], sig_body[0])
            
            dx = np.sin(theta_s) * np.cos(phi_s)
            dy = np.sin(theta_s) * np.sin(phi_s)
            dz = np.cos(theta_s)
            d_s = np.array([dx, dy, dz])
            true_sig_v[i] = np.exp(1j * 2.0 * np.pi * (d_s @ pos_lambda.T))
        
        # Time in seam tracking
        seam_mask = np.zeros(N_STEPS, dtype=bool)
        rel_jam_az_deg = np.degrees(rel_jam_az)
        rel_jam_el_deg = np.degrees(rel_jam_el)
        for i in range(N_STEPS):
            w = get_blend_weights(rel_jam_az_deg[i], rel_jam_el_deg[i])
            if w > 0.0: seam_mask[i] = True
        time_in_seam = np.sum(seam_mask[SCORE_START_STEP:]) / FS
        results['time_in_seam'].append(time_in_seam)
        
        # AI Input Prep
        inputs = np.stack([
            np.cos(rel_jam_el) * np.cos(rel_jam_az),
            np.cos(rel_jam_el) * np.sin(rel_jam_az),
            np.sin(rel_jam_el)
        ], axis=1)
        X = torch.tensor(inputs, dtype=torch.float32, device=device)
        with torch.no_grad():
            U_ai = ai_model(X).to(torch.complex128).cpu().numpy()
            
        sinr_a = np.zeros(N_STEPS)
        sinr_b = np.zeros(N_STEPS)
        sinr_c = np.zeros(N_STEPS)
        sinr_d = np.zeros(N_STEPS)
        
        P_S = 100.0
        P_J = 10000.0
        sigma2 = 1.0
        
        for i in range(N_STEPS):
            # Arm A: Free Space Baseline
            # v_ideal for jammer
            dx = np.sin(np.pi/2 - rel_jam_el[i]) * np.cos(rel_jam_az[i])
            dy = np.sin(np.pi/2 - rel_jam_el[i]) * np.sin(rel_jam_az[i])
            dz = np.cos(np.pi/2 - rel_jam_el[i])
            d = np.array([dx, dy, dz])
            v_jam_ideal = np.exp(1j * 2.0 * np.pi * (d @ pos_lambda.T))
            
            R_A = P_J * np.outer(v_jam_ideal, v_jam_ideal.conj()) + (sigma2 + 390.0) * np.eye(32)
            w_A = np.linalg.solve(R_A, true_sig_v[i])
            w_A = w_A / (true_sig_v[i].conj() @ w_A)
            
            # Arm B: Oracle (1.5s delay)
            delay_frames = int(1.5 * FS)
            if i < delay_frames:
                w_B = w_A # warm-up
            else:
                idx_delay = i - delay_frames
                v_oracle = true_jam_v[idx_delay]
                R_B = P_J * np.outer(v_oracle, v_oracle.conj()) + (sigma2 + 390.0) * np.eye(32)
                w_B = np.linalg.solve(R_B, true_sig_v[i]) # still aim at true current signal
                w_B = w_B / (true_sig_v[i].conj() @ w_B)
                
            # Arm D: Oracle-Instant (0 frame delay)
            v_oracle_instant = true_jam_v[i]
            R_D = P_J * np.outer(v_oracle_instant, v_oracle_instant.conj()) + (sigma2 + 390.0) * np.eye(32)
            w_D = np.linalg.solve(R_D, true_sig_v[i])
            w_D = w_D / (true_sig_v[i].conj() @ w_D)
                
            # Arm C: AI Surrogate with Fallback (0 frame delay)
            Ui = U_ai[i]
            R_C_ai = P_J * (Ui @ Ui.conj().T)
            
            in_seam = seam_mask[i]
            if in_seam:
                v_fallback = physics.interpolate(np.array([rel_jam_az_deg[i]]), np.array([rel_jam_el_deg[i]]))[0]
                R_C_fallback = P_J * np.outer(v_fallback, v_fallback.conj())
                
                blend_w = get_blend_weights(rel_jam_az_deg[i], rel_jam_el_deg[i])
                R_C_blend = blend_w * R_C_fallback + (1 - blend_w) * R_C_ai
            else:
                R_C_blend = R_C_ai
                
            R_C = R_C_blend + (sigma2 + 390.0) * np.eye(32)
            w_C = np.linalg.solve(R_C, true_sig_v[i])
            w_C = w_C / (true_sig_v[i].conj() @ w_C)
            
            # Evaluate all true SINRs
            R_true_sig = P_S * np.outer(true_sig_v[i], true_sig_v[i].conj())
            R_true_jam = P_J * np.outer(true_jam_v[i], true_jam_v[i].conj()) + sigma2 * np.eye(32)
            
            def calc_sinr_debug(w, name=""):
                S = np.real(w.conj().T @ R_true_sig @ w)
                L = np.real(w.conj().T @ (R_true_jam - sigma2 * np.eye(32)) @ w)
                N = np.real(w.conj().T @ (sigma2 * np.eye(32)) @ w)
                NJ = L + N
                sinr_db = 10 * np.log10(S / NJ)
                return sinr_db
                
            sinr_a[i] = calc_sinr_debug(w_A, 'A')
            sinr_b[i] = calc_sinr_debug(w_B, 'B')
            sinr_c[i] = calc_sinr_debug(w_C, 'C')
            sinr_d[i] = calc_sinr_debug(w_D, 'D')
            
            if i < 5:
                v_fallback_true = physics.get_ground_truth(np.array([rel_jam_az[i]]), np.array([rel_jam_el[i]]))[0]
                R_true_fallback = P_J * np.outer(v_fallback_true, v_fallback_true.conj()) + (sigma2 + 390.0) * np.eye(32)
                w_C_true = np.linalg.solve(R_true_fallback, true_sig_v[i])
                w_C_true = w_C_true / (true_sig_v[i].conj() @ w_C_true)
                L_true = np.real(w_C_true.conj().T @ (R_true_jam - sigma2 * np.eye(32)) @ w_C_true)
                print(f"  Frame {i}: Jitter: {rel_jam_az_deg[i] - 180.0:.2f} deg, SINR_C: {sinr_c[i]:.2f} dB, L: {np.real(w_C.conj().T @ (R_true_jam - sigma2 * np.eye(32)) @ w_C):.2f}, L_true: {L_true:.2f}")
            
            if i < 5:
                print(f"  Frame {i}: Jitter: {rel_jam_az_deg[i] - 180.0:.2f} deg, SINR_C: {sinr_c[i]:.2f} dB, SINR_D: {sinr_d[i]:.2f} dB")
            
        # Scoring
        score_a = sinr_a[SCORE_START_STEP:]
        score_b = sinr_b[SCORE_START_STEP:]
        score_c = sinr_c[SCORE_START_STEP:]
        score_d = sinr_d[SCORE_START_STEP:]
        
        results['A'].append(np.sum(score_a < 15.0) / FS)
        results['B'].append(np.sum(score_b < 15.0) / FS)
        results['C'].append(np.sum(score_c < 15.0) / FS)
        results['D'].append(np.sum(score_d < 15.0) / FS)
        
        seam_mask_score = seam_mask[SCORE_START_STEP:]
        results['C_in_seam'].append(np.sum((score_c < 15.0) & seam_mask_score) / FS)
        results['C_out_seam'].append(np.sum((score_c < 15.0) & ~seam_mask_score) / FS)
        results['C_min_sinr'].append(np.min(sinr_c[SCORE_START_STEP:]))
        
    print("\n======================================================")
    print(" RESULTS BREAKDOWN")
    print("======================================================")
    
    total_time = T_END_SCORE - T_START_SCORE
    
    modes = np.array(results['mode'])
    for m in ["Center-Seeking (180, 0)", "Off-Axis Pocket (175, 5)", "Off-Axis Pocket (185, -5)"]:
        idx = (modes == m)
        if not np.any(idx): continue
        print(f"\n--- {m.upper()} MANEUVERS ---")
        print(f"Arm A (Baseline)      : Mean {np.mean(np.array(results['A'])[idx]):.2f}s | Worst {np.max(np.array(results['A'])[idx]):.2f}s")
        print(f"Arm B (Oracle 1.5s)   : Mean {np.mean(np.array(results['B'])[idx]):.2f}s | Worst {np.max(np.array(results['B'])[idx]):.2f}s")
        print(f"Arm D (Oracle Instant): Mean {np.mean(np.array(results['D'])[idx]):.2f}s | Worst {np.max(np.array(results['D'])[idx]):.2f}s")
        print(f"Arm C (AI+Fallback)   : Mean {np.mean(np.array(results['C'])[idx]):.2f}s | Worst {np.max(np.array(results['C'])[idx]):.2f}s")
        
        m_seam = np.mean(np.array(results['time_in_seam'])[idx])
        m_c_in = np.mean(np.array(results['C_in_seam'])[idx])
        m_c_out = np.mean(np.array(results['C_out_seam'])[idx])
        
        print(f"  -> Avg Time in Seam: {m_seam:.2f}s")
        print(f"  -> Arm C failures inside seam:  {m_c_in:.2f}s")
        print(f"  -> Arm C failures outside seam: {m_c_out:.2f}s")
        m_sinr = np.mean(np.array(results['C_min_sinr'])[idx])
        print(f"  -> Arm C Min SINR (dB): {m_sinr:.2f} dB")

    print("\n--- GLOBAL AGGREGATE ---")
    for arm, res in [('Arm A (Baseline)', results['A']), 
                     ('Arm B (Oracle 1.5s)', results['B']), 
                     ('Arm D (Oracle Instant)', results['D']),
                     ('Arm C (AI+Fallback)', results['C'])]:
        print(f"{arm}:")
        print(f"  Mean Link-Down: {np.mean(res):.2f}s ({(np.mean(res)/total_time)*100:.1f}%)")
        print(f"  Worst Case:     {np.max(res):.2f}s ({(np.max(res)/total_time)*100:.1f}%)")
        
    mean_seam = np.mean(results['time_in_seam'])
    mean_c_in = np.mean(results['C_in_seam'])
    mean_c_out = np.mean(results['C_out_seam'])
    print(f"\nAverage Time in Seam Region: {mean_seam:.2f}s")
    print(f"Arm C Average Link-Down IN seam:  {mean_c_in:.2f}s")
    print(f"Arm C Average Link-Down OUT seam: {mean_c_out:.2f}s")
    
if __name__ == '__main__':
    main()
