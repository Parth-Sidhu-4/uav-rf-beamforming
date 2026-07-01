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
T_END_SCORE = 10.0
FS = 100
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
    
    results = {'A': [], 'B': [], 'C': [], 'D': [], 'C_in_seam': [], 'C_out_seam': [], 'time_in_seam': [], 'mode': []}
    
    t = np.linspace(0, T_END_SCORE, N_STEPS)
    
    for seed in [9]:
        np.random.seed(seed)
        mode = "Center-Seeking" if seed < 5 else "Seam-Hunting"
        results['mode'].append(mode)
        print(f"\n--- Running Seed {seed+1}/10 ({mode}) ---")
        
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
            if mode == "Center-Seeking":
                target_az = 180.0
                target_el = 0.0
            else: # Seam-Hunting
                # Oscillate around seam
                target_az = 180.0 + 10.0 * np.sign(np.sin(t[i] * 2 * np.pi * 0.2)) # bang-bang between 170 and 190
                target_el = 25.0 * np.sign(np.cos(t[i] * 2 * np.pi * 0.15)) # +/- 25
            
            # The jammer isn't perfectly instantaneous, add some noise or smooth tracking
            # But the adversary specifically chases geometry *relative* to the drone.
            rel_jam_az[i] = np.radians(target_az + np.random.normal(0, 1.0))
            rel_jam_el[i] = np.radians(target_el + np.random.normal(0, 1.0))
            
        # Ground Truth Evaluation for Jammer
        print("  Generating continuous true ground truth (batched ray-tracing)...")
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
            
            from shadow_engine_batched import compute_shadow_mask_batched
            g_batch = compute_shadow_mask_batched(mesh, pos_body, normals_body, jam_body_chunk)
            
            for j in range(end_idx - chunk_idx):
                idx = chunk_idx + j
                v_ideal = np.exp(1j * 2.0 * np.pi * (jam_body_chunk[j] @ pos_lambda.T))
                true_jam_v[idx] = g_batch[j] * v_ideal
                
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
        w_blend_history = np.zeros(N_STEPS)
        for i in range(N_STEPS):
            w = get_blend_weights(rel_jam_az_deg[i], rel_jam_el_deg[i])
            w_blend_history[i] = w
            if 0.0 < w < 1.0: seam_mask[i] = True
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
            
            # Arm B: Oracle (150 frame delay)
            if i < 150:
                w_B = w_A # warm-up
            else:
                idx_delay = i - 150
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
            
            w_blend = get_blend_weights(rel_jam_az_deg[i], rel_jam_el_deg[i])
            if w_blend > 0.0:
                v_fallback = physics.interpolate(np.array([rel_jam_az_deg[i]]), np.array([rel_jam_el_deg[i]]))[0]
                R_C_oracle = P_J * np.outer(v_fallback, v_fallback.conj())
                R_C_blend = (1 - w_blend) * R_C_ai + w_blend * R_C_oracle
            else:
                R_C_blend = R_C_ai
                
            R_C = R_C_blend + (sigma2 + 390.0) * np.eye(32)
            w_C = np.linalg.solve(R_C, true_sig_v[i])
            w_C = w_C / (true_sig_v[i].conj() @ w_C)
            
            # Evaluate all true SINRs
            R_true_sig = P_S * np.outer(true_sig_v[i], true_sig_v[i].conj())
            R_true_jam = P_J * np.outer(true_jam_v[i], true_jam_v[i].conj()) + sigma2 * np.eye(32)
            
            def calc_sinr(w):
                S = np.real(w.conj().T @ R_true_sig @ w)
                NJ = np.real(w.conj().T @ R_true_jam @ w)
                return 10 * np.log10(S / NJ)
                
            sinr_a[i] = calc_sinr(w_A)
            sinr_b[i] = calc_sinr(w_B)
            sinr_c[i] = calc_sinr(w_C)
            sinr_d[i] = calc_sinr(w_D)
            
        # Scoring
        score_a = sinr_a[SCORE_START_STEP:]
        score_b = sinr_b[SCORE_START_STEP:]
        score_c = sinr_c[SCORE_START_STEP:]
        score_d = sinr_d[SCORE_START_STEP:]
        
        # Plotting for diagnostic
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        
        t_score = t[SCORE_START_STEP:]
        
        ax1.plot(t_score, score_c, color='blue', label='Arm C SINR')
        ax1.axhline(y=15.0, color='red', linestyle='--', label='15 dB Threshold')
        
        failures = score_c < 15.0
        ax1.scatter(t_score[failures], [15.0]*np.sum(failures), color='red', zorder=5, label='Link Down (Points)')
        
        ax2.plot(t_score, w_blend_history[SCORE_START_STEP:], color='green', alpha=0.7, label='Blend Weight (w_blend)')
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
        break
        results['A'].append(np.sum(score_a < 15.0) / FS)
        results['B'].append(np.sum(score_b < 15.0) / FS)
        results['C'].append(np.sum(score_c < 15.0) / FS)
        results['D'].append(np.sum(score_d < 15.0) / FS)
        
        seam_mask_score = seam_mask[SCORE_START_STEP:]
        results['C_in_seam'].append(np.sum((score_c < 15.0) & seam_mask_score) / FS)
        results['C_out_seam'].append(np.sum((score_c < 15.0) & ~seam_mask_score) / FS)
        
        print(f"  [Seed {seed+1} Results] Arm A: {results['A'][-1]:.2f}s, Arm B: {results['B'][-1]:.2f}s, Arm C: {results['C'][-1]:.2f}s, Arm D: {results['D'][-1]:.2f}s")
        print(f"  [Seed {seed+1} Seam] In seam: {results['C_in_seam'][-1]:.2f}s, Out seam: {results['C_out_seam'][-1]:.2f}s")
        import json
        with open("stage21_intermediate_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
    print("\n======================================================")
    print(" RESULTS BREAKDOWN")
    print("======================================================")
    
    total_time = T_END_SCORE - T_START_SCORE
    
    modes = np.array(results['mode'])
    for m in ["Center-Seeking", "Seam-Hunting"]:
        idx = (modes == m)
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
