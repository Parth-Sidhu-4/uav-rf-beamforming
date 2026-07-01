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
    pos_body, _ = get_conformal_array_parametric(mesh, N=32)
    pos_lambda = pos_body / 0.125
    
    results = {'A': [], 'B': [], 'C': [], 'D': [], 'C_in_seam': [], 'C_out_seam': [], 'time_in_seam': [], 'mode': []}
    
    t = np.linspace(0, T_END_SCORE, N_STEPS)
    
    import matplotlib.pyplot as plt
    
    seeds_to_run = [0, 5]
    for seed in seeds_to_run:
        np.random.seed(seed)
        mode = "Center-Seeking" if seed < 5 else "Seam-Hunting"
        print(f"\n--- Running Seed {seed} ({mode}) for Plotting ---")
        
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
        true_jam_v = physics.get_ground_truth(rel_jam_az, rel_jam_el) # (N_STEPS, 32)
        
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
            if 0.0 < w < 1.0: seam_mask[i] = True
        
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
            
        # Plotting
        t_plot = t[SCORE_START_STEP:]
        
        plt.figure(figsize=(12, 6))
        plt.plot(t_plot, sinr_a[SCORE_START_STEP:], label='Arm A (Baseline)', alpha=0.5, color='gray')
        plt.plot(t_plot, sinr_b[SCORE_START_STEP:], label='Arm B (Oracle 1.5s)', alpha=0.7, color='orange')
        plt.plot(t_plot, sinr_d[SCORE_START_STEP:], label='Arm D (Oracle Instant)', alpha=0.9, color='green', linestyle='--')
        plt.plot(t_plot, sinr_c[SCORE_START_STEP:], label='Arm C (AI+Fallback)', color='blue', linewidth=1.5)
        
        # Shade seam regions
        seam_starts = []
        seam_ends = []
        in_seam = False
        for idx, in_s in enumerate(seam_mask[SCORE_START_STEP:]):
            if in_s and not in_seam:
                seam_starts.append(t_plot[idx])
                in_seam = True
            elif not in_s and in_seam:
                seam_ends.append(t_plot[idx])
                in_seam = False
        if in_seam:
            seam_ends.append(t_plot[-1])
            
        for st, en in zip(seam_starts, seam_ends):
            plt.axvspan(st, en, color='red', alpha=0.1, lw=0)
            
        plt.axhline(15, color='red', linestyle=':', label='Link-Down Threshold (15dB)')
        plt.title(f"SINR Trajectory: {mode} Maneuver")
        plt.xlabel("Time (s)")
        plt.ylabel("Output SINR (dB)")
        plt.ylim(0, 30)
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = f"{mode.lower().replace('-', '_')}_plot.png"
        from pathlib import Path
        save_path = Path(r"C:\Users\parth\.gemini\antigravity\brain\0a845480-19bf-4d34-837c-274a6d3970c7") / filename
        plt.savefig(save_path, dpi=300)
        print(f"Saved plot to {save_path}")
        plt.close()

if __name__ == '__main__':
    main()
