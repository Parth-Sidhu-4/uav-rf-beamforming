import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(__file__))
import phase_b_beamforming as pbb
import task_b_jamming_threats as jts

def generate_trajectory(dy=800.0):
    x_vals = np.linspace(-5000, -2000, 75)
    y_vals = np.zeros_like(x_vals)
    path = list(zip(x_vals, y_vals))
    path.append((-2000, dy))
    x_vals_end = np.linspace(-1900, -750, 25)
    y_vals_end = np.zeros_like(x_vals_end)
    path.extend(list(zip(x_vals_end, y_vals_end)))
    return np.array(path)

def run_c2_beamforming_comparison():
    print("Running Task C2: Closed-Loop Null Steering Comparison...")
    jammer_pos = (15000.0, 0.0)
    target_pos = (0.0, 2000.0) # Offset target to avoid collinear spatial singularity with jammer
    
    # Array setup
    N = 8
    L = 100
    sigma_theta_rad = np.radians(0.16)
    
    path = generate_trajectory(800.0)
    N_steps = len(path)
    
    # EKF Setup
    x_est = np.array([10000.0, 1000.0, 0.0, 0.0])
    P_est = np.diag([1e8, 1e8, 1e4, 1e4])
    Q = np.diag([1.0, 1.0, 0.1, 0.1])
    R = np.array([[sigma_theta_rad**2]])
    F = np.eye(4); F[0,2]=1.0; F[1,3]=1.0
    xj_true, yj_true = jammer_pos
    
    # Storage for results
    null_depth_music = []
    null_depth_ekf = []
    null_depth_oracle = []
    
    np.random.seed(42)
    
    for i in range(N_steps):
        xu, yu = path[i]
        
        # True bearings
        theta_s_true = np.arctan2(target_pos[1] - yu, target_pos[0] - xu)
        theta_j_true = np.arctan2(yj_true - yu, xj_true - xu)
        
        # MUSIC instantaneously noisy bearing
        z_music = theta_j_true + np.random.normal(0, sigma_theta_rad)
        
        # EKF Filter Update
        x_pred = F @ x_est
        P_pred = F @ P_est @ F.T + Q
        xj_pred, yj_pred = x_pred[0], x_pred[1]
        dx = xj_pred - xu; dy = yj_pred - yu
        r2 = max(dx**2 + dy**2, 1e-6)
        H = np.array([[-dy/r2, dx/r2, 0.0, 0.0]])
        h_x = np.arctan2(dy, dx)
        y_res = (z_music - h_x + np.pi) % (2*np.pi) - np.pi
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        x_est = x_pred + K.flatten() * y_res
        I_KH = np.eye(4) - K @ H
        P_est = I_KH @ P_pred @ I_KH.T + K @ R @ K.T
        
        # EKF output bearing
        theta_j_ekf = np.arctan2(x_est[1] - yu, x_est[0] - xu)
        
        # Beamforming (LCMV) using identity covariance to isolate spatial nulling pattern
        R_xx_dummy = np.eye(N)
        # 1. Oracle (Perfect)
        w_oracle = pbb.lcmv_beamformer(R_xx_dummy, theta_s_true, [theta_j_true])
        # 2. MUSIC (Instantaneous)
        w_music = pbb.lcmv_beamformer(R_xx_dummy, theta_s_true, [z_music])
        # 3. EKF (Smoothed)
        w_ekf = pbb.lcmv_beamformer(R_xx_dummy, theta_s_true, [theta_j_ekf])
        
        # Evaluate true null depth
        a_j_true = pbb.ula_steering_vector(N, theta_j_true)
        
        depth_oracle = 10 * np.log10(np.abs(w_oracle.conj().T @ a_j_true)**2 + 1e-12)
        depth_music = 10 * np.log10(np.abs(w_music.conj().T @ a_j_true)**2 + 1e-12)
        depth_ekf = 10 * np.log10(np.abs(w_ekf.conj().T @ a_j_true)**2 + 1e-12)
        
        null_depth_oracle.append(depth_oracle)
        null_depth_music.append(depth_music)
        null_depth_ekf.append(depth_ekf)

    # Statistics (skipping first 10 steps for EKF convergence)
    start_idx = 10
    nd_o = np.array(null_depth_oracle[start_idx:])
    nd_m = np.array(null_depth_music[start_idx:])
    nd_e = np.array(null_depth_ekf[start_idx:])
    
    print("\\n=== NULL DEPTH STATISTICS ===")
    print(f"Oracle -> Mean: {np.mean(nd_o):.2f} dB | Var: {np.var(nd_o):.2f} dB^2")
    print(f"MUSIC  -> Mean: {np.mean(nd_m):.2f} dB | Var: {np.var(nd_m):.2f} dB^2")
    print(f"EKF    -> Mean: {np.mean(nd_e):.2f} dB | Var: {np.var(nd_e):.2f} dB^2")
    
    plt.figure(figsize=(12, 6))
    plt.plot(null_depth_music, 'r-', alpha=0.6, label='Case A: Instantaneous MUSIC')
    plt.plot(null_depth_ekf, 'b-', linewidth=2, label='Case B: EKF-Smoothed')
    plt.plot(null_depth_oracle, 'k--', linewidth=2, label='Case C: Perfect Oracle')
    plt.axvline(10, color='g', linestyle=':', label='EKF Convergence Point')
    plt.ylim(-100, 0)
    plt.xlabel('Mission Step')
    plt.ylabel('Null Depth at True Jammer Angle (dB)')
    plt.title('Adaptive LCMV Null Depth Comparison')
    plt.grid(True)
    plt.legend()
    plt.savefig('task_c2_null_depth.png', dpi=300)
    print("\\nSaved task_c2_null_depth.png")

if __name__ == "__main__":
    run_c2_beamforming_comparison()
