import sys
import os
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Phase 2 Track 1', 'em_realism')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Phase 2 Track 2', 'array_structures')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Stage 12 (Operational Analysis)')))

from mesh_loader import load_uav_mesh
from conformal_array import get_conformal_array
from shadow_engine_batched import compute_shadow_mask_batched
from attitude import euler_to_quaternion, rotate_points
from lcmv_stage8 import get_steering_vector
from stage14a_train_pe import ShadowNet, positional_encoding

# Local robust MVDR
def get_R_matrices(v_sig, v_jam):
    JAM_POW = 10000.0
    NOISE_POW = 1.0
    SIG_POW = 100.0
    R_j = JAM_POW * np.outer(v_jam, np.conj(v_jam))
    R_n = NOISE_POW * np.eye(len(v_jam))
    R_s = SIG_POW * np.outer(v_sig, np.conj(v_sig))
    return R_s, R_j, R_n

def compute_mvdr_robust(v_sig, v_jam, dl_factor):
    R_s, R_j, R_n = get_R_matrices(v_sig, v_jam)
    R_xx = R_j + R_n
    
    trace_val = np.real(np.trace(R_xx))
    diag_load = dl_factor * trace_val / R_xx.shape[0] if dl_factor > 0 else 1e-12
    R_reg = R_xx + diag_load * np.eye(R_xx.shape[0])
    
    try:
        R_inv = np.linalg.inv(R_reg)
    except np.linalg.LinAlgError:
        R_inv = np.linalg.pinv(R_reg)

    num = R_inv @ v_sig
    den = np.conj(v_sig) @ num
    w = num / max(abs(den), 1e-12)
    
    S = np.real(np.conj(w) @ R_s @ w)
    N_J = np.real(np.conj(w) @ (R_j + R_n) @ w)
    sinr = S / max(N_J, 1e-12)
    return w, 10 * np.log10(sinr)

def main():
    # 1. Parse and plot training loss curve
    log_path = r"C:\Users\parth\.gemini\antigravity\brain\262e56d0-07ef-4f86-81fb-69ef858784e6\.system_generated\tasks\task-13156.log"
    epochs = []
    train_loss = []
    val_loss = []
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            for line in f:
                if "Train Loss:" in line and "Val Loss:" in line:
                    parts = line.split("|")
                    ep = int(parts[0].split("/")[0].split()[-1])
                    tl = float(parts[1].split(":")[1].strip())
                    vl = float(parts[2].split(":")[1].strip())
                    epochs.append(ep)
                    train_loss.append(tl)
                    val_loss.append(vl)
        
        if epochs:
            plt.figure(figsize=(8,5))
            plt.plot(epochs, train_loss, label="Train Loss")
            plt.plot(epochs, val_loss, label="Val Loss")
            plt.yscale('log')
            plt.title("Training Loss Curve (Log Scale)")
            plt.xlabel("Epochs")
            plt.ylabel("MSE Loss")
            plt.legend()
            plt.grid(True)
            plt.savefig("diag0_loss_curve_pe.png")
            print("Saved diag0_loss_curve_pe.png")

    # Load geometry and model
    MESH_PATH = Path(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone_prepared.stl")
    print("Loading mesh...")
    mesh = load_uav_mesh(MESH_PATH)
    pos_body, normals_body = get_conformal_array(mesh)
    
    model = ShadowNet()
    try:
        model.load_state_dict(torch.load("shadow_net_pe.pt"))
    except Exception:
        print("Model shadow_net_pe.pt not found. Ensure training is complete.")
        return
    model.eval()

    # Generate Test 1 Trajectory
    headings = np.linspace(0, 360, 360, endpoint=False)
    q_inv = euler_to_quaternion(np.deg2rad(15.0), 0, 0).conjugate()
    jam_bodies = np.zeros((360, 3))
    for i, h in enumerate(headings):
        jam_world = np.array([np.cos(np.deg2rad(h)), np.sin(np.deg2rad(h)), 0.0])
        jam_bodies[i] = rotate_points(jam_world.reshape(1, 3), q_inv)[0]
    
    sig_world = np.array([1.0, 0.0, 0.0])
    sig_body = rotate_points(sig_world.reshape(1, 3), q_inv)[0]

    # Exact Physics
    print("Computing exact physics (batched)...")
    g_exact_all = compute_shadow_mask_batched(mesh, pos_body, normals_body, jam_bodies)
    
    # AI Predictions
    with torch.no_grad():
        preds = model(torch.tensor(jam_bodies, dtype=torch.float32)).numpy()
    g_ai_all = preds[:, 0::2] + 1j * preds[:, 1::2]

    # 2. Magnitude vs Phase Error Separation
    mag_exact = np.abs(g_exact_all)
    phase_exact = np.angle(g_exact_all)
    
    mag_preds = np.abs(g_ai_all)
    phase_preds = np.angle(g_ai_all)
    
    mag_err = np.mean(np.abs(mag_exact - mag_preds), axis=1)
    
    # Phase error must account for wrapping
    phase_diff = np.abs(phase_exact - phase_preds)
    phase_diff = np.minimum(phase_diff, 2 * np.pi - phase_diff)
    phase_err = np.mean(phase_diff, axis=1)
    
    fig, ax1 = plt.subplots(figsize=(10,6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Jammer Azimuth (deg)')
    ax1.set_ylabel('Magnitude MAE', color=color)
    ax1.plot(headings, mag_err, label="Magnitude MAE", color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Phase MAE (rad)', color=color)
    ax2.plot(headings, phase_err, label="Phase MAE (wrapped)", color=color, linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title("Diagnostic 1: Magnitude vs Phase Prediction Error")
    fig.tight_layout()
    plt.grid(True)
    plt.savefig("diag1_mag_phase_error_pe.png")
    print("Saved diag1_mag_phase_error_pe.png")

    # 3. Scoped Null-Placement Angular Error
    print("Computing scoped null-placement errors...")
    LAM = 0.15
    K = 2 * np.pi / LAM
    v_sig = get_steering_vector(pos_body, K * sig_body)
    
    null_errors_deg = np.zeros(360)
    for i in range(360):
        # AI weights
        v_jam_ai = g_ai_all[i] * get_steering_vector(pos_body, jam_bodies[i] * K)
        w_ai, _ = compute_mvdr_robust(v_sig, v_jam_ai, dl_factor=0.0) # Base MVDR
        
        # Scoped search (+/- 10 deg azimuth and elevation relative to true jammer body direction)
        j_body = jam_bodies[i]
        j_az = np.arctan2(j_body[1], j_body[0])
        j_el = np.arcsin(j_body[2])
        
        az_grid = np.linspace(j_az - np.deg2rad(10), j_az + np.deg2rad(10), 40)
        el_grid = np.linspace(j_el - np.deg2rad(10), j_el + np.deg2rad(10), 40)
        
        min_resp = float('inf')
        best_ang_err = 0.0
        
        for az in az_grid:
            for el in el_grid:
                test_dir = np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])
                # Assume g=1.0 for pattern search near jammer, or use exact g. 
                # Since we want to find where the deep null in the pattern is, 
                # the null is formed by w_ai. The array response is w_ai^H * (g_exact_test * a(test_dir)).
                # However, the physical null direction is purely determined by the pattern w^H a(theta).
                # Actually, the transmission gain g modifies the effective array manifold. 
                # To be exact, the response to a physical source at test_dir is w^H (g_exact * a).
                # To save time in the 10 deg sweep, we can approximate g as constant (g_exact_all[i]) 
                # because Fresnel coefficients vary smoothly over 10 degrees except at boundaries.
                v_test = g_exact_all[i] * get_steering_vector(pos_body, test_dir * K)
                resp = np.abs(np.conj(w_ai) @ v_test)
                if resp < min_resp:
                    min_resp = resp
                    best_ang_err = np.rad2deg(np.arccos(np.clip(np.dot(j_body, test_dir), -1.0, 1.0)))
                    
        null_errors_deg[i] = best_ang_err

    plt.figure(figsize=(10,6))
    plt.plot(headings, null_errors_deg, color="purple")
    plt.title("Diagnostic 2: Angular Null-Placement Error (AI Hybrid)")
    plt.xlabel("Jammer Azimuth (deg)")
    plt.ylabel("Null Displacement from True Jammer (deg)")
    plt.grid(True)
    plt.savefig("diag2_null_error_pe.png")
    print("Saved diag2_null_error_pe.png")

    # 4. Theoretical Gaussian Noise Sensitivity Sweep
    print("Running Gaussian Noise Sensitivity Sweep...")
    sigmas = np.logspace(-4, -1, 20)
    avg_sinr_noise = np.zeros(len(sigmas))
    
    # We evaluate on just 20 sample headings to save time for the sensitivity sweep
    test_idx = np.linspace(0, 359, 20, dtype=int)
    for s_i, sigma in enumerate(sigmas):
        sinr_sum = 0
        for i in test_idx:
            g_noisy = g_exact_all[i] + np.random.normal(0, sigma, 16) + 1j * np.random.normal(0, sigma, 16)
            v_jam_noisy = g_noisy * get_steering_vector(pos_body, jam_bodies[i] * K)
            w_noisy, _ = compute_mvdr_robust(v_sig, v_jam_noisy, dl_factor=0.0)
            
            # Evaluate against exact environment
            v_jam_exact = g_exact_all[i] * get_steering_vector(pos_body, jam_bodies[i] * K)
            R_s, R_j, R_n = get_R_matrices(v_sig, v_jam_exact)
            S = np.real(np.conj(w_noisy) @ R_s @ w_noisy)
            N_J = np.real(np.conj(w_noisy) @ (R_j + R_n) @ w_noisy)
            sinr_sum += 10 * np.log10(S / max(N_J, 1e-12))
            
        avg_sinr_noise[s_i] = sinr_sum / len(test_idx)
        
    plt.figure(figsize=(8,5))
    plt.semilogx(sigmas, avg_sinr_noise, marker='o')
    plt.title("Diagnostic 3: MVDR Sensitivity to Complex Gaussian Noise")
    plt.xlabel("Noise Std Dev (σ) on Exact Mask")
    plt.ylabel("Average SINR (dB)")
    plt.grid(True)
    plt.savefig("diag3_noise_sensitivity_pe.png")
    print("Saved diag3_noise_sensitivity_pe.png")

    # 5. Empirical DL Parameter Sweep
    print("Running Empirical Diagonal Loading Sweep on AI Predictions...")
    dl_factors = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5]
    worst_sinr_dl = []
    
    for dl in dl_factors:
        min_sinr = float('inf')
        for i in range(360):
            v_jam_ai = g_ai_all[i] * get_steering_vector(pos_body, jam_bodies[i] * K)
            w_ai, _ = compute_mvdr_robust(v_sig, v_jam_ai, dl_factor=dl)
            
            # Evaluate against exact environment
            v_jam_exact = g_exact_all[i] * get_steering_vector(pos_body, jam_bodies[i] * K)
            R_s, R_j, R_n = get_R_matrices(v_sig, v_jam_exact)
            S = np.real(np.conj(w_ai) @ R_s @ w_ai)
            N_J = np.real(np.conj(w_ai) @ (R_j + R_n) @ w_ai)
            sinr = 10 * np.log10(S / max(N_J, 1e-12))
            if sinr < min_sinr:
                min_sinr = sinr
        worst_sinr_dl.append(min_sinr)
        print(f"  DL: {dl:.3f} -> Worst SINR: {min_sinr:.2f} dB")
        
    plt.figure(figsize=(8,5))
    plt.plot(dl_factors, worst_sinr_dl, marker='s', color="green")
    plt.xscale('symlog', linthresh=0.001)
    plt.title("Diagnostic 4: Empirical Diagonal Loading Optimization")
    plt.xlabel("Diagonal Loading Factor (Fraction of Trace)")
    plt.ylabel("Worst-Case SINR over Azimuth Sweep (dB)")
    plt.grid(True)
    plt.savefig("diag4_dl_sweep_pe.png")
    print("Saved diag4_dl_sweep_pe.png")

if __name__ == '__main__':
    main()
