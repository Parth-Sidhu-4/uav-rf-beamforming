import sys
import os
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Phase 2 Track 1', 'em_realism')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Phase 2 Track 2', 'array_structures')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Stage 12 (Operational Analysis)')))

from mesh_loader import load_uav_mesh
from conformal_array import get_conformal_array
from shadow_engine import compute_shadow_mask
from shadow_engine_batched import compute_shadow_mask_batched
from mvdr_beamformer import compute_mvdr_weights_robust, compute_sinr
from attitude import euler_to_quaternion, rotate_points
from lcmv_stage8 import get_steering_vector
from pathlib import Path

# Constants
K = 2 * np.pi / 0.125
SIG_POW = 10**(20/10)
JAM_POW = 10**(40/10)
NOISE_POW = 10**(0/10)

class ShadowNet(nn.Module):
    def __init__(self):
        super(ShadowNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 32)
        )
    def forward(self, x):
        return self.net(x)

def run_evaluation(mesh, pos_body, normals_body, model, test_name, x_axis, jam_bodies, sig_body):
    N_ELEMENTS = pos_body.shape[0]
    N_TESTS = len(jam_bodies)
    
    sinr_exact = np.zeros(N_TESTS)
    sinr_ai = np.zeros(N_TESTS)
    
    mae_mag = np.zeros((N_TESTS, N_ELEMENTS))
    
    time_exact = 0.0
    time_ai = 0.0
    
    k_sig = K * sig_body
    a_sig = get_steering_vector(pos_body, k_sig)
    mae_mag = np.zeros(N_TESTS)
    
    v_sig = get_steering_vector(pos_body, K * sig_body)
    
    # Benchmark exact physics on just 10 samples for fair single-query latency
    print("Benchmarking exact physics single-query latency...")
    t0_exact_bench = time.perf_counter()
    for i in range(min(10, N_TESTS)):
        _ = compute_shadow_mask(mesh, pos_body, normals_body, jam_bodies[i])
    t1_exact_bench = time.perf_counter()
    avg_exact_ms = (t1_exact_bench - t0_exact_bench) * 1000 / min(10, N_TESTS)

    # Compute exact physics for all using batched version for evaluation speed
    print("Computing exact physics (batched)...")
    g_exact_all = compute_shadow_mask_batched(mesh, pos_body, normals_body, jam_bodies)
    
    print("Running exact MVDR...")
    for i, j_body in enumerate(jam_bodies):
        g_exact = g_exact_all[i]
        v_jam_exact = g_exact * get_steering_vector(pos_body, j_body * K)
        
        w_exact = compute_mvdr_weights_robust(v_sig, v_jam_exact)
        sinr_exact[i] = 10 * np.log10(compute_sinr(w_exact, v_sig, v_jam_exact))
        
    # --- 2. AI Hybrid Pipeline ---
    print("Running AI Hybrid pipeline...")
    # Warm up Torch to avoid cold-start penalization
    dummy_input = torch.tensor(jam_bodies[0:1], dtype=torch.float32)
    _ = model(dummy_input)
    
    t0_ai = time.perf_counter()
    for i, j_body in enumerate(jam_bodies):
        input_tensor = torch.tensor(j_body.reshape(1, 3), dtype=torch.float32)
        with torch.no_grad():
            pred = model(input_tensor)[0]
            mag_pred = torch.sigmoid(pred[0::2]).numpy()
            phase_pred = pred[1::2].numpy()
        
        g_ai = mag_pred * np.exp(1j * phase_pred)
        mae_mag[i] = np.mean(np.abs(np.abs(g_exact_all[i]) - np.abs(g_ai)))
        
        v_jam_ai = g_ai * get_steering_vector(pos_body, j_body * K)
        w_ai = compute_mvdr_weights_robust(v_sig, v_jam_ai)
        v_jam_exact = g_exact_all[i] * get_steering_vector(pos_body, j_body * K)
        sinr_ai[i] = 10 * np.log10(compute_sinr(w_ai, v_sig, v_jam_exact))
        
    t1_ai = time.perf_counter()
    avg_ai_ms = (t1_ai - t0_ai) * 1000 / N_TESTS
    
    # --- Speedup Report ---
    print(f"--- {test_name} ---")
    print(f"Exact Time (per query): {avg_exact_ms:.3f} ms")
    print(f"AI Hybrid Time (per query): {avg_ai_ms:.3f} ms")
    print(f"Speedup Ratio (Eager): {avg_exact_ms / avg_ai_ms:.2f}x")
    
    # Batch throughput measurement
    t0 = time.perf_counter()
    batch_tensor = torch.tensor(jam_bodies, dtype=torch.float32)
    with torch.no_grad():
        preds = model(batch_tensor).numpy()
    g_ai_batch = preds[:, 0::2] + 1j * preds[:, 1::2]
    t1 = time.perf_counter()
    batch_time_ms = (t1 - t0) * 1000
    print(f"Batch inference of {N_TESTS} queries took: {batch_time_ms:.3f} ms ({(batch_time_ms/N_TESTS):.4f} ms/query)")
    print(f"Batch Speedup Ratio vs Exact Physics: {avg_exact_ms / (batch_time_ms/N_TESTS):.2f}x\n")
    
    return sinr_exact, sinr_ai, mae_mag

def main():
    MESH_PATH = Path(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone_prepared.stl")
    print("Loading mesh...")
    mesh = load_uav_mesh(MESH_PATH)
    pos_body, normals_body = get_conformal_array(mesh)
    # Load Model
    model = ShadowNet()
    try:
        model.load_state_dict(torch.load("shadow_net_polar.pt"))
    except Exception:
        print("Model shadow_net_polar.pt not found. Ensure training is complete.")
        return
    print("Running Test 1: Azimuth Interpolation")
    headings = np.linspace(0, 360, 360, endpoint=False)
    sig_world = np.array([1.0, 0.0, 0.0])
    jam_bodies_t1 = np.zeros((360, 3))
    
    q = euler_to_quaternion(np.deg2rad(15.0), 0, 0)
    q_inv = q.conjugate()
    sig_body_t1 = rotate_points(sig_world.reshape(1, 3), q_inv)[0]
    
    for i, h in enumerate(headings):
        jam_world = np.array([np.cos(np.deg2rad(h)), np.sin(np.deg2rad(h)), 0.0])
        jam_bodies_t1[i] = rotate_points(jam_world.reshape(1, 3), q_inv)[0]
        
    s1_ex, s1_ai, mae1 = run_evaluation(mesh, pos_body, normals_body, model, "Test 1 (Azimuth Interpolation)", headings, jam_bodies_t1, sig_body_t1)
    
    # Test 2: Elevation Interpolation (Fixed Azimuth 120 in body frame)
    print("Running Test 2: Elevation Interpolation")
    elevations = np.linspace(-90, 90, 180)
    jam_bodies_t2 = np.zeros((180, 3))
    az = np.deg2rad(120)
    for i, el in enumerate(elevations):
        el_rad = np.deg2rad(el)
        jam_bodies_t2[i] = np.array([
            np.cos(az) * np.cos(el_rad),
            np.sin(az) * np.cos(el_rad),
            np.sin(el_rad)
        ])
    sig_body_t2 = np.array([1.0, 0.0, 0.0]) # Forward
    s2_ex, s2_ai, mae2 = run_evaluation(mesh, pos_body, normals_body, model, "Test 2 (Elevation Interpolation)", elevations, jam_bodies_t2, sig_body_t2)
    
    # --- PLOTTING ---
    plt.figure(figsize=(10,6))
    plt.plot(headings, s1_ex, label="Exact Physics (MVDR)", linewidth=2)
    plt.plot(headings, s1_ai, '--', label="AI Hybrid (DNN + MVDR)", linewidth=2)
    plt.title("Test 1: Azimuth Interpolation (15° Bank)")
    plt.xlabel("Jammer Azimuth (deg)")
    plt.ylabel("Worst-Case SINR (dB)")
    plt.legend()
    plt.grid(True)
    plt.savefig("test1_azimuth_sinr.png")
    
    plt.figure(figsize=(10,6))
    plt.plot(elevations, s2_ex, label="Exact Physics (MVDR)", linewidth=2)
    plt.plot(elevations, s2_ai, '--', label="AI Hybrid (DNN + MVDR)", linewidth=2)
    plt.title("Test 2: Elevation Interpolation (Body Azimuth 120°)")
    plt.xlabel("Jammer Elevation (deg)")
    plt.ylabel("Worst-Case SINR (dB)")
    plt.legend()
    plt.grid(True)
    plt.savefig("test2_elevation_sinr.png")
    
    plt.figure(figsize=(10,6))
    plt.plot(headings, mae1, label="Mean Absolute Error of |g|", color='red')
    plt.title("Test 1: Mask Magnitude MAE")
    plt.xlabel("Jammer Azimuth (deg)")
    plt.ylabel("MAE of Transmission Gains")
    plt.grid(True)
    plt.savefig("test1_azimuth_mae.png")

if __name__ == '__main__':
    main()
