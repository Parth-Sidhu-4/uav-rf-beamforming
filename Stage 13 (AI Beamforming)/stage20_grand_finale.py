import numpy as np
import torch
import time
import matplotlib.pyplot as plt
import os
import sys

# Append necessary paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Stage 8 (Mesh Ray-Tracing)'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Stage 12 (Operational Analysis)'))

from conformal_array import get_conformal_array
from em_physics import compute_steering_vectors
from shadow_engine_batched import batched_shadow_mask

class SirenCovarianceNet(torch.nn.Module):
    def __init__(self, in_features=2, hidden_features=256, hidden_layers=3, out_features=160, w0=30.0):
        super().__init__()
        self.net = []
        self.net.append(torch.nn.Linear(in_features, hidden_features))
        for _ in range(hidden_layers):
            self.net.append(torch.nn.Linear(hidden_features, hidden_features))
        self.net.append(torch.nn.Linear(hidden_features, out_features))
        self.net = torch.nn.ModuleList(self.net)
        self.w0 = w0

    def forward(self, x):
        out = self.net[0](x)
        out = torch.sin(self.w0 * out)
        for i in range(1, len(self.net) - 1):
            out = self.net[i](out)
            out = torch.sin(self.w0 * out)
        out = self.net[-1](out)
        return out

def euler_to_matrix(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    
    return Rz @ Ry @ Rx

def run_grand_finale():
    print("======================================================")
    print(" STAGE 20: THE GRAND FINALE - FULL KINEMATIC INTERCEPT")
    print("======================================================")
    
    # 1. Trajectory Setup (60 seconds, 100 Hz = 6000 steps)
    t = np.linspace(0, 60, 6000)
    
    # Drone kinematics: Flying forward, but executes a severe banking maneuver (roll)
    # and a pitch up maneuver, sweeping the tail exactly across the jammer LOS.
    drone_yaw = np.zeros_like(t)
    drone_pitch = np.sin(t * np.pi / 30) * np.radians(15)  # Pitch oscillates +/- 15 deg
    drone_roll = np.sin(t * np.pi / 15) * np.radians(35)   # Roll oscillates +/- 35 deg
    
    # Jammer is fixed in global space, exactly behind the drone
    global_jammer_vec = np.array([-1, 0, -0.2]) # Behind, slightly below
    global_jammer_vec = global_jammer_vec / np.linalg.norm(global_jammer_vec)
    
    # Signal is fixed from the front
    global_sig_vec = np.array([1, 0, 0])
    
    print("[1/5] Calculating Drone Kinematics (6000 flight steps)...")
    rel_jammer_az = np.zeros_like(t)
    rel_jammer_el = np.zeros_like(t)
    
    for i in range(len(t)):
        R = euler_to_matrix(drone_roll[i], drone_pitch[i], drone_yaw[i])
        R_inv = R.T
        
        # Transform global jammer vector to drone body frame
        local_jam = R_inv @ global_jammer_vec
        rel_jammer_az[i] = np.arctan2(local_jam[1], local_jam[0])
        rel_jammer_el[i] = np.arcsin(local_jam[2])
        
    # 2. True Physics (The Ray-Tracer "Reality")
    print("\n[2/5] Running Mesh Ray-Tracer to generate True Physics (The 'Real World')...")
    print("      (This replicates the Stage 8 bottleneck...)")
    elements = get_conformal_array()
    element_positions = np.array([e['position'] for e in elements])
    element_normals = np.array([e['normal'] for e in elements])
    
    # We calculate the true shadow for the jammer at all 6000 steps using the batched engine
    mesh_path = os.path.join(os.path.dirname(__file__), '..', 'Stage 8 (Mesh Ray-Tracing)', 'drone_mesh.stl')
    true_jam_v_free = compute_steering_vectors(element_positions, rel_jammer_az, rel_jammer_el)
    
    start_rt = time.perf_counter()
    # Batch process in chunks of 500 to avoid memory issues
    true_jam_v = np.zeros_like(true_jam_v_free)
    chunk_size = 500
    for i in range(0, 6000, chunk_size):
        az_chunk = rel_jammer_az[i:i+chunk_size]
        el_chunk = rel_jammer_el[i:i+chunk_size]
        _, shadows = batched_shadow_mask(mesh_path, element_positions, az_chunk, el_chunk, return_full_shadow=True)
        true_jam_v[i:i+chunk_size] = true_jam_v_free[i:i+chunk_size] * shadows
    end_rt = time.perf_counter()
    rt_time = end_rt - start_rt
    print(f"      [Ray-Tracer completed 6000 steps in {rt_time:.2f} seconds]")
    
    # True Signal is always front, so no shadow (free space)
    true_sig_v = compute_steering_vectors(element_positions, np.zeros_like(t), np.zeros_like(t))
    
    # 3. Stage 1 Baseline: Free-Space Analytical MVDR
    print("\n[3/5] Running Stage 1 Baseline (Theoretical Array Math)...")
    mvdr_sinr = np.zeros_like(t)
    for i in range(len(t)):
        # Assumption: Jammer is in free space (what the Stage 1 math thinks)
        v_j_assumed = true_jam_v_free[i]
        R_assumed = np.outer(v_j_assumed, v_j_assumed.conj()) + np.eye(32) * 1e-3
        w = np.linalg.solve(R_assumed, true_sig_v[i])
        
        # Calculate TRUE SINR in reality
        sig_p = np.abs(w.conj().T @ true_sig_v[i])**2
        jam_p = np.abs(w.conj().T @ true_jam_v[i])**2
        mvdr_sinr[i] = 10 * np.log10(sig_p / (jam_p + 1e-3))
        
    # 4. Stage 14 AI Surrogate (Rank-5 Covariance)
    print("\n[4/5] Running Stage 14 AI Surrogate (Rank-5 SIREN)...")
    device = torch.device('cpu')
    model = SirenCovarianceNet().to(device)
    model.load_state_dict(torch.load('siren_beamformer_d3_cov_K5_3D.pt', map_location=device))
    model.eval()
    
    ai_sinr = np.zeros_like(t)
    
    # Prepare AI inputs (azimuth, elevation)
    X = torch.tensor(np.column_stack([rel_jammer_az, rel_jammer_el]), dtype=torch.float32)
    
    start_ai = time.perf_counter()
    with torch.no_grad():
        # A. AI Forward Pass for all 6000 steps instantly
        out = model(X) # (6000, 160)
        
        # B. Construct Covariance and Invert
        V = out.view(6000, 32, 5)
        R_ai = torch.bmm(V, V.transpose(1, 2))
        eye = torch.eye(32).unsqueeze(0).expand(6000, -1, -1)
        R_ai = R_ai + 1e-3 * eye
        
        v_sig = torch.tensor(true_sig_v, dtype=torch.complex64) # (6000, 32)
        v_sig_real = torch.ones(6000, 32, 1, dtype=torch.float32) # For inverse trick
        
        w_ai = torch.linalg.solve(R_ai, v_sig_real).squeeze(-1).numpy()
    end_ai = time.perf_counter()
    ai_time = end_ai - start_ai
    print(f"      [AI Surrogate completed 6000 steps in {ai_time:.3f} seconds!]")
    
    # Calculate True SINR for AI
    for i in range(len(t)):
        w = w_ai[i]
        sig_p = np.abs(w.conj().T @ true_sig_v[i])**2
        jam_p = np.abs(w.conj().T @ true_jam_v[i])**2
        ai_sinr[i] = 10 * np.log10(sig_p / (jam_p + 1e-3))
        
    # 5. Plotting the Grand Finale
    print("\n[5/5] Generating Grand Finale Plot...")
    plt.figure(figsize=(14, 8))
    
    # Subplot 1: SINR over time
    plt.subplot(2, 1, 1)
    plt.plot(t, mvdr_sinr, label='Stage 1: Theoretical MVDR (Fails in Shadow)', color='red', alpha=0.7)
    plt.plot(t, ai_sinr, label='Stage 14: AI Surrogate (Real-time Compensation)', color='blue', linewidth=2)
    plt.axhline(15, color='green', linestyle='--', label='Operational Requirement (>15 dB)')
    plt.axvspan(10, 20, color='grey', alpha=0.2, label='Drone Banks into Deep Shadow')
    plt.axvspan(40, 50, color='grey', alpha=0.2)
    plt.ylabel("True SINR (dB)")
    plt.title(f"The Grand Finale: 60-Second Kinematic Intercept\nRay-Tracer Time: {rt_time:.2f}s | AI Time: {ai_time:.3f}s")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(-25, 35)
    
    # Subplot 2: Jammer Relative Angle
    plt.subplot(2, 1, 2)
    plt.plot(t, np.degrees(rel_jammer_az), label='Jammer Relative Azimuth', color='purple')
    plt.plot(t, np.degrees(rel_jammer_el), label='Jammer Relative Elevation', color='orange')
    plt.axhline(180, color='black', linestyle=':', label='Center of Rear Tail')
    plt.axhline(-180, color='black', linestyle=':')
    plt.ylabel("Angle (degrees)")
    plt.xlabel("Flight Time (seconds)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('stage20_grand_finale.png', dpi=300)
    print("Done. Saved as 'stage20_grand_finale.png'")

if __name__ == "__main__":
    run_grand_finale()
