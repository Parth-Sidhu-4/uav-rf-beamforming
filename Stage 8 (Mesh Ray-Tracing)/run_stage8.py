import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

from mesh_loader import load_uav_mesh
from conformal_array import get_conformal_array
from attitude import euler_to_quaternion, rotate_points
from shadow_engine import compute_shadow_mask
from lcmv_stage8 import compute_lcmv_weights, get_steering_vector

def main():
    logging.basicConfig(level=logging.INFO)
    
    # 1. Load Mesh
    mesh_path = Path(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone_prepared.stl")
    mesh = load_uav_mesh(mesh_path)
    
    # 2. Get Conformal Array
    # These are in the body frame
    antennas_body, normals = get_conformal_array(mesh)
    N = len(antennas_body)
    
    # 3. Simulation Parameters
    freq = 2.4e9
    c = 3e8
    lam = c / freq
    k = 2 * np.pi / lam
    
    # Jammer position in world coordinates (East of the drone)
    jammer_world_dir = np.array([0.0, 1.0, 0.0]) # Directly to the right (+Y)
    
    # Signal parameters
    snr_db = 20
    inr_db = 40
    noise_power = 1.0
    signal_power = noise_power * (10 ** (snr_db / 10))
    jammer_power = noise_power * (10 ** (inr_db / 10))
    
    # Maneuver: 0 to 45 degrees right roll
    time_steps = 50
    roll_angles_deg = np.linspace(0, 45, time_steps)
    
    sinr_history = []
    mask_history = []
    
    for i, roll_deg in enumerate(roll_angles_deg):
        roll_rad = np.deg2rad(roll_deg)
        
        # We assume Pitch and Yaw are 0 for this maneuver
        q = euler_to_quaternion(roll_rad, 0.0, 0.0)
        
        # 4. Ray Tracing
        # The jammer is fixed in the world. As the drone rolls right, the world jammer
        # appears to move *up* relative to the drone's body.
        # So we rotate the world vector into the body frame using the inverse quaternion.
        q_inv = q.conjugate()
        # Rotate a single vector using our rotate_points function (expects Nx3)
        jammer_body_dir = rotate_points(jammer_world_dir.reshape(1, 3), q_inv)[0]
        
        mask = compute_shadow_mask(mesh, antennas_body, normals, jammer_body_dir)
        mask_history.append(mask)
        
        # Sanity check at roll = 0
        if i == 0:
            print("--- SANITY CHECK AT 0 DEG ROLL ---")
            for j, (pos, normal) in enumerate(zip(antennas_body, normals)):
                h_dot = np.dot(jammer_body_dir, normal)
                print(f"Element {j:2d} | pos={pos} | normal={normal} | jammer_dot={h_dot:.3f}")
            print("----------------------------------")
            
        # 5. LCMV Beamforming
        # Signal direction (assume it comes from ahead, +X in world)
        signal_world_dir = np.array([1.0, 0.0, 0.0])
        signal_body_dir = rotate_points(signal_world_dir.reshape(1, 3), q_inv)[0]
        
        # Steering vectors
        k_sig = k * signal_body_dir
        k_jam = k * jammer_body_dir
        
        a_sig = get_steering_vector(antennas_body, k_sig)
        a_jam = get_steering_vector(antennas_body, k_jam)
        
        # Covariance Matrix R_xx
        R_s = signal_power * np.outer(a_sig, np.conj(a_sig))
        R_j = jammer_power * np.outer(a_jam, np.conj(a_jam))
        R_n = noise_power * np.eye(N)
        R_xx = R_s + R_j + R_n
        
        w = compute_lcmv_weights(R_xx, a_sig, mask)
        
        # Output powers
        P_s = signal_power * np.abs(np.conj(w).T @ a_sig)**2
        P_j = jammer_power * np.abs(np.conj(w).T @ a_jam)**2
        P_n = noise_power * np.linalg.norm(w)**2
        
        # If fewer than 3 elements are unshadowed, LCMV has 0 DoF remaining.
        # This causes ill-conditioned R_xx and numerical spikes.
        MIN_ACTIVE_ELEMENTS = 3
        if np.sum(mask) < MIN_ACTIVE_ELEMENTS:
            sinr = np.nan
        # If weights are completely zeroed (or near zero), SINR is 0
        elif (P_j + P_n) < 1e-12 or P_s < 1e-12:
            sinr = 0.0
        else:
            sinr = 10 * np.log10(P_s / (P_j + P_n))
            
        sinr_history.append(sinr)

    # 6. Plotting
    mask_history = np.array(mask_history)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [1, 2]})
    
    # Plot SINR
    ax1.plot(roll_angles_deg, sinr_history, 'b-', linewidth=2)
    ax1.set_title("Stage 8: Dynamic Shadowing LCMV Performance (Right Roll)")
    ax1.set_ylabel("Output SINR (dB)")
    ax1.grid(True)
    
    # Plot Heatmap
    cax = ax2.imshow(mask_history.T, aspect='auto', cmap='RdYlGn', 
                     extent=[roll_angles_deg[0], roll_angles_deg[-1], N-0.5, -0.5])
    ax2.set_title("Conformal Array Shadow Mask $\mathbf{m}(t)$ (1=LoS, 0=Blocked)")
    ax2.set_xlabel("Roll Angle (degrees)")
    ax2.set_ylabel("Antenna Element Index")
    ax2.set_yticks(np.arange(N))
    fig.colorbar(cax, ax=ax2, orientation='vertical', label='Mask State')
    
    plt.tight_layout()
    plt.savefig(Path(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\stage8_results.png"))
    logging.info("Saved plot to stage8_results.png")
    
if __name__ == "__main__":
    main()
