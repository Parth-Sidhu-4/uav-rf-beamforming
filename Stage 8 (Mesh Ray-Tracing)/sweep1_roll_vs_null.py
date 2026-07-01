import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

from mesh_loader import load_uav_mesh
from conformal_array import get_conformal_array
from attitude import euler_to_quaternion, rotate_points
from shadow_engine import compute_shadow_mask
from lcmv_stage8 import compute_lcmv_weights, get_steering_vector

def validate_geometry(mesh, antennas, normals, jammer_body_dir, roll_deg):
    mask = compute_shadow_mask(mesh, antennas, normals, jammer_body_dir)
    active = int(mask.sum())
    print(f"[VALIDATION] Roll={roll_deg}° | Active={active}/{len(antennas)} | "
          f"Jammer body-frame: {jammer_body_dir.round(3)}")
    if active < 3:
        print("[WARNING] Fewer than 3 active elements at initial condition — "
              "check coordinate frame before running full sweep")
    return active

def main():
    logging.basicConfig(level=logging.INFO)
    mesh_path = Path(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone_prepared.stl")
    mesh = load_uav_mesh(mesh_path)
    
    antennas_body, normals = get_conformal_array(mesh)
    N = len(antennas_body)

    freq, c = 2.4e9, 3e8
    k = 2 * np.pi / (c / freq)
    
    jammer_world_dir = np.array([0.0, 1.0, 0.0])
    signal_world_dir = np.array([1.0, 0.0, 0.0])
    
    noise_power = 1.0
    signal_power = noise_power * (10 ** (20 / 10))
    jammer_power = noise_power * (10 ** (40 / 10))
    
    roll_angles_deg = np.linspace(0, 90, 91)
    
    sinrs = []
    actives = []
    
    q0 = euler_to_quaternion(0.0, 0.0, 0.0)
    jam_body_0 = rotate_points(jammer_world_dir.reshape(1, 3), q0.conjugate())[0]
    validate_geometry(mesh, antennas_body, normals, jam_body_0, 0.0)
    
    for roll_deg in roll_angles_deg:
        q = euler_to_quaternion(np.deg2rad(roll_deg), 0.0, 0.0)
        q_inv = q.conjugate()
        
        jammer_body_dir = rotate_points(jammer_world_dir.reshape(1, 3), q_inv)[0]
        signal_body_dir = rotate_points(signal_world_dir.reshape(1, 3), q_inv)[0]
        
        mask = compute_shadow_mask(mesh, antennas_body, normals, jammer_body_dir)
        active_count = np.sum(mask)
        actives.append(active_count)
        
        if active_count < 3:
            sinrs.append(np.nan)
            continue
            
        k_sig, k_jam = k * signal_body_dir, k * jammer_body_dir
        a_sig = get_steering_vector(antennas_body, k_sig)
        a_jam = get_steering_vector(antennas_body, k_jam)
        
        R_xx = (signal_power * np.outer(a_sig, np.conj(a_sig)) + 
                jammer_power * np.outer(a_jam, np.conj(a_jam)) + 
                noise_power * np.eye(N))
        
        w = compute_lcmv_weights(R_xx, a_sig, mask)
        
        P_s = signal_power * np.abs(np.conj(w).T @ a_sig)**2
        P_j = jammer_power * np.abs(np.conj(w).T @ a_jam)**2
        P_n = noise_power * np.linalg.norm(w)**2
        
        if (P_j + P_n) < 1e-12 or P_s < 1e-12:
            sinrs.append(0.0)
        else:
            sinrs.append(10 * np.log10(P_s / (P_j + P_n)))

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    ax1.plot(roll_angles_deg, sinrs, 'b-', linewidth=2)
    ax1.set_ylabel('LCMV SINR (dB)')
    ax1.set_title('Sweep 1: Absolute Operational Limits (Roll 0° to 90°)')
    ax1.axvspan(45, 90, alpha=0.1, color='red', label='Non-operational flight regime')
    ax1.axvline(x=45, color='red', linestyle='--', linewidth=1)
    ax1.grid(True)
    ax1.legend()
    
    ax2.plot(roll_angles_deg, actives, 'g-', linewidth=2)
    ax2.set_xlabel('Roll Angle (degrees)')
    ax2.set_ylabel('Active Element Count')
    ax2.axvspan(45, 90, alpha=0.1, color='red', label='Non-operational flight regime')
    ax2.axvline(x=45, color='red', linestyle='--', linewidth=1)
    ax2.axhline(3, color='orange', linestyle='--', label='Minimum for beamformer validity')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(Path(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\sweep1_results.png"))
    logging.info("Saved plot to sweep1_results.png")

if __name__ == "__main__":
    main()
