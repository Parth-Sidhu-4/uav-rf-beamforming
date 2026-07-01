import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

from mesh_loader import load_uav_mesh
from conformal_array import get_wingtip_array
from attitude import euler_to_quaternion, rotate_points
from shadow_engine import compute_shadow_mask
from lcmv_stage8 import compute_lcmv_weights, get_steering_vector

def main():
    logging.basicConfig(level=logging.INFO)
    mesh_path = Path(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone_prepared.stl")
    mesh = load_uav_mesh(mesh_path)
    
    antennas_body, normals = get_wingtip_array(mesh)
    N = len(antennas_body)

    freq, c = 2.4e9, 3e8
    k = 2 * np.pi / (c / freq)
    
    # Desired signal is fixed nose-on (0 deg elevation)
    signal_world_dir = np.array([1.0, 0.0, 0.0])
    
    noise_power = 1.0
    signal_power = noise_power * (10 ** (20 / 10))
    jammer_power = noise_power * (10 ** (40 / 10))
    
    # Fixed roll at 30 degrees
    fixed_roll = 30.0
    q = euler_to_quaternion(np.deg2rad(fixed_roll), 0.0, 0.0)
    q_inv = q.conjugate()
    signal_body_dir = rotate_points(signal_world_dir.reshape(1, 3), q_inv)[0]
    
    elevation_angles_deg = np.linspace(-10, 90, 101)
    
    sinrs = []
    
    # Validations:
    # 1. At -10 degrees elevation
    phi_rad_neg10 = np.radians(-10)
    jam_world_neg10 = np.array([0.0, np.cos(phi_rad_neg10), np.sin(phi_rad_neg10)])
    jam_body_neg10 = rotate_points(jam_world_neg10.reshape(1, 3), q_inv)[0]
    mask_neg10 = compute_shadow_mask(mesh, antennas_body, normals, jam_body_neg10)
    print(f"[VALIDATION] At -10° elevation, 30° roll | Active={int(mask_neg10.sum())}/{N} | Jammer body={jam_body_neg10.round(3)}")
    
    # 2. At +90 degrees elevation (overhead)
    phi_rad_90 = np.radians(90)
    jam_world_90 = np.array([0.0, np.cos(phi_rad_90), np.sin(phi_rad_90)])
    jam_body_90 = rotate_points(jam_world_90.reshape(1, 3), q_inv)[0]
    mask_90 = compute_shadow_mask(mesh, antennas_body, normals, jam_body_90)
    print(f"[VALIDATION] At +90° elevation, 30° roll | Active={int(mask_90.sum())}/{N} | Jammer body={jam_body_90.round(3)}")
    if jam_body_90[2] < 0:
        print("[WARNING] Overhead jammer has negative Z in body frame! Check coordinate conventions.")
        
    # 3. Transition zone diagnostic
    for phi_deg in [70, 71, 72, 73]:
        p_rad = np.radians(phi_deg)
        j_world = np.array([0.0, np.cos(p_rad), np.sin(p_rad)])
        j_body = rotate_points(j_world.reshape(1, 3), q_inv)[0]
        m = compute_shadow_mask(mesh, antennas_body, normals, j_body)
        print(f"[TRANSITION] At {phi_deg}° elevation | Active={int(m.sum())}")
    
    for phi_deg in elevation_angles_deg:
        phi_rad = np.radians(phi_deg)
        jammer_world_dir = np.array([0.0, np.cos(phi_rad), np.sin(phi_rad)])
        
        jammer_body_dir = rotate_points(jammer_world_dir.reshape(1, 3), q_inv)[0]
        
        mask = compute_shadow_mask(mesh, antennas_body, normals, jammer_body_dir)
        active_count = np.sum(mask)
        
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

    # Find first valid phi index
    first_valid_phi = elevation_angles_deg[0]
    for p, s in zip(elevation_angles_deg, sinrs):
        if not np.isnan(s):
            first_valid_phi = p
            break

    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    ax1.plot(elevation_angles_deg, sinrs, 'b-', linewidth=2)
    ax1.axvspan(-10, first_valid_phi, alpha=0.15, color='gray', label='Beamformer failure (< MIN_ACTIVE elements)')
    ax1.set_xlim(-10, 90)
    ax1.set_xlabel('Jammer Elevation Angle $\phi$ (degrees)')
    ax1.set_ylabel('LCMV SINR (dB)')
    ax1.set_title(f'Sweep 2: Wingtip Array Resilience to Threat Elevation at Roll={fixed_roll}°')
    ax1.grid(True)
    ax1.legend()
    
    plt.tight_layout()
    plt.savefig(Path(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\sweep2_results.png"))
    logging.info("Saved plot to sweep2_results.png")

if __name__ == "__main__":
    main()
