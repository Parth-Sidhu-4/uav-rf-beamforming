import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

from mesh_loader import load_uav_mesh
from conformal_array import get_conformal_array, get_semi_distributed_array, get_clustered_array, get_wingtip_array
from attitude import euler_to_quaternion, rotate_points
from shadow_engine import compute_shadow_mask
from lcmv_stage8 import compute_lcmv_weights, get_steering_vector

def validate_geometry(mesh, antennas, normals, jammer_body_dir, roll_deg):
    """Run at start of every sweep script to catch coordinate bugs early."""
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
    
    # Define geometries
    # NOTE: Semi-Distributed uses RIGHT wing intentionally — this represents
    # a starboard-biased placement, demonstrating worst-case vulnerability
    # for a right-roll maneuver. Left-wing bias would be the mirror result.
    geometries = {
        "Nominal (Fully Distributed)": get_conformal_array(mesh),
        "Semi-Distributed (Right Wing + Spine)": get_semi_distributed_array(mesh),
        "Clustered (Dorsal Spine)": get_clustered_array(mesh),
        "Wingtip Array (Extremes)": get_wingtip_array(mesh)
    }
    
    # Verify clustered elements are genuinely dorsal (Z > 0 after snapping)
    clustered_pos, clustered_norms = geometries["Clustered (Dorsal Spine)"]
    for i, (pos, norm) in enumerate(zip(clustered_pos, clustered_norms)):
        if pos[2] < 0.02 or norm[2] < 0.5:
            print(f"[WARNING] Clustered element {i} may not be dorsal: "
                  f"Z={pos[2]:.3f}m, normal_Z={norm[2]:.3f}")

    # Simulation Parameters
    freq, c = 2.4e9, 3e8
    k = 2 * np.pi / (c / freq)
    
    jammer_world_dir = np.array([0.0, 1.0, 0.0])
    signal_world_dir = np.array([1.0, 0.0, 0.0])
    
    noise_power = 1.0
    signal_power = noise_power * (10 ** (20 / 10))
    jammer_power = noise_power * (10 ** (40 / 10))
    
    roll_angles_deg = np.linspace(0, 45, 50)
    
    results_sinr = {name: [] for name in geometries}
    results_active = {name: [] for name in geometries}
    
    for name, (antennas_body, normals) in geometries.items():
        logging.info(f"Running simulation for {name}...")
        N = len(antennas_body)
        
        # Validation at roll=0
        q0 = euler_to_quaternion(0.0, 0.0, 0.0)
        jam_body_0 = rotate_points(jammer_world_dir.reshape(1, 3), q0.conjugate())[0]
        print(f"--- {name} Validation ---")
        validate_geometry(mesh, antennas_body, normals, jam_body_0, 0.0)
        
        # Diagnostic at 5 degrees roll for wing-root blockage
        q5 = euler_to_quaternion(np.deg2rad(5), 0.0, 0.0)
        jam_body_5 = rotate_points(jammer_world_dir.reshape(1, 3), q5.conjugate())[0]
        block_count = 0
        for i, (pos, norm) in enumerate(zip(antennas_body, normals)):
            hit = mesh.ray.intersects_location([pos + norm*1e-4], [jam_body_5])
            if len(hit[0]) > 0:
                dist = np.linalg.norm(hit[0][0] - pos)
                if 0.001 < dist < 1.0: # Ignore self-hits and very far hits
                    block_count += 1
        print(f"[{name}] Wing-root/Airframe blockages at 5° roll: {block_count}")
        
        for roll_deg in roll_angles_deg:
            q = euler_to_quaternion(np.deg2rad(roll_deg), 0.0, 0.0)
            q_inv = q.conjugate()
            
            jammer_body_dir = rotate_points(jammer_world_dir.reshape(1, 3), q_inv)[0]
            signal_body_dir = rotate_points(signal_world_dir.reshape(1, 3), q_inv)[0]
            
            mask = compute_shadow_mask(mesh, antennas_body, normals, jammer_body_dir)
            active_count = np.sum(mask)
            results_active[name].append(active_count)
            
            if active_count < 3:
                results_sinr[name].append(np.nan)
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
                results_sinr[name].append(0.0)
            else:
                results_sinr[name].append(10 * np.log10(P_s / (P_j + P_n)))

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    colors = ['blue', 'orange', 'green', 'purple']
    for (name, sinrs), color in zip(results_sinr.items(), colors):
        ax1.plot(roll_angles_deg, sinrs, label=name, color=color, linewidth=2)
    ax1.set_ylabel('LCMV SINR (dB)')
    ax1.set_title('Sweep 3: Spatial Diversity & Array Geometry (Right Roll)')
    ax1.grid(True)
    ax1.legend()
    
    for (name, actives), color in zip(results_active.items(), colors):
        ax2.plot(roll_angles_deg, actives, label=name, color=color, linewidth=2)
    ax2.set_xlabel('Roll Angle (degrees)')
    ax2.set_ylabel('Active Element Count')
    ax2.axhline(3, color='red', linestyle='--', label='Minimum for beamformer validity')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(Path(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\sweep3_results.png"))
    logging.info("Saved plot to sweep3_results.png")

if __name__ == "__main__":
    main()
