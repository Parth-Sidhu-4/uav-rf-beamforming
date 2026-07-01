import numpy as np
import matplotlib.pyplot as plt
import trimesh
from pathlib import Path
import sys

sys.path.append(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)")
sys.path.append(r"D:\UAV Internship project\Phase 2 Track 1")

from mesh_loader import load_uav_mesh
from conformal_array import get_conformal_array
from mesh_aware_lcmv import (
    get_steering_vector, compute_reduced_lcmv, LAMBDA_M, 
    world_to_body_bearing, CognitiveAutopilot, generate_lut, G
)

def main():
    print("Loading mesh and setting up elements...")
    mesh_path = Path(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone_prepared.stl")
    mesh = load_uav_mesh(mesh_path)
    antennas_body, normals = get_conformal_array(mesh)
    
    intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
    k = 2 * np.pi / LAMBDA_M
    N_ant = len(antennas_body)
    
    # Far-field geometry
    jammer_pos_xy = np.array([0.0, 100000.0])
    gcs_pos_xy    = np.array([0.0, -100000.0]) # WEST
    
    lut = generate_lut()
    
    # Define configurations
    configs = [
        {"name": "Baseline", "autopilot": None, "color": "red"},
        {"name": "Legacy Cognitive (N=6)", "autopilot": CognitiveAutopilot(lut, min_active=6), "color": "green"},
        {"name": "Full Array Cognitive (N=16)", "autopilot": CognitiveAutopilot(lut, min_active=16), "color": "blue"}
    ]
    
    V, dt, desired_phi = 25.0, 0.1, 30.0
    heading_target = 90.0
    num_seeds = 10
    sigma_phase = 0.1
    
    results = {}
    
    for config in configs:
        name = config["name"]
        autopilot = config["autopilot"]
        print(f"\n--- Running {name} ---")
        
        # 1. Precompute Kinematics
        headings, phis, az_jams = [], [], []
        ac_pos = np.zeros(2)
        h = 0.0
        
        while h < heading_target and len(headings) < 2000: # 200 second timeout
            headings.append(h)
            az_gcs, _ = world_to_body_bearing(gcs_pos_xy, ac_pos, h)
            az_jam, _ = world_to_body_bearing(jammer_pos_xy, ac_pos, h)
            az_jams.append(az_jam)
            
            if autopilot is None:
                phi_cmd = desired_phi
            else:
                phi_cmd = autopilot.command_bank(desired_phi, az_gcs, 0.0, V=V, dt=dt)
                
            phis.append(phi_cmd)
            h += np.degrees(G * np.tan(np.radians(phi_cmd)) / V) * dt
            
        N_steps = len(headings)
        headings = np.array(headings)
        phis = np.array(phis)
        az_jams = np.array(az_jams)
        
        if N_steps == 2000:
            print(f"WARNING: {name} timed out after 200s! Max heading reached: {headings[-1]:.2f} deg")
            
        # 2. Run multi-seed dynamic solver
        sinr_matrix = np.zeros((num_seeds, N_steps))
        n_act_trace = np.zeros(N_steps) # Deterministic since mesh/kinematics are noiseless
        
        # Precompute active elements (deterministic)
        print("Ray-casting element occlusion based on GCS LOS...")
        
        # Exact Track 1 3D Geometry
        from attitude import euler_to_quaternion, rotate_points
        jam_world = np.array([jammer_pos_xy[0], jammer_pos_xy[1], 0.0])
        jam_world /= np.linalg.norm(jam_world)
        gcs_world = np.array([gcs_pos_xy[0], gcs_pos_xy[1], 0.0])
        gcs_world /= np.linalg.norm(gcs_world)
        
        all_jam_body = np.zeros((N_steps, 3))
        all_gcs_body = np.zeros((N_steps, 3))
        
        for i in range(N_steps):
            q = euler_to_quaternion(np.deg2rad(phis[i]), 0, np.deg2rad(headings[i]))
            q_inv = q.conjugate()
            all_jam_body[i] = rotate_points(jam_world.reshape(1,3), q_inv)[0]
            all_gcs_body[i] = rotate_points(gcs_world.reshape(1,3), q_inv)[0]

        for step in range(N_steps):
            dir_jam = all_jam_body[step]
            dir_gcs = all_gcs_body[step]
            
            # The active_elements must be based on LOS to the GCS!
            origins = antennas_body + normals * 0.01
            directions_gcs = np.tile(dir_gcs, (len(antennas_body), 1))
            hits_gcs = intersector.intersects_any(origins, directions_gcs)
            active_elements = ~hits_gcs
            n_act_trace[step] = np.sum(active_elements)
            
            a_sig_perc = get_steering_vector(antennas_body, normals, k * dir_gcs, add_noise=False)
            a_jam_perc = get_steering_vector(antennas_body, normals, k * dir_jam, add_noise=False)
            
            # Multi-seed loop for this timestep
            for seed in range(num_seeds):
                np.random.seed(seed * 10000 + step)
                phase_error = np.random.normal(0, sigma_phase, N_ant)
                phase_rotation = np.exp(1j * phase_error)
                
                a_sig_true = a_sig_perc * phase_rotation
                a_jam_true = a_jam_perc * phase_rotation
                
                w, sinr_db, nd_db = compute_reduced_lcmv(
                    a_sig_perc, a_jam_perc,
                    a_sig_true, a_jam_true,
                    active_elements
                )
                sinr_matrix[seed, step] = sinr_db

        # Calculate Dropout Metrics
        # Dropout: any step where SINR < 0 dB
        total_steps = N_steps * num_seeds
        dropout_steps = np.sum(sinr_matrix < 0.0)
        dropout_prob = (dropout_steps / total_steps) * 100.0
        
        # Max continuous dropout
        max_duration = 0.0
        for seed in range(num_seeds):
            current_run = 0
            max_run = 0
            for step in range(N_steps):
                if sinr_matrix[seed, step] < 0.0:
                    current_run += 1
                    max_run = max(max_run, current_run)
                else:
                    current_run = 0
            max_duration = max(max_duration, max_run * dt)
            
        print(f"Dropout Probability: {dropout_prob:.2f}%")
        print(f"Max Continuous Dropout: {max_duration:.2f} s")
        
        results[name] = {
            "headings": headings,
            "az_jams": az_jams,
            "phis": phis,
            "n_act": n_act_trace,
            "sinr_median": np.median(sinr_matrix, axis=0),
            "sinr_10": np.percentile(sinr_matrix, 10, axis=0),
            "sinr_90": np.percentile(sinr_matrix, 90, axis=0),
            "color": config["color"]
        }

    # Plotting
    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    
    for name, data in results.items():
        headings = data["headings"]
        c = data["color"]
        
        axs[0].plot(headings, data["az_jams"], label=name, color=c, linewidth=2)
        axs[1].plot(headings, data["phis"], label=name, color=c, linewidth=2)
        axs[2].plot(headings, data["n_act"], label=name, color=c, linewidth=2)
        
        axs[3].plot(headings, data["sinr_median"], label=f"{name} (Median)", color=c, linewidth=2)
        axs[3].fill_between(headings, data["sinr_10"], data["sinr_90"], color=c, alpha=0.2)
        
    axs[0].set_ylabel('Jammer Rel Az (°)')
    axs[0].grid(True)
    axs[0].legend()
    axs[0].set_title('Path B Dynamic Simulation under EM Realism (σ=0.1 rad, 10 Seeds)')
    
    axs[1].set_ylabel('Bank Angle (°)')
    axs[1].grid(True)
    
    axs[2].set_ylabel('Active Elements')
    axs[2].grid(True)
    axs[2].axhline(y=6, color='green', linestyle=':', label='Legacy Threshold (6)')
    axs[2].axhline(y=16, color='blue', linestyle=':', label='Full Array Threshold (16)')
    
    axs[3].set_xlabel('Aircraft Heading (°)')
    axs[3].set_ylabel('SINR (dB)')
    axs[3].axhline(y=0, color='red', linestyle='--', linewidth=2, label='Failure Threshold (0 dB)')
    axs[3].set_ylim(-10, 25)
    axs[3].grid(True)
    axs[3].legend()
    
    fig.tight_layout()
    out_path = r"C:\Users\parth\.gemini\antigravity\brain\262e56d0-07ef-4f86-81fb-69ef858784e6\dynamic_em_realism_results.png"
    plt.savefig(out_path)
    print(f"\nSaved plot to {out_path}")

if __name__ == "__main__":
    main()
