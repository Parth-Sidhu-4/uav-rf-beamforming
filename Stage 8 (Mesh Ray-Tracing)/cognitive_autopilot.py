import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from scipy.interpolate import RegularGridInterpolator

from mesh_loader import load_uav_mesh
from conformal_array import get_conformal_array
from shadow_engine import compute_shadow_mask
from attitude import euler_to_quaternion, rotate_points

G = 9.81

def world_to_body_bearing(jammer_pos_xy, ac_pos_xy, ac_heading_deg):
    """Jammer azimuth (deg, 0-360) and elevation in aircraft body frame."""
    rel = jammer_pos_xy - ac_pos_xy
    az_world = np.degrees(np.arctan2(rel[1], rel[0]))
    # Body azimuth: relative to nose (0=ahead, 90=right)
    # The world frame has Y=East (if heading=0 is North/X-axis).
    # Actually, let's just use standard navigation:
    az_body  = (az_world - ac_heading_deg) % 360
    el_body  = 0.0 # 2D demo assumption
    return az_body, el_body

class CognitiveAutopilot:
    def __init__(self, lut, min_active=3):
        self.lut = lut
        self.min_active = min_active

    def bank_limit(self, az_body, desired_phi, el_body=0.0, V=25.0, dt=0.1):
        # We only check bank angles up to desired_phi
        max_check = int(np.ceil(abs(desired_phi)))
        if max_check == 0:
            return desired_phi
        bank_grid = np.arange(0, max_check + 1, 1)
        
        pts_curr = np.column_stack((bank_grid, np.full_like(bank_grid, az_body)))
        counts_curr = self.lut(pts_curr)
        
        # Look-ahead: check projected azimuth after 1 timestep
        d_heading = np.degrees(G * np.tan(np.radians(bank_grid)) / V) * dt
        az_proj = (az_body - d_heading) % 360
        pts_proj = np.column_stack((bank_grid, az_proj))
        counts_proj = self.lut(pts_proj)
        
        feasible_mask = (counts_curr >= self.min_active) & (counts_proj >= self.min_active)
        
        if not feasible_mask[0]:
            # No bank angle meets the threshold — fall back to unconstrained desired_phi
            # so the drone continues turning and doesn't deadlock at heading=const.
            return float(desired_phi)
        else:
            first_false = np.argmin(feasible_mask)
            if first_false == 0 and feasible_mask[0]:  # all True
                return float(bank_grid[-1])
            else:
                return float(bank_grid[first_false - 1])

    def command_bank(self, desired_phi, az_body, el_body=0.0, V=25.0, dt=0.1):
        limit = self.bank_limit(az_body, desired_phi, el_body, V, dt)
        return float(np.clip(desired_phi, -limit, limit))

def generate_lut():
    print("Generating Cognitive Autopilot LUT with batched ray-tracing...")
    mesh = load_uav_mesh(Path(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone_prepared.stl"))
    antennas_body, normals = get_conformal_array(mesh)
    
    intersector = __import__('trimesh').ray.ray_triangle.RayMeshIntersector(mesh)
    
    phi_grid = np.arange(0, 46, 1)    # 0 to 45 deg
    az_grid = np.arange(0, 361, 10)   # 0 to 360 deg
    
    lut_data = np.zeros((len(phi_grid), len(az_grid)))
    
    # Precompute all directions
    N_phi = len(phi_grid)
    N_az = len(az_grid) - 1 # We skip 360 and copy 0 later
    N_ant = len(antennas_body)
    
    all_ray_origins = []
    all_ray_dirs = []
    
    # We will just build the big list of directions
    for phi in phi_grid:
        q = euler_to_quaternion(np.deg2rad(phi), 0, 0)
        q_inv = q.conjugate()
        for az in az_grid[:-1]:
            az_rad = np.radians(az)
            jam_world = np.array([np.cos(az_rad), np.sin(az_rad), 0.0])
            jam_body = rotate_points(jam_world.reshape(1, 3), q_inv)[0]
            
            # Push origin out 1cm to escape dense local mesh geometry
            ray_origins = antennas_body + normals * 0.01
            ray_dirs = np.tile(jam_body, (N_ant, 1))
            
            all_ray_origins.append(ray_origins)
            all_ray_dirs.append(ray_dirs)
            
    all_ray_origins = np.vstack(all_ray_origins)
    all_ray_dirs = np.vstack(all_ray_dirs)
    
    print(f"Tracing {len(all_ray_origins)} rays for LUT in chunks...")
    
    chunk_size = 16
    hits_list = []
    for k_chunk in range(0, len(all_ray_origins), chunk_size):
        end = min(k_chunk + chunk_size, len(all_ray_origins))
        chunk_origins = all_ray_origins[k_chunk:end]
        chunk_dirs = all_ray_dirs[k_chunk:end]
        hits_list.append(intersector.intersects_any(chunk_origins, chunk_dirs))
        
    hits = np.concatenate(hits_list)
    print("Ray tracing complete!")
    masks = (~hits).astype(float).reshape(N_phi, N_az, N_ant)
    
    # Sum over antennas
    lut_data[:, :-1] = masks.sum(axis=2)
    
    # Guarantee perfect periodicity at the 360 degree seam
    lut_data[:, -1] = lut_data[:, 0]
    
    assert np.array_equal(lut_data[:, -1], lut_data[:, 0]), "Periodicity assert failed!"
            
    interp = RegularGridInterpolator((phi_grid, az_grid), lut_data, method='nearest', bounds_error=False, fill_value=0)
    logging.info("LUT Generation Complete.")
    return interp

def simulate_turn(autopilot, jammer_pos_xy, heading0, heading_target,
                  V=25.0, dt=0.1, desired_phi=30.0, max_steps=2000):
    heading = heading0
    ac_pos = np.zeros(2) # Drone stationary in XY for this demo, just turning
    log = []
    
    for step in range(max_steps):
        az, el = world_to_body_bearing(jammer_pos_xy, ac_pos, heading)
        
        if autopilot is None:
            phi_cmd = desired_phi
            # Hack: We still want to log active elements, so we pass a dummy interpolator or run the true LUT
            # For simplicity, if autopilot is None, we just assume active=0 if > 13 deg (based on Sweep 1)
            # Actually, let's require the LUT to be passed to log it properly.
            pass # See the loop below
        else:
            phi_cmd = autopilot.command_bank(desired_phi, az, el)
            
        # Update heading
        heading += np.degrees(G * np.tan(np.radians(phi_cmd)) / V) * dt
        
        # Log
        log.append({'t': step*dt, 'heading': heading, 'phi': phi_cmd, 'az': az})
        
        if abs(heading - heading_target) < 1.0 or heading > heading_target:
            break
            
    return log

def main():
    logging.basicConfig(level=logging.INFO)
    lut = generate_lut()
    
    # Setup
    jammer_pos_xy = np.array([0.0, 1000.0]) # Jammer 1km to the East (Y-axis)
    heading0 = 0.0 # Facing North
    heading_target = 90.0 # Turning right towards East
    
    # Baseline (No Autopilot constraint)
    # We use the LUT just to check active count, without constraining the bank.
    baseline_log = []
    heading = heading0
    ac_pos = np.zeros(2)
    V, dt, desired_phi = 25.0, 0.1, 30.0
    for step in range(2000):
        az, el = world_to_body_bearing(jammer_pos_xy, ac_pos, heading)
        phi_cmd = desired_phi
        heading += np.degrees(G * np.tan(np.radians(phi_cmd)) / V) * dt
        new_az, _ = world_to_body_bearing(jammer_pos_xy, ac_pos, heading)
        active = lut(np.array([[phi_cmd, new_az]]))[0]
        baseline_log.append({'t': step*dt, 'heading': heading, 'phi': phi_cmd, 'active': active})
        if heading >= heading_target: break

    # Cognitive Autopilot
    cognitive = CognitiveAutopilot(lut, min_active=3)
    cog_log = []
    heading = heading0
    for step in range(2000):
        az, el = world_to_body_bearing(jammer_pos_xy, ac_pos, heading)
        phi_cmd = cognitive.command_bank(desired_phi, az, el, V=V, dt=dt)
        heading += np.degrees(G * np.tan(np.radians(phi_cmd)) / V) * dt
        new_az, _ = world_to_body_bearing(jammer_pos_xy, ac_pos, heading)
        active = lut(np.array([[phi_cmd, new_az]]))[0]
        cog_log.append({'t': step*dt, 'heading': heading, 'phi': phi_cmd, 'active': active})
        if heading >= heading_target: break

    # Analysis
    base_uptime = sum(1 for x in baseline_log if x['active'] >= 3) / len(baseline_log) * 100
    cog_uptime = sum(1 for x in cog_log if x['active'] >= 3) / len(cog_log) * 100
    
    base_time = baseline_log[-1]['t']
    cog_time = cog_log[-1]['t']
    
    print(f"Baseline Turn: {base_time:.1f}s | Comms Uptime: {base_uptime:.1f}%")
    print(f"Cognitive Turn: {cog_time:.1f}s | Comms Uptime: {cog_uptime:.1f}%")

    # Plotting first trajectory
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    
    t_base = [x['t'] for x in baseline_log]
    h_base = [x['heading'] for x in baseline_log]
    p_base = [x['phi'] for x in baseline_log]
    a_base = [x['active'] for x in baseline_log]
    
    t_cog = [x['t'] for x in cog_log]
    h_cog = [x['heading'] for x in cog_log]
    p_cog = [x['phi'] for x in cog_log]
    a_cog = [x['active'] for x in cog_log]
    
    ax1.plot(t_base, h_base, 'r--', label='Baseline')
    ax1.plot(t_cog, h_cog, 'g-', label='Cognitive Autopilot')
    ax1.set_ylabel('Heading (deg)')
    ax1.grid(True)
    ax1.legend()
    
    ax2.plot(t_base, p_base, 'r--')
    ax2.plot(t_cog, p_cog, 'g-')
    ax2.set_ylabel('Bank Angle (deg)')
    ax2.grid(True)
    
    ax3.plot(t_base, a_base, 'r--')
    ax3.plot(t_cog, a_cog, 'g-')
    ax3.axhline(y=3, color='orange', linestyle='--')
    ax3.set_ylabel('Active Elements')
    ax3.set_xlabel('Time (s)')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig("cognitive_autopilot_results.png")
    logging.info("Saved plot to cognitive_autopilot_results.png")
    
    # ---------------------------------------------------------
    # SECONDARY TRAJECTORY TEST (Confirming Lack of Chattering)
    # ---------------------------------------------------------
    # Jammer East, Turn from South (180) to West (270)
    # ---------------------------------------------------------
    jammer_pos_xy2 = np.array([0.0, 1000.0]) # Jammer 1km East
    heading0_2 = 180.0 # Facing South
    heading_target_2 = 270.0 # Turning right towards West
    
    cog_log2 = []
    heading2 = heading0_2
    ac_pos2 = np.zeros(2)
    for step in range(2000):
        az2, _ = world_to_body_bearing(jammer_pos_xy2, ac_pos2, heading2)
        phi_cmd2 = cognitive.command_bank(desired_phi, az2, 0.0, V=V, dt=dt)
        heading2 += np.degrees(G * np.tan(np.radians(phi_cmd2)) / V) * dt
        new_az2, _ = world_to_body_bearing(jammer_pos_xy2, ac_pos2, heading2)
        active2 = lut(np.array([[phi_cmd2, new_az2]]))[0]
        cog_log2.append({'t': step*dt, 'heading': heading2, 'phi': phi_cmd2, 'active': active2})
        if heading2 >= heading_target_2: break
        
    cog_uptime2 = sum(1 for x in cog_log2 if x['active'] >= 3) / len(cog_log2) * 100
    
    print(f"Trajectory 2 (180->270): Turn time: {cog_log2[-1]['t']:.1f}s | Comms Uptime: {cog_uptime2:.1f}%")
    
    fig2, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    ax1.plot([x['t'] for x in cog_log2], [x['heading'] for x in cog_log2], 'g-')
    ax1.set_ylabel('Heading (deg)')
    ax1.grid(True)
    ax1.set_title("Trajectory 2 (South to West) against East Jammer")
    ax2.plot([x['t'] for x in cog_log2], [x['phi'] for x in cog_log2], 'g-')
    ax2.set_ylabel('Bank Angle (deg)')
    ax2.grid(True)
    ax3.plot([x['t'] for x in cog_log2], [x['active'] for x in cog_log2], 'g-')
    ax3.axhline(y=3, color='orange', linestyle='--')
    ax3.set_ylabel('Active Elements')
    ax3.set_xlabel('Time (s)')
    ax3.grid(True)
    plt.tight_layout()
    plt.savefig("cognitive_autopilot_traj2.png")
    logging.info("Saved plot to cognitive_autopilot_traj2.png")
    
    # ---------------------------------------------------------
    # TERTIARY TRAJECTORY TEST (Stress-testing the boundary)
    # ---------------------------------------------------------
    # Jammer East, Turn from West (-180) to East (0)
    # This sweeps az_body from 180 down to 0, where the LUT limit fluctuates wildly!
    # ---------------------------------------------------------
    jammer_pos_xy3 = np.array([1000.0, 0.0]) # Jammer East
    heading0_3 = -180.0 # Facing West
    heading_target_3 = 0.0 # Turning right towards East
    
    cog_log3 = []
    heading3 = heading0_3
    ac_pos3 = np.zeros(2)
    for step in range(4000):
        az3, _ = world_to_body_bearing(jammer_pos_xy3, ac_pos3, heading3)
        phi_cmd3 = cognitive.command_bank(desired_phi, az3, 0.0, V=V, dt=dt)
        heading3 += np.degrees(G * np.tan(np.radians(phi_cmd3)) / V) * dt
        new_az3, _ = world_to_body_bearing(jammer_pos_xy3, ac_pos3, heading3)
        active3 = lut(np.array([[phi_cmd3, new_az3]]))[0]
        cog_log3.append({'t': step*dt, 'heading': heading3, 'phi': phi_cmd3, 'active': active3})
        if heading3 >= heading_target_3: break
        
    cog_uptime3 = sum(1 for x in cog_log3 if x['active'] >= 3) / len(cog_log3) * 100
    
    print(f"Trajectory 3 (-180->0): Turn time: {cog_log3[-1]['t']:.1f}s | Comms Uptime: {cog_uptime3:.1f}%")
    
    fig3, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    ax1.plot([x['t'] for x in cog_log3], [x['heading'] for x in cog_log3], 'g-')
    ax1.set_ylabel('Heading (deg)')
    ax1.grid(True)
    ax1.set_title("Trajectory 3 (West to East) against East Jammer")
    ax2.plot([x['t'] for x in cog_log3], [x['phi'] for x in cog_log3], 'g-')
    ax2.set_ylabel('Bank Angle (deg)')
    ax2.grid(True)
    ax3.plot([x['t'] for x in cog_log3], [x['active'] for x in cog_log3], 'g-')
    ax3.axhline(y=3, color='orange', linestyle='--')
    ax3.set_ylabel('Active Elements')
    ax3.set_xlabel('Time (s)')
    ax3.grid(True)
    plt.tight_layout()
    plt.savefig("cognitive_autopilot_traj3.png")
    logging.info("Saved plot to cognitive_autopilot_traj3.png")

if __name__ == "__main__":
    main()
