import numpy as np
import matplotlib.pyplot as plt

def compute_fim_and_gdop(uav_positions, jammer_pos, sigma_theta_rad):
    xj, yj = jammer_pos
    J = np.zeros((2, 2))
    
    for i in range(len(uav_positions)):
        xi, yi = uav_positions[i]
        ri_sq = (xj - xi)**2 + (yj - yi)**2
        
        # Partial derivatives
        dtheta_dxj = -(yj - yi) / ri_sq
        dtheta_dyj = (xj - xi) / ri_sq
        
        hi = np.array([dtheta_dxj, dtheta_dyj])
        J += (1.0 / sigma_theta_rad**2) * np.outer(hi, hi)
        
    det_J = np.linalg.det(J)
    
    if det_J < 1e-15:
        return J, np.inf
        
    J_inv = np.linalg.inv(J)
    gdop = np.sqrt(np.trace(J_inv)) * sigma_theta_rad
    return J, gdop

def run_gdop_analysis():
    # Jammer at 15km standoff
    jammer_pos = (15000.0, 0.0)
    target_pos = (0.0, 0.0)
    
    # UAV approaches from 5000m to 750m along x-axis
    # Let's say UAV starts at x = -5000, target is at 0, jammer is at +15000
    # Wait, the baseline defeat is 5000m. If jammer is 15km from target,
    # and UAV approaches target from the opposite side, it's 20km from jammer at start.
    # Let's use x_uav from -5000 to -750, y_uav = 0.
    
    # We need sigma_theta from Phase B. 
    # Let's assume sigma_theta_deg = 0.5 degrees for now (we'll update this if needed)
    # The exact value doesn't change GDOP, only the FIM magnitude and CRLB.
    # From Phase B, MUSIC RMSE at operational point was around 0.16 deg
    sigma_theta_deg = 0.16
    sigma_theta_rad = np.radians(sigma_theta_deg)
    print(f"Using MUSIC RMSE: {sigma_theta_deg} deg")
    
    x_vals = np.linspace(-5000, -750, 100)
    y_vals = np.zeros_like(x_vals)
    uav_trajectory = list(zip(x_vals, y_vals))
    
    gdops = []
    measurement_counts = []
    
    for i in range(1, len(uav_trajectory)+1):
        path_so_far = uav_trajectory[:i]
        J, gdop = compute_fim_and_gdop(path_so_far, jammer_pos, sigma_theta_rad)
        gdops.append(gdop)
        measurement_counts.append(i)
        
        # Check trigger condition
        # If we have taken enough measurements and GDOP is still > 5 before 2000m from target
        # (x = -2000), we trigger S-curve.
        
    # We expect GDOP to be infinity everywhere because of collinearity
    print(f"Collinear GDOP at step 10: {gdops[9]}")
    print(f"Collinear GDOP at step 50: {gdops[49]}")
    
    # -------------------------------------------------------------
    # TRIGGER S-CURVE
    # -------------------------------------------------------------
    # At x = -2000, we check. If GDOP > 5, we sweep dy.
    trigger_x = -2000.0
    # Find index closest to -2000
    trigger_idx = np.argmin(np.abs(x_vals - trigger_x))
    
    if gdops[trigger_idx] > 5.0:
        print("\\n--- GDOP > 5 Detected. Triggering S-Curve ---")
        path_before_trigger = uav_trajectory[:trigger_idx+1]
        
        dy_sweep = np.arange(100.0, 2001.0, 100.0)
        best_dy = 0
        max_det = -1.0
        best_J = None
        best_gdop = np.inf
        
        for dy in dy_sweep:
            # S-curve: fly to (trigger_x, dy) for one measurement
            # Actually, to make it a realistic maneuver, we fly there. 
            # We'll just evaluate the FIM at the deviation waypoint combined with past measurements.
            test_path = path_before_trigger + [(trigger_x, dy)]
            J, gdop = compute_fim_and_gdop(test_path, jammer_pos, sigma_theta_rad)
            det_J = np.linalg.det(J)
            
            if det_J > max_det:
                max_det = det_J
                best_dy = dy
                best_J = J
                best_gdop = gdop
                
        print(f"Optimal Lateral Deviation (dy): {best_dy} m")
        print(f"Resulting GDOP at deviation waypoint: {best_gdop:.4f}")
        
        # CRLB output
        J_inv = np.linalg.inv(best_J)
        sigma_xj = np.sqrt(J_inv[0, 0])
        sigma_yj = np.sqrt(J_inv[1, 1])
        print(f"CRLB sigma_xj: {sigma_xj:.2f} m")
        print(f"CRLB sigma_yj: {sigma_yj:.2f} m")
        
        # P_mcs penalty
        # Extra distance flown: instead of straight from -2000 to -1900, 
        # it flies from (-2000,0) to (-2000, best_dy) and then back to (-1900, 0).
        # Normal distance: 100m. 
        # S-curve distance: sqrt(100^2 + best_dy^2) + sqrt(100^2 + best_dy^2) approx = 2 * best_dy
        # Let's say it just adds 2 * best_dy to the total flight path.
        extra_dist = 2 * best_dy
        V_MPS = 50.0
        extra_time = extra_dist / V_MPS
        # Penalty = exp(-extra_time / 120.0) factor reduction in survival.
        # But wait, this assumes it happens during blind flight. 
        # Actually, the S-curve happens at 2000m, where the link is STILL ALIVE (because MUSIC maintains it).
        # So there is NO kinetic penalty from blind flight! It's just a delay.
        # But it does use up fuel or adds time to the mission.
        print(f"Trajectory deviation cost: {extra_dist:.1f} m extra flight distance ({extra_time:.1f} s delay)")
        print(f"Since this occurs at 2000m (link active), kinetic survival penalty is 0, but total mission time increases.")

    else:
        print("Trajectory is observable.")
        
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(measurement_counts, gdops, 'b-', label='Collinear GDOP')
    plt.axhline(y=5.0, color='r', linestyle='--', label='Usable Threshold (GDOP=5)')
    plt.yscale('log')
    plt.xlabel('Measurement Count (Progressing Inward)')
    plt.ylabel('GDOP')
    plt.title('GDOP vs Measurement Count (Collinear Approach)')
    plt.grid(True)
    plt.legend()
    plt.savefig('task_c1a_gdop.png', dpi=300)
    print("Saved task_c1a_gdop.png")

if __name__ == "__main__":
    run_gdop_analysis()
