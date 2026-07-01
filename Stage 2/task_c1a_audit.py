import numpy as np
import matplotlib.pyplot as plt
import os

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
        return J, np.inf, det_J
        
    J_inv = np.linalg.inv(J)
    gdop = np.sqrt(np.trace(J_inv)) * sigma_theta_rad
    return J, gdop, det_J

def run_gdop_sensitivity_audit():
    print("Running Task C1.a GDOP Sensitivity Audit...")
    jammer_pos = (15000.0, 0.0)
    sigma_theta_deg = 0.16
    sigma_theta_rad = np.radians(sigma_theta_deg)
    
    # Base straight-line approach from 5000m inward to 2000m
    x_vals = np.linspace(-5000, -2000, 75)
    y_vals = np.zeros_like(x_vals)
    base_trajectory = list(zip(x_vals, y_vals))
    
    # We trigger the S-curve at x = -2000
    trigger_x = -2000.0
    
    dy_sweep = np.arange(10.0, 2001.0, 10.0)
    
    gdop_list = []
    det_list = []
    sig_x_list = []
    sig_y_list = []
    penalty_list = []
    
    V_MPS = 50.0
    tau_exposure = 120.0 # Time constant for kinetic exposure penalty
    
    smallest_dy_gdop5 = None
    
    for dy in dy_sweep:
        # Evaluate FIM at the deviation waypoint
        test_path = base_trajectory + [(trigger_x, dy)]
        J, gdop, det_J = compute_fim_and_gdop(test_path, jammer_pos, sigma_theta_rad)
        
        gdop_list.append(gdop)
        det_list.append(det_J)
        
        if det_J > 1e-15:
            J_inv = np.linalg.inv(J)
            sig_x = np.sqrt(J_inv[0, 0])
            sig_y = np.sqrt(J_inv[1, 1])
        else:
            sig_x = np.inf
            sig_y = np.inf
            
        sig_x_list.append(sig_x)
        sig_y_list.append(sig_y)
        
        # Extra flight distance: straight-line hypotenuse to waypoint and back
        # Actually from (-2000, 0) to (-2000, dy) is dy. Then back is dy.
        extra_dist = 2 * dy
        extra_time = extra_dist / V_MPS
        survival_penalty = 1.0 - np.exp(-extra_time / tau_exposure) # Fraction of P_mcs lost
        penalty_list.append(survival_penalty * 100.0) # In percent
        
        if smallest_dy_gdop5 is None and gdop < 5.0:
            smallest_dy_gdop5 = dy
            
    print(f"\\nSmallest lateral deviation achieving GDOP < 5.0: {smallest_dy_gdop5} m")
    
    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # 1. GDOP vs dy
    axes[0].plot(dy_sweep, gdop_list, 'b-', linewidth=2)
    axes[0].axhline(5.0, color='r', linestyle='--', label='GDOP = 5')
    if smallest_dy_gdop5:
        axes[0].axvline(smallest_dy_gdop5, color='g', linestyle=':', label=f'dy={smallest_dy_gdop5}m')
    axes[0].set_ylim(0, 20)
    axes[0].set_xlabel('Lateral Deviation dy (m)')
    axes[0].set_ylabel('GDOP')
    axes[0].set_title('GDOP vs Lateral Deviation')
    axes[0].grid(True)
    axes[0].legend()
    
    # 2. det(FIM) vs dy
    axes[1].plot(dy_sweep, det_list, 'k-', linewidth=2)
    axes[1].set_xlabel('Lateral Deviation dy (m)')
    axes[1].set_ylabel('det(FIM)')
    axes[1].set_title('Determinant of FIM vs Lateral Deviation')
    axes[1].grid(True)
    
    # 3. sigma_x and sigma_y vs dy
    axes[2].plot(dy_sweep, sig_x_list, 'r-', label='sigma_x (down-range)')
    axes[2].plot(dy_sweep, sig_y_list, 'g-', label='sigma_y (cross-range)')
    axes[2].set_yscale('log')
    axes[2].set_xlabel('Lateral Deviation dy (m)')
    axes[2].set_ylabel('CRLB Error (m)')
    axes[2].set_title('CRLB Errors vs Lateral Deviation')
    axes[2].grid(True)
    axes[2].legend()
    
    # 4. P_mcs penalty vs dy
    axes[3].plot(dy_sweep, penalty_list, 'm-', linewidth=2)
    axes[3].set_xlabel('Lateral Deviation dy (m)')
    axes[3].set_ylabel('Survival Penalty (% absolute)')
    axes[3].set_title('Mission Survival Penalty vs Deviation')
    axes[3].grid(True)
    
    plt.tight_layout()
    plt.savefig('task_c1a_audit.png', dpi=300)
    print("Saved plot to task_c1a_audit.png")

if __name__ == "__main__":
    run_gdop_sensitivity_audit()
