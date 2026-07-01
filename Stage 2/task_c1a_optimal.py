import numpy as np

def compute_fim_and_gdop(uav_positions, jammer_pos, sigma_theta_rad):
    xj, yj = jammer_pos
    J = np.zeros((2, 2))
    
    for i in range(len(uav_positions)):
        xi, yi = uav_positions[i]
        ri_sq = (xj - xi)**2 + (yj - yi)**2
        dtheta_dxj = -(yj - yi) / ri_sq
        dtheta_dyj = (xj - xi) / ri_sq
        hi = np.array([dtheta_dxj, dtheta_dyj])
        J += (1.0 / sigma_theta_rad**2) * np.outer(hi, hi)
        
    det_J = np.linalg.det(J)
    if det_J < 1e-15:
        return J, np.inf, np.inf, np.inf
        
    J_inv = np.linalg.inv(J)
    gdop = np.sqrt(np.trace(J_inv)) * sigma_theta_rad
    sig_x = np.sqrt(J_inv[0, 0])
    sig_y = np.sqrt(J_inv[1, 1])
    return J, gdop, sig_x, sig_y

def run_table():
    jammer_pos = (15000.0, 0.0)
    sigma_theta_deg = 0.16
    sigma_theta_rad = np.radians(sigma_theta_deg)
    
    x_vals = np.linspace(-5000, -2000, 75)
    y_vals = np.zeros_like(x_vals)
    base_trajectory = list(zip(x_vals, y_vals))
    trigger_x = -2000.0
    
    dy_sweep = [460, 600, 800, 1000, 1200, 1500, 2000]
    V_MPS = 50.0
    tau_exposure = 120.0
    
    print("| dy (m) | GDOP | sig_x (m) | sig_y (m) | Extra Time (s) | Survival Penalty (%) |")
    print("| ------ | ---- | --------- | --------- | -------------- | -------------------- |")
    
    for dy in dy_sweep:
        test_path = base_trajectory + [(trigger_x, dy)]
        J, gdop, sig_x, sig_y = compute_fim_and_gdop(test_path, jammer_pos, sigma_theta_rad)
        
        extra_dist = 2 * dy
        extra_time = extra_dist / V_MPS
        penalty = (1.0 - np.exp(-extra_time / tau_exposure)) * 100.0
        
        print(f"| {dy} | {gdop:.2f} | {sig_x:.1f} | {sig_y:.1f} | {extra_time:.1f} | {penalty:.1f} |")

if __name__ == "__main__":
    run_table()
