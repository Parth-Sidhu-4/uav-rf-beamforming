import numpy as np
import matplotlib.pyplot as plt

def generate_trajectory(dy):
    x_vals = np.linspace(-5000, -2000, 75)
    y_vals = np.zeros_like(x_vals)
    path = list(zip(x_vals, y_vals))
    
    # Add S-curve
    path.append((-2000, dy))
    
    # Continue towards target
    x_vals_end = np.linspace(-1900, -750, 25)
    y_vals_end = np.zeros_like(x_vals_end)
    path.extend(list(zip(x_vals_end, y_vals_end)))
    return np.array(path)

def run_ekf(path, jammer_pos, sigma_theta_rad):
    N_steps = len(path)
    
    # State: [x, y, vx, vy]
    # Initialize somewhere reasonably ahead of the UAV to prevent EKF linearization divergence
    x_est = np.array([10000.0, 1000.0, 0.0, 0.0]) # Rough guess 10km ahead
    P_est = np.diag([1e8, 1e8, 1e4, 1e4]) # High initial uncertainty
    
    Q = np.diag([1.0, 1.0, 0.1, 0.1]) # Small process noise (assume jammer is mostly stationary)
    R = np.array([[sigma_theta_rad**2]])
    
    dt = 1.0 # arbitrary time step, since we care about geometry mostly
    F = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    xj_true, yj_true = jammer_pos
    
    history_x = []
    history_P = []
    nees_history = []
    raw_errors = []
    
    for i in range(N_steps):
        xu, yu = path[i]
        
        # 1. True measurement + noise
        true_bearing = np.arctan2(yj_true - yu, xj_true - xu)
        z = true_bearing + np.random.normal(0, sigma_theta_rad)
        
        # Predict
        x_pred = F @ x_est
        P_pred = F @ P_est @ F.T + Q
        
        # 2. Measurement Update (Jacobian evaluated at PREDICTED state)
        xj_pred, yj_pred = x_pred[0], x_pred[1]
        
        dx = xj_pred - xu
        dy = yj_pred - yu
        r2 = dx**2 + dy**2
        
        # Protect against exact singularity
        if r2 < 1e-6:
            r2 = 1e-6
            
        H = np.array([[-dy/r2, dx/r2, 0.0, 0.0]])
        
        # Expected measurement
        h_x = np.arctan2(dy, dx)
        
        # Innovation
        y_res = z - h_x
        # Wrap to pi
        y_res = (y_res + np.pi) % (2*np.pi) - np.pi
        
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        
        x_est = x_pred + K.flatten() * y_res
        
        # Joseph form for numerical stability
        I_KH = np.eye(4) - K @ H
        P_est = I_KH @ P_pred @ I_KH.T + K @ R @ K.T
        
        history_x.append(x_est.copy())
        history_P.append(P_est.copy())
        
        # NEES: (x - x_true)^T P^-1 (x - x_true)
        err_vec = x_est[:2] - np.array([xj_true, yj_true])
        P_pos = P_est[:2, :2]
        nees = err_vec.T @ np.linalg.inv(P_pos) @ err_vec
        nees_history.append(nees)
        
        # Compute "raw" position error from single bearing line?
        # A single bearing line does not give a position fix. 
        # But we can calculate the cross-range error at the true distance:
        true_dist = np.sqrt((xj_true - xu)**2 + (yj_true - yu)**2)
        raw_cross_range_err = true_dist * np.tan(z - true_bearing)
        raw_errors.append(np.abs(raw_cross_range_err))

    return np.array(history_x), np.array(history_P), nees_history, raw_errors

def plot_ekf_results():
    jammer_pos = (15000.0, 0.0)
    sigma_theta_rad = np.radians(0.16)
    
    dy_vals = [460, 800, 1200]
    
    final_rmse_list = []
    
    plt.figure(figsize=(15, 10))
    
    for idx, dy in enumerate(dy_vals):
        np.random.seed(42) # For reproducible comparisons
        path = generate_trajectory(dy)
        hist_x, hist_P, nees, raw_err = run_ekf(path, jammer_pos, sigma_theta_rad)
        
        # Compute RMSE
        err_x = hist_x[:, 0] - jammer_pos[0]
        err_y = hist_x[:, 1] - jammer_pos[1]
        pos_rmse = np.sqrt(err_x**2 + err_y**2)
        
        final_rmse_list.append(pos_rmse[-1])
        
        plt.subplot(2, 2, 1)
        plt.plot(pos_rmse, label=f'dy={dy}m')
        plt.yscale('log')
        plt.xlabel('Measurement Step')
        plt.ylabel('Position RMSE (m)')
        plt.title('EKF Position RMSE vs Time')
        
        if dy == 800:
            plt.subplot(2, 2, 2)
            plt.plot(nees, 'b-', label='NEES')
            plt.axhline(2.0, color='r', linestyle='--', label='Expected Mean (dof=2)')
            # 95% chi-square bounds for dof=2 are approx 0.051 to 7.378
            plt.axhline(7.378, color='g', linestyle=':', label='95% Upper Bound')
            plt.ylim(0, 15)
            plt.xlabel('Measurement Step')
            plt.ylabel('NEES')
            plt.title('NEES Consistency Check (dy=800m)')
            plt.legend()
            
            plt.subplot(2, 2, 3)
            # Compare EKF vs raw
            plt.plot(pos_rmse, 'b-', label='EKF Position RMSE')
            plt.plot(raw_err, 'r.', alpha=0.3, label='Raw Single-Bearing Cross-Range Error')
            plt.yscale('log')
            plt.xlabel('Measurement Step')
            plt.ylabel('Error (m)')
            plt.title('Smoothing: EKF vs Raw Measurement (dy=800m)')
            plt.legend()
            
            plt.subplot(2, 2, 4)
            # Covariance bounds over time for dy=800
            sig_x = np.sqrt(hist_P[:, 0, 0])
            sig_y = np.sqrt(hist_P[:, 1, 1])
            plt.plot(sig_x, 'm-', label='sigma_x (EKF P)')
            plt.plot(sig_y, 'c-', label='sigma_y (EKF P)')
            plt.yscale('log')
            plt.xlabel('Measurement Step')
            plt.ylabel('Standard Deviation (m)')
            plt.title('EKF Filter Covariance Convergence (dy=800m)')
            plt.legend()
            
    plt.subplot(2, 2, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig('task_c1b_ekf.png', dpi=300)
    print("Saved EKF plots to task_c1b_ekf.png")
    
    print("\\nEKF Final RMSE vs Maneuver Size:")
    for dy, rmse in zip(dy_vals, final_rmse_list):
        print(f"dy = {dy}m  -> Final RMSE = {rmse:.1f} m")

if __name__ == "__main__":
    plot_ekf_results()
