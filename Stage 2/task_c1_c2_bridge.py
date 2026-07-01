import numpy as np
import matplotlib.pyplot as plt

def generate_trajectory(dy):
    x_vals = np.linspace(-5000, -2000, 75)
    y_vals = np.zeros_like(x_vals)
    path = list(zip(x_vals, y_vals))
    path.append((-2000, dy))
    x_vals_end = np.linspace(-1900, -750, 25)
    y_vals_end = np.zeros_like(x_vals_end)
    path.extend(list(zip(x_vals_end, y_vals_end)))
    return np.array(path)

def run_ekf(path, jammer_pos, sigma_theta_rad):
    N_steps = len(path)
    x_est = np.array([10000.0, 1000.0, 0.0, 0.0])
    P_est = np.diag([1e8, 1e8, 1e4, 1e4])
    Q = np.diag([1.0, 1.0, 0.1, 0.1])
    R = np.array([[sigma_theta_rad**2]])
    dt = 1.0
    F = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    xj_true, yj_true = jammer_pos
    
    ekf_angles = []
    raw_angles = []
    true_angles = []
    
    for i in range(N_steps):
        xu, yu = path[i]
        
        true_bearing = np.arctan2(yj_true - yu, xj_true - xu)
        z = true_bearing + np.random.normal(0, sigma_theta_rad)
        
        x_pred = F @ x_est
        P_pred = F @ P_est @ F.T + Q
        
        xj_pred, yj_pred = x_pred[0], x_pred[1]
        dx = xj_pred - xu
        dy = yj_pred - yu
        r2 = dx**2 + dy**2
        if r2 < 1e-6: r2 = 1e-6
            
        H = np.array([[-dy/r2, dx/r2, 0.0, 0.0]])
        h_x = np.arctan2(dy, dx)
        
        y_res = z - h_x
        y_res = (y_res + np.pi) % (2*np.pi) - np.pi
        
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        x_est = x_pred + K.flatten() * y_res
        I_KH = np.eye(4) - K @ H
        P_est = I_KH @ P_pred @ I_KH.T + K @ R @ K.T
        
        # Calculate EKF bearing
        ekf_bearing = np.arctan2(x_est[1] - yu, x_est[0] - xu)
        
        ekf_angles.append(ekf_bearing)
        raw_angles.append(z)
        true_angles.append(true_bearing)
        
    return np.array(ekf_angles), np.array(raw_angles), np.array(true_angles)

def run_bridge_experiment():
    print("Running C1->C2 Bridge Experiment...")
    jammer_pos = (15000.0, 0.0)
    sigma_theta_rad = np.radians(0.16)
    
    dy = 800.0
    path = generate_trajectory(dy)
    
    trials = 1000
    ekf_errors = []
    raw_errors = []
    
    np.random.seed(42)
    
    for _ in range(trials):
        ekf_ang, raw_ang, true_ang = run_ekf(path, jammer_pos, sigma_theta_rad)
        
        # Angular error in degrees
        err_ekf = np.degrees((ekf_ang - true_ang + np.pi) % (2*np.pi) - np.pi)
        err_raw = np.degrees((raw_ang - true_ang + np.pi) % (2*np.pi) - np.pi)
        
        ekf_errors.extend(err_ekf)
        raw_errors.extend(err_raw)
        
    ekf_errors = np.array(ekf_errors)
    raw_errors = np.array(raw_errors)
    
    mean_ekf = np.mean(np.abs(ekf_errors))
    rms_ekf = np.sqrt(np.mean(ekf_errors**2))
    
    p1 = np.mean(np.abs(ekf_errors) > 1.0) * 100
    p2 = np.mean(np.abs(ekf_errors) > 2.0) * 100
    p5 = np.mean(np.abs(ekf_errors) > 5.0) * 100
    
    print("\\nEKF Bearing Error Statistics (1000 trials, dy=800m):")
    print(f"Mean Absolute Error: {mean_ekf:.4f}°")
    print(f"RMS Error:           {rms_ekf:.4f}°")
    print(f"P(|err| > 1°):       {p1:.2f}%")
    print(f"P(|err| > 2°):       {p2:.2f}%")
    print(f"P(|err| > 5°):       {p5:.2f}%")
    
    # Compare to raw
    rms_raw = np.sqrt(np.mean(raw_errors**2))
    print(f"\\nRaw MUSIC RMS Error: {rms_raw:.4f}°")

if __name__ == "__main__":
    run_bridge_experiment()
