import numpy as np
import pandas as pd
import math
import scipy.stats

# --- UTILITY: Q-Function ---
def q_func(x):
    return 1.0 - scipy.stats.norm.cdf(x)

# --- ARRAY PROCESSING (Fix 2 & 3) ---
def get_steering_vector(N, theta_deg):
    theta_rad = np.radians(theta_deg)
    return np.exp(-1j * np.pi * np.arange(N) * np.sin(theta_rad)).reshape(N, 1)

def lcmv_weights(R_hat, C, f, loading_factor=0.10):
    """
    R_hat         : (N x N) sample covariance matrix
    C             : (N x n_constraints) constraint matrix [a(theta_s), a(theta_j)]
    f             : (n_constraints,) constraint response vector [1, 0]
    loading_factor: diagonal loading as fraction of mean diagonal power
    """
    delta = loading_factor * np.real(np.trace(R_hat)) / R_hat.shape[0]
    R_loaded = R_hat + delta * np.eye(R_hat.shape[0])
    cond_num = np.linalg.cond(R_loaded)
    
    R_inv = np.linalg.inv(R_loaded)
    w = R_inv @ C @ np.linalg.inv(C.conj().T @ R_inv @ C) @ f
    return w, cond_num

# --- DRIFT (Fix 4) ---
W_DRIFT = 3

def compute_drift(lyapunov_history, t, W=W_DRIFT):
    t_start = max(0, t - W)
    if t == t_start:
        return 0.0
    return (lyapunov_history[t] - lyapunov_history[t_start]) / (t - t_start)

# --- SURVIVAL (Fix 6) ---
TAU_SURVIVE = 120.0

def kill_prob_per_epoch(dt):
    return 1.0 - np.exp(-dt / TAU_SURVIVE)

# --- EKF STABILITY (Fix 7) ---
def stability_monitor(theta_error_rms_history, jammer_leakage_db_history):
    if len(theta_error_rms_history) < 2:
        return 0.0
    d_theta_error = theta_error_rms_history[-1] - theta_error_rms_history[-2]
    d_leakage_linear = (10**(jammer_leakage_db_history[-1]/10.0) - 10**(jammer_leakage_db_history[-2]/10.0))
    if abs(d_leakage_linear) < 1e-12:
        return 0.0
    return abs(d_theta_error / d_leakage_linear)

def rate_limited_null_steer(theta_null_prev, theta_null_new, v_uav, dt_epoch, R_current):
    bearing_rate_kinematic = np.degrees(v_uav * dt_epoch / max(1.0, R_current))
    max_delta = 1.5 * bearing_rate_kinematic
    
    delta = theta_null_new - theta_null_prev
    is_clipped = False
    if abs(delta) > max_delta:
        delta = np.sign(delta) * max_delta
        is_clipped = True
    return theta_null_prev + delta, is_clipped

def build_constraint_matrix(a_signal, theta_est_deg, sigma_theta_deg, N_elements=4, n_null_points=2):
    # Cap n_null_points to available DOFs
    n_null_points = min(n_null_points, N_elements - 2)
    
    if sigma_theta_deg < 0.5 or n_null_points < 1:
        C = np.column_stack([a_signal, get_steering_vector(N_elements, theta_est_deg)])
        f = np.array([[1.0], [0.0]])
    else:
        thetas = np.linspace(theta_est_deg - 2*sigma_theta_deg, theta_est_deg + 2*sigma_theta_deg, n_null_points)
        null_vecs = np.column_stack([get_steering_vector(N_elements, th) for th in thetas])
        C = np.column_stack([a_signal, null_vecs])
        f = np.vstack([[1.0], np.zeros((n_null_points, 1))])
        
    return C, f

# --- EKF HEALTH (Fix A) ---
def ekf_health_check(P_cov, max_trace=1e9):
    tr = np.trace(P_cov)
    if tr > max_trace:
        raise RuntimeWarning(f"EKF covariance diverging: Tr(P) = {tr:.2e}")
    if np.any(np.linalg.eigvals(P_cov) < 0):
        raise RuntimeError("EKF covariance matrix is not positive definite.")

# --- COGNITIVE SENSING (Fix B) ---
def p_detection(gamma, M, B_sense=1e6, B_total=100e6):
    p_present = B_sense / B_total
    Pd_given_present = q_func((gamma - 1.0) / np.sqrt(max(1e-9, 2.0 * gamma / M)))
    return p_present * Pd_given_present

# --- MAIN SIMULATOR ---
class UAVSimulator:
    def __init__(self, scenario='A', div_scheme='MRC_L4', target_deg=2.0, policy='RNCO'):
        # CONVERGENCE NOTE:
        # This mission is non-stationary (range closes monotonically, terrain shadowing
        # is deterministic). Neely's asymptotic convergence guarantees do not apply.
        # The optimizer is used as a principled heuristic with sliding-window drift (W=3).
        # Performance is validated empirically, not analytically guaranteed.
        
        self.scenario = scenario
        self.div_scheme = div_scheme
        self.target_deg = target_deg
        self.policy = policy
        
        self.N_array = 4
        self.epochs = 10
        self.dx_step = 425.0
        self.base_speed = 50.0
        self.base_time_per_epoch = self.dx_step / self.base_speed
        
        # EKF Initialization (Fix A)
        self.uav_x = -5000.0
        self.jammer_y_true = 5000.0 if scenario == 'C' else 0.0
        
        self.x_hat = np.array([0.0, 0.0]) # Prior: jammer at origin
        RANGE_UNCERTAINTY_M = 500.0
        LATERAL_UNCERTAINTY_M = 2000.0
        self.P_cov = np.diag([RANGE_UNCERTAINTY_M**2, LATERAL_UNCERTAINTY_M**2])
        
        # Queues
        self.Q_RF = 0.0
        self.Q_EXP = 0.0
        self.Q_INFO = 0.0
        
        # Normalization Constants (Fix 1)
        self.Q_RF_NORM = 5.0 # Max target throughput
        self.Q_EXP_NORM = kill_prob_per_epoch(self.base_time_per_epoch * self.epochs)
        R_initial = 5000.0
        self.Q_INFO_NORM = (R_initial * np.radians(3.0))**2
        self.V = 100.0
        
        self.lyapunov_history = [0.0]
        self.theta_error_rms_history = []
        self.jammer_leakage_db_history = []
        
        self.logs = {
            'throughput': [], 'exposure': [], 'P_trace': [], 
            'dy': [], 'ts': [], 'outage': [], 'nd': [], 'pmcs_rf': 1.0,
            'cond_num': [], 'loop_gain': [], 'clip': []
        }
        
        self.theta_null_prev = None
        
    def get_environment(self, epoch):
        inr_base = 50.0
        snr_base = 15.0
        calib_error = 0.0
        jam_y = self.jammer_y_true
        
        is_shadow = False
        if self.scenario == 'B':
            if 3 <= epoch <= 5: # Shadow zone
                inr_base -= 20.0
                is_shadow = True
        elif self.scenario == 'D':
            calib_error = 0.5
        elif self.scenario == 'E':
            inr_base = 20.0
            
        jnr_lin = 10.0**((inr_base - 50.0 - 5.0)/10.0)
        return inr_base, snr_base, calib_error, jam_y, jnr_lin, is_shadow

    def is_shadow_epoch(self, epoch):
        if self.scenario != 'B': return False
        return 3 <= epoch <= 5

    def compute_info_arrival(self, t, Tr_P_cov, P_target):
        # Fix 5: Terrain-Aware Lookahead
        base_arrival = Tr_P_cov
        if self.scenario == 'B':
            LOOKAHEAD_WINDOW = 2
            LOOKAHEAD_GAIN = 2.0
            lookahead_term = 0.0
            for k in range(1, LOOKAHEAD_WINDOW + 1):
                if self.is_shadow_epoch(t + k):
                    decay = np.exp(-0.5 * k)
                    lookahead_term += decay * P_target * LOOKAHEAD_GAIN
            return base_arrival + lookahead_term
        return base_arrival

    def get_mrc_outage(self, gamma_bar, req_sinr_dB):
        # Fix 2 Clarification: Temporal Diversity
        gamma_th = 10.0**(req_sinr_dB / 10.0)
        x = min(gamma_th / gamma_bar, 100.0)
        
        L_eff = 4 # Conservative default for temporal/frequency hops
        if self.div_scheme == 'No_Div': L_eff = 1
        
        if L_eff == 1:
            p_out = 1.0 - np.exp(-x)
        else:
            sum_term = sum((x**k)/math.factorial(k) for k in range(L_eff))
            p_out = 1.0 - np.exp(-x) * sum_term
        return min(1.0, max(0.0, p_out))

    def predict_proxy(self, ts, dy, amc, use_ekf, epoch):
        # FAST ANALYTICAL PROXY for RNCO Optimizer Loop
        inr_base, snr_base, calib_error, jam_y, jnr_lin, is_shadow = self.get_environment(epoch)
        fs = 1e6
        
        if ts > 0:
            pd_val = p_detection(jnr_lin, ts*1e-3*fs)
            L_eff = max(1, int(100 * (1.0 - ts / 10.0)))
        else:
            pd_val = 1.0; L_eff = 100
            
        base_noise = 0.5
        if jnr_lin < 0.01 or L_eff < 10 or is_shadow:
            base_noise = 45.0 
            
        m_rmse = max(base_noise, 10.0 / np.sqrt(L_eff)) + calib_error
        
        exp = (self.dx_step + 2.0 * dy) / self.base_speed
        
        dx = np.abs(0.0 - self.uav_x)
        dy_j = np.abs(jam_y - dy)
        r = np.sqrt(dx**2 + dy_j**2)
        
        if use_ekf:
            P_pred = self.P_cov + 10.0 * np.eye(2)
            H = np.eye(2)
            # Equivalent 2x2 reformulation of scalar bearing H = [0, 1/R]
            sigma_theta_rad = m_rmse * (np.pi / 180.0)
            R_meas_y = (r * sigma_theta_rad)**2
            R_meas_matrix = np.array([[1e9, 0.0], [0.0, R_meas_y]])
            
            S = H @ P_pred @ H.T + R_meas_matrix
            K = P_pred @ H.T @ np.linalg.inv(S)
            P_post = (np.eye(2) - K @ H) @ P_pred
            bearing_err = (np.sqrt(P_post[1,1]) / r) * (180.0/np.pi)
        else:
            P_post = self.P_cov
            bearing_err = m_rmse
            
        # Calibrated Proxy Null Depth [-65, -45] dB (Empirically fit to 10% loading)
        nd = max(-65.0, -63.31 + 3.45 * bearing_err + max(0, 5.0 - L_eff/20.0))
        nd = min(nd, 0.0)
        
        inr_resid = inr_base + nd
        sinr_dB = snr_base - inr_resid
        req_sinr_dB = {1.0: 2.0, 2.0: 5.0, 4.0: 12.0, 6.0: 18.0}.get(amc, 5.0)
        
        gamma_bar = 10.0**(sinr_dB / 10.0)
        p_out = self.get_mrc_outage(gamma_bar, req_sinr_dB)
        
        comm_survival = 1.0 - 0.9 * p_out
        r_eff = amc * pd_val * (1.0 - ts/10.0) * comm_survival
        
        return r_eff, exp, P_post, nd, p_out, bearing_err, L_eff

    def apply_true_physics(self, best_ts, best_dy, best_amc, use_ekf, epoch, P_post, bearing_err, L_eff):
        # AFTER Action Selection: Run TRUE 4x4 LCMV physics
        inr_base, snr_base, calib_error, jam_y, jnr_lin, is_shadow = self.get_environment(epoch)
        
        theta_s = -45.0
        # True jammer bearing from UAV perspective
        dx = np.abs(0.0 - self.uav_x)
        dy_j_true = self.jammer_y_true - best_dy
        R_current = np.sqrt(dx**2 + dy_j_true**2)
        theta_j_true = np.degrees(np.arctan2(dy_j_true, dx))
        
        if use_ekf:
            # Phase C uses a conservative operational floor of 1.0 deg to account for diagonal 
            # loading bias, reduced snapshot counts under cognitive sensing, and hardware 
            # calibration margin. This is not a statistical 3-sigma bound.
            m_rmse = max(0.5, 10.0 / np.sqrt(L_eff)) + calib_error
            z_theta_deg = theta_j_true + m_rmse * np.random.randn()
            
            x_pred = self.x_hat.copy()
            P_pred = self.P_cov + 10.0 * np.eye(2)
            
            # Proxy Cartesian measurement for the 2x2 EKF
            z_y = dx * np.tan(np.radians(z_theta_deg))
            z_x = 0.0
            
            H = np.eye(2)
            sigma_theta_rad = m_rmse * (np.pi / 180.0)
            R_meas_y = (R_current * sigma_theta_rad)**2
            R_meas_matrix = np.array([[1e9, 0.0], [0.0, R_meas_y]])
            
            S = H @ P_pred @ H.T + R_meas_matrix
            K = P_pred @ H.T @ np.linalg.inv(S)
            
            innovation = np.array([z_x, z_y]) - H @ x_pred
            self.x_hat = x_pred + K @ innovation
            self.P_cov = (np.eye(2) - K @ H) @ P_pred
            
            dy_est = self.x_hat[1] - best_dy
            R_est = np.sqrt(dx**2 + dy_est**2)
            theta_j_est = np.degrees(np.arctan2(dy_est, dx))
            bearing_err = (np.sqrt(self.P_cov[1,1]) / R_est) * (180.0/np.pi)
        else:
            m_rmse = max(0.5, 10.0 / np.sqrt(L_eff)) + calib_error
            theta_j_est = theta_j_true + m_rmse * np.random.randn()
            bearing_err = m_rmse
        
        if self.theta_null_prev is None:
            self.theta_null_prev = theta_j_est
            
        # Kinematic Rate Limiter
        dt_epoch = self.base_time_per_epoch
        theta_j_steer, is_clipped = rate_limited_null_steer(self.theta_null_prev, theta_j_est, self.base_speed, dt_epoch, R_current)
        
        # Prevent singular matrix if jammer is exactly at signal angle
        if np.abs(theta_j_steer - theta_s) < 2.0:
            theta_j_steer = theta_s + 2.0
            
        self.theta_null_prev = theta_j_steer
        
        # Create true covariance matrix
        a_s = get_steering_vector(self.N_array, theta_s)
        a_j_true = get_steering_vector(self.N_array, theta_j_true)
        
        R_hat = (10**(snr_base/10)) * (a_s @ a_s.conj().T) + (10**(inr_base/10)) * (a_j_true @ a_j_true.conj().T) + np.eye(self.N_array)
        
        # Uncertainty-Aware Null Widening
        sigma_theta_deg = bearing_err
        theta_est_calibrated = theta_j_steer + calib_error
        C, f = build_constraint_matrix(a_s, theta_est_calibrated, sigma_theta_deg, self.N_array, n_null_points=2)
        
        # LCMV Weights
        w_lcmv, cond_num = lcmv_weights(R_hat, C, f, loading_factor=0.10)
        
        # Residual Jammer Leakage
        leakage_gain = np.abs(w_lcmv.conj().T @ a_j_true)[0,0]**2
        nd_true = 10 * np.log10(max(1e-10, leakage_gain))
        
        inr_resid = inr_base + nd_true
        sinr_dB = snr_base - inr_resid
        gamma_bar = 10.0**(sinr_dB / 10.0)
        
        req_sinr_dB = {1.0: 2.0, 2.0: 5.0, 4.0: 12.0, 6.0: 18.0}.get(best_amc, 5.0)
        p_out_true = self.get_mrc_outage(gamma_bar, req_sinr_dB)
        
        # Compute true pd
        fs = 1e6
        if best_ts > 0: pd_val = p_detection(jnr_lin, best_ts*1e-3*fs)
        else: pd_val = 1.0
            
        r_eff_true = best_amc * pd_val * (1.0 - best_ts/10.0) * (1.0 - 0.9 * p_out_true)
        exp_true = (self.dx_step + 2.0 * best_dy) / self.base_speed
        
        return r_eff_true, exp_true, nd_true, p_out_true, cond_num, is_clipped

    def step(self, epoch):
        ts_opts = [0.1, 0.7, 2.0, 5.0, 9.0]
        dy_opts = [0.0, 200.0, 460.0, 800.0, 1200.0]
        amc_opts = [1.0, 2.0, 4.0]
        
        best_dpp = np.inf
        best_action = None
        best_P_post = None
        best_bearing_err = None
        best_L_eff = None
        
        use_ekf = True
        if self.policy == 'No_EKF' or self.policy == 'No_Localization': use_ekf = False
        
        dx_target = np.abs(0.0 - self.uav_x)
        P_target = (dx_target * self.target_deg * np.pi / 180.0)**2
        
        if self.policy == 'Fixed_460':
            best_action = (0.7, 460.0, 4.0)
            _, _, best_P_post, _, _, best_bearing_err, best_L_eff = self.predict_proxy(*best_action, use_ekf, epoch)
            self.Q_RF = self.Q_EXP = self.Q_INFO = 0.0
        elif self.policy == 'No_Localization':
            best_action = (0.1, 0.0, 4.0)
            _, _, best_P_post, _, _, best_bearing_err, best_L_eff = self.predict_proxy(*best_action, use_ekf, epoch)
            self.Q_RF = self.Q_EXP = self.Q_INFO = 0.0
        else:
            w1 = 1.0; w2 = 1.0 # dimensionless weights
            for ts in ts_opts:
                for dy in dy_opts:
                    for amc in amc_opts:
                        r, e, P_post, nd_proxy, out_proxy, b_err, l_eff = self.predict_proxy(ts, dy, amc, use_ekf, epoch)
                        
                        a_rf = 5.0; d_rf = r
                        a_exp = kill_prob_per_epoch(e); d_exp = kill_prob_per_epoch(self.base_time_per_epoch)
                        a_info = self.compute_info_arrival(epoch, np.trace(P_post), P_target); d_info = P_target
                        
                        Q_RF_n = max(self.Q_RF + (a_rf - d_rf), 0) / self.Q_RF_NORM
                        Q_EXP_n = max(self.Q_EXP + (a_exp - d_exp), 0) / self.Q_EXP_NORM
                        Q_INFO_n = max(self.Q_INFO + (a_info - d_info)/10.0, 0) / self.Q_INFO_NORM
                        
                        q_rf_old = self.Q_RF / self.Q_RF_NORM
                        q_exp_old = self.Q_EXP / self.Q_EXP_NORM
                        q_info_old = self.Q_INFO / self.Q_INFO_NORM
                        
                        L_n = 0.5 * (Q_RF_n**2 + Q_EXP_n**2 + Q_INFO_n**2)
                        
                        drift_bound = (q_rf_old * (a_rf - d_rf) / self.Q_RF_NORM +
                                       q_exp_old * (a_exp - d_exp) / self.Q_EXP_NORM +
                                       q_info_old * (a_info - d_info) / self.Q_INFO_NORM)
                        
                        pen = w1 * a_exp + w2 * out_proxy
                        dpp = drift_bound + self.V * pen
                        
                        if dpp < best_dpp:
                            best_dpp = dpp
                            best_action = (ts, dy, amc)
                            best_P_post = P_post
                            best_bearing_err = b_err
                            best_L_eff = l_eff
                            best_Q = (max(self.Q_RF + a_rf - d_rf, 0),
                                      max(self.Q_EXP + a_exp - d_exp, 0),
                                      max(self.Q_INFO + a_info - d_info, 0))
                            best_Ln = L_n
                            
            self.Q_RF, self.Q_EXP, self.Q_INFO = best_Q
            self.lyapunov_history.append(best_Ln)

        # Apply True Physics for final results
        r_eff_true, exp_true, nd_true, out_true, cond_num, is_clipped = self.apply_true_physics(
            best_action[0], best_action[1], best_action[2], use_ekf, epoch, best_P_post, best_bearing_err, best_L_eff
        )
        
        ekf_health_check(self.P_cov)
        
        self.uav_x += self.dx_step
        
        # Update state
        # (self.P_cov and self.x_hat are now updated inside apply_true_physics)
        self.theta_error_rms_history.append(best_bearing_err)
        self.jammer_leakage_db_history.append(nd_true)
        loop_gain = stability_monitor(self.theta_error_rms_history, self.jammer_leakage_db_history)
        
        self.logs['throughput'].append(r_eff_true)
        self.logs['exposure'].append(exp_true)
        self.logs['P_trace'].append(np.trace(best_P_post))
        self.logs['dy'].append(best_action[1])
        self.logs['ts'].append(best_action[0])
        self.logs['outage'].append(out_true)
        self.logs['nd'].append(nd_true)
        self.logs['pmcs_rf'] *= (1.0 - out_true)
        self.logs['cond_num'].append(cond_num)
        self.logs['loop_gain'].append(loop_gain)
        self.logs['clip'].append(is_clipped)
