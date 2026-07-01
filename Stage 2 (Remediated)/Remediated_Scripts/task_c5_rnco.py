import sys
import os
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, r'd:\UAV Internship project\Stage 2')

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(__file__))
import phase_b_beamforming_remediated as pbb
import task_c3_cognitive_sensing as c3

class RNCOOptimizer:
    def __init__(self, V_param):
        self.V = V_param
        
        # Virtual Queues
        self.Q_RF = 0.0
        self.Q_EXP = 0.0
        self.Q_INFO = 0.0
        
        # Budgets and Targets (Normalized for balanced drift)
        self.r_target = 10.0      # Target bits/symbol
        self.e_budget = 8.5       # Nominal exposure
        self.P_target = 2000.0    # Nominal Trace
        
        # Weights for penalty
        self.w1 = 1.0   # Exposure cost weight
        self.w2 = 1.0   # Throughput deficit weight
        self.w3 = 0.01  # Covariance penalty weight
        
        # EKF State proxy
        self.P_cov = np.diag([1e6, 1e6]) # Just keeping track of position cov
        
        # Mission state
        self.uav_x = -5000.0
        
        # Action space
        self.tau_s_opts = [0.1, 0.7, 2.0, 5.0, 9.0] # ms
        self.dy_opts = [0.0, 200.0, 460.0, 800.0, 1200.0] # m
        self.amc_opts = [1.0, 2.0, 4.0, 6.0] # bits/symbol
        
        # Tracking logs
        self.logs = {
            'Q_RF': [], 'Q_EXP': [], 'Q_INFO': [],
            'tau_s': [], 'dy': [], 'amc': [],
            'uav_x': [], 'P_trace': [], 'throughput': [],
            'drift_plus_penalty': []
        }

    def predict_metrics(self, tau_s, dy, amc, current_x, epoch_idx):
        # 1. Cognitive & RF
        fs = 1e6; p_fa = 0.05
        
        # Dynamic Terrain Profile
        inr_base = 50.0
        if 10 <= epoch_idx <= 15:
            inr_base = 35.0 # Terrain ridge blocks jammer by 15 dB
            
        jnr_lin = 10.0**((inr_base - 50.0 - 5.0)/10.0) # Base JNR is -5dB when no terrain
        pd = c3.compute_pd(tau_s*1e-3, fs, p_fa, jnr_lin)
        L_eff = max(1, int(100 * (1.0 - tau_s / 10.0)))
        
        m_rmse = max(0.01, 1.0 / np.sqrt(L_eff) - 0.1)
        nd = -120.0 + 35.0 * ((100 - L_eff) / 99.0)**2
        
        snr_base = 15.0
        inr_base = 50.0
        inr_resid = inr_base + nd
        sinr = snr_base - inr_resid
        
        # AMC Outage
        # Simple threshold mapping: 1bps -> SINR>2dB, 2bps -> SINR>5dB, etc.
        req_sinr = {1.0: 2.0, 2.0: 5.0, 4.0: 12.0, 6.0: 18.0}[amc]
        comm_survival = 1.0 if sinr >= req_sinr else 0.1 # 10% chance if below threshold
        
        r_eff = amc * pd * (1.0 - tau_s/10.0) * comm_survival
        
        # 2. Exposure
        base_time = 8.5 # 425m at 50m/s
        extra_time = (2 * dy) / 50.0
        exposure = base_time + extra_time
        
        # 3. Information Update
        # Predict EKF covariance trace reduction
        # GDOP approx
        dx = 15000.0 - current_x
        g = np.sqrt(dx**2) / max(dy, 10.0)
        pos_err = g * m_rmse * dx * (np.pi/180.0)
        
        # Simple 1D Kalman update proxy for trace
        R_meas = pos_err**2
        P_pred = self.P_cov[0,0] + 1000.0 # Process noise
        K = P_pred / (P_pred + R_meas + 1e-3)
        P_post = (1 - K) * P_pred
        
        return r_eff, exposure, P_post*2, P_pred*2 # Multiply by 2 for x,y trace
        
    def step(self, epoch_idx):
        best_dpp = np.inf
        best_action = None
        best_metrics = None
        
        for ts in self.tau_s_opts:
            for dy in self.dy_opts:
                for amc in self.amc_opts:
                    
                    r_eff, exp, P_post, P_pred = self.predict_metrics(ts, dy, amc, self.uav_x, epoch_idx)
                    
                    # Normalized Queue updates to balance drift scales
                    # Scale everything so max growth per epoch is roughly ~10
                    Q_RF_next = max(self.Q_RF + (self.r_target - r_eff), 0)
                    Q_EXP_next = max(self.Q_EXP + (exp - self.e_budget), 0) 
                    Q_INFO_next = max(self.Q_INFO + (P_post - self.P_target)/100.0, 0)
                    
                    # Drift
                    L_t = 0.5 * (self.Q_RF**2 + self.Q_EXP**2 + self.Q_INFO**2)
                    L_next = 0.5 * (Q_RF_next**2 + Q_EXP_next**2 + Q_INFO_next**2)
                    drift = L_next - L_t
                    
                    # Penalty
                    penalty = self.w1 * exp + self.w2 * max(0, self.r_target - r_eff) + self.w3 * P_post
                    
                    dpp = drift + self.V * penalty
                    
                    if dpp < best_dpp:
                        best_dpp = dpp
                        best_action = (ts, dy, amc)
                        best_metrics = (r_eff, exp, P_post, Q_RF_next, Q_EXP_next, Q_INFO_next)
                        
        # Apply best action
        ts, dy, amc = best_action
        r_eff, exp, P_post, Q_RF_next, Q_EXP_next, Q_INFO_next = best_metrics
        
        self.logs['Q_RF'].append(self.Q_RF)
        self.logs['Q_EXP'].append(self.Q_EXP)
        self.logs['Q_INFO'].append(self.Q_INFO)
        self.logs['tau_s'].append(ts)
        self.logs['dy'].append(dy)
        self.logs['amc'].append(amc)
        self.logs['uav_x'].append(self.uav_x)
        self.logs['P_trace'].append(P_post)
        self.logs['throughput'].append(r_eff)
        self.logs['drift_plus_penalty'].append(best_dpp)
        
        # Update states
        self.Q_RF = Q_RF_next
        self.Q_EXP = Q_EXP_next
        self.Q_INFO = Q_INFO_next
        self.P_cov = np.diag([P_post/2, P_post/2])
        self.uav_x += 425.0
        
def run_c5_experiment():
    print("Running Task C5: RNCO Lyapunov Optimizer...")
    
    V_values = [0.1, 10.0, 1000.0]
    results = {}
    
    for V in V_values:
        rnco = RNCOOptimizer(V_param=V)
        # Run 30 epochs
        for ep in range(30):
            rnco.step(ep)
        results[V] = rnco
        
    # Plotting Queue Evolution and Actions for V=10.0 (Balanced)
    opt = results[10.0]
    epochs = range(30)
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    # Q_RF
    axes[0,0].plot(epochs, opt.logs['Q_RF'], 'b-o')
    axes[0,0].set_title('Q_RF Evolution (Communication Deficit)')
    axes[0,0].grid(True)
    
    # Q_EXP
    axes[1,0].plot(epochs, opt.logs['Q_EXP'], 'r-s')
    axes[1,0].set_title('Q_EXP Evolution (Excess Exposure)')
    axes[1,0].grid(True)
    
    # Q_INFO
    axes[2,0].plot(epochs, opt.logs['Q_INFO'], 'g-^')
    axes[2,0].set_title('Q_INFO Evolution (Localization Uncertainty)')
    axes[2,0].grid(True)
    
    # Actions: Sensing Time
    axes[0,1].step(epochs, opt.logs['tau_s'], 'm-', where='mid')
    axes[0,1].set_title('Action: Sensing Time τs (ms)')
    axes[0,1].grid(True)
    
    # Actions: Lateral Deviation
    axes[1,1].step(epochs, opt.logs['dy'], 'c-', where='mid')
    axes[1,1].set_title('Action: Lateral Deviation Δy (m)')
    axes[1,1].grid(True)
    
    # P_trace (Information quality)
    axes[2,1].plot(epochs, opt.logs['P_trace'], 'k--')
    axes[2,1].axhline(1000.0, color='r', linestyle=':', label='P_target')
    axes[2,1].set_yscale('log')
    axes[2,1].set_title('Covariance Trace (Tr(P_k))')
    axes[2,1].legend()
    axes[2,1].grid(True)
    
    plt.tight_layout()
    plt.savefig('task_c5_rnco_evolution.png', dpi=300)
    print("Saved task_c5_rnco_evolution.png")
    
    # Print V-sensitivity
    print("\\n=== V-Sensitivity Analysis (Final Averages) ===")
    print(f"|   V   | Avg Throughput | Avg Exposure | Avg P_trace | Avg dy | Avg ts |")
    print(f"| ----- | -------------- | ------------ | ----------- | ------ | ------ |")
    for V in V_values:
        opt_v = results[V]
        avg_r = np.mean(opt_v.logs['throughput'])
        avg_e = np.mean(opt_v.logs['Q_EXP']) # Or compute actual exposure
        # Re-compute actual exposure for log
        actual_exp = np.mean([8.5 + (2*dy)/50.0 for dy in opt_v.logs['dy']])
        avg_P = np.mean(opt_v.logs['P_trace'])
        avg_dy = np.mean(opt_v.logs['dy'])
        avg_ts = np.mean(opt_v.logs['tau_s'])
        print(f"| {V:5.1f} | {avg_r:14.2f} | {actual_exp:12.2f} | {avg_P:11.1f} | {avg_dy:6.1f} | {avg_ts:6.1f} |")

if __name__ == "__main__":
    run_c5_experiment()
