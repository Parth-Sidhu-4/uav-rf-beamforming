import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(__file__))
import task_c3_cognitive_sensing as c3

class PrecisionSimulator:
    def __init__(self, target_deg=1.0):
        self.target_deg = target_deg
        
        self.epochs = 10
        self.dx_step = 425.0
        self.base_time = 8.5
        
        self.uav_x = -5000.0
        self.P_cov = 1000.0
        
        self.Q_RF = 0.0
        self.Q_EXP = 0.0
        self.Q_INFO = 0.0
        
        self.logs = {
            'throughput': [], 'exposure': [], 'P_trace': [], 
            'dy': [], 'ts': [], 'outage': [], 'nd': [], 'pmcs_rf': 1.0,
            'bearing_err': []
        }
        
    def predict(self, ts, dy, amc, use_ekf):
        inr_base = 50.0
        snr_base = 15.0
        calib_error = 0.0
        jam_y = 0.0
        jnr_lin = 10.0**((inr_base - 50.0 - 5.0)/10.0)
        
        fs = 1e6; p_fa = 0.05
        
        if ts > 0:
            pd_val = c3.compute_pd(ts*1e-3, fs, p_fa, jnr_lin)
            L_eff = max(1, int(100 * (1.0 - ts / 10.0)))
        else:
            pd_val = 1.0; L_eff = 100
            
        base_noise = 0.5
        if jnr_lin < 0.01 or L_eff < 10:
            base_noise = 45.0 
            
        m_rmse = max(base_noise, 10.0 / np.sqrt(L_eff)) + calib_error
        
        exp = self.base_time + (2 * dy) / 50.0
        
        dx = np.abs(0.0 - self.uav_x)
        dy_j = np.abs(jam_y - dy)
        r = np.sqrt(dx**2 + dy_j**2)
        g = r / max(dy_j, 10.0) if dy_j > 0 else r / 10.0
        
        pos_err_meas = g * m_rmse * dx * (np.pi/180.0)
        R_meas = pos_err_meas**2
        P_pred = self.P_cov + 1000.0 
        
        if use_ekf:
            K = P_pred / (P_pred + R_meas + 1e-3)
            P_post = (1 - K) * P_pred
            bearing_err = (np.sqrt(P_post) / dx) * (180.0/np.pi)
        else:
            P_post = P_pred
            bearing_err = m_rmse
            
        nd = max(-120.0, -120.0 + 50.0 * bearing_err)
        nd = min(nd, 0.0)
        inr_resid = inr_base + nd
        sinr = snr_base - inr_resid
        
        req_sinr = {1.0: 2.0, 2.0: 5.0, 4.0: 12.0, 6.0: 18.0}.get(amc, 5.0)
        comm_survival = 1.0 if sinr >= req_sinr else 0.1
        outage = 1.0 - comm_survival
        
        r_eff = amc * pd_val * (1.0 - ts/10.0) * comm_survival
        
        return r_eff, exp, P_post, nd, outage, bearing_err
        
    def step(self):
        ts_opts = [0.1, 0.7, 2.0, 5.0, 9.0]
        dy_opts = [0.0, 200.0, 460.0, 800.0, 1200.0]
        amc_opts = [2.0, 4.0]
        
        best_dpp = np.inf
        best_action = None
        
        V = 100.0
        r_target = 5.0
        e_budget = 8.5
        w1 = 1.0; w2 = 1000.0; w3 = 0.0
        
        dx_target = np.abs(0.0 - self.uav_x)
        P_target = (dx_target * self.target_deg * np.pi / 180.0)**2
        
        for ts in ts_opts:
            for dy in dy_opts:
                for amc in amc_opts:
                    r, e, P, nd_tmp, out_tmp, berr_tmp = self.predict(ts, dy, amc, True)
                    
                    Q_RF_n = max(self.Q_RF + (r_target - r), 0)
                    Q_EXP_n = max(self.Q_EXP + (e - e_budget), 0)
                    Q_INFO_n = max(self.Q_INFO + (P - P_target)/10.0, 0)
                    
                    L_t = 0.5*(self.Q_RF**2 + self.Q_EXP**2 + self.Q_INFO**2)
                    L_n = 0.5*(Q_RF_n**2 + Q_EXP_n**2 + Q_INFO_n**2)
                    drift = L_n - L_t
                    
                    pen = w1*e + w2*max(0, r_target-r) + w3*(P/100.0)
                    dpp = drift + V*pen
                    
                    if dpp < best_dpp:
                        best_dpp = dpp
                        best_action = (ts, dy, amc)
                        r_eff, exp, P_post, nd, out, berr = r, e, P, nd_tmp, out_tmp, berr_tmp
                        best_Q = (Q_RF_n, Q_EXP_n, Q_INFO_n)
                        
        self.Q_RF, self.Q_EXP, self.Q_INFO = best_Q

        self.P_cov = P_post
        self.uav_x += self.dx_step
        
        self.logs['throughput'].append(r_eff)
        self.logs['exposure'].append(exp)
        self.logs['P_trace'].append(P_post)
        self.logs['dy'].append(best_action[1])
        self.logs['ts'].append(best_action[0])
        self.logs['outage'].append(out)
        self.logs['nd'].append(nd)
        self.logs['bearing_err'].append(berr)
        self.logs['pmcs_rf'] *= (1.0 - out)

def run_precision_audit():
    targets = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
    results = []
    
    print("Running Localization Precision Audit...")
    for t in targets:
        sim = PrecisionSimulator(target_deg=t)
        for ep in range(sim.epochs): sim.step()
        
        tot_exp = np.sum(sim.logs['exposure'])
        surv = np.exp(-tot_exp / 120.0)
        pmcs = surv * sim.logs['pmcs_rf']
        outage_rate = np.mean([1.0 if o > 0.0 else 0.0 for o in sim.logs['outage']])
        
        results.append({
            'Target (deg)': t,
            'Pmcs': pmcs,
            'Avg dy': np.mean(sim.logs['dy']),
            'Total Exp': tot_exp,
            'Avg Trace': np.mean(sim.logs['P_trace']),
            'Avg Null Depth': np.mean(sim.logs['nd']),
            'Outage Rate': outage_rate
        })
        
    df = pd.DataFrame(results)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    df.to_csv('task_c6c_precision_audit.csv', index=False)

if __name__ == "__main__":
    run_precision_audit()
