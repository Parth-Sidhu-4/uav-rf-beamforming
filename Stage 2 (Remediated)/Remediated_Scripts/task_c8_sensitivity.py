import sys
import os
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, r'd:\UAV Internship project\Stage 2')

import numpy as np
import pandas as pd
import os
import sys
import math

sys.path.append(os.path.dirname(__file__))
import task_c3_cognitive_sensing as c3

class SensitivitySimulator:
    def __init__(self, sweep_param=None, sweep_val=None):
        self.sweep_param = sweep_param
        self.sweep_val = sweep_val
        
        self.epochs = 10
        self.dx_step = 425.0
        
        # Base parameters
        self.inr_base_nominal = 50.0
        self.calib_error_nominal = 0.0
        self.terrain_att_nominal = 20.0
        self.flight_speed = 50.0
        self.div_scheme = 'MRC_L4'
        self.target_deg = 2.0
        
        if sweep_param == 'Jammer Power': self.inr_base_nominal = sweep_val
        if sweep_param == 'Calib Error': self.calib_error_nominal = sweep_val
        if sweep_param == 'Terrain Att': self.terrain_att_nominal = sweep_val
        if sweep_param == 'Flight Speed': self.flight_speed = sweep_val
        if sweep_param == 'Diversity': self.div_scheme = sweep_val
        if sweep_param == 'Target Deg': self.target_deg = sweep_val
        
        self.uav_x = -5000.0
        self.P_cov = 1000.0
        
        self.Q_RF = 0.0
        self.Q_EXP = 0.0
        self.Q_INFO = 0.0
        
        self.logs = {
            'throughput': [], 'exposure': [], 'P_trace': [], 
            'dy': [], 'ts': [], 'outage': [], 'nd': [], 'pmcs_rf': 1.0
        }
        
    def get_environment(self, epoch):
        inr_base = self.inr_base_nominal
        snr_base = 15.0
        calib_error = self.calib_error_nominal
        jam_y = 0.0
        
        # We run the sensitivity sweep on Scenario B (Shadowing) to engage all subsystems
        if 3 <= epoch <= 6:
            inr_base -= self.terrain_att_nominal
            
        jnr_lin = 10.0**((inr_base - 50.0 - 5.0)/10.0)
        return inr_base, snr_base, calib_error, jam_y, jnr_lin
        
    def predict(self, ts, dy, amc, use_ekf, epoch):
        inr_base, snr_base, calib_error, jam_y, jnr_lin = self.get_environment(epoch)
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
        
        exp = (self.dx_step + 2.0 * dy) / self.flight_speed
        
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
        sinr_dB = snr_base - inr_resid
        req_sinr_dB = {1.0: 2.0, 2.0: 5.0, 4.0: 12.0, 6.0: 18.0}.get(amc, 5.0)
        
        gamma_bar = 10.0**(sinr_dB / 10.0)
        gamma_th = 10.0**(req_sinr_dB / 10.0)
        x = min(gamma_th / gamma_bar, 100.0)
        
        L = int(self.div_scheme.split('_L')[1]) if '_L' in self.div_scheme else 1
        
        if L == 1:
            p_out = 1.0 - np.exp(-x)
        else:
            sum_term = sum((x**k)/math.factorial(k) for k in range(L))
            p_out = 1.0 - np.exp(-x) * sum_term
            
        p_out = min(1.0, max(0.0, p_out))
        comm_survival = 1.0 - 0.9 * p_out
        r_eff = amc * pd_val * (1.0 - ts/10.0) * comm_survival
        
        return r_eff, exp, P_post, nd, p_out
        
    def step(self, epoch):
        ts_opts = [0.1, 0.7, 2.0, 5.0, 9.0]
        dy_opts = [0.0, 200.0, 460.0, 800.0, 1200.0]
        amc_opts = [2.0, 4.0]
        
        best_dpp = np.inf
        best_action = None
        
        V = 100.0
        r_target = 5.0
        e_budget = self.dx_step / self.flight_speed
        w1 = 1.0; w2 = 1000.0; w3 = 0.0
        
        dx_target = np.abs(0.0 - self.uav_x)
        P_target = (dx_target * self.target_deg * np.pi / 180.0)**2
        
        for ts in ts_opts:
            for dy in dy_opts:
                for amc in amc_opts:
                    r, e, P, nd_tmp, out_tmp = self.predict(ts, dy, amc, True, epoch)
                    
                    Q_RF_n = max(self.Q_RF + (r_target - r), 0)
                    Q_EXP_n = max(self.Q_EXP + (e - e_budget), 0)
                    Q_INFO_n = max(self.Q_INFO + (P - P_target)/10.0, 0)
                    
                    L_t = 0.5*(self.Q_RF**2 + self.Q_EXP**2 + self.Q_INFO**2)
                    L_n = 0.5*(Q_RF_n**2 + Q_EXP_n**2 + Q_INFO_n**2)
                    drift = L_n - L_t
                    
                    pen = w1*e + w2*out_tmp + w3*(P/100.0)
                    dpp = drift + V*pen
                    
                    if dpp < best_dpp:
                        best_dpp = dpp
                        best_action = (ts, dy, amc)
                        r_eff, exp, P_post, nd, out = r, e, P, nd_tmp, out_tmp
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
        self.logs['pmcs_rf'] *= (1.0 - out)

def run_sensitivity():
    sweeps = {
        'Calib Error': [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    }
    
    results = []
    
    for param, values in sweeps.items():
        print(f"Sweeping {param}...")
        for val in values:
            sim = SensitivitySimulator(sweep_param=param, sweep_val=val)
            for ep in range(sim.epochs): sim.step(ep)
            
            tot_exp = np.sum(sim.logs['exposure'])
            surv = np.exp(-tot_exp / 120.0)
            pmcs = surv * sim.logs['pmcs_rf']
            
            results.append({
                'Sweep Parameter': param,
                'Value': val,
                'Pmcs': pmcs,
                'Avg dy': np.mean(sim.logs['dy']),
                'Total Exp': tot_exp,
                'Avg Null Depth': np.mean(sim.logs['nd']),
                'Avg Outage': np.mean(sim.logs['outage'])
            })
            
    df = pd.DataFrame(results)
    print("\\n=== Task C8: Full Sensitivity Sweep ===")
    for param in sweeps.keys():
        print(f"\\n--- {param} ---")
        sub_df = df[df['Sweep Parameter'] == param].drop(columns=['Sweep Parameter'])
        print(sub_df.to_string(index=False, float_format=lambda x: f"{x:.3f}" if isinstance(x, float) else str(x)))
        
    df.to_csv('task_c8_sensitivity_results.csv', index=False)

if __name__ == "__main__":
    run_sensitivity()
