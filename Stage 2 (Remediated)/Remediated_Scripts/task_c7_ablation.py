import sys
import os
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, r'd:\UAV Internship project\Stage 2')

import numpy as np
import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(__file__))
import task_c3_cognitive_sensing as c3

class AblationSimulator:
    def __init__(self, scenario='A', ablation=None):
        self.scenario = scenario
        self.ablation = ablation # None, 'No_RNCO', 'No_Diversity', 'No_EKF', 'No_Terrain', 'No_Cognitive', 'No_MUSIC', 'No_Localization'
        
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
            'dy': [], 'ts': [], 'outage': [], 'nd': [], 'pmcs_rf': 1.0
        }
        
    def get_environment(self, epoch):
        inr_base = 50.0
        snr_base = 15.0
        calib_error = 0.0
        jam_y = 0.0
        
        if self.scenario == 'B' and self.ablation != 'No_Terrain':
            if 3 <= epoch <= 6:
                inr_base -= 20.0
        elif self.scenario == 'C':
            jam_y = 5000.0
        elif self.scenario == 'D':
            calib_error = 0.5
        elif self.scenario == 'E':
            inr_base = 20.0
            
        jnr_lin = 10.0**((inr_base - 50.0 - 5.0)/10.0)
        return inr_base, snr_base, calib_error, jam_y, jnr_lin
        
    def predict(self, ts, dy, amc, use_ekf, epoch):
        if self.ablation == 'No_EKF' or self.ablation == 'No_Localization':
            use_ekf = False
            
        inr_base, snr_base, calib_error, jam_y, jnr_lin = self.get_environment(epoch)
        fs = 1e6; p_fa = 0.05
        
        if self.ablation == 'No_Cognitive':
            pd_val = 1.0; L_eff = 100
        else:
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
            
        if self.ablation == 'No_MUSIC':
            nd = 0.0
        else:
            nd = max(-120.0, -120.0 + 50.0 * bearing_err)
            nd = min(nd, 0.0)
            
        inr_resid = inr_base + nd
        sinr_dB = snr_base - inr_resid
        
        req_sinr_dB = {1.0: 2.0, 2.0: 5.0, 4.0: 12.0, 6.0: 18.0}.get(amc, 5.0)
        
        gamma_bar = 10.0**(sinr_dB / 10.0)
        gamma_th = 10.0**(req_sinr_dB / 10.0)
        x = min(gamma_th / gamma_bar, 100.0)
        
        if self.ablation == 'No_Diversity':
            p_out = 1.0 - np.exp(-x)
        else:
            # Baseline is MRC L=4
            p_out = 1.0 - np.exp(-x) * (1.0 + x + (x**2)/2.0 + (x**3)/6.0)
            
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
        
        if self.ablation == 'No_RNCO':
            # Fixed 460 policy
            best_action = (0.7, 460.0, 4.0)
            r, e, P, nd_tmp, out_tmp = self.predict(*best_action, True, epoch)
            r_eff, exp, P_post, nd, out = r, e, P, nd_tmp, out_tmp
            self.Q_RF = self.Q_EXP = self.Q_INFO = 0.0
            
        elif self.ablation == 'No_Localization':
            # No EKF (handled in predict) and no S-Curve (dy = 0)
            best_action = (0.1, 0.0, 4.0)
            r, e, P, nd_tmp, out_tmp = self.predict(*best_action, False, epoch)
            r_eff, exp, P_post, nd, out = r, e, P, nd_tmp, out_tmp
            self.Q_RF = self.Q_EXP = self.Q_INFO = 0.0
            
        else:
            # Full RNCO Optimization
            V = 100.0
            r_target = 5.0
            e_budget = 8.5
            w1 = 1.0; w2 = 1000.0; w3 = 0.0
            target_deg = 2.0 # Baseline shifted to 2.0 deg under MRC L=4
            
            dx_target = np.abs(0.0 - self.uav_x)
            P_target = (dx_target * target_deg * np.pi / 180.0)**2
            
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

def run_ablation():
    scenarios = ['A', 'B', 'C', 'D', 'E']
    ablations = [
        None, 'No_MUSIC', 'No_Localization', 'No_RNCO', 
        'No_Diversity', 'No_EKF', 'No_Terrain', 'No_Cognitive'
    ]
    
    results = {}
    for s in scenarios:
        results[s] = {}
        for a in ablations:
            sim = AblationSimulator(scenario=s, ablation=a)
            for ep in range(sim.epochs): sim.step(ep)
            
            tot_exp = np.sum(sim.logs['exposure'])
            surv = np.exp(-tot_exp / 120.0)
            pmcs = surv * sim.logs['pmcs_rf']
            results[s][a] = pmcs
            
    # Level 2 Matrix
    print("=== Level 2: Scenario-Specific Importance Matrix ===")
    matrix_data = []
    baseline_pmcs = {}
    for s in scenarios:
        baseline_pmcs[s] = results[s][None]
        
    for a in ablations:
        if a is None: continue
        row = {'Component Removed': a}
        for s in scenarios:
            drop = baseline_pmcs[s] - results[s][a]
            row[s] = drop
        matrix_data.append(row)
        
    df_matrix = pd.DataFrame(matrix_data)
    print(df_matrix.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    df_matrix.to_csv('task_c7_scenario_matrix.csv', index=False)
    
    # Level 1 Aggregate Ranking
    print("\\n=== Level 1: Aggregate Ranking (Primary Result) ===")
    agg_data = []
    for a in ablations:
        if a is None: continue
        avg_drop = np.mean([baseline_pmcs[s] - results[s][a] for s in scenarios])
        agg_data.append({'Component Removed': a, 'Avg Drop Pmcs': avg_drop})
        
    df_agg = pd.DataFrame(agg_data).sort_values(by='Avg Drop Pmcs', ascending=False)
    df_agg['Rank'] = range(1, len(df_agg) + 1)
    print(df_agg.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    df_agg.to_csv('task_c7_aggregate_ranking.csv', index=False)

if __name__ == "__main__":
    run_ablation()
