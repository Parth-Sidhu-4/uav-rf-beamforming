import numpy as np
import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(__file__))
import task_c3_cognitive_sensing as c3

class DiversitySimulator:
    def __init__(self, target_deg=1.5, div_scheme='No_Div'):
        self.target_deg = target_deg
        self.div_scheme = div_scheme
        
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
        sinr_dB = snr_base - inr_resid
        
        req_sinr_dB = {1.0: 2.0, 2.0: 5.0, 4.0: 12.0, 6.0: 18.0}.get(amc, 5.0)
        
        gamma_bar = 10.0**(sinr_dB / 10.0)
        gamma_th = 10.0**(req_sinr_dB / 10.0)
        
        x = gamma_th / gamma_bar
        
        # Prevent math overflow if x is massive
        x = min(x, 100.0)
        
        if self.div_scheme == 'No_Div':
            p_out = 1.0 - np.exp(-x)
        elif self.div_scheme == 'SC_L2':
            p_out = (1.0 - np.exp(-x))**2
        elif self.div_scheme == 'MRC_L2':
            p_out = 1.0 - np.exp(-x) * (1.0 + x)
        elif self.div_scheme == 'EGC_L2':
            # Fast empirical approximation: EGC is approx 1.5 dB worse than MRC
            gamma_th_egc = gamma_th * (10.0**(1.5/10.0))
            x_egc = min(gamma_th_egc / gamma_bar, 100.0)
            p_out = 1.0 - np.exp(-x_egc) * (1.0 + x_egc)
        elif self.div_scheme == 'MRC_L4':
            p_out = 1.0 - np.exp(-x) * (1.0 + x + (x**2)/2.0 + (x**3)/6.0)
            
        p_out = min(1.0, max(0.0, p_out))
        
        comm_survival = 1.0 - 0.9 * p_out
        r_eff = amc * pd_val * (1.0 - ts/10.0) * comm_survival
        
        return r_eff, exp, P_post, nd, p_out, bearing_err
        
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
                    
                    pen = w1*e + w2*out_tmp + w3*(P/100.0)
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

def run_diversity_audit():
    schemes = ['No_Div', 'SC_L2', 'EGC_L2', 'MRC_L2', 'MRC_L4']
    results_div = []
    
    print("=== Phase 1: Evaluating Combining Schemes (Fixed 1.5 deg Target) ===")
    
    baseline_dy = 0
    baseline_exp = 0
    
    for s in schemes:
        sim = DiversitySimulator(target_deg=1.5, div_scheme=s)
        for _ in range(sim.epochs): sim.step()
        
        tot_exp = np.sum(sim.logs['exposure'])
        surv = np.exp(-tot_exp / 120.0)
        pmcs = surv * sim.logs['pmcs_rf']
        avg_outage = np.mean(sim.logs['outage'])
        avg_dy = np.mean(sim.logs['dy'])
        
        if s == 'No_Div':
            baseline_dy = avg_dy
            baseline_exp = tot_exp
            
        results_div.append({
            'Scheme': s,
            'Outage Rate': avg_outage,
            'Avg Null Depth': np.mean(sim.logs['nd']),
            'Avg dy': avg_dy,
            'dy_saved': baseline_dy - avg_dy,
            'Total Exp': tot_exp,
            'Exp_saved': baseline_exp - tot_exp,
            'Pmcs': pmcs
        })
        
    df1 = pd.DataFrame(results_div)
    print(df1.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    df1.to_csv('task_c9_diversity_schemes.csv', index=False)
    
    print("\\n=== Phase 2: Rerunning Localization Precision Audit with MRC (L=4) ===")
    targets = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    results_sweep = []
    
    for t in targets:
        sim = DiversitySimulator(target_deg=t, div_scheme='MRC_L4')
        for _ in range(sim.epochs): sim.step()
        
        tot_exp = np.sum(sim.logs['exposure'])
        surv = np.exp(-tot_exp / 120.0)
        pmcs = surv * sim.logs['pmcs_rf']
        avg_outage = np.mean(sim.logs['outage'])
        
        results_sweep.append({
            'Target (deg)': t,
            'Pmcs': pmcs,
            'Avg dy': np.mean(sim.logs['dy']),
            'Total Exp': tot_exp,
            'Avg Null Depth': np.mean(sim.logs['nd']),
            'Outage Rate': avg_outage
        })
        
    df2 = pd.DataFrame(results_sweep)
    print(df2.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    df2.to_csv('task_c9_precision_shift.csv', index=False)

if __name__ == "__main__":
    run_diversity_audit()
