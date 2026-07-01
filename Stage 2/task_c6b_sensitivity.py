import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(__file__))
import task_c3_cognitive_sensing as c3

class MissionSimulator:
    def __init__(self, scenario='A', policy='RNCO', weights=(1.0, 1000.0, 10.0)):
        self.scenario = scenario
        self.policy = policy
        self.weights = weights # w1 (EXP), w2 (RF), w3 (INFO)
        
        self.epochs = 10
        self.dx_step = 425.0
        self.base_time = 8.5
        
        self.oracle_P = 10.0 
        
        # Start at 5000m, end at 750m
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
        
    def get_environment(self, epoch):
        inr_base = 50.0
        snr_base = 15.0
        calib_error = 0.0
        jam_y = 0.0
        
        if self.scenario == 'B': # Shadowing
            if 3 <= epoch <= 6:
                inr_base -= 20.0
        elif self.scenario == 'C': # Rapid jammer geometry
            jam_y = 5000.0
        elif self.scenario == 'D': # Reduced array
            calib_error = 0.5
        elif self.scenario == 'E': # Weak Jammer
            inr_base = 20.0
            
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
        
    def step(self, epoch):
        ts_opts = [0.1, 0.7, 2.0, 5.0, 9.0]
        dy_opts = [0.0, 200.0, 460.0, 800.0, 1200.0]
        amc_opts = [2.0, 4.0]
        
        best_action = None
        
        if self.policy == 'Oracle':
            best_action = (0.7, 0.0, 4.0)
            r_eff, exp, _, nd, out, berr = self.predict(*best_action, True, epoch)
            exp = self.base_time 
            P_post = self.oracle_P
            berr = 0.0
            nd = -120.0
            r_eff = 4.0 * 1.0 * (1.0 - 0.7/10.0)
            out = 0.0
            
        elif self.policy == 'Fixed_800':
            best_action = (0.7, 800.0, 2.0)
            r_eff, exp, P_post, nd, out, berr = self.predict(*best_action, True, epoch)
            
        elif self.policy == 'Fixed_460':
            best_action = (0.7, 460.0, 2.0)
            r_eff, exp, P_post, nd, out, berr = self.predict(*best_action, True, epoch)
            
        elif self.policy == 'No_Loc_No_Cog':
            best_action = (0.1, 0.0, 2.0)
            r_eff, exp, P_post, nd, out, berr = self.predict(*best_action, False, epoch)
            
        elif self.policy == 'H_MRSM':
            if self.P_cov > 2000.0: dy = 460.0
            else: dy = 0.0
            best_action = (0.7, dy, 2.0)
            r_eff, exp, P_post, nd, out, berr = self.predict(*best_action, True, epoch)
            
        elif self.policy == 'RNCO':
            best_dpp = np.inf
            V = 1000.0
            r_target = 5.0
            e_budget = 8.5
            
            dx_target = np.abs(0.0 - self.uav_x)
            P_target = (dx_target * np.pi / 180.0)**2
            
            w1, w2, w3 = self.weights
            
            for ts in ts_opts:
                for dy in dy_opts:
                    for amc in amc_opts:
                        r, e, P, nd_tmp, out_tmp, berr_tmp = self.predict(ts, dy, amc, True, epoch)
                        
                        Q_RF_n = max(self.Q_RF + (r_target - r), 0)
                        Q_EXP_n = max(self.Q_EXP + (e - e_budget), 0)
                        Q_INFO_n = max(self.Q_INFO + (P - P_target)/100.0, 0)
                        
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
        self.logs['pmcs_rf'] *= (1.0 - out)

def run_c6_audit():
    print("Running C6 10-epoch mission baseline update...")
    scenarios = ['A', 'B', 'C', 'D', 'E']
    policies = ['Oracle', 'RNCO', 'Fixed_800', 'Fixed_460', 'H_MRSM', 'No_Loc_No_Cog']
    
    results = []
    
    for s in scenarios:
        for p in policies:
            sim = MissionSimulator(scenario=s, policy=p)
            for ep in range(sim.epochs): sim.step(ep)
            
            tot_exp = np.sum(sim.logs['exposure'])
            surv = np.exp(-tot_exp / 120.0)
            pmcs = surv * sim.logs['pmcs_rf']
            
            results.append({
                'Scenario': s, 'Policy': p, 'Pmcs': pmcs,
                'Avg_dy': np.mean(sim.logs['dy'])
            })
            
    df = pd.DataFrame(results)
    pivot = df[~df['Policy'].str.contains('No_') | (df['Policy'] == 'No_Loc_No_Cog')].pivot(index='Policy', columns='Scenario', values='Pmcs')
    print("\\n=== Corrected 10-Epoch Mission Success (Pmcs) ===")
    print(pivot.to_string(float_format=lambda x: f"{x:.3f}"))
    
    print("\\nRunning Queue Weight Sensitivity Analysis...")
    base_w1, base_w2, base_w3 = 1.0, 1000.0, 10.0
    multipliers = [0.5, 1.0, 2.0]
    
    # We test on Scenario A (nominal) to see how the weights alter resource allocation
    sens_results = []
    
    # 1. EXP Weight sweep
    for mult in multipliers:
        w_curr = (base_w1 * mult, base_w2, base_w3)
        sim = MissionSimulator(scenario='A', policy='RNCO', weights=w_curr)
        for ep in range(sim.epochs): sim.step(ep)
        sens_results.append({
            'Sweep': 'w_EXP', 'Mult': f"{mult}x",
            'Avg_Tput': np.mean(sim.logs['throughput']),
            'Total_Exp': np.sum(sim.logs['exposure']),
            'Avg_P_trace': np.mean(sim.logs['P_trace']),
            'Avg_dy': np.mean(sim.logs['dy'])
        })
        
    # 2. RF Weight sweep
    for mult in multipliers:
        w_curr = (base_w1, base_w2 * mult, base_w3)
        sim = MissionSimulator(scenario='A', policy='RNCO', weights=w_curr)
        for ep in range(sim.epochs): sim.step(ep)
        sens_results.append({
            'Sweep': 'w_RF', 'Mult': f"{mult}x",
            'Avg_Tput': np.mean(sim.logs['throughput']),
            'Total_Exp': np.sum(sim.logs['exposure']),
            'Avg_P_trace': np.mean(sim.logs['P_trace']),
            'Avg_dy': np.mean(sim.logs['dy'])
        })

    # 3. INFO Weight sweep
    for mult in multipliers:
        w_curr = (base_w1, base_w2, base_w3 * mult)
        sim = MissionSimulator(scenario='A', policy='RNCO', weights=w_curr)
        for ep in range(sim.epochs): sim.step(ep)
        sens_results.append({
            'Sweep': 'w_INFO', 'Mult': f"{mult}x",
            'Avg_Tput': np.mean(sim.logs['throughput']),
            'Total_Exp': np.sum(sim.logs['exposure']),
            'Avg_P_trace': np.mean(sim.logs['P_trace']),
            'Avg_dy': np.mean(sim.logs['dy'])
        })
        
    df_sens = pd.DataFrame(sens_results)
    print("\\n=== Queue Weight Sensitivity Analysis ===")
    print(df_sens.to_string(index=False, float_format=lambda x: f"{x:.2f}" if isinstance(x, float) else x))

if __name__ == "__main__":
    run_c6_audit()
