import sys
import os
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, r'd:\UAV Internship project\Stage 2')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(__file__))
import task_c3_cognitive_sensing as c3

class MissionSimulator:
    def __init__(self, scenario='A', policy='RNCO', ablation=None):
        self.scenario = scenario
        self.policy = policy
        self.ablation = ablation
        
        self.epochs = 30
        self.dx_step = 425.0
        self.base_time = 8.5
        
        # Oracle stats
        self.oracle_P = 10.0 # Perfect tracking
        
        # Initialize state
        self.uav_x = -15000.0 + (30 * 425.0) # Start far away
        self.uav_x = -15000.0 # Actually, let's start at -15000 and go to ~ -2000
        
        self.P_cov = 1000.0
        
        # Queues
        self.Q_RF = 0.0
        self.Q_EXP = 0.0
        self.Q_INFO = 0.0
        
        # Logs
        self.logs = {
            'throughput': [], 'exposure': [], 'P_trace': [], 
            'dy': [], 'ts': [], 'outage': [], 'nd': [], 'pmcs_rf': 1.0,
            'bearing_err': []
        }
        
    def get_environment(self, epoch):
        # Default
        inr_base = 50.0
        snr_base = 15.0
        calib_error = 0.0
        jam_y = 0.0
        
        if self.scenario == 'B': # Extended Shadowing
            if 10 <= epoch <= 20:
                inr_base -= 20.0
                
        elif self.scenario == 'C': # Rapid jammer geometry
            # Jammer is offset by 5000m, making GDOP huge if UAV doesn't maneuver
            jam_y = 5000.0
            
        elif self.scenario == 'D': # Reduced array
            calib_error = 0.5 # Extra 0.5 deg RMSE
            
        elif self.scenario == 'E': # Weak Jammer
            inr_base = 20.0
            
        jnr_lin = 10.0**((inr_base - snr_base - 5.0)/10.0)
        return inr_base, snr_base, calib_error, jam_y, jnr_lin
        
    def predict(self, ts, dy, amc, use_ekf, epoch):
        inr_base, snr_base, calib_error, jam_y, jnr_lin = self.get_environment(epoch)
        fs = 1e6; p_fa = 0.05
        
        if ts > 0:
            pd_val = c3.compute_pd(ts*1e-3, fs, p_fa, jnr_lin)
            L_eff = max(1, int(100 * (1.0 - ts / 10.0)))
        else:
            pd_val = 1.0; L_eff = 100
            
        # Instantaneous MUSIC noise floor
        base_noise = 0.5
        # If jammer is heavily shadowed or we don't sense enough, instantaneous MUSIC breaks down
        jnr_lin = 10.0**((inr_base - 50.0 - 5.0)/10.0)
        if jnr_lin < 0.01 or L_eff < 10:
            base_noise = 45.0 # Complete loss of instantaneous bearing
            
        m_rmse = max(base_noise, 10.0 / np.sqrt(L_eff)) + calib_error
        
        # Exposure
        exp = self.base_time + (2 * dy) / 50.0
        
        # Information update
        dx = np.abs(0.0 - self.uav_x)
        dy_j = np.abs(jam_y - dy)
        r = np.sqrt(dx**2 + dy_j**2)
        # Proper GDOP calculation: depends on lateral deviation
        g = r / max(dy_j, 10.0) if dy_j > 0 else r / 10.0
        
        pos_err_meas = g * m_rmse * dx * (np.pi/180.0)
        R_meas = pos_err_meas**2
        P_pred = self.P_cov + 1000.0 # Process noise
        
        if use_ekf:
            K = P_pred / (P_pred + R_meas + 1e-3)
            P_post = (1 - K) * P_pred
            # Convert pos error to bearing error
            bearing_err = (np.sqrt(P_post) / dx) * (180.0/np.pi)
        else:
            P_post = P_pred # No update
            bearing_err = m_rmse # Instantaneous MUSIC
            
        # RF Metrics
        # Null depth depends on bearing error (50 dB loss per degree of error)
        nd = max(-120.0, -120.0 + 50.0 * bearing_err)
        # Cap worst-case null depth at 0 dB (no null)
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
        
        use_ekf = True
        best_action = None
        
        if self.policy == 'Oracle':
            best_action = (0.7, 0.0, 4.0) # Always perfect
            r_eff, exp, _, nd, out, berr = self.predict(*best_action, True, epoch)
            import simulator_core_remediated as scr
            import phase_b_beamforming_remediated as pbb
            
            exp = self.base_time
            P_post = self.oracle_P
            berr = 0.0
            
            theta_s_rad = np.radians(-45.0)
            dx = np.abs(0.0 - self.uav_x)
            jam_y = 5000.0 if self.scenario == 'C' else 0.0
            inr_base = 50.0
            if self.scenario == 'B' and 10 <= epoch <= 20: inr_base -= 20.0
            if self.scenario == 'E': inr_base = 20.0
            snr_base = 15.0
            
            theta_j_true_rad = np.arctan2(jam_y, max(abs(dx), 1.0))
            
            R_xx_true = pbb.build_rician_covariance_matrix(
                N=4, theta_s_rad=theta_s_rad, sigma2_s=10.0**(snr_base/10.0), K_signal=10.0,
                theta_j_rad=theta_j_true_rad, sigma2_j=10.0**(inr_base/10.0), K_jammer=12.0,
                sigma2_noise=1.0
            )
            
            a_s = pbb.ula_steering_vector_3d(4, -45.0, 0.0)
            a_j = pbb.ula_steering_vector_3d(4, np.degrees(theta_j_true_rad), 0.0)
            
            w_lcmv, _, _, _ = scr.lcmv_with_fallback(
                R_xx_true, a_s, np.degrees(theta_j_true_rad), 0.0, 4, loading_factor=0.10
            )
            leakage = np.abs(np.conj(w_lcmv).T @ a_j)**2
            nd = 10 * np.log10(max(1e-10, float(np.squeeze(leakage).item())))
            
            r_eff = 4.0 * 1.0 * (1.0 - 0.7/10.0)
            out = 0.0
        elif self.policy == 'Fixed_800':
            best_action = (0.7, 800.0, 2.0)
            r_eff, exp, P_post, nd, out, berr = self.predict(*best_action, True, epoch)
            
        elif self.policy == 'Fixed_460':
            best_action = (0.7, 460.0, 2.0)
            r_eff, exp, P_post, nd, out, berr = self.predict(*best_action, True, epoch)
            
        elif self.policy == 'No_Loc_No_Cog':
            use_ekf = False
            best_action = (0.1, 0.0, 2.0) # Barely sense, no move
            r_eff, exp, P_post, nd, out, berr = self.predict(*best_action, False, epoch)
            
        elif self.policy == 'H_MRSM':
            # Heuristic: if trace > threshold, do 460m, else 0
            if self.P_cov > 2000.0:
                dy = 460.0
            else:
                dy = 0.0
            best_action = (0.7, dy, 2.0)
            r_eff, exp, P_post, nd, out, berr = self.predict(*best_action, True, epoch)
            
        elif self.policy == 'RNCO':
            # Run Lyapunov Optimizer
            best_dpp = np.inf
            V = 1000.0
            r_target = 5.0
            e_budget = 8.5
            
            # P_target is now dynamic! We want a maximum of 1.0 degree angular error.
            # Convert 1.0 degree to Cartesian variance at current distance.
            dx_target = np.abs(0.0 - self.uav_x)
            P_target = (dx_target * np.pi / 180.0)**2
            
            w1 = 1.0; w2 = 1000.0; w3 = 10.0
            
            for ts in ts_opts:
                for dy in dy_opts:
                    for amc in amc_opts:
                        r, e, P, nd_tmp, out_tmp, berr_tmp = self.predict(ts, dy, amc, True, epoch)
                        
                        Q_RF_n = max(self.Q_RF + (r_target - r), 0)
                        Q_EXP_n = max(self.Q_EXP + (e - e_budget), 0)
                        Q_INFO_n = max(self.Q_INFO + (P - P_target)/100.0, 0)
                        
                        rf_w = 0.0 if self.ablation == 'No_RF' else 1.0
                        exp_w = 0.0 if self.ablation == 'No_EXP' else 1.0
                        info_w = 0.0 if self.ablation == 'No_INFO' else 1.0
                        
                        L_t_abl = 0.5*(rf_w*self.Q_RF**2 + exp_w*self.Q_EXP**2 + info_w*self.Q_INFO**2)
                        L_n_abl = 0.5*(rf_w*Q_RF_n**2 + exp_w*Q_EXP_n**2 + info_w*Q_INFO_n**2)
                        drift = L_n_abl - L_t_abl
                        
                        pen = w1*e + w2*max(0, r_target-r) + w3*(P/100.0)
                        dpp = drift + V*pen
                        
                        if dpp < best_dpp:
                            best_dpp = dpp
                            best_action = (ts, dy, amc)
                            # Store metrics
                            r_eff, exp, P_post, nd, out, berr = r, e, P, nd_tmp, out_tmp, berr_tmp
                            best_Q = (Q_RF_n, Q_EXP_n, Q_INFO_n)
                            
            self.Q_RF, self.Q_EXP, self.Q_INFO = best_Q

        # Apply state updates
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
        
        # Cumulative RF survival
        self.logs['pmcs_rf'] *= (1.0 - out)

def run_c6():
    scenarios = ['A', 'B', 'C', 'D', 'E']
    policies = ['Oracle', 'RNCO', 'Fixed_800', 'Fixed_460', 'H_MRSM', 'No_Loc_No_Cog']
    ablations = [None, 'No_RF', 'No_EXP', 'No_INFO']
    
    results = []
    
    print("Running C6 Competitions...")
    
    for s in scenarios:
        for p in policies:
            if p == 'RNCO':
                # Run main + ablations
                for a in ablations:
                    sim = MissionSimulator(scenario=s, policy=p, ablation=a)
                    for ep in range(30): sim.step(ep)
                    pol_name = f"RNCO_{a}" if a else "RNCO"
                    
                    tot_exp = np.sum(sim.logs['exposure'])
                    surv = np.exp(-tot_exp / 120.0)
                    pmcs = surv * sim.logs['pmcs_rf']
                    
                    results.append({
                        'Scenario': s, 'Policy': pol_name, 'Pmcs': pmcs,
                        'Avg_Tput': np.mean(sim.logs['throughput']),
                        'Avg_Exp': np.mean(sim.logs['exposure']),
                        'Avg_Trace': np.mean(sim.logs['P_trace']),
                        'Avg_dy': np.mean(sim.logs['dy']),
                        'Avg_ts': np.mean(sim.logs['ts']),
                        'Survival': surv,
                        'Avg_ND': np.mean(sim.logs['nd'])
                    })
            else:
                sim = MissionSimulator(scenario=s, policy=p)
                for ep in range(30): sim.step(ep)
                tot_exp = np.sum(sim.logs['exposure'])
                surv = np.exp(-tot_exp / 120.0)
                pmcs = surv * sim.logs['pmcs_rf']
                
                results.append({
                    'Scenario': s, 'Policy': p, 'Pmcs': pmcs,
                    'Avg_Tput': np.mean(sim.logs['throughput']),
                    'Avg_Exp': np.mean(sim.logs['exposure']),
                    'Avg_Trace': np.mean(sim.logs['P_trace']),
                    'Avg_dy': np.mean(sim.logs['dy']),
                    'Avg_ts': np.mean(sim.logs['ts']),
                    'Survival': surv,
                    'Avg_ND': np.mean(sim.logs['nd'])
                })
                
    df = pd.DataFrame(results)
    df.to_csv('task_c6_results.csv', index=False)
    
    # Analyze and print PMCS Matrix
    print("\\n=== Mission Success Probability (Pmcs) Matrix ===")
    pivot = df[~df['Policy'].str.contains('No_') | (df['Policy'] == 'No_Loc_No_Cog')].pivot(index='Policy', columns='Scenario', values='Pmcs')
    print(pivot.to_string(float_format=lambda x: f"{x:.3f}"))
    
    # Calculate Gaps for RNCO
    print("\\n=== Optimality Gaps (RNCO vs Best Fixed vs Oracle) ===")
    for s in scenarios:
        sub = df[(df['Scenario'] == s) & (~df['Policy'].str.contains('No_'))]
        oracle_p = sub[sub['Policy'] == 'Oracle']['Pmcs'].values[0]
        rnco_p = sub[sub['Policy'] == 'RNCO']['Pmcs'].values[0]
        fixed = sub[sub['Policy'].isin(['Fixed_800', 'Fixed_460', 'H_MRSM'])]['Pmcs'].max()
        
        print(f"Scenario {s}:")
        print(f"  Oracle: {oracle_p:.3f} | RNCO: {rnco_p:.3f} | Best Fixed: {fixed:.3f}")
        print(f"  Gap to Oracle: {oracle_p - rnco_p:.3f}")
        print(f"  Gain over Fixed: {rnco_p - fixed:.3f}\\n")
        
    print("=== Queue Ablation Impact (Average Pmcs across all Scenarios) ===")
    ablation_df = df[df['Policy'].str.contains('RNCO')]
    abl_mean = ablation_df.groupby('Policy')['Pmcs'].mean()
    base_rnco = abl_mean['RNCO']
    for p in ['RNCO_No_RF', 'RNCO_No_EXP', 'RNCO_No_INFO']:
        drop = base_rnco - abl_mean[p]
        print(f"{p}: Pmcs = {abl_mean[p]:.3f} (Degradation: -{drop:.3f})")

if __name__ == "__main__":
    run_c6()
