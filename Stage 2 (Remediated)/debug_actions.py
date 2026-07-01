import sys
import os
sys.path.insert(0, r'd:\UAV Internship project\Stage 2')
sys.path.insert(0, r'd:\UAV Internship project\Stage 2 (Remediated)')

from Remediated_Scripts.task_c6_competition import MissionSimulator

sim = MissionSimulator(scenario='A', policy='RNCO')
sim.Q_RF = 1000.0
sim.Q_EXP = 100000.0
sim.Q_INFO = 1000.0
sim.ablation = None

V = 1000.0
r_target = 5.0
e_budget = 8.5
P_target = 0.0

ts_opts = [0.1, 0.7, 2.0, 5.0, 9.0]
dy_opts = [0.0, 200.0, 460.0, 800.0, 1200.0]
amc_opts = [2.0, 4.0]

for ts in ts_opts:
    for dy in dy_opts:
        r, e, P, nd_tmp, out_tmp, berr_tmp = sim.predict(ts, dy, amc_opts[0], True, 0)
        
        Q_RF_n = max(sim.Q_RF + (r_target - r), 0)
        Q_EXP_n = max(sim.Q_EXP + (e - e_budget), 0)
        Q_INFO_n = max(sim.Q_INFO + (P - P_target)/100.0, 0)
        
        L_t = 0.5*(sim.Q_RF**2 + sim.Q_EXP**2 + sim.Q_INFO**2)
        L_n = 0.5*(Q_RF_n**2 + Q_EXP_n**2 + Q_INFO_n**2)
        drift = L_n - L_t
        
        pen = 1.0*e + 1000.0*max(0, r_target-r) + 10.0*(P/100.0)
        dpp = drift + V*pen
        
        print(f"ts={ts}, dy={dy} | r={r:.2f}, e={e:.2f}, P={P:.2f} | drift={drift:.1f}, pen={pen:.1f}, dpp={dpp:.1f}")
        
