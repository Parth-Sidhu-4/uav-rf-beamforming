import numpy as np
import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(__file__))
from simulator_core import UAVSimulator

def run_3d_sweep():
    JNR_sweep = [30.0, 50.0]        
    SNR_sweep = [15.0]            
    N_sweep = [2, 4]               
    theta_sweep = [0.5, 1.5, 3.0]   
    
    results = []
    
    for jnr in JNR_sweep:
        for snr in SNR_sweep:
            for n in N_sweep:
                best_pmcs = -1.0
                best_theta = None
                
                for t in theta_sweep:
                    sim = UAVSimulator(scenario='B', target_deg=t)
                    
                    # Hack simulator to use specific sweeps
                    sim.N_array = n
                    
                    # Run sim
                    for ep in range(sim.epochs):
                        sim.step(ep)
                        
                    tot_exp = np.sum(sim.logs['exposure'])
                    surv = np.exp(-tot_exp / 120.0)
                    pmcs = surv * sim.logs['pmcs_rf']
                    
                    if pmcs > best_pmcs:
                        best_pmcs = pmcs
                        best_theta = t
                        
                results.append({
                    'JNR (dB)': jnr,
                    'SNR (dB)': snr,
                    'N_elements': n,
                    'theta_star (deg)': best_theta,
                    'Max Pmcs': best_pmcs
                })
                
    df = pd.DataFrame(results)
    print("\\n=== Fix 8: Empirical Precision Characterization ===")
    print(df.to_string(index=False))
    df.to_csv('theta_star_table.csv', index=False)

if __name__ == "__main__":
    run_3d_sweep()
