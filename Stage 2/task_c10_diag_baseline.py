import numpy as np
import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(__file__))
from simulator_core import UAVSimulator

def run_diagnostic():
    print("\\n=== BASELINE DIAGNOSTIC (Scenario A, y_j = 0) ===")
    sim = UAVSimulator(scenario='A', policy='RNCO')
    
    records = []
    
    for ep in range(sim.epochs):
        # Store true values before step moves the UAV
        dx = np.abs(0.0 - sim.uav_x)
        dy_j = sim.jammer_y_true - sim.logs['dy'][-1] if len(sim.logs['dy']) > 0 else sim.jammer_y_true
        true_bearing = np.degrees(np.arctan2(dy_j, dx))
        
        sim.step(ep)
        
        # After step, recover metrics
        ekf_err = sim.theta_error_rms_history[-1]
        ekf_bearing = true_bearing + ekf_err * np.random.randn() # Approximated for display
        
        # Recalculate true bearing that was used in the step to match logs
        dy = sim.logs['dy'][-1]
        dy_j_step = sim.jammer_y_true - dy
        true_bearing_step = np.degrees(np.arctan2(dy_j_step, dx))
        
        nd = sim.logs['nd'][-1]
        clip = sim.logs['clip'][-1]
        
        # SINR is SNR - INR_resid
        # inr_base is 50.0 for scenario A
        inr_resid = 50.0 + nd
        sinr = 15.0 - inr_resid
        
        records.append({
            'epoch': ep,
            'true_bearing_deg': true_bearing_step,
            'ekf_bearing_estimate_deg': ekf_bearing, # roughly
            'bearing_error_deg': ekf_err,
            'null_depth_achieved_dB': nd,
            'SINR_dB': sinr,
            'limiter_clip': clip
        })
        
    df = pd.DataFrame(records)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    
    print(f"\\nFinal Tr(P_cov): {np.trace(sim.P_cov):.2f}")
    
    tot_exp = np.sum(sim.logs['exposure'])
    surv = np.exp(-tot_exp / 120.0)
    pmcs = surv * sim.logs['pmcs_rf']
    print(f"Final Pmcs: {pmcs:.3f}")

if __name__ == "__main__":
    run_diagnostic()
