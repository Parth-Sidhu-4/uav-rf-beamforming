import os
import sys
import math
import numpy as np
import scipy.linalg as la

sys.path.append(os.path.dirname(__file__))
import task_b10_grand_integration as b10

def resume_experiment_3():
    taus = [240.0]
    rng = np.random.default_rng(42)
    # Fast forward the RNG state to match what it would have been approximately
    # Since we don't need exact reproducibility, just good random samples:
    for tau in taus:
        p_base, _, _ = b10.run_mission_monte_carlo('A_Baseline', trials=300, tau_kinetic=tau, rng=rng)
        p_music, _, _ = b10.run_mission_monte_carlo('C_MUSIC', trials=300, tau_kinetic=tau, rng=rng)
        print(f"{tau:<10.1f} | {p_base:<15.2f} | {p_music:<15.2f}")

if __name__ == "__main__":
    resume_experiment_3()
    b10.run_experiment_5_amc_outage()
    b10.run_experiment_6_k_factor()
    b10.run_experiment_7_fspl_baseline()
