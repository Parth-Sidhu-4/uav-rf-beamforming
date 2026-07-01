"""Stage 2 Task B9: Array Element Count Sweep and Defeat Boundaries.

This script sweeps the ULA element count N in {1, 4, 8, 16} and solves for the
defeat ranges under Barrage, PBJ, and Follower jamming.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Ensure Stage 2 directory is in path
sys.path.append(os.path.dirname(__file__))
import phase_b_beamforming as pbb
import task_b_jamming_threats as jts
import task_b7_amc as t7

def solve_defeat_range_amc(N, theta_s, theta_j, alpha, jammer_type='pbj', tau_delay=0.0, use_amc=True, trials=1000):
    """Solve for the defeat range (distance where link outage probability is exactly 10%)."""
    lo, hi = 100.0, 250000.0
    for _ in range(16):
        mid = (lo + hi) / 2.0
        p_b, _, _, _, _ = t7.simulate_amc_hop_sequence(
            mid, N, theta_s, theta_j, alpha, jammer_type, tau_delay, use_amc=use_amc, trials=trials
        )
        outage = 1.0 - p_b
        if outage <= 0.10:
            lo = mid
        else:
            hi = mid
    return lo

def run_array_size_sweep():
    print("=" * 80)
    print("         PHASE B SPRINT 3: GCS ARRAY SIZE SWEEP & DEFEAT BOUNDARIES")
    print("=" * 80)
    
    n_elements = [1, 4, 8, 16]
    theta_s = np.radians(0.0)
    theta_j = np.radians(30.0)
    
    threats = [
        ('Barrage', 1.0, 'pbj', 0.0),
        ('PBJ', 0.11, 'pbj', 0.0),
        ('Follower', 1.0, 'follower', 5e-3)
    ]
    
    results = {}
    
    for N in n_elements:
        print(f"\nSimulating N = {N} elements...")
        for name, alpha, j_type, delay in threats:
            print(f"  Threat: {name}")
            # Fixed
            dr_fs = solve_defeat_range_amc(N, theta_s, theta_j, alpha, j_type, delay, use_amc=False, trials=500)
            # AMC
            dr_amc = solve_defeat_range_amc(N, theta_s, theta_j, alpha, j_type, delay, use_amc=True, trials=500)
            
            results[(N, name)] = (dr_fs / 1000.0, dr_amc / 1000.0)
            
    # Print results table
    print("\n" + "=" * 90)
    print("                      ARRAY SWEEP DEFEAT RANGE MATRIX (km)")
    print("=" * 90)
    print(f"{'Array Size':<12} | {'Threat':<12} | {'Fixed BFSK (km)':^18} | {'AMC (km)':^15} | {'Gain Factor':^12}")
    print("-" * 90)
    for N in n_elements:
        for name, _, _, _ in threats:
            dr_fs, dr_amc = results[(N, name)]
            gain = dr_amc / max(dr_fs, 0.01)
            print(f"N = {N:<8} | {name:<12} | {dr_fs:14.2f} km | {dr_amc:11.2f} km | {gain:10.2f}x")
        print("-" * 90)
    print("=" * 90)

    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors_fs = 'blue'
    colors_amc = 'green'
    
    for i, (name, _, _, _) in enumerate(threats):
        ax = axes[i]
        dr_fs_list = [results[(N, name)][0] for N in n_elements]
        dr_amc_list = [results[(N, name)][1] for N in n_elements]
        
        x = np.arange(len(n_elements))
        width = 0.35
        
        ax.bar(x - width/2, dr_fs_list, width, label='Fixed BFSK', color=colors_fs)
        ax.bar(x + width/2, dr_amc_list, width, label='AMC', color=colors_amc)
        
        ax.set_title(f"Defeat Range vs. Array Size\n(Threat: {name})", fontsize=11, fontweight="bold")
        ax.set_xlabel("ULA Element Count N", fontsize=10)
        ax.set_ylabel("Defeat Range [km]", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([f"N={N}" for N in n_elements])
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper left", frameon=True)
        
    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), "phase_b_array_sweep_amc.png")
    plt.savefig(plot_path, dpi=300)
    print(f"\n[SUCCESS] Completed array sweep and saved plots to: {plot_path}")

if __name__ == "__main__":
    run_array_size_sweep()
