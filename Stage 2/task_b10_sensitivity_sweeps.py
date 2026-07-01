"""Stage 2 Task B10: Follower Jammer Reaction Delay Sensitivity Sweep.

This script sweeps the follower reaction delay tau_delay in {0, 2, 5, 8, 10} ms,
solves for the defeat ranges of Fixed BFSK and AMC, and reports the range-extension factor.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Ensure Stage 2 directory is in path for imports
sys.path.append(os.path.dirname(__file__))
import task_b7_amc as t7

def solve_defeat_range(N, theta_s, theta_j, alpha, jammer_type, tau_delay, use_amc, trials=10000):
    """Solve for the defeat range (distance where link outage probability is exactly 10%)."""
    lo, hi = 100.0, 20000.0  # Search range in meters
    # Use 12 bisection steps to get ~5m resolution
    for step in range(12):
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

def run_follower_delay_sweep():
    print("=" * 80)
    print("      PHASE B SPRINT 3: FOLLOWER JAMMER DELAY SWEEP & RESILIENCE")
    print("=" * 80)
    
    N = 8
    theta_s = np.radians(0.0)
    theta_j = np.radians(30.0)
    alpha = 1.0 # Follower active on all jammed channels
    jammer_type = 'follower'
    
    delays_ms = np.array([0.0, 2.0, 5.0, 8.0, 10.0])
    delays_sec = delays_ms / 1000.0
    
    results_fixed = []
    results_amc = []
    
    print("\nStarting Follower Reaction Delay Sweep (10,000 trials per point)...")
    
    for d_ms, d_sec in zip(delays_ms, delays_sec):
        print(f"\nSimulating Reaction Delay: {d_ms} ms...")
        
        # Fixed BFSK
        print("  Solving for Fixed BFSK defeat range...")
        dr_fixed = solve_defeat_range(N, theta_s, theta_j, alpha, jammer_type, d_sec, use_amc=False, trials=10000)
        results_fixed.append(dr_fixed / 1000.0)
        print(f"    Fixed BFSK Defeat Range: {dr_fixed / 1000.0:.3f} km")
        
        # AMC
        print("  Solving for AMC defeat range...")
        dr_amc = solve_defeat_range(N, theta_s, theta_j, alpha, jammer_type, d_sec, use_amc=True, trials=10000)
        results_amc.append(dr_amc / 1000.0)
        print(f"    AMC Defeat Range: {dr_amc / 1000.0:.3f} km")
        
    results_fixed = np.array(results_fixed)
    results_amc = np.array(results_amc)
    gain_factors = results_amc / results_fixed
    
    # Print results table
    print("\n" + "=" * 90)
    print("                  FOLLOWER JAMMER DELAY SENSITIVITY MATRIX")
    print("=" * 90)
    print(f"{'Delay (ms)':<12} | {'Fixed BFSK (km)':^18} | {'AMC (km)':^15} | {'Range Extension':^18}")
    print("-" * 90)
    for d_ms, r_fs, r_amc, gain in zip(delays_ms, results_fixed, results_amc, gain_factors):
        print(f"{d_ms:<12.1f} | {r_fs:14.3f} km | {r_amc:11.3f} km | {gain:14.2f}x")
    print("=" * 90)
    
    # Plotting
    plt.figure(figsize=(9, 6))
    plt.plot(delays_ms, results_fixed, 'o-', label="Fixed BFSK Baseline", color="blue", linewidth=2.5, markersize=8)
    plt.plot(delays_ms, results_amc, 's-', label="AMC (BPSK/QPSK/16QAM)", color="green", linewidth=2.5, markersize=8)
    
    # Highlight 5 ms reference point
    plt.axvline(5.0, color="red", linestyle="--", alpha=0.7)
    plt.text(5.2, results_fixed[2], "Reference Delay (5 ms)", color="red", fontweight="bold")
    
    plt.title("UAV Link Defeat Range vs. Jammer Reaction Delay\n(N=8 Phased Array, 10% Outage Criterion)", fontsize=12, fontweight="bold")
    plt.xlabel("Jammer Reaction Delay (tau_delay) [ms]", fontsize=11)
    plt.ylabel("Defeat Range [km]", fontsize=11)
    plt.xticks(delays_ms)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper left", frameon=True, fontsize=10)
    
    plot_path = os.path.join(os.path.dirname(__file__), "phase_b_follower_delay_sweep.png")
    plt.savefig(plot_path, dpi=300)
    print(f"\n[SUCCESS] Saved follower delay sweep plot to: {plot_path}")
    
if __name__ == "__main__":
    run_follower_delay_sweep()
