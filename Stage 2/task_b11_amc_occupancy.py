"""Stage 2 Task B11: AMC Mode Occupancy vs. Distance.

This script sweeps distance and records the percentage of time spent in each AMC mode
(Outage, BPSK, QPSK 1/2, QPSK 3/4, 16QAM) across four critical operational scenarios:
1. Follower Jammer (tau_delay = 5 ms) - Reference Scenario
2. Follower Jammer (tau_delay = 0 ms) - Hysteresis Lockout Study
3. Partial-Band Jamming (alpha = 0.11) - PBJ Outage Study
4. Follower Jammer (tau_delay = 10 ms) - Missed-Hop Study
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Ensure Stage 2 directory is in path for imports
sys.path.append(os.path.dirname(__file__))
import task_b7_amc as t7
import task_b_jamming_threats as jts

def get_mode_occupancy(d_m, N, theta_s, theta_j, alpha, jammer_type, tau_delay, trials=2000):
    """Run simulation at range d_m and return the fraction of hops spent in each mode."""
    _, _, _, _, modes = t7.simulate_amc_hop_sequence(
        d_m, N, theta_s, theta_j, alpha, jammer_type, tau_delay, use_amc=True, trials=trials
    )
    modes = np.array(modes)
    total = len(modes)
    
    # Modes: -1 (Outage), 0 (BPSK 1/2), 1 (QPSK 1/2), 2 (QPSK 3/4), 3 (16QAM 1/2)
    occ = {
        -1: np.sum(modes == -1) / total,
        0: np.sum(modes == 0) / total,
        1: np.sum(modes == 1) / total,
        2: np.sum(modes == 2) / total,
        3: np.sum(modes == 3) / total
    }
    return occ

def run_occupancy_study():
    print("=" * 80)
    print("      PHASE B SPRINT 3: AMC MODE OCCUPANCY VS. DISTANCE DIAGNOSTIC")
    print("=" * 80)
    
    N = 8
    theta_s = np.radians(0.0)
    theta_j = np.radians(30.0)
    
    # 4 Scenarios to study
    scenarios = [
        {
            'title': '1. Follower Jammer (5 ms Delay - Ref)',
            'alpha': 1.0, 'jammer_type': 'follower', 'delay': 5e-3
        },
        {
            'title': '2. Follower Jammer (0 ms Delay - Reactive)',
            'alpha': 1.0, 'jammer_type': 'follower', 'delay': 0.0
        },
        {
            'title': '3. Partial-Band Jamming (alpha = 0.11)',
            'alpha': 0.11, 'jammer_type': 'pbj', 'delay': 0.0
        },
        {
            'title': '4. Follower Jammer (10 ms Delay - Miss)',
            'alpha': 1.0, 'jammer_type': 'follower', 'delay': 10e-3
        }
    ]
    
    distances_km = np.linspace(0.1, 15.0, 40)
    distances_m = distances_km * 1000.0
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    colors = ['#8b0000', '#1f77b4', '#aec7e8', '#ff7f0e', '#2ca02c'] # Outage, BPSK, QPSK1/2, QPSK3/4, 16QAM
    labels = ['Outage (Mode -1)', 'BPSK 1/2 (Mode 0)', 'QPSK 1/2 (Mode 1)', 'QPSK 3/4 (Mode 2)', '16QAM 1/2 (Mode 3)']
    
    for i, sc in enumerate(scenarios):
        print(f"\nSimulating Scenario {i+1}: {sc['title']}...")
        
        # Initialize occupancy histories
        hist_outage = []
        hist_bpsk = []
        hist_qpsk_half = []
        hist_qpsk_three_quarter = []
        hist_qam = []
        
        for d in distances_m:
            occ = get_mode_occupancy(d, N, theta_s, theta_j, sc['alpha'], sc['jammer_type'], sc['delay'], trials=2000)
            hist_outage.append(occ[-1])
            hist_bpsk.append(occ[0])
            hist_qpsk_half.append(occ[1])
            hist_qpsk_three_quarter.append(occ[2])
            hist_qam.append(occ[3])
            
        # Convert to numpy arrays
        y = np.vstack([hist_outage, hist_bpsk, hist_qpsk_half, hist_qpsk_three_quarter, hist_qam])
        
        # Plot stacked area chart
        ax = axes[i]
        ax.stackplot(distances_km, y * 100.0, labels=labels, colors=colors, alpha=0.85)
        
        # Add ground reflection notch vertical line at 3.2 km
        ax.axvline(3.2, color='white', linestyle='--', linewidth=1.5, alpha=0.8)
        ax.text(3.3, 50.0, "Two-Ray Notch\n(3.2 km)", color='white', fontsize=9, fontweight='bold')
        
        ax.set_title(sc['title'], fontsize=12, fontweight='bold')
        ax.set_xlabel("Link Range [km]", fontsize=10)
        ax.set_ylabel("Mode Occupancy Percentage [%]", fontsize=10)
        ax.set_xlim([0.1, 15.0])
        ax.set_ylim([0, 100])
        ax.grid(True, linestyle=':', alpha=0.5)
        
        if i == 0:
            ax.legend(loc='lower left', frameon=True, facecolor='white', edgecolor='gray', fontsize=9)
            
    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), "phase_b_amc_mode_occupancy.png")
    plt.savefig(plot_path, dpi=300)
    print(f"\n[SUCCESS] Completed occupancy study and saved plots to: {plot_path}")

if __name__ == "__main__":
    run_occupancy_study()
