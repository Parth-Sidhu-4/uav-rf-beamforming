"""Stage 2 Task B12: Phase B Sprint 4 - Risk-Aware Adaptation Policy Comparison.

This script compares 5 mode selection policies under 5 jamming scenarios.
Policies:
- Fixed BFSK (Baseline legacy)
- Policy A (Conservative: worst-case jammed SINR)
- Policy B (Observation-based: one-hop lag actual SINR)
- Policy C (Hybrid: Confidence-weighted SINR)
- Policy D (Risk-constrained: 30-hop rolling empirical outage budget)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(__file__))
import phase_b_beamforming as pbb
import task_b_jamming_threats as jts
import task_b7_amc as b7

def simulate_policy_hop_sequence(policy, d_m, N, theta_s, theta_j, alpha, jammer_type='pbj', tau_delay=0.0, trials=10000, seed=2026, track_time_series=False, W=30, transition_scenario=False):
    # Nominal received powers
    Ps = jts.get_received_signal_power(d_m, 'two-ray')
    P_j_total = jts.get_received_jammer_power('fspl')
    
    if jammer_type == 'pbj':
        P_j_ch = P_j_total / (alpha * jts.N_h)
    else:
        P_j_ch = P_j_total
        
    SNR_lin = Ps / jts.P_N_lin
    INR_lin = P_j_ch / jts.P_N_lin
    
    theta_elev = np.arctan(jts.h_UAV / d_m)
    K_s_db = 10.0 + 10.0 * np.sin(theta_elev)
    K_s = 10 ** (K_s_db / 10.0)
    
    rng_cal = np.random.default_rng(42)
    amp_err = np.clip(rng_cal.normal(0, jts.amp_err_std, N), -0.9, 0.9)
    phase_err = rng_cal.normal(0, np.radians(jts.phase_err_std_deg), N)
    cal_err = (1.0 + amp_err) * np.exp(1j * phase_err)
    
    a_s_nom = pbb.ula_steering_vector(N, theta_s)
    a_j_nom = pbb.ula_steering_vector(N, theta_j)
    a_s_true = a_s_nom * cal_err
    a_j_true = a_j_nom * cal_err
    
    if N > 1:
        R_xx_dl = SNR_lin * np.outer(a_s_nom, np.conj(a_s_nom)) + INR_lin * np.outer(a_j_nom, np.conj(a_j_nom)) + np.eye(N) + 1e-5 * np.eye(N)
        w = pbb.lcmv_beamformer(R_xx_dl, theta_s, [theta_j])
    else:
        w = np.array([1.0], dtype=complex)
        
    noise_out = np.sum(np.abs(w)**2)
    
    # Nominal SINRs
    sig_out_nom = SNR_lin * (np.abs(np.conj(w).T @ a_s_true) ** 2)
    jam_out_nom = INR_lin * (np.abs(np.conj(w).T @ a_j_true) ** 2)
    sinr_jammed_nom = sig_out_nom / (noise_out + jam_out_nom)
    sinr_unjammed_nom = sig_out_nom / noise_out
    
    if policy == 'fixed':
        prev_mode = 0
    else:
        sinr_jammed_nom_db = 10.0 * np.log10(sinr_jammed_nom)
        prev_mode = -1
        for idx in range(3, -1, -1):
            if sinr_jammed_nom_db >= b7.amc_thresholds_db[idx]:
                prev_mode = idx
                break
                
    gamma_obs_prev_lin = sinr_unjammed_nom
    buffer_gamma_obs_lin = [sinr_unjammed_nom] * W
    
    rng_fad = np.random.default_rng(seed)
    los_s = np.sqrt(K_s / (K_s + 1.0))
    diff_s = np.sqrt(1.0 / (K_s + 1.0))
    
    success_ber_count = 0
    success_sinr_count = 0
    total_throughput = 0.0
    total_ber = 0.0
    
    modes_selected = []
    throughputs = []
    
    for k in range(trials):
        cur_j_type = jammer_type
        cur_tau = tau_delay
        cur_alpha = alpha
        
        if transition_scenario:
            if 100 <= k < 200:
                cur_j_type = 'follower'
                cur_tau = 5e-3
            else:
                cur_j_type = 'none'
                
        if cur_j_type == 'pbj':
            rho = cur_alpha
            I_k = 1 if (rng_fad.random() < cur_alpha) else 0
        elif cur_j_type == 'follower':
            rho = np.max([0.0, 1.0 - cur_tau / jts.T_hop])
            I_k = 1 if rho > 0 else 0
        elif cur_j_type == 'barrage':
            rho = 1.0
            I_k = 1
        else: # none
            rho = 0.0
            I_k = 0
            
        phi_s = rng_fad.uniform(-np.pi, np.pi)
        u_s = rng_fad.normal(0, np.sqrt(0.5)) + 1j * rng_fad.normal(0, np.sqrt(0.5))
        h_s = los_s * np.exp(1j * phi_s) + diff_s * u_s
        h_j = rng_fad.normal(0, np.sqrt(0.5)) + 1j * rng_fad.normal(0, np.sqrt(0.5))
        
        sig_out = SNR_lin * (np.abs(h_s)**2) * (np.abs(np.conj(w).T @ a_s_true) ** 2)
        jam_out = INR_lin * (np.abs(h_j)**2) * (np.abs(np.conj(w).T @ a_j_true) ** 2)
        
        gamma_jammed = sig_out / (noise_out + jam_out)
        gamma_unjammed = sig_out / noise_out
        gamma_obs = gamma_jammed if I_k == 1 else gamma_unjammed
        
        # Policy Selection
        if policy == 'fixed':
            mode = 0
        elif policy == 'A':
            g_db = 10.0 * np.log10(gamma_jammed)
            mode = b7.select_amc_mode_hysteresis(g_db, prev_mode)
        elif policy == 'B':
            g_db = 10.0 * np.log10(gamma_obs_prev_lin)
            mode = b7.select_amc_mode_hysteresis(g_db, prev_mode)
        elif policy == 'C':
            g_prev_db = 10.0 * np.log10(gamma_obs_prev_lin)
            p_hat = 1.0 if g_prev_db < 5.64 else 0.0
            g_lin = p_hat * gamma_jammed + (1.0 - p_hat) * gamma_obs_prev_lin
            g_db = 10.0 * np.log10(g_lin)
            mode = b7.select_amc_mode_hysteresis(g_db, prev_mode)
        elif policy == 'D':
            if k < W:
                g_db = 10.0 * np.log10(gamma_jammed)
                mode = b7.select_amc_mode_hysteresis(g_db, prev_mode)
            else:
                best_mode = -1
                for m in [3, 2, 1, 0]:
                    outages = sum(1 for g in buffer_gamma_obs_lin if b7.get_ber_for_mode(m, g) > b7.target_ber)
                    if outages / W < 0.10:
                        best_mode = m
                        break
                mode = best_mode
                
        if mode == -1:
            total_ber += 0.5
            modes_selected.append(-1)
            if track_time_series: throughputs.append(0.0)
            
            gamma_obs_prev_lin = gamma_obs
            buffer_gamma_obs_lin.pop(0)
            buffer_gamma_obs_lin.append(gamma_obs)
            if policy != 'D': prev_mode = -1
            continue
            
        ber_hop = rho * b7.get_ber_for_mode(mode, gamma_jammed) + (1.0 - rho) * b7.get_ber_for_mode(mode, gamma_unjammed)
        ok_ber = (ber_hop <= b7.target_ber)
        
        # SINR check for backward compatibility
        if policy == 'fixed':
            ok_sinr = (10.0 * np.log10(gamma_jammed) >= 10.0)
        else:
            ok_sinr = (10.0 * np.log10(gamma_jammed) >= b7.amc_thresholds_db[mode] - 2.0)
            
        if ok_ber: success_ber_count += 1
        if ok_sinr: success_sinr_count += 1
        
        total_ber += ber_hop
        modes_selected.append(mode)
        
        eff_rate = b7.amc_spectral_efficiencies[mode] if policy != 'fixed' else 0.5
        hop_thr = eff_rate * (1.0 - ber_hop) if ok_ber else 0.0
        
        total_throughput += hop_thr
        if track_time_series: throughputs.append(hop_thr)
        
        prev_mode = mode
        gamma_obs_prev_lin = gamma_obs
        buffer_gamma_obs_lin.pop(0)
        buffer_gamma_obs_lin.append(gamma_obs)
        
    if track_time_series:
        return modes_selected, throughputs
        
    p_link_ber = success_ber_count / trials
    p_link_sinr = success_sinr_count / trials
    avg_throughput = total_throughput / trials
    avg_ber = total_ber / trials
    
    return p_link_ber, p_link_sinr, avg_throughput, avg_ber, modes_selected

def solve_defeat_range(policy, N, theta_s, theta_j, alpha, jammer_type, tau_delay, trials=2000, target_outage=0.10):
    d_min = 100.0
    d_max = 25000.0
    d_opt = d_max
    
    p_link_max, _, _, _, _ = simulate_policy_hop_sequence(policy, d_max, N, theta_s, theta_j, alpha, jammer_type, tau_delay, trials=trials)
    if 1.0 - p_link_max < target_outage:
        return d_max
        
    for _ in range(12):
        d_mid = (d_min + d_max) / 2.0
        p_link, _, _, _, _ = simulate_policy_hop_sequence(policy, d_mid, N, theta_s, theta_j, alpha, jammer_type, tau_delay, trials=trials)
        if 1.0 - p_link >= target_outage:
            d_max = d_mid
        else:
            d_min = d_mid
            d_opt = d_mid
    return d_opt

def run_policy_comparison():
    policies = ['fixed', 'A', 'B', 'C', 'D']
    policy_names = ['Fixed BFSK', 'Policy A', 'Policy B', 'Policy C', 'Policy D']
    scenarios = [
        ('Barrage', 1.0, 'barrage', 0.0),
        ('PBJ', 0.11, 'pbj', 0.0),
        ('Follower', 1.0, 'follower', 5e-3),
        ('Missed-Hop', 1.0, 'follower', 10e-3)
    ]
    
    N = 8
    d_ref = 2490.0
    theta_s = np.radians(0.0)
    theta_j = np.radians(30.0)
    
    print("="*110)
    print("     PHASE B SPRINT 4: RISK-AWARE ADAPTATION POLICY COMPARISON")
    print("="*110)
    
    # 1. Performance Matrix (10,000 trials)
    results = {}
    print("\n[Step 1] Running Performance Matrix (10,000 trials at d = 2.49 km)...")
    for p, p_name in zip(policies, policy_names):
        for s_name, alpha, j_type, tau in scenarios:
            p_b, p_s, thr, ber, _ = simulate_policy_hop_sequence(p, d_ref, N, theta_s, theta_j, alpha, j_type, tau, trials=10000)
            results[(p, s_name)] = {'p_b': p_b, 'thr': thr}
            
    print("\n" + "="*80)
    print(" STEADY-STATE PERFORMANCE MATRIX (d = 2.49 km) ")
    print("="*80)
    print(f"{'Scenario':<12} | {'Scheme':<12} | {'Reliability':<14} | {'Throughput':<14}")
    print("-" * 80)
    for s_name, _, _, _ in scenarios:
        for p, p_name in zip(policies, policy_names):
            res = results[(p, s_name)]
            print(f"{s_name:<12} | {p_name:<12} | {res['p_b']*100.0:13.2f}% | {res['thr']:10.4f} bps/Hz")
        print("-" * 80)
        
    # 2. Defeat Range Sweeps (Stage 1: 2,000 trials)
    print("\n[Step 2] Running Defeat Range Sweeps (Stage 1: 2,000 trials)...")
    ranges = {}
    for p, p_name in zip(policies, policy_names):
        for s_name, alpha, j_type, tau in scenarios:
            d_opt = solve_defeat_range(p, N, theta_s, theta_j, alpha, j_type, tau, trials=2000)
            ranges[(p, s_name)] = d_opt
            
    print("\n" + "="*80)
    print(" DEFEAT RANGE MATRIX (10% OUTAGE, 2,000 trials) ")
    print("="*80)
    print(f"{'Scenario':<12} | {'Scheme':<12} | {'Defeat Range (km)':<18}")
    print("-" * 80)
    for s_name, _, _, _ in scenarios:
        for p, p_name in zip(policies, policy_names):
            print(f"{s_name:<12} | {p_name:<12} | {ranges[(p, s_name)]/1000.0:18.2f}")
        print("-" * 80)
        
    # 3. Throughput vs Distance Plot (4 panels)
    print("\n[Step 3] Generating Throughput vs Distance curves (30 points, 2,000 trials)...")
    distances = np.linspace(100.0, 20000.0, 30)
    fig_dist, axes_dist = plt.subplots(2, 2, figsize=(14, 10))
    axes_dist = axes_dist.flatten()
    
    colors = ['black', 'red', 'orange', 'blue', 'green']
    ls = ['--', '-', '-', '-', '-']
    
    for i, (s_name, alpha, j_type, tau) in enumerate(scenarios):
        ax = axes_dist[i]
        for p, p_name, c, l in zip(policies, policy_names, colors, ls):
            thr_curve = []
            for d in distances:
                _, _, thr, _, _ = simulate_policy_hop_sequence(p, d, N, theta_s, theta_j, alpha, j_type, tau, trials=2000)
                thr_curve.append(thr)
            ax.plot(distances/1000.0, thr_curve, color=c, linestyle=l, label=p_name, linewidth=2.0)
            
        ax.set_title(f"Scenario: {s_name}", fontweight='bold')
        ax.set_xlabel("Link Range [km]")
        ax.set_ylabel("Throughput [bps/Hz]")
        ax.set_xlim([0, 20])
        ax.grid(True, linestyle=":", alpha=0.6)
        if i == 0:
            ax.legend(loc="upper right")
            
    plt.tight_layout()
    fig_dist.savefig(os.path.join(os.path.dirname(__file__), "phase_b_policy_comparison.png"), dpi=300)
    
    # 4. S5 Threat Transition
    print("\n[Step 4] Running S5 Threat Transition Time-Series (300 hops)...")
    fig_s5, axes_s5 = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
    
    for i, (p, p_name) in enumerate(zip(policies, policy_names)):
        modes, _ = simulate_policy_hop_sequence(
            p, d_ref, N, theta_s, theta_j, 1.0, 'none', 0.0, 
            trials=300, transition_scenario=True, track_time_series=True
        )
        ax = axes_s5[i]
        ax.step(range(300), modes, where='mid', color='purple', linewidth=2.0)
        ax.axvspan(100, 199, color='red', alpha=0.1, label='Follower Jammer Active')
        ax.set_ylabel("Mode Index")
        ax.set_yticks([-1, 0, 1, 2, 3])
        ax.set_ylim([-1.5, 3.5])
        ax.set_title(f"Adaptation Latency: {p_name}", fontweight='bold', fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6)
        if i == 0:
            ax.legend(loc="upper right")
            
    axes_s5[-1].set_xlabel("Hop Number")
    plt.tight_layout()
    fig_s5.savefig(os.path.join(os.path.dirname(__file__), "phase_b_policy_transition.png"), dpi=300)
    
    # 5. Occupancy Grid
    print("\n[Step 5] Generating Occupancy Grid (20 panels, 40 dist points x 2000 trials)...")
    fig_occ, axes_occ = plt.subplots(5, 4, figsize=(16, 12), sharex=True, sharey=True)
    d_occ = np.linspace(100.0, 20000.0, 40)
    
    for r, (p, p_name) in enumerate(zip(policies, policy_names)):
        for c, (s_name, alpha, j_type, tau) in enumerate(scenarios):
            ax = axes_occ[r, c]
            occupancy = {m: [] for m in [-1, 0, 1, 2, 3]}
            for d in d_occ:
                _, _, _, _, modes = simulate_policy_hop_sequence(p, d, N, theta_s, theta_j, alpha, j_type, tau, trials=2000)
                counts = {-1: 0, 0: 0, 1: 0, 2: 0, 3: 0}
                for m in modes: counts[m] += 1
                for m in [-1, 0, 1, 2, 3]: occupancy[m].append(counts[m]/2000.0)
            
            y = np.vstack([occupancy[-1], occupancy[0], occupancy[1], occupancy[2], occupancy[3]])
            ax.stackplot(d_occ/1000.0, y, labels=['Outage', 'M0', 'M1', 'M2', 'M3'], 
                         colors=['#d62728', '#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd'], alpha=0.8)
                         
            if r == 0: ax.set_title(s_name, fontweight='bold')
            if c == 0: ax.set_ylabel(p_name, fontweight='bold')
            if r == 4: ax.set_xlabel("Range [km]")
            ax.set_ylim([0, 1])
            ax.set_xlim([0, 20])
            
    handles, labels = axes_occ[0,0].get_legend_handles_labels()
    fig_occ.legend(handles, labels, loc='upper center', ncol=5, bbox_to_anchor=(0.5, 0.98))
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig_occ.savefig(os.path.join(os.path.dirname(__file__), "phase_b_policy_occupancy.png"), dpi=300)

    print("\n[SUCCESS] Sprint 4 Complete. Plots saved.")

if __name__ == "__main__":
    run_policy_comparison()
