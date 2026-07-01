"""Stage 2 Task B13: Validation Audit of Sprint 4 Conclusions.

This script performs a robustness and sensitivity audit on the control policies
to determine if the findings (Variance Trap, Follower Ambush) are fundamental
or hyperparameter-dependent artifacts.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(__file__))
import phase_b_beamforming as pbb
import task_b_jamming_threats as jts
import task_b7_amc as b7

def simulate_policy_hop_sequence(policy, d_m, N, theta_s, theta_j, alpha, jammer_type='pbj', tau_delay=0.0, trials=10000, seed=2026, track_time_series=False, W=30, risk_budget=0.10, lag=1, log_internal=False):
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
                
    buffer_gamma_obs_lin = [sinr_unjammed_nom] * W
    lag_buffer_obs_lin = [sinr_unjammed_nom] * max(1, lag + 1)
    lag_buffer_Ik = [0] * max(1, lag + 1)
    
    rng_fad = np.random.default_rng(seed)
    los_s = np.sqrt(K_s / (K_s + 1.0))
    diff_s = np.sqrt(1.0 / (K_s + 1.0))
    
    success_ber_count = 0
    total_throughput = 0.0
    
    modes_selected = []
    
    # Internal logging for Policy C
    log_p_hat = []
    log_I_k = []
    log_gamma_adapt = []
    
    for k in range(trials):
        cur_j_type = jammer_type
        cur_tau = tau_delay
        cur_alpha = alpha
                
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
            gamma_lagged = lag_buffer_obs_lin[-(lag+1)] if lag > 0 else gamma_obs
            g_db = 10.0 * np.log10(gamma_lagged)
            mode = b7.select_amc_mode_hysteresis(g_db, prev_mode)
        elif policy == 'C':
            gamma_lagged = lag_buffer_obs_lin[-2] # Always 1-hop lag for Policy C's observation
            g_prev_db = 10.0 * np.log10(gamma_lagged)
            p_hat = 1.0 if g_prev_db < 5.64 else 0.0 # BPSK detection threshold
            g_lin = p_hat * gamma_jammed + (1.0 - p_hat) * gamma_lagged
            g_db = 10.0 * np.log10(g_lin)
            mode = b7.select_amc_mode_hysteresis(g_db, prev_mode)
            
            if log_internal:
                log_p_hat.append(p_hat)
                log_I_k.append(lag_buffer_Ik[-2]) # True state of the PREVIOUS hop (which p_hat is trying to guess)
                log_gamma_adapt.append(g_db)
                
        elif policy == 'D':
            if k < W:
                g_db = 10.0 * np.log10(gamma_jammed)
                mode = b7.select_amc_mode_hysteresis(g_db, prev_mode)
            else:
                best_mode = -1
                for m in [3, 2, 1, 0]:
                    outages = sum(1 for g in buffer_gamma_obs_lin if b7.get_ber_for_mode(m, g) > b7.target_ber)
                    if outages / W < risk_budget:
                        best_mode = m
                        break
                mode = best_mode
                
        if mode == -1:
            modes_selected.append(-1)
            buffer_gamma_obs_lin.pop(0)
            buffer_gamma_obs_lin.append(gamma_obs)
            lag_buffer_obs_lin.pop(0)
            lag_buffer_obs_lin.append(gamma_obs)
            lag_buffer_Ik.pop(0)
            lag_buffer_Ik.append(I_k)
            if policy != 'D': prev_mode = -1
            continue
            
        ber_hop = rho * b7.get_ber_for_mode(mode, gamma_jammed) + (1.0 - rho) * b7.get_ber_for_mode(mode, gamma_unjammed)
        ok_ber = (ber_hop <= b7.target_ber)
        
        if ok_ber: success_ber_count += 1
        
        modes_selected.append(mode)
        
        eff_rate = b7.amc_spectral_efficiencies[mode] if policy != 'fixed' else 0.5
        hop_thr = eff_rate * (1.0 - ber_hop) if ok_ber else 0.0
        
        total_throughput += hop_thr
        
        prev_mode = mode
        buffer_gamma_obs_lin.pop(0)
        buffer_gamma_obs_lin.append(gamma_obs)
        lag_buffer_obs_lin.pop(0)
        lag_buffer_obs_lin.append(gamma_obs)
        lag_buffer_Ik.pop(0)
        lag_buffer_Ik.append(I_k)
        
    if log_internal:
        return success_ber_count / trials, total_throughput / trials, log_p_hat, log_I_k, log_gamma_adapt, modes_selected
        
    p_link_ber = success_ber_count / trials
    avg_throughput = total_throughput / trials
    
    return p_link_ber, avg_throughput

def solve_defeat_range(policy, N, theta_s, theta_j, alpha, jammer_type, tau_delay, trials=2000, target_outage=0.10, W=30, risk_budget=0.10, lag=1):
    d_min = 100.0
    d_max = 25000.0
    d_opt = d_max
    
    p_link_max, _ = simulate_policy_hop_sequence(policy, d_max, N, theta_s, theta_j, alpha, jammer_type, tau_delay, trials=trials, W=W, risk_budget=risk_budget, lag=lag)
    if 1.0 - p_link_max < target_outage:
        return d_max
        
    for _ in range(12):
        d_mid = (d_min + d_max) / 2.0
        p_link, _ = simulate_policy_hop_sequence(policy, d_mid, N, theta_s, theta_j, alpha, jammer_type, tau_delay, trials=trials, W=W, risk_budget=risk_budget, lag=lag)
        if 1.0 - p_link >= target_outage:
            d_max = d_mid
        else:
            d_min = d_mid
            d_opt = d_mid
    return d_opt

def audit_policy_D():
    print("======================================================================")
    print(" STUDY 1: POLICY D ROBUSTNESS (THE VARIANCE TRAP)")
    print("======================================================================")
    N = 8
    theta_s = 0.0
    theta_j = np.radians(30.0)
    jammer_type = 'pbj'
    tau_delay = 0.0
    
    W_vals = [10, 30, 50, 100, 200]
    eps_vals = [0.05, 0.10, 0.15, 0.20]
    
    # 1. 2D Sweep of W and eps at fixed alpha=0.11
    print("\\n[Step 1A] W vs Risk Budget (eps) Defeat Range Sweep (alpha = 0.11)")
    alpha = 0.11
    print(f"{'W':<5} | " + " | ".join([f"e={e:<4.2f}" for e in eps_vals]))
    print("-" * 45)
    for w in W_vals:
        row_str = f"{w:<5} | "
        for e in eps_vals:
            dr = solve_defeat_range('D', N, theta_s, theta_j, alpha, jammer_type, tau_delay, W=w, risk_budget=e, trials=2000)
            row_str += f"{dr/1000.0:6.2f} | "
        print(row_str)
        
    # 2. Extended alpha sweep mapping alpha/eps
    print("\\n[Step 1B] PBJ Fraction (alpha) vs Risk Budget (eps) mapping (W=30 and W=100)")
    alpha_vals = [0.05, 0.10, 0.11, 0.15, 0.20, 0.30, 0.50]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for i, w in enumerate([30, 100]):
        ax = axes[i]
        for e in eps_vals:
            ranges = []
            ratios = []
            for a in alpha_vals:
                dr = solve_defeat_range('D', N, theta_s, theta_j, a, jammer_type, tau_delay, W=w, risk_budget=e, trials=2000)
                ranges.append(dr / 1000.0)
                ratios.append(a / e)
            
            # Sort by ratio for plotting
            ratios_sorted, ranges_sorted = zip(*sorted(zip(ratios, ranges)))
            ax.plot(ratios_sorted, ranges_sorted, marker='o', linewidth=2.0, label=f"eps={e:.2f}")
            
        ax.axvline(1.0, color='red', linestyle='--', label='alpha = eps')
        ax.set_title(f"Policy D Defeat Range vs alpha/eps (W={w})", fontweight='bold')
        ax.set_xlabel("Ratio: alpha / epsilon")
        ax.set_ylabel("Defeat Range [km]")
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend()
        
    plt.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(__file__), "audit_policy_D_alpha_eps.png"), dpi=300)
    print("Saved audit_policy_D_alpha_eps.png")

def audit_policy_B():
    print("\\n======================================================================")
    print(" STUDY 2: POLICY B SENSITIVITY (THE FOLLOWER AMBUSH)")
    print("======================================================================")
    N = 8
    theta_s = 0.0
    theta_j = np.radians(30.0)
    jammer_type = 'follower'
    tau_delay = 5e-3 # 5ms
    alpha = 1.0
    
    lag_vals = [0, 1, 2, 3, 5]
    print(f"{'Lag [hops]':<12} | {'Defeat Range [km]':<18} | {'Throughput (d=2.49km) [bps/Hz]':<30}")
    print("-" * 65)
    
    ranges = []
    thrs = []
    for lag in lag_vals:
        dr = solve_defeat_range('B', N, theta_s, theta_j, alpha, jammer_type, tau_delay, lag=lag, trials=2000)
        _, thr = simulate_policy_hop_sequence('B', 2490.0, N, theta_s, theta_j, alpha, jammer_type, tau_delay, lag=lag, trials=2000)
        print(f"{lag:<12} | {dr/1000.0:<18.2f} | {thr:<30.4f}")
        ranges.append(dr / 1000.0)
        thrs.append(thr)
        
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(lag_vals, ranges, 'bo-', linewidth=2.0, label='Defeat Range [km]')
    ax1.set_xlabel("Observation Lag [hops]")
    ax1.set_ylabel("Defeat Range [km]", color='b')
    ax1.tick_params('y', colors='b')
    ax1.set_title("Policy B Sensitivity to Lag (Follower Jamming)", fontweight='bold')
    
    ax2 = ax1.twinx()
    ax2.plot(lag_vals, thrs, 'rx--', linewidth=2.0, label='Throughput [bps/Hz]')
    ax2.set_ylabel("Throughput at 2.49km [bps/Hz]", color='r')
    ax2.tick_params('y', colors='r')
    
    ax1.grid(True, linestyle=':', alpha=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(__file__), "audit_policy_B_lag.png"), dpi=300)
    print("Saved audit_policy_B_lag.png")

def audit_policy_C():
    print("\\n======================================================================")
    print(" STUDY 3: POLICY C ABLATION (THE HYBRID FAILURE)")
    print("======================================================================")
    N = 8
    theta_s = 0.0
    theta_j = np.radians(30.0)
    
    # Run under PBJ
    print("Evaluating under PBJ (alpha=0.11) at d=10.0km...")
    _, _, log_p_hat, log_I_k, _, _ = simulate_policy_hop_sequence('C', 10000.0, N, theta_s, theta_j, 0.11, 'pbj', 0.0, trials=5000, log_internal=True)
    
    p_hat = np.array(log_p_hat[1:]) # Drop first hop to align lag
    I_k = np.array(log_I_k[1:])
    
    TP = np.sum((p_hat == 1.0) & (I_k == 1))
    FP = np.sum((p_hat == 1.0) & (I_k == 0))
    TN = np.sum((p_hat == 0.0) & (I_k == 0))
    FN = np.sum((p_hat == 0.0) & (I_k == 1))
    
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
    
    print("\\nConfusion Matrix (PBJ, d=10km):")
    print(f"True Positives:  {TP:<5} | False Positives: {FP:<5}")
    print(f"False Negatives: {FN:<5} | True Negatives:  {TN:<5}")
    print(f"TPR (Sensitivity): {TPR:.2%} | FPR: {FPR:.2%}")
    print(f"Total Jams: {TP+FN} | Total Clean: {FP+TN}")
    
    print("\\nWhy does Policy C degenerate into Policy B?")
    # Check if p_hat is exactly boolean 1 or 0
    unique_p_hat = np.unique(p_hat)
    print(f"Unique values of p_hat generated by estimator: {unique_p_hat}")

if __name__ == "__main__":
    audit_policy_C()
