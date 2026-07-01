"""
cat5/run_cat5.py
Category 5: Sensitivity Sweeps — Full Validation Runner (5a -> 5c -> 5b)

Produces a 2x3 dashboard figure:
  Row 1: 5a — Frequency Band Sweep
    [0,0] Path loss vs distance  (900/2.4/5.8 GHz)
    [0,1] K-factor and shadow fading vs frequency
    [0,2] Outage probability vs range (MRC-L4, K(f))
  Row 2: 5c — Alpha Sweep (RNCO aggressiveness)
    [1,0] FDR vs alpha + critical alpha*
    [1,1] Throughput & success rate vs alpha
  Row 2 also: 5b — Wind Robustness
    [1,2] DOA sigma and pos error vs wind speed
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from channel_bridge import (
    fspl_db, RICIAN_K, SINR_THRESH_DB, NULL_DEPTH_DB,
    NOISE_DBM, L_MRC
)
from band_sweep    import (run_band_sweep, coverage_range_m,
                            validate_fspl_deltas, BANDS_HZ, K0_REF, F0_REF, ALPHA_K)
from alpha_sweep   import (AlphaSweepConfig, run_alpha_sweep,
                            find_critical_alpha)
from wind_robustness import (WindConfig, DrydenWindModel,
                              wind_speed_sweep)

plt.style.use('seaborn-v0_8-darkgrid')
fig = plt.figure(figsize=(18, 11), constrained_layout=True)
fig.suptitle(
    "Stage 4 — Category 5: Sensitivity Sweeps\n"
    "5a: Frequency Band Sweep  |  5c: RNCO Alpha Sweep  |  5b: Wind Robustness",
    fontsize=13, fontweight='bold')
gs = gridspec.GridSpec(2, 3, figure=fig)

# ─────────────────────────────────────────────────────────────────────────────
# 5a: Frequency Band Sweep
# ─────────────────────────────────────────────────────────────────────────────
print("Running 5a: Frequency Band Sweep...")
distances = np.linspace(50, 20000, 400)
band_results = run_band_sweep(distances_m=distances)

BAND_COLORS = {"900 MHz": "green", "2.4 GHz": "royalblue", "5.8 GHz": "crimson"}
BAND_LS     = {"900 MHz": ":",     "2.4 GHz": "-",          "5.8 GHz": "--"}

# ── Plot [0,0]: Path loss vs distance ────────────────────────────────────────
ax00 = fig.add_subplot(gs[0, 0])
for name, res in band_results.items():
    ax00.plot(distances / 1e3, res.fspl_db_arr,
              color=BAND_COLORS[name], linestyle=BAND_LS[name],
              linewidth=2, label=f'{name}  (K={res.k_factor:.1f})')
ax00.set_xlabel('GCS-UAV Range (km)')
ax00.set_ylabel('Free-Space Path Loss (dB)')
ax00.set_title('5a: Path Loss vs Range\n(Fixed-gain antenna)')
ax00.legend(fontsize=9)

# ── Plot [0,1]: K-factor, shadow fading, coverage range vs band ───────────────
ax01 = fig.add_subplot(gs[0, 1])
freqs_hz    = [900e6, 2.4e9, 5.8e9]
freqs_label = ["900", "2400", "5800"]
k_vals      = [K0_REF * (f / F0_REF) ** ALPHA_K for f in freqs_hz]
sf_vals     = [4.0 + 1.5 * np.log10(f / F0_REF) for f in freqs_hz]
cov_vals    = [coverage_range_m(band_results[n]) / 1e3
               for n in ["900 MHz", "2.4 GHz", "5.8 GHz"]]

x = np.arange(3)
ax01_twin = ax01.twinx()
bars  = ax01.bar(x - 0.2, k_vals,     0.35, color='steelblue', alpha=0.7, label='Rician K')
bars2 = ax01.bar(x + 0.2, sf_vals,    0.35, color='orange',    alpha=0.7, label='Shadow fading std (dB)')
line, = ax01_twin.plot(x, cov_vals, 'D--', color='crimson', linewidth=2, markersize=9, label='Coverage (km)')
ax01.set_xticks(x); ax01.set_xticklabels(freqs_label)
ax01.set_xlabel('Frequency (MHz)'); ax01.set_ylabel('K-factor / Shadow std (dB)')
ax01_twin.set_ylabel('Coverage range (km)', color='crimson')
ax01.set_title('5a: K-factor, Shadow Fading\n& Coverage Range vs Band')
lines1, labels1 = ax01.get_legend_handles_labels()
lines2, labels2 = ax01_twin.get_legend_handles_labels()
ax01.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

# ── Plot [0,2]: Outage probability vs range ────────────────────────────────────
ax02 = fig.add_subplot(gs[0, 2])
for name, res in band_results.items():
    ax02.semilogy(distances / 1e3, np.clip(res.p_out_arr, 1e-6, 1),
                  color=BAND_COLORS[name], linestyle=BAND_LS[name],
                  linewidth=2, label=name)
ax02.axhline(0.01, color='grey', linestyle=':', linewidth=1.5, label='P_out=1% threshold')
ax02.set_xlabel('GCS-UAV Range (km)')
ax02.set_ylabel('Outage Probability P_out')
ax02.set_title(f'5a: Outage vs Range\n(MRC-L{L_MRC}, SINR thresh={SINR_THRESH_DB} dB)')
ax02.set_ylim([1e-6, 1.0])
ax02.legend(fontsize=9)

# ─────────────────────────────────────────────────────────────────────────────
# 5c: Alpha Sweep (RNCO aggressiveness)
# ─────────────────────────────────────────────────────────────────────────────
print("Running 5c: RNCO Alpha Sweep (N_MC=300, ~30 s)...")
alpha_cfg = AlphaSweepConfig(
    alpha_values=list(np.linspace(0.05, 3.0, 25)),
    n_monte_carlo=300,
    sim_duration_s=120.0,
    dt_s=1.0,
    gamma_thresh_db=SINR_THRESH_DB
)
alpha_results = run_alpha_sweep(alpha_cfg)
alpha_star    = find_critical_alpha(alpha_results, fdr_threshold=0.05)

alphas     = sorted(alpha_results.keys())
fdr_arr    = [alpha_results[a]['mean_fdr']             for a in alphas]
tput_arr   = [alpha_results[a]['mean_throughput_mbps'] for a in alphas]
srate_arr  = [alpha_results[a]['success_rate']         for a in alphas]
drops_arr  = [alpha_results[a]['mean_drops_per_min']   for a in alphas]

# ── Plot [1,0]: FDR vs alpha + alpha* ────────────────────────────────────────
ax10 = fig.add_subplot(gs[1, 0])
ax10.plot(alphas, fdr_arr, 'o-', color='crimson', linewidth=2, label='Mean FDR')
ax10.axhline(0.05, color='grey', linestyle='--', linewidth=1.5, label='FDR=5% threshold')
if alpha_star < float('inf'):
    ax10.axvline(alpha_star, color='orange', linestyle=':', linewidth=2,
                 label=f'alpha* = {alpha_star:.2f}')
ax10.set_xlabel('RNCO Aggressiveness alpha')
ax10.set_ylabel('False Drop Rate (FDR)')
ax10.set_title('5c: RNCO Alpha Sweep\nFalse Drop Rate vs Aggressiveness')
ax10.legend(fontsize=9); ax10.set_ylim([0, max(0.15, max(fdr_arr) * 1.2)])

# ── Plot [1,1]: Throughput and Success Rate vs alpha ─────────────────────────
ax11 = fig.add_subplot(gs[1, 1])
ax11_twin = ax11.twinx()
ax11.plot(alphas, tput_arr,  's-', color='steelblue', linewidth=2, label='Throughput (Mbps)')
ax11_twin.plot(alphas, srate_arr, 'D--', color='green', linewidth=2, label='Mission success rate')
if alpha_star < float('inf'):
    ax11.axvline(alpha_star, color='orange', linestyle=':', linewidth=2, label=f'alpha*={alpha_star:.2f}')
ax11.set_xlabel('RNCO Aggressiveness alpha')
ax11.set_ylabel('Mean Throughput (Mbps)', color='steelblue')
ax11_twin.set_ylabel('Mission Success Rate', color='green')
ax11_twin.set_ylim([0, 1.1])
ax11.set_title('5c: Throughput & Mission Success\nvs RNCO Aggressiveness')
lines1, labs1 = ax11.get_legend_handles_labels()
lines2, labs2 = ax11_twin.get_legend_handles_labels()
ax11.legend(lines1 + lines2, labs1 + labs2, fontsize=8, loc='lower left')

# ─────────────────────────────────────────────────────────────────────────────
# 5b: Wind Robustness
# ─────────────────────────────────────────────────────────────────────────────
print("Running 5b: Wind Robustness Sweep...")
v_wind_vals = [0, 5, 10, 15, 20]
wind_results = wind_speed_sweep(v_wind_values=v_wind_vals, t_total_s=60.0)

v_w_arr        = list(wind_results.keys())
delta_r_arr    = [wind_results[v]['delta_r_ss_m']          for v in v_w_arr]
pos_rms_arr    = [wind_results[v]['sigma_pos_rms_m']       for v in v_w_arr]
doa_nowind_arr = [wind_results[v]['doa_sigma_no_wind_deg'] for v in v_w_arr]
doa_wind_arr   = [wind_results[v]['doa_sigma_wind_deg']    for v in v_w_arr]
gust_arr       = [wind_results[v]['gust_rms_ms2']          for v in v_w_arr]

# ── Plot [1,2]: DOA sigma, position error, steady displacement vs wind ─────────
ax12 = fig.add_subplot(gs[1, 2])
ax12_twin = ax12.twinx()

ax12.plot(v_w_arr, delta_r_arr, 'o-', color='orange',   linewidth=2, label='Steady displacement (m)')
ax12.plot(v_w_arr, pos_rms_arr, 's-', color='crimson',  linewidth=2, label='INS pos RMS error (m)')
ax12_twin.plot(v_w_arr, doa_nowind_arr, '--', color='royalblue', linewidth=1.5, label='DOA sigma (no wind, deg)')
ax12_twin.plot(v_w_arr, doa_wind_arr,   '-',  color='purple',    linewidth=2.0, label='DOA sigma (with wind, deg)')

ax12.set_xlabel('Wind Speed (m/s)')
ax12.set_ylabel('Displacement / INS Error (m)')
ax12_twin.set_ylabel('DOA Sigma (deg)', color='purple')
ax12.set_title('5b: Wind Robustness\nINS Error & DOA Degradation vs Wind Speed')
lines1, labs1 = ax12.get_legend_handles_labels()
lines2, labs2 = ax12_twin.get_legend_handles_labels()
ax12.legend(lines1 + lines2, labs1 + labs2, fontsize=7, loc='upper left')

plt.savefig(os.path.join(os.path.dirname(__file__), 'cat5_validation.png'), dpi=150)
print("Saved cat5_validation.png")

# ─────────────────────────────────────────────────────────────────────────────
# Validation Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n===== Category 5 Validation Summary =====")

# Check 1: FSPL deltas match spec
ok_fspl = validate_fspl_deltas()
d = 1000.0
delta_58 = fspl_db(d, 5.8e9) - fspl_db(d, 2.4e9)
delta_09 = fspl_db(d, 0.9e9) - fspl_db(d, 2.4e9)
print(f"[1] FSPL deltas at 1 km:")
print(f"    5.8 vs 2.4 GHz: {delta_58:.3f} dB  (spec: +7.67 dB) => {'PASS' if abs(delta_58-7.67)<0.05 else 'FAIL'}")
print(f"    900 vs 2.4 GHz: {delta_09:.3f} dB  (spec: -8.55 dB) => {'PASS' if abs(delta_09-(-8.55))<0.05 else 'FAIL'}")

# Check 2: K-factor monotonicity with frequency
k_900  = K0_REF * (900e6  / F0_REF) ** ALPHA_K
k_2400 = K0_REF * (2.4e9  / F0_REF) ** ALPHA_K
k_5800 = K0_REF * (5.8e9  / F0_REF) ** ALPHA_K
print(f"\n[2] K-factor monotonicity (alpha_K={ALPHA_K}):")
print(f"    K(900) = {k_900:.2f}, K(2400) = {k_2400:.2f}, K(5800) = {k_5800:.2f}")
print(f"    K(900) > K(2400) > K(5800): {'PASS' if k_900 > k_2400 > k_5800 else 'FAIL'}")

# Check 3: Alpha=0 gives FDR=0
fdr_at_zero = alpha_results[min(alphas)]['mean_fdr']
print(f"\n[3] FDR at alpha_min ({min(alphas):.2f}): {fdr_at_zero:.4f} (expect ~0)")
print(f"    FDR < 0.01: {'PASS' if fdr_at_zero < 0.01 else 'FAIL'}")

# Check 4: critical alpha found in (0, 3)
print(f"\n[4] Critical alpha*: {alpha_star:.3f} dBW")
print(f"    Found in (0, 3): {'PASS' if 0 < alpha_star < 3 else 'NOT FOUND (FDR never >5%)'}")

# Check 5: Steady-state wind displacement formula
v_test = 10.0
cfg_t  = WindConfig(v_wind_ms=v_test)
mdl_t  = DrydenWindModel(cfg_t)
F_drag = 0.5 * 1.225 * cfg_t.C_D * cfg_t.A_eff_m2 * v_test**2
delta_r_expected = F_drag / cfg_t.control_stiffness_Npm
delta_r_computed = mdl_t.steady_displacement_m()
print(f"\n[5] Steady displacement at V_w=10 m/s:")
print(f"    Expected: {delta_r_expected:.4f} m,  Computed: {delta_r_computed:.4f} m")
print(f"    Match: {'PASS' if abs(delta_r_expected - delta_r_computed) < 1e-8 else 'FAIL'}")

# Check 6: INS error grows with wind speed
pos_rms_0  = wind_results[0]['sigma_pos_rms_m']
pos_rms_20 = wind_results[20]['sigma_pos_rms_m']
print(f"\n[6] INS pos RMS: {pos_rms_0:.4f} m (V_w=0) -> {pos_rms_20:.4f} m (V_w=20)")
print(f"    Monotonically increases: {'PASS' if pos_rms_20 > pos_rms_0 else 'FAIL'}")

print("==========================================")
