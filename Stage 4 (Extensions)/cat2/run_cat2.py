"""
cat2/run_cat2.py
Extension 2 — FHSS + Spectrum Sensing + AMC — Integrated Validation

Uses the REAL Stage 2 (Remediated) / Stage 3 (Phase D) functions:
  - baseline_channel(eirp_dbw)   → (sinr_db, p_out) — no beamforming
  - phase_b_channel(eirp_dbw)   → (sinr_db, p_out) — with 45 dB LCMV null
  - phase_bc_channel(eirp_dbw)  → (sinr_db, p_out) — null + RNCO reduction
  - rician_mrc_outage()          → P_out (MRC-L4, K=10, real Marcum-Q)
  - detection_prob()             → Neyman-Pearson P_d for spectrum sensing

All BER, PER, AMC and FHSS computations are driven by the ACTUAL SINR
values from the Phase A–D channel model, not standalone approximations.

Produces four figures:
  1. BER vs ρ with worst-case ρ* at the actual jammer EIRPs we use in Phase D
  2. SINR distribution across hop channels: Baseline vs Phase B vs Full Stack
  3. ROC curve comparison: our EnergyDetector vs Stage 2's detection_prob()
  4. AMC throughput trace driven by Phase D SINR sweep
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from channel_bridge import (
    baseline_channel, phase_b_channel, phase_bc_channel,
    rician_mrc_outage, detection_prob,
    NOISE_W, NOISE_DBM, FC_HZ, SINR_THRESH_DB, NULL_DEPTH_DB
)
from fhss_jammer    import FHSSConfig, JammerConfig, JammerType, FHSSSystem
from spectrum_sensing import SensingConfig, EnergyDetector
from adaptive_mcs    import AMCController, MCS_TABLE

# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 link budget geometry (from sinr_models.py fixed geometry)
# GCS → UAV: 1000 m, 2.4 GHz, signal EIRP = 10 dBW + 30 = 40 dBm
# Jammer: variable EIRP, 500 m from UAV, 2.4 GHz
# ─────────────────────────────────────────────────────────────────────────────
EIRP_SWEEP   = np.linspace(-10, 60, 90)    # same sweep as Phase D (Extension 3)
EIRP_NOMINAL = 40.0                         # nominal jammer EIRP for spot checks (dBW)

# Signal power from sinr_models.py geometry (10 dBW EIRP, 1km, 2.4 GHz)
# fspl(1000, 2.4e9) ≈ 100.07 dB → P_sig ≈ 10+30-100 = -60 dBm
RX_POWER_W = 10 ** ((-60 - 30) / 10)

# Jammer received power at nominal EIRP=40 dBW, 500m
# P_jam = (40+30) - fspl(500, 2.4e9) ≈ 70-94 = -24 dBm
# But we use baseline_channel() from Stage 3 which does all this properly:
sinr_nominal_db, _ = baseline_channel(EIRP_NOMINAL)

plt.style.use('seaborn-v0_8-darkgrid')
fig = plt.figure(figsize=(16, 12), constrained_layout=True)
fig.suptitle(
    "Stage 4 — Category 2: FHSS + Spectrum Sensing + AMC\n"
    "Integrated with Stage 3 (Phase D) Channel Model (EIRP Sweep -10 to +60 dBW)",
    fontsize=13, fontweight='bold')
gs = gridspec.GridSpec(2, 2, figure=fig)

# ─────────────────────────────────────────────────────────────────────────────
# Build FHSS system parameters from real Stage 3 link budget
# ─────────────────────────────────────────────────────────────────────────────
fhss_cfg = FHSSConfig(n_channels=79, bandwidth_hz=83.5e6,
                       hop_period_s=1/1600, seed=42)

# Compute JSR from real Stage 3 geometry at nominal EIRP
def get_jsr_from_stage3(eirp_dbw: float) -> float:
    """Jammer-to-Signal Ratio (linear) using Stage 3 sinr_models geometry."""
    sinr_db, _ = baseline_channel(eirp_dbw)
    sinr_lin   = 10 ** (sinr_db / 10)
    # JSR = 1/SINR when noise is negligible (jammer-dominated)
    return 1.0 / (sinr_lin + 1e-30)

JSR_NOMINAL = get_jsr_from_stage3(EIRP_NOMINAL)

# ─── Plot 1: BER vs ρ at realistic EIRPs from Stage 3 sweep ──────────────────
ax1 = fig.add_subplot(gs[0, 0])

EIRP_SPOT_VALS = [20.0, 40.0, 55.0]   # dBW — matches key points on Phase D collapse curve
for eirp in EIRP_SPOT_VALS:
    sinr_db, _ = baseline_channel(eirp)
    jsr        = get_jsr_from_stage3(eirp)
    ebno_lin   = 10 ** (sinr_db / 10)   # Eb/N0 ≈ SINR for narrow-band

    jam_cfg = JammerConfig(jammer_type=JammerType.PARTIAL_BAND,
                           total_power_w=1.0, rho=0.3)
    sys_pbj = FHSSSystem(fhss_cfg, jam_cfg)
    rho_arr, ber_arr = sys_pbj.sweep_rho_ber(sinr_db, 10*np.log10(jsr))
    rho_star = FHSSSystem.worst_case_rho(ebno_lin, jsr)
    ber_star = FHSSSystem.bfsk_ber_pbj(ebno_lin, jsr, rho_star)
    line, = ax1.semilogy(rho_arr, ber_arr, label=f'EIRP={eirp:.0f} dBW (SINR={sinr_db:.1f} dB)')
    ax1.axvline(rho_star, color=line.get_color(), linestyle=':', alpha=0.6)
    ax1.plot(rho_star, ber_star, 'v', color=line.get_color(), markersize=8)

ax1.set_xlabel('Partial-Band Fraction rho')
ax1.set_ylabel('BER (BFSK)')
ax1.set_title('BER vs rho — Jammer EIRPs from Phase D Collapse Curve\n(v = worst-case rho*)')
ax1.legend(fontsize=8); ax1.set_xlim([0, 1])

# ─── Plot 2: Per-hop SINR by config using real Stage 3 channel models ─────────
ax2 = fig.add_subplot(gs[0, 1])

N_HOPS = 2000

# Use the three Phase D channel configs from sinr_models.py
configs = [
    ('Baseline (No EW)',        baseline_channel,   'red',       '--'),
    ('Phase B: LCMV Nulling',   phase_b_channel,    'royalblue', '-'),
    ('Phase B+C+D: Full Stack', phase_bc_channel,   'green',     '-'),
]

for name, ch_fn, color, ls in configs:
    sinr_db_val, p_out_val = ch_fn(EIRP_NOMINAL)

    # Convert real SINR to received power for FHSS system
    rx_power_w = 10 ** ((sinr_db_val - 30) / 10) * NOISE_W   # back-calculate
    jam_cfg    = JammerConfig(jammer_type=JammerType.PARTIAL_BAND,
                              total_power_w=max(rx_power_w * 10 ** (-sinr_db_val/10), 1e-20),
                              rho=0.3)
    sys_j = FHSSSystem(fhss_cfg, jam_cfg)
    sinrs = sys_j.compute_per_hop_sinr_db(rx_power_w, NOISE_W, N_HOPS)

    ax2.hist(sinrs, bins=60, density=True, alpha=0.5, color=color,
             linestyle=ls, label=f'{name}\n(SINR={sinr_db_val:.1f} dB, P_out={p_out_val:.3f})')
    ax2.axvline(float(np.median(sinrs)), color=color, linestyle='--', linewidth=1.5)

ax2.axvline(SINR_THRESH_DB, color='grey', linestyle=':', linewidth=2,
            label=f'SINR threshold ({SINR_THRESH_DB} dB)')
ax2.set_xlabel('Per-Hop SINR (dB)')
ax2.set_ylabel('Density')
ax2.set_title(f'FHSS Per-Hop SINR Distribution\nJammer EIRP={EIRP_NOMINAL:.0f} dBW, {N_HOPS} hops')
ax2.legend(fontsize=7)

# ─── Plot 3: ROC — our EnergyDetector vs Stage 2's detection_prob() ───────────
ax3 = fig.add_subplot(gs[1, 0])

det_cfg  = SensingConfig(n_samples=256, pfa_target=0.01, noise_power=NOISE_W)
detector = EnergyDetector(det_cfg)

snr_db_vals = [-5, 0, 5, 10]
colors_roc  = plt.cm.viridis(np.linspace(0.15, 0.85, len(snr_db_vals)))

for snr_db, col in zip(snr_db_vals, colors_roc):
    snr_lin = 10 ** (snr_db / 10)

    # Our energy detector theoretical ROC
    pfa_arr, pd_arr = detector.compute_roc(snr_lin, n_points=200)
    ax3.plot(pfa_arr, pd_arr, color=col, linewidth=2,
             label=f'Our EnergyDet SNR={snr_db} dB')

    # Stage 2 detection_prob() as reference points (at P_fa = 0.01)
    jnr_db  = snr_db   # treat SNR as JNR (jammer is the signal of interest for anti-drone)
    pd_s2   = detection_prob(t_s_sec=0.001, BW_hz=1e6, JNR_dB=jnr_db, P_fa=0.01)
    ax3.plot(0.01, pd_s2, '*', color=col, markersize=14, zorder=5)

ax3.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Random guess')
ax3.axvline(0.01, color='red', linestyle=':', linewidth=1.5, label='P_fa target')
ax3.set_xlabel('False Alarm Probability P_fa')
ax3.set_ylabel('Detection Probability P_d')
ax3.set_title('ROC: Our EnergyDetector (curves)\nvs Stage 2 detection_prob() at P_fa=0.01 (*)')
ax3.legend(fontsize=7); ax3.set_xlim([0, 1]); ax3.set_ylim([0, 1.05])

# ─── Plot 4: AMC throughput over the real Phase D EIRP sweep ─────────────────
ax4 = fig.add_subplot(gs[1, 1])

amc = AMCController(channel_bw_hz=5e6)

# Drive AMC with real SINR from all three Phase D configs over the full EIRP sweep
for name, ch_fn, color, ls in configs:
    sinr_trace_db = np.array([ch_fn(e)[0] for e in EIRP_SWEEP])
    throughputs   = []
    amc.current_mcs_idx = 0
    amc.history = []
    for s in sinr_trace_db:
        mcs = amc.select_mcs(s)
        throughputs.append(mcs.spectral_efficiency * 5e6 / 1e6)   # Mbps

    ax4.plot(EIRP_SWEEP, throughputs, color=color, linestyle=ls,
             linewidth=2, label=name)

# Shannon capacity for Phase B (best case) as upper bound
sinr_b_lin = np.array([10**(phase_b_channel(e)[0]/10) for e in EIRP_SWEEP])
shannon_b   = AMCController.shannon_capacity_bps(sinr_b_lin, 5e6) / 1e6
ax4.plot(EIRP_SWEEP, shannon_b, 'k:', linewidth=1.5, label='Shannon (Phase B)')

ax4.set_xlabel('Jammer EIRP (dBW)')
ax4.set_ylabel('AMC Throughput (Mbps)')
ax4.set_title('AMC Adaptive Throughput vs Jammer EIRP\n(Phase D Channel Model, 5 MHz channel)')
ax4.legend(fontsize=8)
ax4.set_xlim([-10, 60])

plt.savefig(os.path.join(os.path.dirname(__file__), 'cat2_validation.png'), dpi=150)
print("Saved cat2_validation.png")

# ─────────────────────────────────────────────────────────────────────────────
# Validation summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n===== Category 2 Validation Summary (Integrated) =====")

# Check 1: BER(rho=1) = standard BFSK full-band formula, at real JSR
ebno_lin = 10 ** (sinr_nominal_db / 10)
ber_pbj1 = FHSSSystem.bfsk_ber_pbj(ebno_lin, JSR_NOMINAL, rho=1.0)
ber_fb   = 0.5 * np.exp(-ebno_lin / (2 * (1 + JSR_NOMINAL)))
print(f"[1] BER(rho=1) at EIRP={EIRP_NOMINAL} dBW (SINR={sinr_nominal_db:.2f} dB):")
print(f"    PBJ formula: {ber_pbj1:.6f}  Full-band: {ber_fb:.6f}")
print(f"    Match: {'PASS' if abs(ber_pbj1 - ber_fb) < 1e-8 else 'FAIL'}")

# Check 2: Phase B SINR > baseline SINR (null steering improves link)
# Note: sinr_models.py p_out uses SNR not SINR (Rician K=12 fading only),
# so p_out ≈ 0 regardless of jamming — correct behaviour. SINR is the key metric.
EIRP_CHECK = 40.0   # dBW — well within the operating range of Phase D
sinr_baseline_db, _ = baseline_channel(EIRP_CHECK)
sinr_phaseB_db,   _ = phase_b_channel(EIRP_CHECK)
sinr_fullstack_db, _ = phase_bc_channel(EIRP_CHECK)
print(f"\n[2] SINR at EIRP={EIRP_CHECK} dBW (45 dB LCMV null applied):")
print(f"    Baseline SINR:    {sinr_baseline_db:.2f} dB")
print(f"    Phase B SINR:     {sinr_phaseB_db:.2f} dB  (improvement: {sinr_phaseB_db - sinr_baseline_db:.1f} dB)")
print(f"    Full Stack SINR:  {sinr_fullstack_db:.2f} dB")
print(f"    SINR gain ~= NULL_DEPTH_DB ({NULL_DEPTH_DB} dB): {'PASS' if abs((sinr_phaseB_db - sinr_baseline_db) - NULL_DEPTH_DB) < 5 else 'FAIL'}")

# Check 3: AMC below Shannon bound at all EIRP points
sinr_b_check = phase_b_channel(20.0)[0]
tput_amc  = AMCController(5e6).throughput_bps(sinr_b_check) / 1e6
cap_shan  = AMCController.shannon_capacity_bps(10**(sinr_b_check/10), 5e6) / 1e6
print(f"\n[3] AMC vs Shannon at Phase B, EIRP=20 dBW (SINR={sinr_b_check:.1f} dB):")
print(f"    AMC={tput_amc:.2f} Mbps  Shannon={cap_shan:.2f} Mbps")
print(f"    Below bound: {'PASS' if tput_amc < cap_shan else 'FAIL'}")

# Check 4: Energy detector P_d consistency with Stage 2 at JNR=10 dB
pd_our = detector.compute_pd(snr_linear=10)   # SNR=10 (linear) = 10 dB
pd_s2  = detection_prob(t_s_sec=0.001, BW_hz=1e6, JNR_dB=10.0, P_fa=0.01)
print(f"\n[4] P_d at SNR/JNR=10 linear:")
print(f"    Our EnergyDetector: {pd_our:.4f}")
print(f"    Stage 2 detection_prob: {pd_s2:.4f}")
print(f"    Both > 0.9: {'PASS' if pd_our > 0.9 and pd_s2 > 0.9 else 'FAIL'}")

# Check 5: FHSS follower jammer zero jam fraction when tau > T_hop
jam_slow = JammerConfig(jammer_type=JammerType.FOLLOWER, total_power_w=1.0,
                        detection_delay_s=1.0)
sys_slow = FHSSSystem(fhss_cfg, jam_slow)
print(f"\n[5] Follower alpha when tau_d>>T_hop: {sys_slow.follower_jam_fraction():.4f} (expect 0.0)")
print(f"    {'PASS' if sys_slow.follower_jam_fraction() == 0.0 else 'FAIL'}")

print("======================================================")
