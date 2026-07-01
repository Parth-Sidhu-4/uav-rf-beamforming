"""
cat1/run_cat1.py
Extension 1a: Jammer Localization + Anti-Drone Tracker — Integrated Validation

Uses the REAL Stage 2 (Remediated) functions:
  - generate_received_signal_rician()  → real signal snapshots with Rician fading
  - music_doa() + find_music_peaks()   → real MUSIC DOA estimates with physical noise
  - compute_crlb_doa_snr_aware()       → real MUSIC CRLB (RSS of statistical + calibration)
  - rician_mrc_outage()                → real outage probability (MRC-L4, K=10)
  - detection_prob()                   → Neyman-Pearson P_d for anti-drone sensing

Produces four figures:
  1. Triangulation geometry with real MUSIC DOA estimates and CRLB ellipse
  2. MC RMSE vs N_uavs, compared against real MUSIC CRLB bounds
  3. Anti-Drone Kalman tracker fed by real MUSIC measurements
  4. Directed beam gain vs real pointing error distribution
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse

from channel_bridge import (
    music_doa, find_music_peaks,
    generate_received_signal_rician,
    compute_crlb_doa_snr_aware,
    rician_mrc_outage, detection_prob,
    N_ARRAY, L_SNAPSHOTS, RICIAN_K,
    SINR_THRESH_DB, FC_HZ, NULL_DEPTH_DB
)
from jammer_localization import (
    UAVObservation, triangulate_jammer_2uav,
    triangulate_jammer_ls, compute_crlb,
    AntidroneBridgeTracker
)

# ─────────────────────────────────────────────────────────────────────────────
# Scenario constants — consistent with Stage 2 (Remediated)
# ─────────────────────────────────────────────────────────────────────────────
RNG          = np.random.default_rng(42)
P_JAMMER_M   = np.array([200.0, 150.0])   # true jammer position (metres)
N_MC         = 3000                        # Monte Carlo trials

# Stage 2 link budget parameters
SNR_DB   = 15.0    # signal SNR at array (consistent with simulator_core scenario A)
INR_DB   = 50.0    # jammer INR before nulling (consistent with simulator_core scenario A)
N        = N_ARRAY # 4-element ULA (Stage 2 default)
L        = L_SNAPSHOTS  # 100 snapshots

# ─────────────────────────────────────────────────────────────────────────────
# Helper: get a real MUSIC DOA estimate from Stage 2 signal model
# ─────────────────────────────────────────────────────────────────────────────
def get_real_music_doa(uav_pos_m: np.ndarray,
                       jammer_pos_m: np.ndarray,
                       snr_db: float = SNR_DB,
                       inr_db: float = INR_DB,
                       rng_seed: int = None,
                       is_anti_drone: bool = False) -> tuple:
    """
    Compute real MUSIC DOA estimate using Stage 2 Rician signal model.

    1. Compute true geometry angles (UAV → GCS signal, UAV → Jammer)
    2. Call generate_received_signal_rician() to build Rician snapshots
    3. Run music_doa() + find_music_peaks() on the sample covariance
    4. Return estimated jammer DOA and MUSIC CRLB sigma

    Returns
    -------
    doa_est_deg  : float — MUSIC estimated jammer angle (degrees)
    doa_true_deg : float — true geometric angle (degrees)
    sigma_deg    : float — MUSIC CRLB uncertainty (degrees)
    """
    # Jammer angle relative to UAV position
    delta = jammer_pos_m - uav_pos_m
    jam_angle_deg = float(np.degrees(np.arctan2(delta[1], delta[0])))

    # Clip to ±90 deg (ULA endfire limit)
    jam_angle_deg = float(np.clip(jam_angle_deg, -89.0, 89.0))

    rng_local = np.random.default_rng(rng_seed)

    if is_anti_drone:
        # For Anti-Drone, the target (UAV) IS the signal of interest. No other jammer.
        _, R_sample, _ = generate_received_signal_rician(
            N=N,
            theta_s_deg=jam_angle_deg,
            theta_j_list_deg=[],
            SNR_dB=snr_db,
            INR_dB_list=[],
            L_snapshots=L,
            K_signal=RICIAN_K,
            K_jammer=RICIAN_K,
            rng=rng_local
        )
    else:
        # Standard case: GCS is at 0 degrees, Jammer is at jam_angle_deg
        sig_angle_deg = 0.0
        _, R_sample, _ = generate_received_signal_rician(
            N=N,
            theta_s_deg=sig_angle_deg,
            theta_j_list_deg=[jam_angle_deg],
            SNR_dB=snr_db,
            INR_dB_list=[inr_db],
            L_snapshots=L,
            K_signal=RICIAN_K,
            K_jammer=RICIAN_K,
            rng=rng_local
        )

    # Run real MUSIC on the Stage 2 sample covariance
    scan_angles, spectrum = music_doa(R_sample, num_sources=1, scan_resolution_deg=0.2)
    est_peaks = find_music_peaks(scan_angles, spectrum, num_sources=1)
    doa_est_deg = float(est_peaks[0])

    # Real CRLB from Stage 2 (RSS of statistical + calibration error)
    sigma_deg = compute_crlb_doa_snr_aware(N, L, snr_db, jam_angle_deg,
                                            calib_error_deg=0.0)

    return doa_est_deg, jam_angle_deg, sigma_deg


# ─────────────────────────────────────────────────────────────────────────────
# Build UAV observation positions (ring geometry)
# ─────────────────────────────────────────────────────────────────────────────
def make_ring_observations(n_uavs: int,
                            radius: float = 400.0,
                            rng_seeds: list = None) -> list:
    """
    Place n_uavs on a ring and get real MUSIC DOA observations for each.
    """
    angles  = np.linspace(0, 2*np.pi, n_uavs, endpoint=False)
    obs_list = []
    for k, ang in enumerate(angles):
        upos = radius * np.array([np.cos(ang), np.sin(ang)])
        seed = rng_seeds[k] if rng_seeds else None
        doa_est, doa_true, sigma = get_real_music_doa(upos, P_JAMMER_M,
                                                       rng_seed=seed)
        doa_rad = float(np.radians(doa_est))
        obs_list.append(UAVObservation(
            position=upos,
            doa_rad=doa_rad,
            doa_var=float(np.radians(sigma))**2
        ))
    return obs_list


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────
plt.style.use('seaborn-v0_8-darkgrid')
fig = plt.figure(figsize=(16, 12), constrained_layout=True)
fig.suptitle(
    "Stage 4 — Category 1: Jammer Localization & Anti-Drone Tracker\n"
    "Integrated with Stage 2 (Remediated) MUSIC DOA + Rician Channel",
    fontsize=13, fontweight='bold')
gs = gridspec.GridSpec(2, 2, figure=fig)

# ─── Plot 1: Triangulation geometry with real MUSIC DOA ──────────────────────
ax1 = fig.add_subplot(gs[0, 0])

uav_positions = [np.array([-300.0, 0.0]),
                 np.array([300.0, -200.0]),
                 np.array([0.0,  400.0])]
colors_uav = ['royalblue', 'orange', 'green']
obs_list_3 = []

for k, (upos, col) in enumerate(zip(uav_positions, colors_uav)):
    doa_est, doa_true, sigma = get_real_music_doa(upos, P_JAMMER_M, rng_seed=k)
    doa_rad = float(np.radians(doa_est))
    obs_list_3.append(UAVObservation(
        position=upos,
        doa_rad=doa_rad,
        doa_var=float(np.radians(sigma))**2
    ))

    ray_len = 900
    ray_dir = np.array([np.cos(doa_rad), np.sin(doa_rad)])
    # Ray may point away from jammer if DOA is ambiguous — show full ray
    ray_end = upos + ray_len * ray_dir
    ax1.annotate('', xy=ray_end, xytext=upos,
                 arrowprops=dict(arrowstyle='->', color=col, lw=1.5))
    ax1.plot(*upos, 's', color=col, markersize=10, zorder=5)
    ax1.text(upos[0]+15, upos[1]+15,
             f'UAV {k+1}\nDOA={doa_est:.1f}° (true={doa_true:.1f}°)',
             color=col, fontsize=7)

p_est, cov_est = triangulate_jammer_ls(obs_list_3)
crlb_mat = compute_crlb(obs_list_3, P_JAMMER_M)
eigvals, eigvecs = np.linalg.eigh(crlb_mat)
angle_ell = float(np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1])))
ellipse = Ellipse(xy=P_JAMMER_M,
                  width=2*3*np.sqrt(max(eigvals[0], 1e-6)),
                  height=2*3*np.sqrt(max(eigvals[1], 1e-6)),
                  angle=angle_ell, fill=False, edgecolor='red',
                  linestyle='--', linewidth=2, label='3-sigma CRLB')
ax1.add_patch(ellipse)
ax1.plot(*P_JAMMER_M, '*', color='red',    markersize=18, zorder=6, label='True jammer')
ax1.plot(*p_est,      'D', color='purple', markersize=11, zorder=6, label='WLS estimate')
err_m = float(np.linalg.norm(p_est - P_JAMMER_M))
ax1.set_title(f'Triangulation (real MUSIC DOA)\nWLS error = {err_m:.1f} m')
ax1.set_xlabel('x (m)'); ax1.set_ylabel('y (m)')
ax1.legend(fontsize=8); ax1.set_aspect('equal')
ax1.set_xlim([-500, 600]); ax1.set_ylim([-450, 650])

# ─── Plot 2: MC RMSE vs N UAVs vs real MUSIC CRLB ───────────────────────────
ax2 = fig.add_subplot(gs[0, 1])

n_uav_range = [2, 3, 4, 6, 8]
rmse_mc     = []
crlb_bound  = []

print("Running Monte Carlo RMSE sweep (this takes ~30 s)...")
for N_u in n_uav_range:
    angles   = np.linspace(0, 2*np.pi, N_u, endpoint=False)
    radius   = 400.0
    upositions = [radius * np.array([np.cos(a), np.sin(a)]) for a in angles]

    # Real MUSIC CRLB for this geometry (using Stage 2's compute_crlb_doa_snr_aware)
    crlb_sigmas_rad = []
    for upos in upositions:
        _, doa_true, sigma_deg = get_real_music_doa(upos, P_JAMMER_M, rng_seed=99)
        crlb_sigmas_rad.append(float(np.radians(sigma_deg)))

    obs_ideal = [UAVObservation(
        position=upositions[i],
        doa_rad=float(np.radians(
            np.degrees(np.arctan2(P_JAMMER_M[1]-upositions[i][1],
                                  P_JAMMER_M[0]-upositions[i][0])))),
        doa_var=crlb_sigmas_rad[i]**2
    ) for i in range(N_u)]

    crlb_pos = compute_crlb(obs_ideal, P_JAMMER_M)
    crlb_bound.append(float(np.sqrt(np.trace(crlb_pos))))

    # MC with real MUSIC DOA each trial
    errors = []
    for trial in range(N_MC):
        obs_trial = []
        for ui, upos in enumerate(upositions):
            doa_est, _, sigma = get_real_music_doa(upos, P_JAMMER_M,
                                                    rng_seed=trial * 100 + ui)
            obs_trial.append(UAVObservation(
                position=upos,
                doa_rad=float(np.radians(doa_est)),
                doa_var=float(np.radians(sigma))**2
            ))
        try:
            p_hat, _ = triangulate_jammer_ls(obs_trial)
            errors.append(float(np.linalg.norm(p_hat - P_JAMMER_M)))
        except np.linalg.LinAlgError:
            pass

    rmse_mc.append(float(np.sqrt(np.mean(np.array(errors)**2))))
    print(f"  N={N_u}: MC RMSE={rmse_mc[-1]:.2f} m, CRLB={crlb_bound[-1]:.2f} m")

ax2.plot(n_uav_range, rmse_mc,    'o-',  color='royalblue', linewidth=2, label='MC RMSE (real MUSIC)')
ax2.plot(n_uav_range, crlb_bound, 's--', color='crimson',   linewidth=2, label='CRLB bound')
ax2.fill_between(n_uav_range, crlb_bound, rmse_mc, alpha=0.15, color='steelblue', label='Efficiency gap')
ax2.set_xlabel('Number of UAVs'); ax2.set_ylabel('Position RMSE (m)')
ax2.set_title(f'WLS Accuracy vs N UAVs\n(Stage 2 Rician channel, SNR={SNR_DB} dB, INR={INR_DB} dB)')
ax2.legend(fontsize=9)

# ─── Plot 3: Anti-Drone Kalman tracker with real MUSIC ───────────────────────
ax3 = fig.add_subplot(gs[1, 0])

DT      = 0.5      # seconds per step (slower than before — real MUSIC takes time)
T_SIM   = 20.0
N_STEPS = int(T_SIM / DT)
t_axis  = np.arange(N_STEPS) * DT

# Simulate UAV flying slowly (azimuth changing gradually)
true_az_deg = 5.0 * np.sin(2 * np.pi * t_axis / 10.0) + 20.0   # slow oscillation in deg

tracker = AntidroneBridgeTracker(dt=DT, meas_noise_std=0.02,
                                  process_noise_std=2e-3,
                                  g_max_dbi=25.0, bw_3db_rad=0.05)
tracker.x[0] = float(np.radians(true_az_deg[0]))
tracker.x[2] = float(np.radians(10.0))   # fixed elevation guess

# Ground anti-drone station at origin, UAV at 500m
ANTIDRONE_POS = np.array([0.0, 0.0])
UAV_RADIUS    = 500.0

est_az_deg  = []
meas_az_deg = []

for k in range(N_STEPS):
    # True UAV position from angle
    uav_pos = UAV_RADIUS * np.array([
        np.cos(np.radians(true_az_deg[k])),
        np.sin(np.radians(true_az_deg[k]))
    ])

    # Real MUSIC DOA from anti-drone ground station
    doa_est, doa_true, sigma_meas = get_real_music_doa(
        ANTIDRONE_POS, uav_pos,
        snr_db=20.0,   # anti-drone station has high SNR (elevated, fixed)
        inr_db=0.0,    # no competing jamming on anti-drone's own array
        rng_seed=k * 7,
        is_anti_drone=True
    )
    meas_rad = float(np.radians(doa_est))
    meas_el  = float(np.radians(5.0))   # flat trajectory

    x_hat, _ = tracker.update(np.array([meas_rad, meas_el]))
    est_az_deg.append(float(np.degrees(x_hat[0])))
    meas_az_deg.append(doa_est)

ax3.plot(t_axis, true_az_deg,  'k-',  linewidth=2,  label='True UAV azimuth')
ax3.plot(t_axis, meas_az_deg,  '.',   color='orange', markersize=4, alpha=0.7, label='MUSIC measurement')
ax3.plot(t_axis, est_az_deg,   '-',   color='royalblue', linewidth=1.5, label='Kalman estimate')
ax3.set_xlabel('Time (s)'); ax3.set_ylabel('Azimuth (deg)')
ax3.set_title('Anti-Drone Tracker: Kalman on Real MUSIC DOA\n(Anti-drone station tracks flying UAV)')
ax3.legend(fontsize=9)

# ─── Plot 4: Anti-Drone effectiveness — P_d of Neyman-Pearson sensor ─────────
ax4 = fig.add_subplot(gs[1, 1])

# Use Stage 2's detection_prob() — the real NP radiometer
t_sense_values = [0.001, 0.005, 0.01, 0.05, 0.1]   # sensing dwell times (s)
jnr_range_db   = np.linspace(-10, 30, 100)

for t_s, ls in zip(t_sense_values, ['-', '--', ':', '-.', '-']):
    pd_arr = [detection_prob(t_s, BW_hz=1e6, JNR_dB=jnr, P_fa=0.01)
              for jnr in jnr_range_db]
    ax4.plot(jnr_range_db, pd_arr, linestyle=ls, linewidth=2,
             label=f't_sense={int(t_s*1000)} ms')

ax4.axhline(0.9, color='grey', linestyle=':', linewidth=1.5, label='P_d = 0.9 target')
ax4.set_xlabel('Jammer-to-Noise Ratio JNR (dB)')
ax4.set_ylabel('Detection Probability P_d')
ax4.set_title('Anti-Drone Detection: Neyman-Pearson Radiometer\n(Stage 2 detection_prob, P_fa=0.01, BW=1 MHz)')
ax4.legend(fontsize=8)
ax4.set_ylim([0, 1.05])

plt.savefig(os.path.join(os.path.dirname(__file__), 'cat1_validation.png'), dpi=150)
print("Saved cat1_validation.png")

# ─────────────────────────────────────────────────────────────────────────────
# Validation summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n===== Category 1 Validation Summary (Integrated) =====")

# Check 1: 2-UAV real MUSIC intersection
upos1 = np.array([0.0, 0.0])
upos2 = np.array([400.0, 0.0])
d1, _, s1 = get_real_music_doa(upos1, P_JAMMER_M, rng_seed=101)
d2, _, s2 = get_real_music_doa(upos2, P_JAMMER_M, rng_seed=102)
obs1 = UAVObservation(upos1, float(np.radians(d1)), float(np.radians(s1))**2)
obs2 = UAVObservation(upos2, float(np.radians(d2)), float(np.radians(s2))**2)
p_hat2, cond2, rel2 = triangulate_jammer_2uav(obs1, obs2)
err2 = float(np.linalg.norm(p_hat2 - P_JAMMER_M))
print(f"[1] 2-UAV real MUSIC error: {err2:.2f} m  (MUSIC noise-limited, expect < 3*CRLB)")
expected_bound = 3 * max(s1, s2) * 400.0  # rough bound: 3*sigma_doa * range
print(f"    3-sigma bound: {expected_bound:.1f} m  => {'PASS' if err2 < expected_bound else 'FAIL'}")

# Check 2: MC RMSE vs CRLB (fundamental physics)
print(f"\n[2] MC RMSE vs CRLB at N=3 UAVs:")
print(f"    RMSE = {rmse_mc[1]:.2f} m,  CRLB = {crlb_bound[1]:.2f} m")
print(f"    RMSE >= 0.8 * CRLB (allowing for WLS bias): {'PASS' if rmse_mc[1] >= 0.8 * crlb_bound[1] else 'FAIL'}")

# Check 3: Collinear geometry flagged as unreliable
obs_col1 = UAVObservation(np.array([0.0,   0.0]), doa_rad=0.15)
obs_col2 = UAVObservation(np.array([0.0, 100.0]), doa_rad=0.15)  # same direction!
_, cond_col, rel_col = triangulate_jammer_2uav(obs_col1, obs_col2)
print(f"\n[3] Collinear geometry: cond={cond_col:.0e}  reliable={rel_col}")
print(f"    Correctly flagged: {'PASS' if not rel_col else 'FAIL'}")

# Check 4: Anti-Drone P_d at JNR=10 dB with t_sense=10ms should be >0.9
pd_check = detection_prob(t_s_sec=0.01, BW_hz=1e6, JNR_dB=10.0, P_fa=0.01)
print(f"\n[4] Anti-Drone P_d at JNR=10dB, t_sense=10ms: {pd_check:.4f}")
print(f"    > 0.9 target: {'PASS' if pd_check > 0.9 else 'FAIL'}")

# Check 5: Tracker RMSE vs real measurement noise
est_arr  = np.array(est_az_deg)
true_arr = true_az_deg
rmse_tracker = float(np.sqrt(np.mean((est_arr[N_STEPS//2:] - true_arr[N_STEPS//2:])**2)))
sigma_meas_deg = float(compute_crlb_doa_snr_aware(N, L, 20.0, 20.0))
print(f"\n[5] Tracker RMSE: {rmse_tracker:.3f} deg  |  MUSIC sigma: {sigma_meas_deg:.3f} deg")
print(f"    Tracker better than 5*sigma_meas ({5*sigma_meas_deg:.3f} deg): "
      f"{'PASS' if rmse_tracker < 5*sigma_meas_deg else 'FAIL'}")

print("======================================================")
