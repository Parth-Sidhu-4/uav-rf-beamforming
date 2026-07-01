import sys, os
import numpy as np
import matplotlib.pyplot as plt

# Add dependencies to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Stage 3 (Phase D)')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Stage 4 (Extensions)')))

from ew_channel import EWChannelBridge, ChannelSnapshot
from mavlink_rx import MAVLinkReceiver, RxStats
from cat3.advanced_ins import AdvancedINS
from cat4.conformal_array import CylindricalConformalArray

class SystemState:
    def __init__(self, x, y, z, vx, vy, vz):
        self.pos = np.array([x, y, z], dtype=float)
        self.vel = np.array([vx, vy, vz], dtype=float)
        self.attitude = np.array([0., 0., 0.], dtype=float)

def get_radial_velocity(pos1, vel1, pos2, vel2):
    """Radial velocity of pos1 relative to pos2 (projection onto LOS)."""
    dp = pos1 - pos2
    dist = np.linalg.norm(dp)
    if dist < 1e-6: return 0.0
    dv = vel1 - vel2
    return np.dot(dv, dp) / dist

def two_ray_path_loss(d_3d, h_tx, h_rx, carrier_freq):
    """3D Two-Ray Ground Reflection Model."""
    lam = 3e8 / carrier_freq
    d_direct = d_3d
    d_reflect = np.sqrt(d_3d**2 - (h_tx - h_rx)**2 + (h_tx + h_rx)**2)
    
    # Complex amplitudes
    E_d = (lam / (4 * np.pi * d_direct)) * np.exp(-1j * 2 * np.pi * d_direct / lam)
    # Assume Gamma = -1 for simple grazing angle ground reflection
    E_r = -1.0 * (lam / (4 * np.pi * d_reflect)) * np.exp(-1j * 2 * np.pi * d_reflect / lam)
    
    P_rx = np.abs(E_d + E_r)**2
    if P_rx == 0: return 1000.0
    return -10 * np.log10(P_rx)

def main():
    # Simulation Parameters
    dt = 0.01  # 100 Hz
    duration = 400.0  
    n_steps = int(duration / dt)
    
    # Environment
    gcs_pos = np.array([0, 0, 10])
    gcs_vel = np.array([0, 0, 0])
    
    # We place the jammer EXACTLY at the GCS to defeat LCMV angular resolution
    jammer_pos = np.copy(gcs_pos)
    jammer_vel = np.array([0, 0, 0])
    # jammer_eirp is dynamic
    
    carrier_freq = 2.4e9
    lam = 3e8 / carrier_freq
    
    # True State vs Estimated State Split
    true_state = SystemState(0, 0, 100, 20, 0, 0)
    est_state  = SystemState(0, 0, 100, 20, 0, 0)
    
    # Components
    # We set dt=dt for the INS to match the 100Hz loop
    ins = AdvancedINS(dt_s=dt, use_schuler=True)
    ins.x[0:4] = [0, 0, 20, 0] # Init true
    # Inject tactical-grade IMU bias to drive the Schuler drift
    ins.x[4] = 0.5  # bias x (m/s^2)
    ins.x[5] = 0.5  # bias y (m/s^2)
    array = CylindricalConformalArray(N=16, R=0.2)
    
    # Separate RNG streams for reproducibility
    seed = 42
    rng_fade = np.random.default_rng(seed + 1)
    rng_mc = np.random.default_rng(seed + 2)
    
    # We override the EWChannelBridge's internal rng to ensure separation
    bridge = EWChannelBridge(seed=seed+3)
    rx = MAVLinkReceiver()
    
    # Failsafe Trackers
    per_window = np.zeros(100, dtype=bool)
    per_idx = 0
    mission_state = "OUTBOUND"
    telemetry_healthy = True
    
    # Event Trigger States
    last_heavy_pos = np.copy(true_state.pos)
    last_heavy_jam_angle = 0.0
    last_los = True
    los_persist_ticks = 0
    
    # Macro Caches
    cached_sinr_db = 20.0
    cached_p_out = 0.0
    K_factor_db = 10.0
    
    # Fading State
    fading_state = rng_fade.normal(0, 1/np.sqrt(2), 2).view(np.complex128)[0]
    
    # Logs
    log_t = np.zeros(n_steps)
    log_true_pos = np.zeros((n_steps, 3))
    log_est_pos  = np.zeros((n_steps, 3))
    log_sinr = np.zeros(n_steps)
    log_per  = np.zeros(n_steps)
    log_rtl  = np.zeros(n_steps, dtype=bool)
    
    for k in range(n_steps):
        t = k * dt
        log_t[k] = t
        
        # ==========================================
        # 1. H-MRSM Flight Controller (uses EST STATE)
        # ==========================================
        if mission_state == "RTL":
            # JAPP fast_emergency_rtl(): Head back to origin (0,0) based on perceived position
            rtl_vec = np.array([0, 0, 100]) - est_state.pos
            dist_est = np.linalg.norm(rtl_vec)
            if dist_est > 1e-6:
                est_state.vel = (rtl_vec / dist_est) * 20.0
        else:
            # Nominal JAPP mission route
            if est_state.pos[0] < 2000:
                est_state.vel = np.array([20, 0, 0])
            else:
                est_state.vel = np.array([0, 20, 0])
                
        # The true state simply integrates the commanded velocity perfectly (assuming perfect actuators)
        old_true_vel = np.copy(true_state.vel)
        true_state.vel = np.copy(est_state.vel)
        dv_true = true_state.vel - old_true_vel
        
        true_state.pos += true_state.vel * dt
        log_true_pos[k] = np.copy(true_state.pos)
        
        # ==========================================
        # 2. INS Estimation (updates EST STATE)
        # ==========================================
        # The INS measures the acceleration (dv_true/dt) plus bias.
        # We manually add the true velocity change to the INS before predict() 
        # so it follows the trajectory but accumulates drift.
        ins.x[2] += dv_true[0]
        ins.x[3] += dv_true[1]
        
        ins.predict(inject_bias=True)
        # Assuming the commanded velocity from FC becomes the new physical velocity 
        # (the INS measures acceleration to get there).
        # We simulate the INS integrating to the new position:
        est_state.pos[0] = ins.x[0]
        est_state.pos[1] = ins.x[1]
        est_state.vel[0] = ins.x[2]
        est_state.vel[1] = ins.x[3]
        log_est_pos[k] = np.copy(est_state.pos)
        
        # ==========================================
        # 3. Tier A: Deterministic Geometry (True State)
        # ==========================================
        d_gcs = np.linalg.norm(true_state.pos - gcs_pos)
        d_jam = np.linalg.norm(true_state.pos - jammer_pos)
        
        # Deterministic LOS check (shadowing behind a hypothetical building block)
        current_los = True
        if 1000 < true_state.pos[0] < 1200 and true_state.pos[1] < 100:
            current_los = False 
            
        if current_los != last_los:
            los_persist_ticks += 1
        else:
            los_persist_ticks = 0
            
        trigger_heavy = False
        
        if los_persist_ticks >= 10:
            trigger_heavy = True
            last_los = current_los
            los_persist_ticks = 0
            
        # Two-Ray spatial fringe tracking: distance moved > 2 meters
        if np.linalg.norm(true_state.pos - last_heavy_pos) > 2.0:
            trigger_heavy = True
            
        # Jammer DOA tracking
        vec_j = jammer_pos - true_state.pos
        angle_j = np.degrees(np.arctan2(vec_j[1], vec_j[0]))
        if abs(angle_j - last_heavy_jam_angle) > 1.0:
            trigger_heavy = True
            
        # ==========================================
        # 4. Tier B: Event-Triggered Heavy RF
        # ==========================================
        if trigger_heavy or k == 0:
            # Rician K factor depends on elevation angle
            val = max(0, true_state.pos[2]) / max(d_gcs, 1.0)
            elev_rad = np.arcsin(np.clip(val, -1.0, 1.0))
            K_factor_db = 10.0 * np.exp(0.05 * np.degrees(elev_rad))
            
            # LCMV Beamforming (MUSIC DOA assumed perfect here for array weights)
            phi_s = np.arctan2(-true_state.pos[1], -true_state.pos[0])
            phi_j = np.radians(angle_j)
            
            desired = (np.pi/2, phi_s)
            nulls = [(np.pi/2, phi_j)]
            
            a_s = array.steering_vector(*desired)
            a_j = array.steering_vector(*nulls[0])
            
            # Main beam resolution limit: if GCS and Jammer are within 5 degrees, 
            # the array cannot mathematically resolve them without impossible super-directivity.
            if abs(phi_s - phi_j) < np.radians(5.0):
                gain_s = 1.0
                gain_j = 1.0
            else:
                R_yy = 1e-3 * np.eye(16) + 1.0 * np.outer(a_s, a_s.conj()) + 1e5 * np.outer(a_j, a_j.conj())
                w_lcmv = array.lcmv_weights(R_yy, desired, nulls)
                gain_s = np.abs(w_lcmv.conj().T @ a_s)**2
                gain_j = np.abs(w_lcmv.conj().T @ a_j)**2
            
            # Link Budget with 3D Two-Ray Path Loss
            P_tx = 40 # dBm
            PL_gcs = two_ray_path_loss(d_gcs, true_state.pos[2], gcs_pos[2], carrier_freq)
            P_rx_s = P_tx - PL_gcs + 10*np.log10(max(gain_s, 1e-6))
            
            PL_j = 20*np.log10(max(d_jam,1)) + 20*np.log10(carrier_freq) - 147.55
            if not current_los:
                PL_j += 30 # Knife-edge shadowing
                
            # EW Attack Scenario: Jammer activates at max range (t > 100s)
            if t > 100.0:
                jammer_eirp = 55.0  # Massive jamming
            else:
                jammer_eirp = -100.0 # Silent
                
            P_rx_j = jammer_eirp + 30 - PL_j + 10*np.log10(max(gain_j, 1e-6))
            
            noise_dbm = -100
            P_j_lin = 10**(P_rx_j/10)
            N_lin   = 10**(noise_dbm/10)
            
            cached_sinr_db = P_rx_s - 10*np.log10(P_j_lin + N_lin)
            
            last_heavy_pos = np.copy(true_state.pos)
            last_heavy_jam_angle = angle_j
            
        # ==========================================
        # 5. Tier C: Packet Engine & AR(1) Fading
        # ==========================================
        # Doppler & Coherence Time via Radial Velocity
        v_rad = get_radial_velocity(true_state.pos, true_state.vel, gcs_pos, gcs_vel)
        f_D = abs(v_rad) / lam
        if f_D < 0.1: f_D = 0.1
        
        # AR(1) correlation parameter (Clarke's model J0)
        # Clamped to >= 0 to model block fading if the channel is undersampled (f_D * dt > ~0.4)
        from scipy.special import j0
        rho = max(0.0, float(j0(2 * np.pi * f_D * dt)))
        
        # AR(1) Complex Gaussian process
        w_fade = rng_fade.normal(0, 1/np.sqrt(2), 2).view(np.complex128)[0]
        fading_state = rho * fading_state + np.sqrt(max(0, 1 - rho**2)) * w_fade
        
        # Rician Envelope Transform
        K_lin = 10**(K_factor_db/10)
        los_amp = np.sqrt(K_lin / (K_lin + 1))
        scat_amp = np.sqrt(1 / (K_lin + 1))
        
        envelope = np.abs(los_amp + scat_amp * fading_state)
        fade_db = 20 * np.log10(max(envelope, 1e-3))
        
        inst_sinr = cached_sinr_db + fade_db
        log_sinr[k] = inst_sinr
        
        # Feed into MAVLink CRC checker
        snap = ChannelSnapshot(sinr_post_db=inst_sinr, p_out=0.0, sigma_theta=0.1)
        pkt = rng_mc.bytes(28) # 28 random bytes
        rx_pkt = bridge.transmit(pkt, snap)
        
        pkt_dropped = (rx_pkt is None) or (rx_pkt != pkt)
        per_window[per_idx] = pkt_dropped
        per_idx = (per_idx + 1) % 100
        
        current_per = np.mean(per_window)
        log_per[k] = current_per
        
        # INS Correction Pathway: First-order low-pass (exponential moving average)
        if not pkt_dropped and telemetry_healthy:
            # We simulate a simple GPS loosely-coupled correction
            alpha = 0.05
            ins.x[0] = (1 - alpha) * ins.x[0] + alpha * true_state.pos[0]
            ins.x[1] = (1 - alpha) * ins.x[1] + alpha * true_state.pos[1]
            ins.x[2] = (1 - alpha) * ins.x[2] + alpha * true_state.vel[0]
            ins.x[3] = (1 - alpha) * ins.x[3] + alpha * true_state.vel[1]
            ins.P[0:4, 0:4] *= 0.99  # slowly reduce covariance 
            
        # Sliding-Window H-MRSM Failsafe
        if current_per > 0.90:
            mission_state = "RTL"  # LATCHED! Once triggered, we never abort RTL.
            telemetry_healthy = False
        elif current_per < 0.50:
            telemetry_healthy = True
            
        log_rtl[k] = (mission_state == "RTL")

    print("Simulation complete. Generating plots...")
    # ==========================================
    # Plotting: 4 Panels including 2D X-Y Map
    # ==========================================
    fig = plt.figure(figsize=(14, 16))
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 1.5])

    # Plot 1: 1D Distance
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(log_t, np.linalg.norm(log_true_pos - gcs_pos, axis=1), label='True Distance to GCS', color='blue', lw=2)
    ax1.axhline(0, color='gray', linestyle='--')
    ax1.fill_between(log_t, 0, np.max(np.linalg.norm(log_true_pos - gcs_pos, axis=1)), where=log_rtl, color='red', alpha=0.15, label='H-MRSM RTL Active (Link Lost)')
    ax1.set_ylabel('Distance (m)')
    ax1.set_title('Stage 5: Integrated Navigation & Telemetry Resilience')
    ax1.legend(loc='upper right')
    ax1.grid(True)

    # Plot 2: SINR & PER
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(log_t, log_sinr, label='Tier B: Instantaneous SINR', color='purple', alpha=0.3)
    ax2.set_ylabel('SINR (dB)', color='purple')
    ax2.tick_params(axis='y', labelcolor='purple')
    ax2.grid(True)
    
    ax2_twin = ax2.twinx()
    ax2_twin.plot(log_t, log_per, label='Tier C: Packet Error Rate', color='orange', lw=2)
    ax2_twin.axhline(0.9, color='red', linestyle=':', label='Link Loss Threshold')
    ax2_twin.axhline(0.5, color='green', linestyle=':', label='Link Recovery Threshold')
    ax2_twin.set_ylabel('Packet Error Rate', color='orange')
    ax2_twin.tick_params(axis='y', labelcolor='orange')
    ax2_twin.set_ylim([-0.05, 1.05])

    # Plot 3: INS Drift
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    drift = np.linalg.norm(log_true_pos - log_est_pos, axis=1)
    ax3.plot(log_t, drift, label='INS Navigation Error (Schuler Drift)', color='green', lw=2)
    ax3.fill_between(log_t, 0, np.max(drift), where=log_rtl, color='red', alpha=0.15)
    ax3.set_ylabel('Error Magnitude (m)')
    ax3.set_xlabel('Time (s)')
    ax3.legend(loc='upper left')
    ax3.grid(True)

    # Plot 4: 2D X-Y Map
    ax4 = fig.add_subplot(gs[3])
    
    # Split true trajectory into outbound and return
    idx100 = int(100.0 / 0.1) # approx index for t=100s
    if idx100 >= len(log_true_pos): idx100 = len(log_true_pos)-1
    
    # Plot true path (outbound)
    ax4.plot(log_true_pos[:idx100, 0], log_true_pos[:idx100, 1], label='Outbound Trajectory', color='gray', linestyle='--', lw=4, alpha=0.5)
    # Plot true path (diverging RTL)
    ax4.plot(log_true_pos[idx100:, 0], log_true_pos[idx100:, 1], label='True Trajectory (RTL)', color='blue', lw=2)
    # Plot perceived path
    ax4.plot(log_est_pos[idx100:, 0], log_est_pos[idx100:, 1], label='Perceived Trajectory (INS)', color='green', linestyle='--', lw=2)
    
    # Plot the "Ideal" RTL path from the trigger point
    # Find index where RTL triggers
    rtl_idx = np.where(log_rtl)[0]
    if len(rtl_idx) > 0:
        idx0 = rtl_idx[0]
        trigger_pos = log_true_pos[idx0]
        ax4.plot([trigger_pos[0], gcs_pos[0]], [trigger_pos[1], gcs_pos[1]], 'k:', lw=2, label='Ideal RTL Path')
        ax4.scatter([trigger_pos[0]], [trigger_pos[1]], color='red', marker='x', s=100, label='EW Attack / RTL Trigger')

    # Markers
    ax4.scatter([gcs_pos[0]], [gcs_pos[1]], color='black', marker='^', s=150, label='GCS / Jammer Location')
    
    # Empty scatter just for the legend, so the user knows what the inset marker is
    ax4.scatter([], [], color='darkred', marker='o', s=100, label='Final Position (t=400s)')
    
    ax4.set_aspect('equal', 'box')
    ax4.set_title('Top-Down 2D Map: Unrecoverable Navigation Divergence')
    ax4.set_xlabel('East (m)')
    ax4.set_ylabel('North (m)')
    ax4.legend(loc='upper right')
    ax4.grid(True)
    
    # --- INSET AXES ---
    # The true path is dwarfed by the massive INS drift scale. 
    # Create an inset in the bottom left to zoom in on the True Path.
    axins = ax4.inset_axes([0.05, 0.05, 0.45, 0.45])
    axins.set_facecolor('white')
    axins.set_zorder(10)
    
    axins.plot(log_true_pos[:idx100, 0], log_true_pos[:idx100, 1], color='gray', linestyle='--', lw=4, alpha=0.5)
    axins.plot(log_true_pos[idx100:, 0], log_true_pos[idx100:, 1], color='blue', lw=2)
    if len(rtl_idx) > 0:
        axins.plot([trigger_pos[0], gcs_pos[0]], [trigger_pos[1], gcs_pos[1]], 'k:', lw=2)
        axins.scatter([trigger_pos[0]], [trigger_pos[1]], color='red', marker='x', s=50)
    axins.scatter([gcs_pos[0]], [gcs_pos[1]], color='black', marker='^', s=100)
    axins.scatter([log_true_pos[-1, 0]], [log_true_pos[-1, 1]], color='darkred', marker='o', s=50)
    
    # Set inset limits to frame ONLY the True Trajectory
    x1, x2 = min(log_true_pos[:,0]) - 500, max(log_true_pos[:,0]) + 500
    y1, y2 = min(log_true_pos[:,1]) - 500, max(log_true_pos[:,1]) + 500
    
    # Ensure square aspect ratio for inset
    dx = x2 - x1
    dy = y2 - y1
    max_range = max(dx, dy)
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2
    
    axins.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    axins.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    axins.set_title('Zoom: True Miss Trajectory', fontsize=10)
    axins.grid(True)
    # Turn off ticks entirely for the inset to prevent overlap
    axins.set_xticks([])
    axins.set_yticks([])
    axins.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    
    # Hide the spine if the ticks are still showing up magically
    for spine in axins.spines.values():
        spine.set_visible(True)

    ax4.indicate_inset_zoom(axins, edgecolor='lightgray', alpha=0.5, zorder=9)

    # Bump legend linewidths for visibility
    leg = ax4.legend(loc='upper right')
    for legobj in leg.legend_handles:
        legobj.set_linewidth(3.0)

    plt.tight_layout()
    
    # --- FINAL DIAGNOSTIC CHECK ---
    print("\n--- INSET DIAGNOSTICS ---")
    print(f"GCS / Jammer Location mathematically placed at: [{gcs_pos[0]}, {gcs_pos[1]}]")
    
    final_pos = log_true_pos[-1]
    gcs_pos_arr = np.array([gcs_pos[0], gcs_pos[1], 10.0])
    dist = np.linalg.norm(final_pos - gcs_pos_arr)
    print(f"Computed final distance from GCS: {dist:.1f} m")
    # ------------------------
    
    plt.savefig('stage5_grand_integration.png', dpi=300)
    plt.close()
    print("Saved stage5_grand_integration.png")

if __name__ == "__main__":
    main()
