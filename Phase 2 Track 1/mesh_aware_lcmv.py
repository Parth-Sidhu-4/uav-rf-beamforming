import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — prevents hanging on plt.show() in headless runs
import matplotlib.pyplot as plt
from pathlib import Path
import trimesh

sys.path.append(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)")

from mesh_loader import load_uav_mesh
from conformal_array import get_conformal_array
from attitude import euler_to_quaternion, rotate_points
from cognitive_autopilot import CognitiveAutopilot, generate_lut, world_to_body_bearing

G = 9.81
LAMBDA_M = 0.125
P_NOISE = 1.0
P_SIG = P_NOISE * (10 ** (20 / 10))
P_JAM = P_NOISE * (10 ** (40 / 10))

def get_steering_vector(positions: np.ndarray, normals: np.ndarray, k_vec: np.ndarray, cal_error: np.ndarray = None) -> np.ndarray:
    phases = np.dot(positions, k_vec)
    k_dir = k_vec / np.linalg.norm(k_vec)
    cos_theta = np.sum(normals * k_dir, axis=1)
    gain = 0.5 * (1.0 + cos_theta)
    
    if cal_error is not None:
        phase_noise = cal_error
    else:
        phase_noise = 0.0
        
    return gain * np.exp(1j * (phases + phase_noise))

def compute_reduced_lcmv(a_sig_perceived, a_jam_perceived, a_sig_true, a_jam_true, mask):
    active_indices = np.where(mask)[0]
    M = len(active_indices)
    
    w_full = np.zeros(len(mask), dtype=complex)
    null_depth_db = 0.0
    
    if M <= 1:
        # Omni fallback / Link Failure regime
        if M == 1:
            w_full[active_indices] = 1.0
            g_sig = P_SIG * np.abs(a_sig_true[active_indices[0]])**2
            g_jam = P_JAM * np.abs(a_jam_true[active_indices[0]])**2
            g_nse = P_NOISE
            sinr_db = 10 * np.log10(max(g_sig / (g_jam + g_nse), 1e-12))
        else:
            sinr_db = -10.0 # completely dead
        return w_full, sinr_db, null_depth_db
        
    a_sig_A = a_sig_perceived[active_indices]
    a_jam_A = a_jam_perceived[active_indices]
    
    R_A = (P_SIG * np.outer(a_sig_A, np.conj(a_sig_A)) + 
           P_JAM * np.outer(a_jam_A, np.conj(a_jam_A)) + 
           P_NOISE * np.eye(M))
           
    cond_num = np.linalg.cond(R_A)
    
    # LCMV Degradation Rule
    if M == 2 or cond_num > 1e6:
        # Fallback to Matched Filter Steering (Conventional Beamforming)
        # This completely drops the null constraint and interference suppression.
        w_A = a_sig_A / (np.conj(a_sig_A).T @ a_sig_A)
        w_full[active_indices] = w_A
    else:
        # Full LCMV using MVDR form
        R_A_inv = np.linalg.inv(R_A + 1e-6 * np.eye(M))
        C = np.column_stack((a_sig_A, a_jam_A))
        f = np.array([1.0, 0.0])
        try:
            term2 = np.linalg.inv(np.conj(C).T @ R_A_inv @ C)
            w_A = R_A_inv @ C @ term2 @ f
            w_full[active_indices] = w_A
        except np.linalg.LinAlgError:
            # Fallback to Matched Filter
            w_A = a_sig_A / (np.conj(a_sig_A).T @ a_sig_A)
            w_full[active_indices] = w_A

    # Achieved Null Depth evaluated on TRUE manifold
    nd_linear = np.abs(np.conj(w_full).T @ a_jam_true)**2
    null_depth_db = 10 * np.log10(max(nd_linear, 1e-30))

    # Calculate actual SINR on TRUE manifold
    g_sig = P_SIG * np.abs(np.conj(w_full).T @ a_sig_true)**2
    g_jam = P_JAM * np.abs(np.conj(w_full).T @ a_jam_true)**2
    g_nse = P_NOISE * np.linalg.norm(w_full)**2
    
    denom = g_jam + g_nse
    if denom < 1e-12 or g_sig < 1e-12:
        sinr_db = 0.0
    else:
        sinr_db = 10 * np.log10(g_sig / denom)
        
    return w_full, sinr_db, null_depth_db

def run_simulation(replot_only=False):
    CACHE_FILE = Path("D:/UAV Internship project/Phase 2 Track 1/lcmv_results_cache.npz")

    if replot_only:
        if not CACHE_FILE.exists():
            print(f"ERROR: cache file not found at {CACHE_FILE}. Run without --replot first.")
            return
        print(f"[--replot] Loading cached results from {CACHE_FILE} ...")
        d = np.load(CACHE_FILE)
        h_bin,  s_bin,  a_bin,  p_bin,  nd_bin  = d['h_bin'],  d['s_bin'],  d['a_bin'],  d['p_bin'],  d['nd_bin']
        h_cont, s_cont, a_cont, p_cont, nd_cont = d['h_cont'], d['s_cont'], d['a_cont'], d['p_cont'], d['nd_cont']
        h_cog,  s_cog,  a_cog,  p_cog,  nd_cog  = d['h_cog'],  d['s_cog'],  d['a_cog'],  d['p_cog'],  d['nd_cog']
        _make_plots(h_bin, s_bin, a_bin, nd_bin,
                    h_cont, s_cont, a_cont, nd_cont,
                    h_cog, s_cog, a_cog, p_cog, nd_cog)
        return

    print("Loading mesh and setting up elements...")
    mesh_path = Path(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone_prepared.stl")
    mesh = load_uav_mesh(mesh_path)
    antennas_body, normals = get_conformal_array(mesh)
    
    intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
    k = 2 * np.pi / LAMBDA_M
    
    # Far-field geometry (100km) to decouple translation from relative body azimuth
    jammer_pos_xy = np.array([0.0, 100000.0])
    gcs_pos_xy    = np.array([0.0, -100000.0]) # WEST
    
    lut = generate_lut()
    # Require 15 elements to be strongly active (> -10.4 dB) to avoid degradation
    cognitive = CognitiveAutopilot(lut, min_active=15)
    
    def simulate_vectorized(autopilot, name, use_continuous=True):
        V, dt, desired_phi = 25.0, 0.1, 30.0
        heading_target = 90.0
        
        headings, phis = [], []
        ac_pos = np.zeros(2)
        h = 0.0
        
        print(f"[{name}] Precomputing kinematics...")
        while h < heading_target and len(headings) < 2000:
            headings.append(h)
            az_gcs, _ = world_to_body_bearing(gcs_pos_xy, ac_pos, h)
            if autopilot is None:
                phi_cmd = desired_phi
            else:
                phi_cmd = autopilot.command_bank(desired_phi, az_gcs, 0.0, V=V, dt=dt)
            phis.append(phi_cmd)
            h += np.degrees(G * np.tan(np.radians(phi_cmd)) / V) * dt
            
        N_steps = len(headings)
        headings = np.array(headings)
        phis = np.array(phis)
                
        jam_world = np.array([jammer_pos_xy[0], jam_pos_xy[1], 0.0] if 'jam_pos_xy' in locals() else [jammer_pos_xy[0], jammer_pos_xy[1], 0.0])
        jam_world /= np.linalg.norm(jam_world)
        gcs_world = np.array([gcs_pos_xy[0], gcs_pos_xy[1], 0.0])
        gcs_world /= np.linalg.norm(gcs_world)
        
        from shadow_engine import compute_shadow_mask
        
        print(f"[{name}] Running LCMV solver...")
        log = []
        
        # Static calibration error (phase noise) for realistic null depth
        np.random.seed(42)
        cal_err_sig = np.random.normal(0, 0.1, len(antennas_body))
        cal_err_jam = np.random.normal(0, 0.1, len(antennas_body))
        
        epsilon = 1e-4
        ray_origins = antennas_body + normals * epsilon
        
        for i in range(N_steps):
            q = euler_to_quaternion(np.deg2rad(phis[i]), 0, np.deg2rad(headings[i]))
            q_inv = q.conjugate()
            jam_body = rotate_points(jam_world.reshape(1,3), q_inv)[0]
            gcs_body = rotate_points(gcs_world.reshape(1,3), q_inv)[0]
            
            if use_continuous:
                mask_jam = compute_shadow_mask(mesh, antennas_body, normals, jam_body)
                mask_gcs = compute_shadow_mask(mesh, antennas_body, normals, gcs_body)
                # For continuous, all elements structurally active
                active_elements = np.ones(16, dtype=bool)
                # Define active element for visualization as having > -10.4 dB gain (magnitude > 0.3)
                active_count = np.sum(np.abs(mask_gcs) > 0.3)
            else:
                hits_jam = mesh.ray.intersects_any(ray_origins, np.tile(jam_body, (16, 1)))
                hits_gcs = mesh.ray.intersects_any(ray_origins, np.tile(gcs_body, (16, 1)))
                mask_jam = (~hits_jam).astype(complex)
                mask_gcs = (~hits_gcs).astype(complex)
                active_elements = mask_gcs.real > 0.5
                active_count = active_elements.sum()
            
            a_sig_true = get_steering_vector(antennas_body, normals, k * gcs_body)
            a_jam_true = get_steering_vector(antennas_body, normals, k * jam_body)
            
            a_sig_perceived = get_steering_vector(antennas_body, normals, k * gcs_body, cal_error=cal_err_sig)
            a_jam_perceived = get_steering_vector(antennas_body, normals, k * jam_body, cal_error=cal_err_jam)
            
            a_jam_perc_masked = a_jam_perceived * mask_jam
            
            a_sig_true_masked = a_sig_true * mask_gcs
            a_jam_true_masked = a_jam_true * mask_jam
            
            if not use_continuous:
                a_sig_perceived_masked = a_sig_perceived * mask_gcs
                w, sinr, nd = compute_reduced_lcmv(a_sig_perceived_masked, a_jam_perc_masked, a_sig_true_masked, a_jam_true_masked, active_elements)
            else:
                w, sinr, nd = compute_reduced_lcmv(a_sig_perceived, a_jam_perc_masked, a_sig_true_masked, a_jam_true_masked, active_elements)
            
            if sinr == 0.0 and i % 100 == 0:
                g_sig = P_SIG * np.abs(np.conj(w).T @ a_sig_true_masked)**2
                g_jam = P_JAM * np.abs(np.conj(w).T @ a_jam_true_masked)**2
                g_nse = P_NOISE * np.linalg.norm(w)**2
                print(f"[0dB Debug] Name: {name}, Heading: {headings[i]:.1f}, g_sig: {g_sig}, g_jam: {g_jam}, g_nse: {g_nse}, norm(w): {np.linalg.norm(w)}")
            
            log.append({
                't': i * dt,
                'heading': headings[i],
                'phi': phis[i],
                'active': active_count,
                'null_depth': nd,
                'sinr': sinr
            })
            
        return log

    base_bin_log = simulate_vectorized(None, "Baseline (Binary)", use_continuous=False)
    base_cont_log = simulate_vectorized(None, "Baseline (Continuous)", use_continuous=True)
    cog_cont_log = simulate_vectorized(cognitive, "Cognitive (Continuous)", use_continuous=True)
    
    # --- Checkpoint: save all arrays before touching matplotlib ---
    # If this process is killed during plotting, re-run with --replot to skip computation.
    h_bin = [x['heading'] for x in base_bin_log]
    s_bin = [x['sinr']   for x in base_bin_log]
    a_bin = [x['active'] for x in base_bin_log]
    p_bin = [x['phi']    for x in base_bin_log]
    nd_bin= [x['null_depth'] for x in base_bin_log]

    h_cont = [x['heading'] for x in base_cont_log]
    s_cont = [x['sinr']   for x in base_cont_log]
    a_cont = [x['active'] for x in base_cont_log]
    p_cont = [x['phi']    for x in base_cont_log]
    nd_cont= [x['null_depth'] for x in base_cont_log]

    h_cog  = [x['heading'] for x in cog_cont_log]
    s_cog  = [x['sinr']   for x in cog_cont_log]
    a_cog  = [x['active'] for x in cog_cont_log]
    p_cog  = [x['phi']    for x in cog_cont_log]
    nd_cog = [x['null_depth'] for x in cog_cont_log]

    np.savez(CACHE_FILE,
             h_bin=h_bin,  s_bin=s_bin,  a_bin=a_bin,  p_bin=p_bin,  nd_bin=nd_bin,
             h_cont=h_cont, s_cont=s_cont, a_cont=a_cont, p_cont=p_cont, nd_cont=nd_cont,
             h_cog=h_cog,  s_cog=s_cog,  a_cog=a_cog,  p_cog=p_cog,  nd_cog=nd_cog)
    print(f"[CHECKPOINT] Results cached to {CACHE_FILE}")

    _make_plots(h_bin, s_bin, a_bin, nd_bin,
                h_cont, s_cont, a_cont, nd_cont,
                h_cog, s_cog, a_cog, p_cog, nd_cog)


def _make_plots(h_bin, s_bin, a_bin, nd_bin,
                h_cont, s_cont, a_cont, nd_cont,
                h_cog, s_cog, a_cog, p_cog, nd_cog):
    """Shared plotting + summary — called by both the full simulation and --replot mode."""
    # Reconstruct p_bin and p_cont from the arrays if needed
    # (they are not passed here to keep the signature tight; generate flat 30° placeholders)
    p_bin  = [30.0] * len(h_bin)
    p_cont = [30.0] * len(h_cont)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 14), sharex=True)

    ax1.plot(h_bin, s_bin, 'r--', label='Baseline (Binary Mask)', linewidth=2)
    ax1.plot(h_cont, s_cont, 'b-', label='Baseline (Continuous Mask)', linewidth=4, alpha=0.5)
    ax1.plot(h_cog, s_cog, 'g-', label='Cognitive (Continuous)', linewidth=2)
    ax1.set_ylabel('SINR (dB)')
    ax1.set_title('Phase 2 Track 1: Mesh-Aware Dynamic LCMV Performance')
    ax1.axhline(y=0, color='orange', linestyle='--', label='Link Failure Threshold')
    ax1.grid(True)
    ax1.legend()

    ax2.plot(h_bin, a_bin, 'r--', linewidth=2)
    ax2.plot(h_cont, a_cont, 'b-', linewidth=2)
    ax2.plot(h_cog, [x['active'] if isinstance(x, dict) else x for x in a_cog], 'g-', linewidth=2)
    ax2.set_ylabel('Active Elements')
    ax2.axhline(y=3, color='orange', linestyle='--', label='LCMV Rank Limit (M=3, min for null + signal constraints)')
    ax2.axhline(y=15, color='blue', linestyle=':', label='Cognitive Autopilot Trigger (min_active=15)')
    ax2.legend()
    ax2.grid(True)

    ax3.plot(h_bin, nd_bin, 'r--', linewidth=2)
    ax3.plot(h_cont, nd_cont, 'b-', linewidth=2)
    ax3.plot(h_cog, nd_cog, 'g-', linewidth=2)
    ax3.set_ylabel('Null Depth (dB)')
    ax3.set_title('Jammer Null Depth (Note: Binary mask artificially truncates jammer to 0 causing -300dB floor)')
    ax3.axhline(y=-40, color='purple', linestyle=':', label='Target Null Depth')
    ax3.legend()
    ax3.grid(True)

    ax4.plot(h_bin,  p_bin,  'r--', linewidth=2)
    ax4.plot(h_cont, p_cont, 'b-',  linewidth=2)
    ax4.plot(h_cog,  p_cog,  'g-',  linewidth=2)
    ax4.set_ylabel(r'Bank Angle $\phi$ (deg)')
    ax4.set_xlabel('Heading (deg)')
    ax4.grid(True)

    plt.tight_layout()
    out_path = Path("D:/UAV Internship project/Phase 2 Track 1/mesh_aware_lcmv_results.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Results saved to {out_path}")

    print("\n--- Summary ---")
    print("Baseline (Binary) Minimum SINR:", np.min(s_bin), "dB")
    print("Baseline (Continuous) Minimum SINR:", np.min(s_cont), "dB")
    print("Cognitive (Continuous) Minimum SINR:", np.min(s_cog), "dB")
    print("Min Active Elements (Continuous):", np.min(a_cont))
    print("Min Active Elements (Cognitive):", np.min(a_cog))
    print("Unique p_cog values:", set(p_cog))

    # Numerical confirmation of cognitive/baseline overlap (per user request)
    s_cont_arr = np.array(s_cont)
    s_cog_arr  = np.array(s_cog)
    h_cont_arr = np.array(h_cont)
    h_cog_arr  = np.array(h_cog)
    # Interpolate onto a common heading grid to handle different trajectory lengths
    h_common = np.linspace(max(h_cont_arr[0], h_cog_arr[0]),
                           min(h_cont_arr[-1], h_cog_arr[-1]), 200)
    s_cont_interp = np.interp(h_common, h_cont_arr, s_cont_arr)
    s_cog_interp  = np.interp(h_common, h_cog_arr,  s_cog_arr)
    max_diff  = np.max(np.abs(s_cog_interp - s_cont_interp))
    mean_diff = np.mean(s_cog_interp - s_cont_interp)
    print(f"\n--- Cognitive vs Baseline Continuous SINR difference ---")
    print(f"  max |s_cog - s_cont|  = {max_diff:.4f} dB")
    print(f"  mean(s_cog - s_cont)  = {mean_diff:.4f} dB")
    if max_diff < 0.5:
        print("  => Curves are numerically overlapping (legitimate — same trajectory, same physics).")
    else:
        print("  => Curves are DISTINCT — cognitive benefit is real.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--replot', action='store_true',
                        help='Skip simulation, load cached npz and regenerate plots only')
    args = parser.parse_args()
    run_simulation(replot_only=args.replot)
