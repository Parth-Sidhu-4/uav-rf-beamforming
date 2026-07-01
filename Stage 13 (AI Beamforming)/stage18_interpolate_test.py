import numpy as np
import torch
import time

def compute_oracle_steering_vectors(headings_deg, elevations_deg):
    """
    Computes true physics steering vectors via the CartesianShadowNet surrogate.
    Since we need 0.1 deg resolution, we use the shadow net surrogate for fast eval, 
    but it represents the "true" physics we are trying to match.
    """
    import sys, os
    sys_path_save = sys.path.copy()
    sys.path.append(os.path.abspath(r'D:\UAV Internship project\Stage 12 (Operational Analysis)'))
    sys.path.append(os.path.abspath(r'D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)'))
    sys.path.append(os.path.abspath(r'D:\UAV Internship project\Phase 2 Track 1\em_realism'))
    from attitude import rotate_points, euler_to_quaternion
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Base geometry setup
    sys.path.append(os.path.abspath(r'D:\UAV Internship project\Phase 2 Track 2\array_structures'))
    from conformal_array import get_conformal_array_parametric
    from mesh_loader import load_uav_mesh
    from pathlib import Path
    mesh = load_uav_mesh(Path(r'D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone_prepared.stl'))
    pos_body, _ = get_conformal_array_parametric(mesh, N=32)
    pos_lambda_np = pos_body / 0.15
    q_inv = euler_to_quaternion(np.deg2rad(15.0), 0, 0).conjugate()
    pos_lambda = torch.tensor(pos_lambda_np, dtype=torch.float64, device=device)
    
    # Shadow net
    from stage15d_train_cartesian_32 import CartesianShadowNet32
    shadow_net = CartesianShadowNet32().to(device)
    shadow_net.load_state_dict(torch.load('shadow_net_cartesian_32.pt', map_location=device))
    shadow_net.eval()
    
    sig_world = np.array([1.0, 0.0, 0.0])
    sig_body = rotate_points(sig_world.reshape(1, 3), q_inv)[0]
    sig_t = torch.tensor(sig_body, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        out_sig = shadow_net(sig_t)
        g_sig_raw = (out_sig[:, 0::2] + 1j * out_sig[:, 1::2]).to(torch.complex128)
        g_sig = g_sig_raw / torch.clamp(torch.abs(g_sig_raw), min=1.0)
    theta_s = torch.acos(sig_t[:, 2] / torch.norm(sig_t, dim=1)).to(torch.float64)
    phi_s = torch.atan2(sig_t[:, 1], sig_t[:, 0]).to(torch.float64)
    
    def get_steering_vector_pt(pos_lambda, theta, phi):
        dx = torch.sin(theta) * torch.cos(phi)
        dy = torch.sin(theta) * torch.sin(phi)
        dz = torch.cos(theta)
        d = torch.stack([dx, dy, dz], dim=-1)
        return torch.exp(1j * 2.0 * torch.pi * (d @ pos_lambda.T))
        
    v_sig_masked = g_sig * get_steering_vector_pt(pos_lambda, theta_s, phi_s)
    v_sig_masked = v_sig_masked[0].cpu().numpy()
    
    # Process inputs
    h_rad = np.deg2rad(headings_deg)
    e_rad = np.deg2rad(elevations_deg)
    
    x = np.cos(e_rad) * np.cos(h_rad)
    y = np.cos(e_rad) * np.sin(h_rad)
    z = np.sin(e_rad)
    jam_world = np.stack([x, y, z], axis=-1)
    
    jam_body = rotate_points(jam_world, q_inv)
    jam_t = torch.tensor(jam_body, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        out_jam = shadow_net(jam_t)
        g_jam_raw = (out_jam[:, 0::2] + 1j * out_jam[:, 1::2]).to(torch.complex128)
        g_jam = g_jam_raw / torch.clamp(torch.abs(g_jam_raw), min=1.0)
        
    jam_t_double = jam_t.to(torch.float64)
    theta_j = torch.acos(jam_t_double[:, 2] / torch.norm(jam_t_double, dim=1))
    phi_j = torch.atan2(jam_t_double[:, 1], jam_t_double[:, 0])
    v_ideal = get_steering_vector_pt(pos_lambda, theta_j, phi_j)
    
    v_jam = (g_jam * v_ideal).cpu().numpy()
    sys.path = sys_path_save
    
    return v_jam, v_sig_masked

def naive_bilinear_interpolate(h, e, grid_h, grid_e, grid_v):
    """
    Interpolates complex vectors component-wise.
    h: N-array of headings
    e: N-array of elevations
    grid_h: 1D array of grid headings
    grid_e: 1D array of grid elevations
    grid_v: 3D array (len(grid_h), len(grid_e), 32) of complex vectors
    """
    N = len(h)
    out = np.zeros((N, 32), dtype=np.complex128)
    
    for i in range(N):
        idx_h = np.searchsorted(grid_h, h[i]) - 1
        idx_e = np.searchsorted(grid_e, e[i]) - 1
        
        idx_h = np.clip(idx_h, 0, len(grid_h) - 2)
        idx_e = np.clip(idx_e, 0, len(grid_e) - 2)
        
        h0, h1 = grid_h[idx_h], grid_h[idx_h+1]
        e0, e1 = grid_e[idx_e], grid_e[idx_e+1]
        
        wh = (h[i] - h0) / (h1 - h0)
        we = (e[i] - e0) / (e1 - e0)
        
        v00 = grid_v[idx_h, idx_e]
        v10 = grid_v[idx_h+1, idx_e]
        v01 = grid_v[idx_h, idx_e+1]
        v11 = grid_v[idx_h+1, idx_e+1]
        
        v_interp = (1-wh)*(1-we)*v00 + wh*(1-we)*v10 + (1-wh)*we*v01 + wh*we*v11
        out[i] = v_interp
        
    return out

def phase_aware_bilinear_interpolate(h, e, grid_h, grid_e, grid_v):
    """
    Interpolates magnitude and unwrapped phase separately.
    """
    N = len(h)
    out = np.zeros((N, 32), dtype=np.complex128)
    
    mag_grid = np.abs(grid_v)
    phase_grid = np.angle(grid_v)
    
    for i in range(N):
        idx_h = np.searchsorted(grid_h, h[i]) - 1
        idx_e = np.searchsorted(grid_e, e[i]) - 1
        
        idx_h = np.clip(idx_h, 0, len(grid_h) - 2)
        idx_e = np.clip(idx_e, 0, len(grid_e) - 2)
        
        h0, h1 = grid_h[idx_h], grid_h[idx_h+1]
        e0, e1 = grid_e[idx_e], grid_e[idx_e+1]
        
        wh = (h[i] - h0) / (h1 - h0)
        we = (e[i] - e0) / (e1 - e0)
        
        # Magnitude interpolation
        m00 = mag_grid[idx_h, idx_e]
        m10 = mag_grid[idx_h+1, idx_e]
        m01 = mag_grid[idx_h, idx_e+1]
        m11 = mag_grid[idx_h+1, idx_e+1]
        m_interp = (1-wh)*(1-we)*m00 + wh*(1-we)*m10 + (1-wh)*we*m01 + wh*we*m11
        
        # Phase interpolation requires unwrapping relative to v00
        p00 = phase_grid[idx_h, idx_e]
        
        # Function to minimize wrap distance to p00
        def unwrap_rel(p):
            diff = p - p00
            diff = (diff + np.pi) % (2 * np.pi) - np.pi
            return p00 + diff
            
        p10 = unwrap_rel(phase_grid[idx_h+1, idx_e])
        p01 = unwrap_rel(phase_grid[idx_h, idx_e+1])
        p11 = unwrap_rel(phase_grid[idx_h+1, idx_e+1])
        
        p_interp = (1-wh)*(1-we)*p00 + wh*(1-we)*p10 + (1-wh)*we*p01 + wh*we*p11
        
        out[i] = m_interp * np.exp(1j * p_interp)
        
    return out

def compute_sinr_from_v(v_pred_list, v_true_list, v_sig):
    P_J = 10000.0
    P_S = 100.0
    sigma2 = 1.0
    alpha = 390.0
    
    sinrs = []
    
    import torch
    v_sig_t = torch.tensor(v_sig, dtype=torch.complex128)
    R_sig = P_S * torch.outer(v_sig_t, torch.conj(v_sig_t))
    R_n = sigma2 * torch.eye(32, dtype=torch.complex128)
    R_sn = R_sig + R_n
    
    for v_pred, v_true in zip(v_pred_list, v_true_list):
        vp_t = torch.tensor(v_pred, dtype=torch.complex128)
        vt_t = torch.tensor(v_true, dtype=torch.complex128)
        
        # Analytic MVDR from predicted v
        R_j_pred = P_J * torch.outer(vp_t, torch.conj(vp_t))
        R_in_pred = R_j_pred + (sigma2 + alpha) * torch.eye(32, dtype=torch.complex128)
        R_in_pred_inv = torch.linalg.inv(R_in_pred)
        
        w = R_in_pred_inv @ v_sig_t
        w = w / (torch.conj(v_sig_t) @ w)
        
        # Evaluate SINR on TRUE v
        S = torch.real(torch.conj(w) @ R_sn @ w)
        NJ = P_J * torch.abs(torch.conj(w) @ vt_t)**2 + sigma2 * torch.real(torch.conj(w) @ w)
        
        sinrs.append(10 * np.log10(float(S / NJ)))
        
    return np.array(sinrs)

def test_resolution(grid_res, test_h, test_e, test_v, v_sig):
    grid_h = np.arange(170.0, 190.0 + 1e-5, grid_res)
    grid_e = np.arange(-25.0, 25.0 + 1e-5, grid_res)
    
    # Compute ground truth for grid nodes
    H_grid, E_grid = np.meshgrid(grid_h, grid_e, indexing='ij')
    flat_h_grid = H_grid.flatten()
    flat_e_grid = E_grid.flatten()
    
    grid_v_flat, _ = compute_oracle_steering_vectors(flat_h_grid, flat_e_grid)
    grid_v = grid_v_flat.reshape(len(grid_h), len(grid_e), 32)
    
    print(f"\n--- Testing Grid Resolution: {grid_res} deg (Nodes: {len(flat_h_grid)}) ---")
    
    v_interp_naive = naive_bilinear_interpolate(test_h, test_e, grid_h, grid_e, grid_v)
    sinrs_naive = compute_sinr_from_v(v_interp_naive, test_v, v_sig)
    print(f"Naive Interpolation -> Min SINR: {np.min(sinrs_naive):.2f} dB, Below 15dB: {np.sum(sinrs_naive < 15.0)}")
    
    v_interp_phase = phase_aware_bilinear_interpolate(test_h, test_e, grid_h, grid_e, grid_v)
    sinrs_phase = compute_sinr_from_v(v_interp_phase, test_v, v_sig)
    print(f"Phase Interpolation -> Min SINR: {np.min(sinrs_phase):.2f} dB, Below 15dB: {np.sum(sinrs_phase < 15.0)}")

def main():
    print("Generating dense 0.1 deg test set over 170-190, +/-25...")
    test_h = np.random.uniform(170.1, 189.9, 5000)
    test_e = np.random.uniform(-24.9, 24.9, 5000)
    
    test_v, v_sig = compute_oracle_steering_vectors(test_h, test_e)
    
    # Test Oracles on their own true vectors (should be perfect)
    oracle_sinrs = compute_sinr_from_v(test_v, test_v, v_sig)
    print(f"Perfect Oracle Min SINR: {np.min(oracle_sinrs):.2f} dB")
    
    test_resolution(1.0, test_h, test_e, test_v, v_sig)
    test_resolution(0.5, test_h, test_e, test_v, v_sig)
    
if __name__ == '__main__':
    main()
