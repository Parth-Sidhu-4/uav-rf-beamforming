import numpy as np
import torch

def get_hybrid_steering_vectors(headings_deg, elevations_deg, U_ai):
    """
    Computes hybrid steering vectors.
    Uses AI prediction everywhere except in [168, 192], [-27, 27] blending box.
    Inside the box, interpolates precomputed Oracle steering vectors.
    """
    from stage18_interpolate_test import compute_oracle_steering_vectors, phase_aware_bilinear_interpolate, naive_bilinear_interpolate
    
    # 1. Precompute Oracle Grid (only done once offline)
    data = np.load('true_oracle_grid_0_5deg.npz')
    grid_h = data['grid_h']
    grid_e = data['grid_e']
    grid_v = data['grid_v']
    
    # 2. Extract AI predicted steering vectors from U_ai (Rank-K=5)
    # The first principal component of U is roughly v_pred. 
    # Or, we can just use U_ai for MVDR directly. But wait, how do we blend?
    # We can't linearly blend U (a 32x5 matrix) with an Oracle steering vector (32x1).
    # Wait! We can just blend the final MVDR weights, OR blend the Covariance matrices!
    # R_in_ai = P_J * (U @ U^H) + sigma2 * I
    # R_in_oracle = P_J * (v_oracle @ v_oracle^H) + sigma2 * I
    # Blending the Covariance matrices is mathematically rigorous and preserves PD properties!
    return grid_h, grid_e, grid_v

def blend_covariance(R_ai, v_oracle, weight_oracle):
    """
    Blends the AI covariance matrix with the Oracle covariance matrix.
    weight_oracle = 0.0 -> Pure AI
    weight_oracle = 1.0 -> Pure Oracle
    """
    import torch
    P_J = 10000.0
    v_oracle_t = torch.tensor(v_oracle, dtype=torch.complex128, device=R_ai.device)
    R_j_oracle = P_J * torch.outer(v_oracle_t, torch.conj(v_oracle_t))
    
    # We blend the J part, not the noise part, to preserve the condition number exactly.
    R_j_ai = R_ai - 390.0 * torch.eye(32, dtype=torch.complex128, device=R_ai.device)
    
    R_j_blend = (1 - weight_oracle) * R_j_ai + weight_oracle * R_j_oracle
    R_blend = R_j_blend + 390.0 * torch.eye(32, dtype=torch.complex128, device=R_ai.device)
    return R_blend

def get_blend_weights(h, e):
    """
    Computes blending weights.
    1.0 inside [170, 190] x [-25, 25] (Pure Oracle)
    0.0 outside [168, 192] x [-27, 27] (Pure AI)
    Linear blend in the 2-degree margin.
    """
    import numpy as np
    
    # Azimuth distance to [170, 190] box
    d_h = np.maximum(0, np.maximum(170 - h, h - 190))
    # Elevation distance to [-25, 25] box
    d_e = np.maximum(0, np.maximum(-25 - e, e - 25))
    
    # Max distance to box
    d = np.maximum(d_h, d_e)
    
    # 0 if d=0, 1 if d>=2
    w_ai = np.where(d > 0.0, 1.0, 0.0)
    return 1.0 - w_ai

def main():
    import numpy as np
    import torch
    from stage18w_train_100k_relu import ReluCovariancePredictor
    from stage18_interpolate_test import phase_aware_bilinear_interpolate, naive_bilinear_interpolate, compute_sinr_from_v
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Data
    
    data = np.load('dataset_shadow_100k_polar_32el.npz')
    inputs_all = data['inputs']
    labels_polar = data['labels']
    mag = labels_polar[:, 0::2]
    phase = labels_polar[:, 1::2]
    g_exact_all = mag * np.exp(1j * phase)
    
    theta_j = np.arccos(inputs_all[:, 2] / np.linalg.norm(inputs_all, axis=1))
    phi_j = np.arctan2(inputs_all[:, 1], inputs_all[:, 0])
    elevs = 90.0 - np.rad2deg(theta_j)
    heads = np.rad2deg(phi_j)
    heads[heads < 0] += 360.0
    
    idx_rear = (heads >= 150) & (heads <= 210) & (elevs >= -25) & (elevs <= 25)
    headings_rear = heads[idx_rear]
    elevations_rear = elevs[idx_rear]
    inputs_rear = inputs_all[idx_rear]
    g_exact_rear = g_exact_all[idx_rear]

    
    # 2. Setup Base Vectors
    import sys, os
    sys_path_save = sys.path.copy()
    sys.path.append(os.path.abspath(r'D:\UAV Internship project\Stage 12 (Operational Analysis)'))
    sys.path.append(os.path.abspath(r'D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)'))
    from conformal_array import get_conformal_array_parametric
    from mesh_loader import load_uav_mesh
    from pathlib import Path
    from attitude import rotate_points, euler_to_quaternion
    mesh = load_uav_mesh(Path(r'D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone_prepared.stl'))
    pos_body, _ = get_conformal_array_parametric(mesh, N=32)
    pos_lambda_np = pos_body / 0.15
    q_inv = euler_to_quaternion(np.deg2rad(15.0), 0, 0).conjugate()
    pos_lambda = torch.tensor(pos_lambda_np, dtype=torch.float64, device=device)
    
    sig_world = np.array([1.0, 0.0, 0.0])
    sig_body = rotate_points(sig_world.reshape(1, 3), q_inv)[0]
    
    # 3. True Physics for Evaluation
    sys.path.append(os.path.abspath(r'D:\UAV Internship project\Phase 2 Track 1\em_realism'))
    from stage15d_train_cartesian_32 import CartesianShadowNet32
    shadow_net = CartesianShadowNet32().to(device)
    shadow_net.load_state_dict(torch.load('shadow_net_cartesian_32.pt', map_location=device))
    shadow_net.eval()
    
    sig_t = torch.tensor(sig_body, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        out_sig = shadow_net(sig_t)
        g_sig_raw = (out_sig[:, 0::2] + 1j * out_sig[:, 1::2]).to(torch.complex128)
        g_sig = g_sig_raw / torch.clamp(torch.abs(g_sig_raw), min=1.0)
    
    def get_steering_vector_pt(pos_lambda, theta, phi):
        dx = torch.sin(theta) * torch.cos(phi)
        dy = torch.sin(theta) * torch.sin(phi)
        dz = torch.cos(theta)
        d = torch.stack([dx, dy, dz], dim=-1)
        return torch.exp(1j * 2.0 * torch.pi * (d @ pos_lambda.T))
        
    theta_s = torch.acos(sig_t[:, 2] / torch.norm(sig_t, dim=1)).to(torch.float64)
    phi_s = torch.atan2(sig_t[:, 1], sig_t[:, 0]).to(torch.float64)
    v_sig_masked = g_sig * get_steering_vector_pt(pos_lambda, theta_s, phi_s)
    
    inputs_t = torch.tensor(inputs_rear, dtype=torch.float32, device=device)
    theta_j = torch.acos(inputs_t[:, 2].to(torch.float64) / torch.norm(inputs_t.to(torch.float64), dim=1))
    phi_j = torch.atan2(inputs_t[:, 1].to(torch.float64), inputs_t[:, 0].to(torch.float64))
    v_ideal = get_steering_vector_pt(pos_lambda, theta_j, phi_j)
    g_exact_t = torch.tensor(g_exact_rear, dtype=torch.complex128, device=device)
    v_true = g_exact_t * v_ideal
    
    # 4. AI Prediction
    model = ReluCovariancePredictor(K_rank=5, hidden_dim=512).to(device)
    model.load_state_dict(torch.load('relu_beamformer_d3_cov_K5_100k_w512.pt', map_location=device))
    model.eval()
    with torch.no_grad():
        U = model(inputs_t).to(torch.complex128)
        
    # 5. Hybrid Evaluation
    print("Generating Offline Lookup Table...")
    grid_h, grid_e, grid_v = get_hybrid_steering_vectors(headings_rear, elevations_rear, U)
    
    print("Evaluating Hybrid System...")
    sinrs = []
    
    for i in range(len(inputs_rear)):
        h = headings_rear[i]
        e = elevations_rear[i]
        
        # 1. AI Weight
        Ui = U[i]
        R_in_ai = 10000.0 * (Ui @ torch.conj(Ui.T)) + 390.0 * torch.eye(32, dtype=torch.complex128, device=device)
        w_ai_raw = torch.linalg.inv(R_in_ai) @ v_sig_masked[0]
        w_ai = w_ai_raw / (torch.conj(v_sig_masked[0]) @ w_ai_raw)
        
        w_oracle_weight = get_blend_weights(h, e)
        
        if w_oracle_weight > 0.0:
            # 2. Oracle Weight
            v_oracle_interp = phase_aware_bilinear_interpolate(np.array([h]), np.array([e]), grid_h, grid_e, grid_v)[0]
            v_oracle_t = torch.tensor(v_oracle_interp, dtype=torch.complex128, device=device)
            R_in_oracle = 10000.0 * torch.outer(v_oracle_t, torch.conj(v_oracle_t)) + 390.0 * torch.eye(32, dtype=torch.complex128, device=device)
            w_or_raw = torch.linalg.inv(R_in_oracle) @ v_sig_masked[0]
            w_or = w_or_raw / (torch.conj(v_sig_masked[0]) @ w_or_raw)
            
            # Blend weights directly
            w = (1.0 - w_oracle_weight) * w_ai + w_oracle_weight * w_or
            w = w / (torch.conj(v_sig_masked[0]) @ w)
        else:
            w = w_ai
            
        Ps = 100.0 * torch.abs(torch.conj(w) @ v_sig_masked[0])**2
        Pj = 10000.0 * torch.abs(torch.conj(w) @ v_true[i])**2
        Pn = torch.real(torch.conj(w) @ w)
        sinrs.append(10 * np.log10(float(Ps / (Pj + Pn))))
        
    sinrs = np.array(sinrs)
    np.savez('hybrid_sinrs.npz', sinrs=sinrs, headings=headings_rear, elevations=elevations_rear)
    print("Saved to hybrid_sinrs.npz")

if __name__ == '__main__':
    main()
