import re
with open('stage18m_eval_dense.py', 'r') as f:
    content = f.read()
content = content.replace('model = SIRENCovariancePredictor(w0=30.0, K_rank=5).to(device)', 'model = SIRENCovariancePredictor(w0=30.0, K_rank=5, hidden_dim=512).to(device)')
content = content.replace('siren_beamformer_d3_cov_K5_3D_dense.pt', 'siren_beamformer_d3_cov_K5_3D_w30_h512.pt')
content = content.replace('diag_180deg_pilot_3D_dense.png', 'diag_180deg_pilot_3D_w30_h512.png')
content = re.sub(r'print\(f\'El.*?\)', 'print(f\"El {el:>5.1f} | Min: {min_sinr:>5.2f} | Below 15dB: {np.sum(sinrs < 15.0)}/601\")', content)
with open('stage18q_eval_capacity_w30.py', 'w') as f:
    f.write(content)
