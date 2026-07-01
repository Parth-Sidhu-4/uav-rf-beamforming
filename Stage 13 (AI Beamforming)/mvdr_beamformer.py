import numpy as np

def compute_mvdr_weights_robust(v_sig, v_jam):
    JAM_POW = 10000.0  # 40 dB
    NOISE_POW = 1.0
    
    R_j = JAM_POW * np.outer(v_jam, np.conj(v_jam))
    R_n = NOISE_POW * np.eye(len(v_jam))
    R_xx = R_j + R_n
    
    R_reg = R_xx + 1e-6 * np.eye(R_xx.shape[0])
    try:
        R_inv = np.linalg.inv(R_reg)
    except np.linalg.LinAlgError:
        R_inv = np.linalg.pinv(R_reg)

    num = R_inv @ v_sig
    den = np.conj(v_sig) @ num
    return num / max(abs(den), 1e-12)

def compute_sinr(w, v_sig, v_jam):
    SIG_POW = 100.0   # 20 dB
    JAM_POW = 10000.0 # 40 dB
    NOISE_POW = 1.0
    
    R_s = SIG_POW * np.outer(v_sig, np.conj(v_sig))
    R_j = JAM_POW * np.outer(v_jam, np.conj(v_jam))
    R_n = NOISE_POW * np.eye(len(v_jam))
    
    S = np.real(np.conj(w) @ R_s @ w)
    N_J = np.real(np.conj(w) @ (R_j + R_n) @ w)
    return S / max(N_J, 1e-12)
