import numpy as np
import matplotlib.pyplot as plt
import task_b_jamming_threats as jts

def get_gamma(d_m):
    Ps = jts.get_received_signal_power(d_m, 'two-ray')
    P_j = jts.get_received_jammer_power('fspl')
    SNR_lin = Ps / jts.P_N_lin
    INR_lin = P_j / jts.P_N_lin
    
    # Rician K factor
    theta_elev = np.arctan(jts.h_UAV / max(d_m, 1.0))
    K_s_db = 10.0 + 10.0 * np.sin(theta_elev)
    K_s = 10 ** (K_s_db / 10.0)
    
    # Mean h_s^2 is 1.0. We just evaluate deterministic mean
    sig_out = SNR_lin
    jam_out = INR_lin
    noise_out = 1.0
    gamma_db = 10.0 * np.log10(max(sig_out / (noise_out + jam_out), 1e-15)) + 19.03
    return gamma_db

distances = np.arange(5000.0, 10.0, -10.0)
gammas = [get_gamma(d) for d in distances]

for d, g in zip(distances, gammas):
    if d in [5000, 3000, 2000, 1500, 1000, 790, 750, 500]:
        print(f"Distance {d}m: Gamma = {g:.2f} dB")
