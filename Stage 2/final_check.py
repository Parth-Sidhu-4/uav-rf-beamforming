import numpy as np

# System parameters
P_tx_dBm = 37.0         # UAV TX Power: 37 dBm (5 W)
G_tx_dBi = 3.0          # UAV TX Antenna Gain: 3 dBi
G_rx_dBi = 10.0         # GCS RX Antenna Gain: 10 dBi (for desired signal)
B_Hz = 500e3            # Receiver/Hop Bandwidth: 500 kHz
NF_dB = 6.0             # Receiver Noise Figure: 6 dB
B_ss_Hz = 40e6          # Hopping bandwidth: 40 MHz
PG_dB = 10 * np.log10(B_ss_Hz / B_Hz)  # Processing Gain: 19.03 dB

P_N_dBm = -174.0 + 10 * np.log10(B_Hz) + NF_dB  # Noise floor: -111.0 dBm
P_N_lin = 10 ** (P_N_dBm / 10.0)

# Class II Jammer parameters
standoff_km = 5.0
eirp_dB = 73.0
G_rx_jam = 0.0          # Baseline

# Geometry
h_GS = 2.0
h_UAV = 100.0
dh = h_UAV - h_GS

# Horizontal range to check (numerical defeat range in A6 at 100 m altitude)
d_horizontal = 996.271  # solved as sqrt(1001.077**2 - 98**2)

def fspl_db(d_m):
    return 40.05 + 20.0 * np.log10(d_m)

# 1. Compute slant range
d_slant = np.sqrt(d_horizontal**2 + dh**2)

# 2. Compute signal power at GCS
L_s = fspl_db(d_slant)
S_dBm = P_tx_dBm + G_tx_dBi + G_rx_dBi - L_s
S_lin = 10 ** (S_dBm / 10.0)

# 3. Compute jammer power at GCS after processing gain
L_j = fspl_db(standoff_km * 1000.0)
J_dBm = eirp_dB + G_rx_jam - L_j
J_after_PG_dBm = J_dBm - PG_dB
J_after_PG_lin = 10 ** (J_after_PG_dBm / 10.0)

# 4. Compute mean SINR
mean_sinr_lin = S_lin / (J_after_PG_lin + P_N_lin)
mean_sinr_dB = 10.0 * np.log10(mean_sinr_lin)

print("="*60)
print(f"VERIFICATION AT d_horizontal = {d_horizontal:.3f} m, h_UAV = {h_UAV:.1f} m")
print(f"Slant range d_slant = {d_slant:.4f} m (expected slant range ~1001.077 m)")
print(f"Signal path loss L_s = {L_s:.4f} dB")
print(f"Received signal S = {S_dBm:.4f} dBm")
print(f"Jammer path loss L_j = {L_j:.4f} dB")
print(f"Received jammer J = {J_dBm:.4f} dBm")
print(f"Effective jammer J_eff (after PG) = {J_after_PG_dBm:.4f} dBm")
print(f"Noise Floor N = {P_N_dBm:.4f} dBm")
print(f"Mean SINR = {mean_sinr_dB:.6f} dB")
print(f"Difference from 10.0 dB = {mean_sinr_dB - 10.0:.6f} dB")
print("="*60)
