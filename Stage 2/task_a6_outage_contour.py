"""Task A6: Range-Altitude Outage Contour Map.

This script consumes the frozen communication, jamming, and fading parameters
validated in Task A5 to generate Range-Altitude Outage Contour Maps and risk
classifications across the 2D grid of UAV horizontal range and altitude.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import chndtr

# ==============================================================================
# 1. FROZEN SYSTEM PARAMETERS (FROM TASK A5)
# ==============================================================================
P_tx_dBm = 37.0         # UAV TX Power: 37 dBm (5 W)
G_tx_dBi = 3.0          # UAV TX Antenna Gain: 3 dBi
G_rx_dBi = 10.0         # GCS RX Antenna Gain: 10 dBi (for desired signal)
B_Hz = 500e3            # Receiver/Hop Bandwidth: 500 kHz
NF_dB = 6.0             # Receiver Noise Figure: 6 dB
f_c_GHz = 2.4           # Carrier Frequency: 2.4 GHz
gamma_th_dB = 10.0      # Demodulation SNR threshold: 10.0 dB

# Derived quantities
P_N_dBm = -174.0 + 10 * np.log10(B_Hz) + NF_dB  # Noise floor: -111.0 dBm
gamma_th_lin = 10 ** (gamma_th_dB / 10.0)
P_N_lin = 10 ** (P_N_dBm / 10.0)

# FHSS Parameters
B_ss_Hz = 40e6          # Hopping bandwidth: 40 MHz
PG_dB = 10 * np.log10(B_ss_Hz / B_Hz)  # Processing Gain: 19.03 dB
PG_lin = 10 ** (PG_dB / 10.0)

# Jammer Baseline (G_rx,jam = 0 dBi, fixed standoff)
G_rx_jam = 0.0          # Frozen GCS receiver antenna gain for jammer
JAMMER_CLASSES = {
    "Class I":  {"eirp_dBm": 73.0, "standoff_km": 15.0},
    "Class II": {"eirp_dBm": 73.0, "standoff_km": 5.0}
}

# Elevation angle geometry
h_GS = 2.0              # Ground station antenna height: 2 m
K_a = 2.3
K_b = 0.035

# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================
def fspl_db(d_m):
    """Free Space Path Loss at 2.4 GHz for distance d in meters."""
    return 40.05 + 20.0 * np.log10(d_m)

def k_factor_suburban(theta_rad):
    """Suburban K-factor model (theta in radians)."""
    theta_deg = np.degrees(theta_rad)
    return K_a * np.exp(K_b * theta_deg)

def compute_rician_pout(mean_sinr_lin, K_val):
    """Compute Rician outage probability using the chndtr CDF."""
    a2 = 2.0 * K_val
    b2 = 2.0 * (K_val + 1.0) * gamma_th_lin / mean_sinr_lin
    return chndtr(b2, 2, a2)

def get_defeat_range_horizontal(h_uav, standoff_km, jammer_eirp):
    """Numerically solve for horizontal defeat range where mean SINR = 10.0 dB."""
    # Received jammer power (fixed standoff)
    L_j = fspl_db(standoff_km * 1000.0)
    J_dBm = jammer_eirp + G_rx_jam - L_j
    J_after_PG_lin = 10 ** ((J_dBm - PG_dB) / 10.0)

    # Required signal power at antenna terminal
    # SINR = S_rx_lin / (J_after_PG_lin + P_N_lin) = 10.0 (gamma_th)
    S_rx_lin_req = gamma_th_lin * (J_after_PG_lin + P_N_lin)
    S_rx_dBm_req = 10 * np.log10(S_rx_lin_req)

    # Required FSPL
    # S_rx = P_tx + G_tx + G_rx - L_FS
    # L_FS_req = P_tx + G_tx + G_rx - S_rx_req
    L_FS_req = P_tx_dBm + G_tx_dBi + G_rx_dBi - S_rx_dBm_req

    # Required slant range
    d_slant = 10 ** ((L_FS_req - 40.05) / 20.0)
    
    # Horizontal range
    dh = h_uav - h_GS
    if d_slant > abs(dh):
        return np.sqrt(d_slant**2 - dh**2)
    return 0.0

# ==============================================================================
# 3. RANGESweep GRID COMPUTATION & REGION CLASSIFICATION
# ==============================================================================
def generate_contour_maps():
    print("=" * 80)
    print("                 RANGE-ALTITUDE OUTAGE CONTOUR MAP GENERATION")
    print("=" * 80)

    # Define the 2D grid
    ranges_m = np.linspace(100, 10000, 400)
    altitudes_m = np.linspace(10, 500, 200)
    R_grid, H_grid = np.meshgrid(ranges_m, altitudes_m)

    # Setup the figure
    fig, axes = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

    for idx, (jam_name, jam_params) in enumerate(JAMMER_CLASSES.items()):
        ax = axes[idx]

        # Standoff and received jammer power
        L_j = fspl_db(jam_params["standoff_km"] * 1000.0)
        J_dBm = jam_params["eirp_dBm"] + G_rx_jam - L_j
        J_after_PG_lin = 10 ** ((J_dBm - PG_dB) / 10.0)

        # Compute slant range and elevation angle for the 2D grid
        dh_grid = H_grid - h_GS
        d_slant_grid = np.sqrt(R_grid**2 + dh_grid**2)
        theta_grid = np.arctan(dh_grid / R_grid)

        # Compute signal power and mean SINR
        S_dBm_grid = P_tx_dBm + G_tx_dBi + G_rx_dBi - fspl_db(d_slant_grid)
        S_lin_grid = 10 ** (S_dBm_grid / 10.0)
        mean_sinr_lin_grid = S_lin_grid / (J_after_PG_lin + P_N_lin)
        mean_sinr_dB_grid = 10 * np.log10(mean_sinr_lin_grid)

        # Compute K-factor and Rician Outage Probability
        K_grid = k_factor_suburban(theta_grid)
        pout_grid = compute_rician_pout(mean_sinr_lin_grid, K_grid)

        # 1. Filled Contour Map for Risk Zoning
        # Zones: Green (<5%), Yellow (5-50%), Orange (50-95%), Red (>95%)
        levels = [0.0, 0.05, 0.50, 0.95, 1.0]
        colors = ["#D4EDDA", "#FFF3CD", "#FFE8D6", "#F8D7DA"]  # Soft Green, Yellow, Orange, Red
        
        # Plot filled contours
        contour_filled = ax.contourf(R_grid, H_grid, pout_grid, levels=levels, colors=colors)

        # 2. Overlay Mean SINR contours (e.g. 20, 15, 10, 5, 0 dB)
        sinr_levels = [0.0, 5.0, 10.0, 15.0, 20.0]
        sinr_contours = ax.contour(R_grid, H_grid, mean_sinr_dB_grid, levels=sinr_levels,
                                   colors="black", linestyles=":", linewidths=1.2)
        ax.clabel(sinr_contours, inline=True, fmt="%1.0f dB", fontsize=9, colors="black")

        # 3. Highlight the Mean-SINR Threshold Boundary (SINR = 10 dB)
        # It should exactly overlay the SINR = 10 dB contour line
        ax.contour(R_grid, H_grid, mean_sinr_dB_grid, levels=[10.0],
                   colors="teal", linestyles="-", linewidths=2.5)

        # 4. Plot the Tactical Handover Reference Point (1037 m, 100 m)
        ax.scatter(1037.0, 100.0, color="purple", marker="*", s=150, zorder=5,
                   label="Tactical Handover Point ($1037$ m, $100$ m)")

        # Legend and Labels
        ax.set_title(f"Range-Altitude Outage Contour Map for {jam_name} barrage jammer\n(Standoff = {jam_params['standoff_km']} km, EIRP = {jam_params['eirp_dBm']} dBm, $G_{{rx,jam}} = 0$ dBi)",
                     fontsize=12, fontweight="bold", pad=10)
        ax.set_ylabel("UAV Altitude $h$ [meters AGL]", fontsize=11)
        ax.set_xlim(100, 10000)
        ax.set_ylim(10, 500)
        ax.grid(True, which="both", linestyle="--", alpha=0.3)

        # Custom Legend entries
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#D4EDDA", edgecolor="#2CA02C", label="Green Zone ($P_{out} < 5\%$)"),
            Patch(facecolor="#FFF3CD", edgecolor="#FF7F0E", label="Yellow Zone ($5\% \\leq P_{out} < 50\%$)"),
            Patch(facecolor="#FFE8D6", edgecolor="#E07A5F", label="Orange Zone ($50\% \\leq P_{out} < 95\%$)"),
            Patch(facecolor="#F8D7DA", edgecolor="#D62728", label="Red Zone ($P_{out} \\geq 95\%$)"),
            plt.Line2D([0], [0], color="teal", lw=2.5, label="Mean-SINR Threshold Boundary ($\\overline{\\text{SINR}} = 10\\text{ dB}$)"),
            plt.Line2D([0], [0], color="purple", marker="*", ls="", ms=12, label="Tactical Handover Point ($1037$ m, $100$ m)"),
            plt.Line2D([0], [0], color="black", ls=":", lw=1.2, label="Mean SINR Contours (dB)")
        ]
        ax.legend(handles=legend_elements, loc="upper right", frameon=True, facecolor="white", edgecolor="none", shadow=False)

    axes[1].set_xlabel("UAV Horizontal Range to GCS $d$ [meters]", fontsize=11)
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), "phase_a_task_a6_outage_contour.png")
    plt.savefig(output_path, dpi=300)
    print(f"\n[A] SAVED RANGE-ALTITUDE CONTOUR PLOT: {output_path}")

# ==============================================================================
# 4. ALTITUDE VS. DEFEAT-RANGE SUMMARY TABLE
# ==============================================================================
def print_altitude_defeat_table():
    print("\n" + "=" * 80)
    print("                  ALTITUDE VS. NUMERICAL DEFEAT RANGE TABLE")
    print("=" * 80)
    
    altitudes = [50.0, 100.0, 200.0, 300.0]
    
    print(f"{'Altitude (m)':<15} | {'Class I Defeat Range (m)':^30} | {'Class II Defeat Range (m)':^30}")
    print("-" * 81)
    
    for h in altitudes:
        d_I = get_defeat_range_horizontal(h, JAMMER_CLASSES["Class I"]["standoff_km"], JAMMER_CLASSES["Class I"]["eirp_dBm"])
        d_II = get_defeat_range_horizontal(h, JAMMER_CLASSES["Class II"]["standoff_km"], JAMMER_CLASSES["Class II"]["eirp_dBm"])
        print(f"{h:<15.1f} | {d_I:^30.1f} | {d_II:^30.1f}")
        
    print("-" * 81)
    print("* Note: Ranges solved numerically from the exact equation SINR(d, h) = 10.0 dB.")

if __name__ == "__main__":
    generate_contour_maps()
    print_altitude_defeat_table()
