"""Task A2: Two-Ray Ground Reflection Model.

This script implements the two-ray ground reflection model, comparing it against
the free-space path loss (FSPL) baseline. It computes path loss vs. range for
different polarization states and generates a 2D received power heatmap in the
mission area.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# 1. PARAMETERS (FROZEN FROM TASK A5 BASELINE)
# ==============================================================================
P_tx_dBm = 37.0         # UAV TX Power: 37 dBm (5 W)
G_tx_dBi = 3.0          # UAV TX Antenna Gain: 3 dBi
G_rx_dBi = 10.0         # GCS RX Antenna Gain: 10 dBi
f_c = 2.4e9             # Carrier Frequency: 2.4 GHz
c = 3.0e8               # Speed of light: 3e8 m/s
h_GS = 2.0              # Ground station antenna height: 2 m
h_UAV = 100.0           # Nominal UAV altitude: 100 m
epsilon_r = 15.0        # Relative permittivity of typical ground (dry soil/suburban)

# Derived variables
wavelength = c / f_c    # 0.125 m
d_c = (4.0 * h_GS * h_UAV) / wavelength  # Crossover distance: 6400 m

# ==============================================================================
# 2. TWO-RAY POWER EQUATION IMPLEMENTATION
# ==============================================================================
def fspl_db(d_m):
    """Free Space Path Loss in dB."""
    return 40.05 + 20.0 * np.log10(np.maximum(d_m, 0.1))

def two_ray_path_loss(d_horizontal, h_gs, h_uav, f, eps_r=15.0, polarization='vertical'):
    """Calculate the Two-Ray Ground Reflection Path Loss.
    
    Args:
        d_horizontal: Horizontal distance in meters.
        h_gs: Height of ground station in meters.
        h_uav: Height of UAV in meters.
        f: Frequency in Hz.
        eps_r: Relative permittivity of ground.
        polarization: 'vertical' or 'horizontal'.
    """
    wl = c / f
    k = 2.0 * np.pi / wl
    
    d_dir = np.sqrt(d_horizontal**2 + (h_uav - h_gs)**2)
    d_ref = np.sqrt(d_horizontal**2 + (h_uav + h_gs)**2)
    
    # Grazing angle theta
    sin_theta = (h_uav + h_gs) / d_ref
    cos_theta = d_horizontal / d_ref
    
    # Fresnel reflection coefficients
    sqrt_term = np.sqrt(eps_r - cos_theta**2 + 0j)
    
    if polarization == 'horizontal':
        Gamma = (sin_theta - sqrt_term) / (sin_theta + sqrt_term)
    else:  # vertical
        Gamma = (eps_r * sin_theta - sqrt_term) / (eps_r * sin_theta + sqrt_term)
        
    delta_phi = k * (d_ref - d_dir)
    
    # Received power term combining direct and reflected rays
    # |1 + Gamma * e^(-j*delta_phi)|^2
    interference = np.abs(1.0 + Gamma * np.exp(-1j * delta_phi))**2
    
    # Free-space amplitude at direct distance
    fs_amp = wl / (4.0 * np.pi * d_dir)
    
    # Total path loss in dB
    # PL = -10 * log10(fs_amp^2 * interference)
    pl_db = -10.0 * np.log10(fs_amp**2 * interference + 1e-20)
    return pl_db

def two_ray_path_loss_ideal(d_horizontal, h_gs, h_uav, f):
    """Calculate Two-Ray path loss using ideal reflector (Gamma = -1)."""
    wl = c / f
    k = 2.0 * np.pi / wl
    
    d_dir = np.sqrt(d_horizontal**2 + (h_uav - h_gs)**2)
    d_ref = np.sqrt(d_horizontal**2 + (h_uav + h_gs)**2)
    
    delta_phi = k * (d_ref - d_dir)
    interference = np.abs(1.0 - np.exp(-1j * delta_phi))**2
    
    fs_amp = wl / (4.0 * np.pi * d_dir)
    pl_db = -10.0 * np.log10(fs_amp**2 * interference + 1e-20)
    return pl_db

# ==============================================================================
# 3. PLOTTING AND GRID HEATMAP COMPUTATION
# ==============================================================================
def run_two_ray_analysis():
    print("=" * 80)
    print("                      RUNNING TASK A2: TWO-RAY BASELINE ANALYSIS")
    print("=" * 80)
    
    # Ranges for curve plotting (100 m to 20 km)
    ranges = np.linspace(100.0, 20000.0, 1000)
    
    # Compute path loss curves
    pl_fspl = fspl_db(np.sqrt(ranges**2 + (h_UAV - h_GS)**2))
    pl_2r_v = [two_ray_path_loss(r, h_GS, h_UAV, f_c, epsilon_r, 'vertical') for r in ranges]
    pl_2r_h = [two_ray_path_loss(r, h_GS, h_UAV, f_c, epsilon_r, 'horizontal') for r in ranges]
    pl_2r_ideal = [two_ray_path_loss_ideal(r, h_GS, h_UAV, f_c) for r in ranges]
    
    # Setup Figure (2 panels: Path Loss comparison and received power heatmap)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Panel 1: Path Loss vs. Range
    ax1.plot(ranges / 1000.0, pl_fspl, label="Free Space Path Loss (FSPL)", color="black", linestyle="--", linewidth=1.5)
    ax1.plot(ranges / 1000.0, pl_2r_v, label="Two-Ray (Vertical Polarization, $\\epsilon_r=15$)", color="red", linewidth=1.5)
    ax1.plot(ranges / 1000.0, pl_2r_h, label="Two-Ray (Horizontal Polarization, $\\epsilon_r=15$)", color="orange", linewidth=1.5)
    ax1.plot(ranges / 1000.0, pl_2r_ideal, label="Two-Ray (Ideal Reflector, $\\Gamma = -1$)", color="blue", alpha=0.5, linewidth=1.2)
    
    # Draw reference line for Crossover distance d_c
    ax1.axvline(d_c / 1000.0, color="purple", linestyle=":", linewidth=2, label=f"Crossover Distance $d_c$ = {d_c/1000:.2f} km")
    
    # Highlight R*_comm
    ax1.axvline(1.037, color="green", linestyle="-.", linewidth=1.5, label="Tactical $R^*_{comm}$ (1.037 km)")
    
    ax1.set_title("Path Loss vs. Range Comparison: FSPL vs. Two-Ray model", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xlabel("Horizontal Range to GCS $d$ [km]", fontsize=11)
    ax1.set_ylabel("Path Loss [dB]", fontsize=11)
    ax1.grid(True, which="both", linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right", frameon=True, facecolor="white")
    ax1.set_xlim(0.1, 20.0)
    ax1.set_ylim(70, 160)
    
    # Panel 2: Received Power Heatmap in 2D Mission Area using Two-Ray (Vertical Polarization)
    grid_size = 10000.0 # 10 km square
    x_vals = np.linspace(-grid_size, grid_size, 250)
    y_vals = np.linspace(-grid_size, grid_size, 250)
    X_grid, Y_grid = np.meshgrid(x_vals, y_vals)
    
    # Compute horizontal range from GCS at (0, 0)
    R_grid = np.sqrt(X_grid**2 + Y_grid**2)
    
    # Flatten grid for fast function evaluation, then reshape back
    r_flat = R_grid.flatten()
    pl_flat = np.array([two_ray_path_loss(r, h_GS, h_UAV, f_c, epsilon_r, 'vertical') for r in r_flat])
    PL_grid = pl_flat.reshape(R_grid.shape)
    
    # Received signal power in dBm = P_tx + G_tx + G_rx - PL
    P_rx_grid = P_tx_dBm + G_tx_dBi + G_rx_dBi - PL_grid
    
    # Plot heatmap
    im = ax2.imshow(P_rx_grid, extent=[-grid_size/1000, grid_size/1000, -grid_size/1000, grid_size/1000],
                    cmap="jet", origin="lower", vmin=-80.0, vmax=-20.0)
    
    cbar = fig.colorbar(im, ax=ax2, pad=0.02)
    cbar.set_label("Received Power $P_{rx}$ [dBm]", fontsize=11)
    
    # Draw contours of constant received power
    contours = ax2.contour(X_grid/1000.0, Y_grid/1000.0, P_rx_grid, levels=[-75.0, -65.0, -55.0, -45.0, -35.0],
                           colors="white", linestyles="--", linewidths=0.8)
    ax2.clabel(contours, inline=True, fmt="%1.0f dBm", fontsize=8, colors="white")
    
    # Draw GCS at center
    ax2.scatter(0, 0, color="yellow", marker="*", s=200, edgecolor="black", zorder=5, label="GCS (0,0)")
    
    # Draw 1.037 km handover circle
    circle = plt.Circle((0, 0), 1.037, color="white", fill=False, linestyle="-.", linewidth=1.5, label="Tactical $R^*_{comm}$ (1.037 km)")
    ax2.add_patch(circle)
    
    ax2.set_title("GCS Received Power Heatmap (2D Mission Area) under Two-Ray model\n(UAV Altitude = 100 m AGL, Vertical Polarization)",
                 fontsize=12, fontweight="bold", pad=10)
    ax2.set_xlabel("X coordinate [km]", fontsize=11)
    ax2.set_ylabel("Y coordinate [km]", fontsize=11)
    ax2.set_xlim(-10, 10)
    ax2.set_ylim(-10, 10)
    ax2.legend(loc="upper right", frameon=True, facecolor="white")
    
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Stage 2", "phase_a_task_a2_two_ray.png")
    plt.savefig(output_path, dpi=300)
    print(f"\n[A2 SUCCESS] Generated two-ray analysis and saved plot to: {output_path}")

if __name__ == "__main__":
    run_two_ray_analysis()
