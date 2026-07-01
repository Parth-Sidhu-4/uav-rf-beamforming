import sys, os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))
from urban_ray_tracer import UrbanRayTracer2D, RayTracerConfig, Building


def main():
    print("Running Category 4b: Urban Ray-Tracing Channel Model\n")
    
    cfg = RayTracerConfig(freq_hz=2.4e9)
    lam = 3e8 / cfg.freq_hz
    
    # Check 1: Free-space limit
    tracer_empty = UrbanRayTracer2D([], cfg)
    tx = np.array([0.0, 50.0])
    rx = np.array([1000.0, 50.0])
    
    pr_rt = tracer_empty.received_power_dbm(30.0, tx, rx)
    
    dist = 1000.0
    fspl = 20*np.log10(dist) + 20*np.log10(cfg.freq_hz) - 147.55
    pr_friis = 30.0 - fspl
    
    print("[Check 1] Free-space limit (1000m):")
    print(f"  Ray Tracer Pr: {pr_rt:.2f} dBm")
    print(f"  Friis Pr:      {pr_friis:.2f} dBm")
    if abs(pr_rt - pr_friis) < 0.1:
        print("  PASS: Ray tracer matches Friis formula in empty space.\n")
    else:
        print("  FAIL: Ray tracer does not match Friis.\n")

    # Check 2: Two-ray model (One flat ground reflection)
    # We model the ground as a very long building just below y=0.
    ground = Building(x_min=-5000, y_min=-10, x_max=500000, y_max=0)
    tracer_2ray = UrbanRayTracer2D([ground], cfg)
    
    # Sweep distance and observe roll-off past crossover distance
    # Crossover distance dc = 4*htx*hrx / lambda = 4*50*50 / 0.125 = 80,000 m
    distances = np.logspace(4, 5.5, 100) # 10km to 316km
    pr_2ray = []
    pr_fspl = []
    
    # Setup plotting
    plt.figure(figsize=(10, 6))
    
    h_tx = 50.0
    h_rx = 50.0
    
    for d in distances:
        tx_p = np.array([0.0, h_tx])
        rx_p = np.array([d, h_rx])
        
        pr = tracer_2ray.received_power_dbm(30.0, tx_p, rx_p)
        pr_2ray.append(pr)
        
        fspl_d = 20*np.log10(d) + 20*np.log10(cfg.freq_hz) - 147.55
        pr_fspl.append(30.0 - fspl_d)
        
    pr_2ray = np.array(pr_2ray)
    pr_fspl = np.array(pr_fspl)
    
    # Calculate empirical slope at large distances (e.g., last 10 points)
    d_log = np.log10(distances[-10:])
    p_log = pr_2ray[-10:] / 10.0
    slope = np.polyfit(d_log, p_log, 1)[0] * 10
    
    print("[Check 2] Two-ray model roll-off at large d:")
    print(f"  Empirical slope: {slope:.1f} dB/decade")
    # Expected slope for 2-ray model is -40 dB/decade
    if slope < -35.0:
        print("  PASS: Slope approaches d^-4.\n")
    else:
        print("  FAIL: Slope does not match 2-ray expectations.\n")

    plt.semilogx(distances, pr_fspl, 'k--', label='Free Space (d^-2)')
    plt.semilogx(distances, pr_2ray, 'b-', label='2-Ray Ground Model')
    plt.xlabel('Distance (m)')
    plt.ylabel('Received Power (dBm)')
    plt.title('4b: 2-Ray Propagation Model vs Free Space')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.savefig('cat4b_2ray.png')
    print("Saved cat4b_2ray.png")

    # Check 3: K-factor estimate
    tx = np.array([0.0, 50.0])
    rx = np.array([1000.0, 50.0])
    
    delays, amps = tracer_2ray.compute_cir(tx, rx)
    k_factor_lin = tracer_2ray.rician_k_from_cir(delays, amps)
    k_factor_db = 10 * np.log10(k_factor_lin + 1e-9)
    
    print("\n[Check 3] K-factor estimate (2 paths):")
    print(f"  LOS amplitude: {abs(amps[0]):.2e}")
    if len(amps) > 1:
        print(f"  Reflected amplitude: {abs(amps[1]):.2e}")
        analytical_k = (abs(amps[0])**2) / (abs(amps[1])**2)
        print(f"  Analytical K: {10*np.log10(analytical_k):.2f} dB")
        print(f"  Measured K:   {k_factor_db:.2f} dB")
        if abs(10*np.log10(analytical_k) - k_factor_db) < 0.1:
            print("  PASS: K-factor matches analytical ratio.\n")
        else:
            print("  FAIL: K-factor mismatch.\n")
    else:
        print("  FAIL: Reflected path not found.\n")


if __name__ == "__main__":
    main()
