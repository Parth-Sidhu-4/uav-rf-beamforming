import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(__file__))
import phase_b_beamforming as pbb

def generate_covariance(scenario, N, theta_j, L, rng=None):
    if rng is None: rng = np.random.default_rng()
    sigma_n2 = 1.0
    
    if scenario == 'A':
        # Direct Path
        INR_dB = 10.0
        sigma_j2 = 10.0**(INR_dB/10.0)
        j_sig = rng.normal(0, np.sqrt(sigma_j2/2.0), L) + 1j * rng.normal(0, np.sqrt(sigma_j2/2.0), L)
        a_j = pbb.ula_steering_vector(N, theta_j)
        X = np.outer(a_j, j_sig)
        
    elif scenario == 'B':
        # Shadowing Attenuation (20 dB loss)
        INR_dB = -10.0 
        sigma_j2 = 10.0**(INR_dB/10.0)
        j_sig = rng.normal(0, np.sqrt(sigma_j2/2.0), L) + 1j * rng.normal(0, np.sqrt(sigma_j2/2.0), L)
        a_j = pbb.ula_steering_vector(N, theta_j)
        X = np.outer(a_j, j_sig)
        
    elif scenario == 'C':
        # Coherent Multipath (Direct + Diffracted/Reflected path)
        INR_dB = 10.0
        sigma_j2 = 10.0**(INR_dB/10.0)
        # Same exact source signal for both paths (fully coherent)
        j_sig = rng.normal(0, np.sqrt(sigma_j2/2.0), L) + 1j * rng.normal(0, np.sqrt(sigma_j2/2.0), L)
        
        a_j1 = pbb.ula_steering_vector(N, theta_j)
        # Multipath arriving from 5 degrees off, attenuated by 3 dB, random phase shift
        theta_m = theta_j + np.radians(5.0)
        a_j2 = pbb.ula_steering_vector(N, theta_m)
        
        phase_shift = np.exp(1j * np.pi / 4.0)
        alpha = 10.0**(-3.0/20.0) # -3 dB amplitude
        
        # Superposition of perfectly coherent paths
        X = np.outer(a_j1, j_sig) + np.outer(a_j2, j_sig * alpha * phase_shift)
        
    else:
        raise ValueError("Invalid scenario")
        
    noise = rng.normal(0, np.sqrt(sigma_n2/2.0), (N, L)) + 1j * rng.normal(0, np.sqrt(sigma_n2/2.0), (N, L))
    X_total = X + noise
    R_xx = (X_total @ X_total.conj().T) / max(L, 1)
    return R_xx

def run_c4a_audit():
    print("Running Task C4.a: Terrain Model Feasibility Audit...")
    N = 8
    L = 100
    theta_j = np.radians(30.0)
    
    scenarios = ['A', 'B', 'C']
    names = {
        'A': 'Scenario A: Direct Path Only',
        'B': 'Scenario B: Shadowing Only (-20dB)',
        'C': 'Scenario C: Coherent Multipath (Rank Collapse)'
    }
    
    np.random.seed(42)
    rng = np.random.default_rng(42)
    
    plt.figure(figsize=(15, 5))
    
    print("| Scenario | Eigenvalues (Top 3) | DOA Estimate | RMSE |")
    print("| -------- | ------------------- | ------------ | ---- |")
    
    for idx, scen in enumerate(scenarios):
        R_xx = generate_covariance(scen, N, theta_j, L, rng)
        
        # 1. Covariance Rank (Eigenvalues)
        eigenvalues, _ = np.linalg.eigh(R_xx)
        eigenvalues = np.sort(np.abs(eigenvalues))[::-1]
        top3 = eigenvalues[:3]
        
        # 2. MUSIC Peak Stability
        scan_angles, P_mu = pbb.music_doa(R_xx, num_sources=1, scan_resolution_deg=0.5)
        peaks = pbb.find_music_peaks(scan_angles, P_mu, num_sources=1)
        
        if len(peaks) > 0:
            est = peaks[0]
            err = np.abs(est - 30.0)
        else:
            est = 90.0
            err = 60.0
            
        print(f"| {scen} | {top3[0]:.1f}, {top3[1]:.1f}, {top3[2]:.1f} | {est:.1f}° | {err:.2f}° |")
        
        plt.subplot(1, 3, idx+1)
        plt.plot(scan_angles, 10*np.log10(P_mu))
        plt.axvline(30.0, color='r', linestyle='--', label='True Jammer')
        if scen == 'C':
            plt.axvline(35.0, color='g', linestyle=':', label='Multipath')
        plt.title(names[scen])
        plt.xlabel('Angle (deg)')
        plt.ylabel('Spatial Spectrum (dB)')
        plt.ylim(-20, 30)
        plt.grid(True)
        plt.legend()
        
    plt.tight_layout()
    plt.savefig('task_c4a_audit.png', dpi=300)
    print("\\nSaved task_c4a_audit.png")

if __name__ == "__main__":
    run_c4a_audit()
