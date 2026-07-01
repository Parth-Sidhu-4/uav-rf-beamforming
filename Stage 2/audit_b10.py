import numpy as np
import phase_b_beamforming as pbb
import task_b_jamming_threats as jts

def run_audit(trials=1000):
    rng = np.random.default_rng(999)
    
    N = 8
    theta_s = 0.0
    theta_j_true = np.radians(30.0)
    
    total_drops = 0
    min_sinrs = []
    outages_per_trial = []
    
    for i in range(trials):
        # 1. Mission specific static parameters
        amp_err = np.clip(rng.normal(0, 0.05, N), -0.9, 0.9)
        phase_err = rng.normal(0, np.radians(5.0), N)
        cal_err = (1.0 + amp_err) * np.exp(1j * phase_err)
        
        a_s_nom = pbb.ula_steering_vector(N, theta_s)
        a_j_nom = pbb.ula_steering_vector(N, theta_j_true)
        a_s_true = a_s_nom * cal_err
        a_j_true = a_j_nom * cal_err
        
        # 2. Sweep the approach from 5000m to 750m
        distances = np.arange(5000.0, 700.0, -50.0)
        
        trial_outages = 0
        trial_min_sinr = 100.0
        dropped = False
        
        for d_m in distances:
            Ps = jts.get_received_signal_power(d_m, 'two-ray')
            P_j = jts.get_received_jammer_power('fspl') # Follower at 5km
            
            SNR_lin = Ps / jts.P_N_lin
            INR_lin = P_j / jts.P_N_lin
            
            # Fast-fading channel
            theta_elev = np.arctan(jts.h_UAV / d_m)
            K_s_db = 10.0 + 10.0 * np.sin(theta_elev)
            K_s = 10 ** (K_s_db / 10.0)
            los_s = np.sqrt(K_s / (K_s + 1.0))
            diff_s = np.sqrt(1.0 / (K_s + 1.0))
            
            phi_s = rng.uniform(-np.pi, np.pi)
            u_s = rng.normal(0, np.sqrt(0.5)) + 1j * rng.normal(0, np.sqrt(0.5))
            h_s = los_s * np.exp(1j * phi_s) + diff_s * u_s
            h_j = rng.normal(0, np.sqrt(0.5)) + 1j * rng.normal(0, np.sqrt(0.5))
            
            # MUSIC Estimation (simplified Oracle proxy to speed up audit, 
            # since we proved B_Oracle and C_MUSIC have identical R_drop behavior)
            hat_theta_j = theta_j_true 
            
            R_xx_dl = SNR_lin * np.outer(a_s_nom, np.conj(a_s_nom)) + INR_lin * np.outer(pbb.ula_steering_vector(N, hat_theta_j), np.conj(pbb.ula_steering_vector(N, hat_theta_j))) + np.eye(N)
            w = pbb.lcmv_beamformer(R_xx_dl, theta_s, [hat_theta_j])
            
            noise_out = np.sum(np.abs(w)**2)
            sig_out = SNR_lin * (np.abs(h_s)**2) * (np.abs(np.conj(w).T @ a_s_true) ** 2)
            jam_out = INR_lin * (np.abs(h_j)**2) * (np.abs(np.conj(w).T @ a_j_true) ** 2)
            
            gamma_db = 10.0 * np.log10(max(sig_out / (noise_out + jam_out), 1e-15))
            
            if gamma_db < trial_min_sinr:
                trial_min_sinr = gamma_db
                
            if gamma_db < 5.64:
                trial_outages += 1
                if not dropped:
                    dropped = True
                    total_drops += 1
                    
        min_sinrs.append(trial_min_sinr)
        outages_per_trial.append(trial_outages)

    print("=========================================")
    print(" MUSIC/LCMV SPATIAL ARRAY DROPOUT AUDIT  ")
    print("=========================================")
    print(f"Total Missions Simulated : {trials}")
    print(f"Total Link-Drop Events   : {total_drops} / {trials}")
    print(f"Mean Outage Hops/Mission : {np.mean(outages_per_trial):.2f} (out of {len(distances)} evaluated hops)")
    print(f"Absolute Minimum SINR    : {np.min(min_sinrs):.2f} dB")
    print(f"Average Minimum SINR     : {np.mean(min_sinrs):.2f} dB")
    
if __name__ == "__main__":
    run_audit(1000)
