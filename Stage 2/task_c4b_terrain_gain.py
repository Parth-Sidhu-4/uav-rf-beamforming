import numpy as np

def run_c4b_terrain_gain():
    print("Running Task C4.b: Terrain vs Beamforming Gain Comparison...")
    
    # Let's define a simple 100-step trajectory
    steps = 100
    
    # Baseline SNR and INR (in dB)
    snr_base = np.full(steps, 10.0) # Constant 10 dB SNR from target
    inr_base = np.full(steps, 50.0) # Constant 50 dB INR from jammer (extremely strong)
    
    # Terrain Profile: A ridge blocks the jammer from step 40 to 60, providing 30 dB attenuation.
    # From step 60 to 70, it's a partial blockage (15 dB).
    terrain_loss = np.zeros(steps)
    terrain_loss[40:60] = 30.0
    terrain_loss[60:70] = 15.0
    
    # Beamforming Profile: LCMV provides an average 60 dB null depth on the jammer.
    # (Assuming perfect oracle for simplicity of the gain comparison)
    lcmv_null_depth = 60.0
    
    # Calculate SINRs
    sinr_baseline = snr_base - inr_base
    sinr_bf_only = snr_base - (inr_base - lcmv_null_depth)
    sinr_terr_only = snr_base - (inr_base - terrain_loss)
    sinr_combined = snr_base - (inr_base - terrain_loss - lcmv_null_depth)
    
    # Calculate Gains relative to Baseline
    gain_bf = sinr_bf_only - sinr_baseline
    gain_terr = sinr_terr_only - sinr_baseline
    gain_combined = sinr_combined - sinr_baseline
    
    # In reality, SINR cannot exceed the thermal SNR limit (i.e. if jammer is fully nulled, SINR -> SNR)
    # So we cap the SINR at the thermal limit (snr_base)
    sinr_bf_capped = np.minimum(sinr_bf_only, snr_base)
    sinr_terr_capped = np.minimum(sinr_terr_only, snr_base)
    sinr_comb_capped = np.minimum(sinr_combined, snr_base)
    
    # Recalculate true effective gains (how much SINR actually improved)
    true_gain_bf = sinr_bf_capped - sinr_baseline
    true_gain_terr = sinr_terr_capped - sinr_baseline
    true_gain_comb = sinr_comb_capped - sinr_baseline
    
    print("\\n| Mechanism   | Mean Effective SINR Gain | Peak Effective SINR Gain |")
    print("| ----------- | ------------------------ | ------------------------ |")
    print(f"| Beamforming | {np.mean(true_gain_bf):.1f} dB                  | {np.max(true_gain_bf):.1f} dB                  |")
    print(f"| Terrain     | {np.mean(true_gain_terr):.1f} dB                   | {np.max(true_gain_terr):.1f} dB                  |")
    print(f"| Combined    | {np.mean(true_gain_comb):.1f} dB                  | {np.max(true_gain_comb):.1f} dB                  |")
    
    # Count how often Terrain outperforms Beamforming
    # Terrain outperforms Beamforming if sinr_terr_capped > sinr_bf_capped
    # In our simple setup, LCMV gives 60dB, Terrain max is 30dB, so BF always wins.
    # BUT if we use a realistic MUSIC null depth from C3 when starved of snapshots (e.g., 20dB null depth)
    # then Terrain (30dB) would beat Beamforming (20dB)!
    
    print("\\nConsider the snapshot-starved scenario (Null Depth drops to 20 dB due to sensing):")
    sinr_bf_starved = np.minimum(snr_base - (inr_base - 20.0), snr_base)
    terr_beats_bf = np.sum(sinr_terr_capped > sinr_bf_starved)
    print(f"When spatial processing is starved, Terrain outperforms Beamforming {terr_beats_bf}% of the time!")

if __name__ == "__main__":
    run_c4b_terrain_gain()
