import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import sys

# Append path to load phase_b_beamforming if needed, though we will use analytical proxies for speed
sys.path.append(os.path.dirname(__file__))
import phase_b_beamforming as pbb
import task_c3_cognitive_sensing as c3

def run_c5_preaudit():
    print("Generating 10,000 Monte Carlo samples for C5 Pre-Audit...")
    np.random.seed(42)
    N_samples = 10000
    
    # 1. Randomize decision variables
    tau_s = np.random.uniform(0.1, 9.0, N_samples) # ms
    dy = np.random.uniform(0, 2000, N_samples) # m
    uav_x = np.random.uniform(-5000, -1000, N_samples) # m
    
    # Constants
    V_MPS = 50.0
    T_f = 10.0 # ms
    fs = 1e6
    p_fa = 0.05
    jnr_lin = 10.0**(-5.0/10.0)
    
    jammer_pos = np.array([15000.0, 0.0])
    
    # Pre-allocate arrays
    pd_array = np.zeros(N_samples)
    leff_array = np.zeros(N_samples)
    music_rmse = np.zeros(N_samples)
    null_depth = np.zeros(N_samples)
    gdop = np.zeros(N_samples)
    
    # 2. Compute variables
    for i in range(N_samples):
        # Cognitive Variables
        pd_array[i] = c3.compute_pd(tau_s[i] * 1e-3, fs, p_fa, jnr_lin)
        L_eff = max(1, int(100 * (1.0 - tau_s[i] / T_f)))
        leff_array[i] = L_eff
        
        # MUSIC RMSE proxy (empirical fit from C3: RMSE roughly scales as 1/sqrt(L_eff))
        # At L=99, RMSE ~ 0.0. At L=9, RMSE ~ 0.3.
        # Let's use a simple analytical curve: RMSE = 1.0 / sqrt(L_eff) - 0.1 (floored to 0.01)
        m_rmse = max(0.01, 1.0 / np.sqrt(L_eff) - 0.1)
        music_rmse[i] = m_rmse
        
        # Null depth proxy (empirical fit from C3: -120 dB at L=99, -90 dB at L=9)
        # depth = -120 + 30 * ((100 - L_eff)/91)
        nd = -120.0 + 35.0 * ((100 - L_eff) / 99.0)**2
        null_depth[i] = nd
        
        # GDOP proxy based on C1a
        # Distance to jammer
        dx = 15000.0 - uav_x[i]
        dy_j = 0.0 - dy[i]
        r = np.sqrt(dx**2 + dy_j**2)
        # If dy is small, GDOP is large. If dy is large, GDOP is small.
        # Max dy = 2000 -> GDOP ~ 1.15. dy = 460 -> GDOP ~ 5. dy < 100 -> GDOP > 20
        # A rough geometrical approximation for GDOP in bearing-only: GDOP ~ r / dy
        g = r / max(dy[i], 10.0)
        gdop[i] = g

    # Survivability Variables
    extra_dist = 2 * dy
    extra_time = extra_dist / V_MPS
    base_time = (0.0 - uav_x) / V_MPS
    total_exposure = base_time + extra_time
    survival_prob = np.exp(-total_exposure / 120.0)
    
    # Information Variables
    bearing_err = music_rmse # essentially the same for single snapshot
    ekf_pos_rmse = gdop * music_rmse * (15000.0 - uav_x) * (np.pi/180.0) # Approx pos error
    ekf_cov_trace = ekf_pos_rmse**2 # Trace of covariance is sum of variances
    
    # RF Variables
    snr_base = 10.0
    inr_base = 50.0
    inr_resid = inr_base + null_depth
    sinr = snr_base - inr_resid
    # Outage prob is 1 if SINR < 0, 0 otherwise, smoothed via sigmoid
    comm_outage = 1.0 / (1.0 + np.exp(0.5 * sinr))
    eff_throughput = pd_array * (1.0 - tau_s / T_f) * (1.0 - comm_outage)
    
    # 3. Create DataFrame
    df = pd.DataFrame({
        'Post-BF SINR': sinr,
        'Null Depth': null_depth,
        'Comm Outage Prob': comm_outage,
        'Effective Throughput': eff_throughput,
        'EKF Cov Trace': ekf_cov_trace,
        'EKF Pos RMSE': ekf_pos_rmse,
        'GDOP': gdop,
        'Bearing Error': bearing_err,
        'MUSIC RMSE': music_rmse,
        'Exposure Time': total_exposure,
        'Survival Prob': survival_prob,
        'Additional Path': extra_dist,
        'Sensing Time ts': tau_s,
        'Effective Snapshots Leff': leff_array,
        'Detection Prob Pd': pd_array
    })
    
    # 4. Correlation Matrix
    corr = df.corr()
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title("Pearson Correlation Heatmap of State Variables")
    plt.tight_layout()
    plt.savefig('task_c5_correlation.png', dpi=300)
    
    # 5. PCA Analysis
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    
    pca = PCA()
    pca.fit(df_scaled)
    
    explained_var = pca.explained_variance_ratio_
    cum_var = np.cumsum(explained_var)
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 16), cum_var, 'b-o', linewidth=2)
    plt.axhline(0.95, color='r', linestyle='--', label='95% Variance Explained')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Variance Plot')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('task_c5_pca.png', dpi=300)
    
    print("=== Correlation Analysis ===")
    highly_correlated = []
    cols = df.columns
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            if abs(corr.iloc[i, j]) > 0.90:
                highly_correlated.append(f"{cols[i]} <-> {cols[j]}: {corr.iloc[i, j]:.3f}")
                
    for pair in highly_correlated:
        print(pair)
        
    print(f"\\n=== PCA Analysis ===")
    print(f"PC1 explains: {explained_var[0]*100:.1f}%")
    print(f"PC2 explains: {explained_var[1]*100:.1f}%")
    print(f"PC3 explains: {explained_var[2]*100:.1f}%")
    print(f"PC4 explains: {explained_var[3]*100:.1f}%")
    
    n_95 = np.argmax(cum_var >= 0.95) + 1
    print(f"Number of components to explain 95% variance: {n_95}")
    
    # PC Loadings
    loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(15)], index=df.columns)
    print("\\nTop variables driving PC1:")
    print(loadings['PC1'].abs().sort_values(ascending=False).head(4))
    print("\\nTop variables driving PC2:")
    print(loadings['PC2'].abs().sort_values(ascending=False).head(4))
    print("\\nTop variables driving PC3:")
    print(loadings['PC3'].abs().sort_values(ascending=False).head(4))

if __name__ == "__main__":
    run_c5_preaudit()
