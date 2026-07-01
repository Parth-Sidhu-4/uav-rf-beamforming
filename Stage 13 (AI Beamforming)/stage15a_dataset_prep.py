import os
import sys
import numpy as np
from pathlib import Path
from scipy.special import sph_harm

sys.path.append(os.path.abspath(r'D:\UAV Internship project\Phase 2 Track 1\em_realism'))
sys.path.append(os.path.abspath(r'D:\UAV Internship project\Phase 2 Track 2\array_structures'))
sys.path.append(os.path.abspath(r'D:\UAV Internship project\Stage 12 (Operational Analysis)'))

from mesh_loader import load_uav_mesh
from conformal_array import get_conformal_array

def real_spherical_harmonics(x, y, z, L=4):
    """Computes real spherical harmonics up to degree L."""
    r = np.sqrt(x**2 + y**2 + z**2)
    x, y, z = x/r, y/r, z/r
    
    phi = np.arccos(z)
    theta = np.arctan2(y, x)
    theta[theta < 0] += 2 * np.pi
    
    Y = []
    for l in range(L + 1):
        for m in range(-l, l + 1):
            if m < 0:
                val = np.sqrt(2) * (-1)**m * np.imag(sph_harm(abs(m), l, theta, phi))
            elif m == 0:
                val = np.real(sph_harm(0, l, theta, phi))
            else:
                val = np.sqrt(2) * (-1)**m * np.real(sph_harm(m, l, theta, phi))
            Y.append(val)
    return np.stack(Y, axis=1)

def main():
    print("Loading mesh and array geometry...")
    MESH_PATH = Path(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone_prepared.stl")
    mesh = load_uav_mesh(MESH_PATH)
    pos_body, _ = get_conformal_array(mesh)
    
    print("Loading 100k dataset...")
    dataset_path = "dataset_shadow_100k_polar.npz"
    data = np.load(dataset_path)
    inputs = data['inputs']
    labels = data['labels']
    
    mag = labels[:, 0::2]
    phase_exact = labels[:, 1::2]
    
    LAM = 0.15
    K = 2.0 * np.pi / LAM
    
    print("Computing geometric phases and phase deviations...")
    phase_geo = K * (inputs @ pos_body.T)
    
    # Phase deviation wrapped to [-pi, pi]
    phase_dev = phase_exact - phase_geo
    phase_dev = (phase_dev + np.pi) % (2 * np.pi) - np.pi
    
    phase_dev_deg = np.abs(np.rad2deg(phase_dev))
    
    print("\n--- Phase Deviation Distribution ---")
    print(f"50th percentile (Median) : {np.percentile(phase_dev_deg, 50):.2f} deg")
    print(f"90th percentile          : {np.percentile(phase_dev_deg, 90):.2f} deg")
    print(f"95th percentile          : {np.percentile(phase_dev_deg, 95):.2f} deg")
    print(f"99th percentile          : {np.percentile(phase_dev_deg, 99):.2f} deg")
    print(f"Max deviation            : {np.max(phase_dev_deg):.2f} deg")
    
    print("\nComputing Spherical Harmonic encoding (L=4)...")
    sh_features = real_spherical_harmonics(inputs[:, 0], inputs[:, 1], inputs[:, 2], L=4)
    print(f"SH features shape: {sh_features.shape}")
    
    out_path = "dataset_shadow_100k_sh.npz"
    np.savez(out_path, inputs=inputs, sh_features=sh_features, labels=labels)
    print(f"Saved precomputed SH dataset to {out_path}")

if __name__ == '__main__':
    main()
