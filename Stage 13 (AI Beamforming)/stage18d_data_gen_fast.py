import os
import sys
import numpy as np
import time
import torch
from pathlib import Path
from tqdm import tqdm

sys.path.append(os.path.abspath(r'D:\UAV Internship project\Phase 2 Track 2\array_structures'))
sys.path.append(os.path.abspath(r'D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)'))
from attitude import euler_to_quaternion, rotate_points
from stage15d_train_cartesian_32 import CartesianShadowNet32

def main():
    print("Loading CartesianShadowNet32...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CartesianShadowNet32().to(device)
    model.load_state_dict(torch.load("shadow_net_cartesian_32.pt", map_location=device))
    model.eval()

    q_inv = euler_to_quaternion(np.deg2rad(15.0), 0, 0).conjugate()

    azimuths = np.linspace(0, 360, 3600, endpoint=False)
    elevations = [-20.0, 0.0, 20.0]
    
    total_points = len(azimuths) * len(elevations)
    
    jam_bodies_list = []
    headings_list = []
    elevations_list = []
    
    print(f"Generating {total_points} positions...")
    
    for el in elevations:
        for az in azimuths:
            jam_world = np.array([
                np.cos(np.deg2rad(az)) * np.cos(np.deg2rad(el)),
                np.sin(np.deg2rad(az)) * np.cos(np.deg2rad(el)),
                np.sin(np.deg2rad(el))
            ])
            jam_body = rotate_points(jam_world.reshape(1, 3), q_inv)[0]
            
            jam_bodies_list.append(jam_body)
            headings_list.append(az)
            elevations_list.append(el)
            
    jam_bodies_np = np.array(jam_bodies_list, dtype=np.float32)
    headings_np = np.array(headings_list, dtype=np.float32)
    elevations_np = np.array(elevations_list, dtype=np.float32)
    
    print("Computing shadow masks via neural network...")
    t0 = time.time()
    
    with torch.no_grad():
        inputs_t = torch.tensor(jam_bodies_np, device=device)
        out = model(inputs_t)
        g_raw = (out[:, 0::2] + 1j * out[:, 1::2]).to(torch.complex128)
        
        mag_raw = torch.abs(g_raw)
        scale = torch.clamp(mag_raw, min=1.0)
        g_bounded = g_raw / scale
        
        g_exact_np = g_bounded.cpu().numpy()
        
    elapsed = time.time() - t0
    print(f"Mask computation took {elapsed:.3f} seconds.")
    
    save_path = "dataset_3d_pilot_masks.npz"
    np.savez(save_path, 
             g_exact=g_exact_np, 
             jam_bodies=jam_bodies_np, 
             headings=headings_np,
             elevations=elevations_np)
             
    print(f"Saved to {save_path}")

if __name__ == '__main__':
    main()
