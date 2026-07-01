"""
omega_0 sensitivity sweep for Phase C SIREN.
Usage: python stage14c_omega_sweep.py <omega_0>
Saves weights to shadow_net_siren_w<omega_0>.pt
"""
import sys
import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Phase 2 Track 1', 'em_realism')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Phase 2 Track 2', 'array_structures')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Stage 12 (Operational Analysis)')))

from mesh_loader import load_uav_mesh
from conformal_array import get_conformal_array

# Import model/loss/trainer from Phase C script
import importlib.util
spec = importlib.util.spec_from_file_location(
    'stage14c', os.path.join(os.path.dirname(__file__), 'stage14c_train_siren.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

def main():
    if len(sys.argv) < 2:
        print("Usage: python stage14c_omega_sweep.py <omega_0>")
        sys.exit(1)
    omega = int(sys.argv[1])
    save_path = f"shadow_net_siren_w{omega}.pt"
    print(f"=== SIREN omega_0={omega} sweep | save -> {save_path} ===")

    MESH_PATH = Path(r"D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone_prepared.stl")
    print("Loading mesh and dataset...")
    mesh = load_uav_mesh(MESH_PATH)
    pos_body, _ = get_conformal_array(mesh)

    data   = np.load("dataset_shadow_30k.npz")
    inputs = data['inputs']
    labels = data['labels']
    np.random.seed(42); torch.manual_seed(42)

    N = inputs.shape[0]; idx = np.random.permutation(N); split = int(N * 0.8)
    Xt = torch.tensor(inputs[idx[:split]], dtype=torch.float32)
    yt = torch.tensor(labels[idx[:split]], dtype=torch.float32)
    Xv = torch.tensor(inputs[idx[split:]], dtype=torch.float32)
    yv = torch.tensor(labels[idx[split:]], dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(Xt, yt), batch_size=256, shuffle=True)
    val_loader   = DataLoader(TensorDataset(Xv, yv), batch_size=256, shuffle=False)

    model = mod.ShadowNet_SIREN(omega_0=omega)
    crit  = mod.NullResponseLoss(pos_body)
    mod.train_model(model, crit, train_loader, val_loader,
                    epochs=500, patience=50, save_path=save_path)
    print(f"Done. Saved to {save_path}")

if __name__ == '__main__':
    main()
