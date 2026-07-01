"""
Two-part analysis:
1. Full per-element comparison v4 vs v5 (find the hidden bucket-count entry)
2. Targeted trace of Ant 2 across its jump heading
"""
import numpy as np, sys, os
from pathlib import Path
sys.path.append(r'D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)')
sys.path.append(r'D:\UAV Internship project\Phase 2 Track 1')

from mesh_loader import load_uav_mesh
from conformal_array import get_conformal_array
from attitude import euler_to_quaternion, rotate_points
from em_physics import compute_distances, fresnel_diffraction_gain
from trimesh.proximity import signed_distance

# ── shared setup ──────────────────────────────────────────────────────────────
mesh = load_uav_mesh(Path(r'D:\UAV Internship project\Stage 8 (Mesh Ray-Tracing)\assets\meshes\drone_prepared.stl'))
antennas, normals = get_conformal_array(mesh)
jam_world = np.array([0.0, 1.0, 0.0])
edges = mesh.face_adjacency_edges
V1_all = mesh.vertices[edges[:,0]]; V2_all = mesh.vertices[edges[:,1]]
vm = np.linalg.norm(V1_all - V2_all, axis=1) > 0.015
V1 = V1_all[vm]; V2 = V2_all[vm]
headings = np.linspace(22, 32, 200)
lam = 0.125; beta = 500.0


def compute_gain(headings, soft_secondary=True):
    g = np.zeros((len(headings), 16), dtype=complex)
    for idx_h, h_ang in enumerate(headings):
        q = euler_to_quaternion(np.deg2rad(30), 0, np.deg2rad(h_ang))
        jb = rotate_points(jam_world.reshape(1,3), q.conjugate())[0]
        ro = antennas + normals * 1e-4
        hd, d1, _ = compute_distances(ro, jb, V1, V2)
        for i in range(16):
            hi = hd[i]; d1i = d1[i]
            lw = -beta*(hi - hi.min()); lw -= lw.max()
            ws = np.exp(lw); ws /= ws.sum()
            e1d = ws @ d1i
            Sj = 1 - np.exp(-(d1i - e1d)**2 / (2*0.2**2))
            hp = hi / (Sj + 1e-8)
            # F1 soft (top-3 primary)
            t3 = np.argsort(hi)[:3]; wt3 = ws[t3]; wt3 /= wt3.sum()
            F1s = 0.0
            for k, ek in enumerate(t3):
                sd = signed_distance(mesh, [ro[i]+d1i[ek]*jb])[0]
                F1s += wt3[k]*fresnel_diffraction_gain(sd*np.sqrt(2/(lam*d1i[ek])))
            # F2
            if soft_secondary:
                lw2 = -beta*(hp - hp.min()); lw2 -= lw2.max()
                ws2 = np.exp(lw2); ws2 /= ws2.sum()
                w2 = ws2 @ Sj
                t3b = np.argsort(hp)[:3]; wt3b = ws2[t3b]; wt3b /= wt3b.sum()
                F2s = 0.0
                for k, ek in enumerate(t3b):
                    sd2 = signed_distance(mesh, [ro[i]+d1i[ek]*jb])[0]
                    F2s += wt3b[k]*fresnel_diffraction_gain(sd2*np.sqrt(2/(lam*(e1d+d1i[ek]))))
            else:
                idx2 = np.argmin(hp)
                sd2 = signed_distance(mesh, [ro[i]+d1i[idx2]*jb])[0]
                F2s = fresnel_diffraction_gain(sd2*np.sqrt(2/(lam*(e1d+d1i[idx2]))))
                w2 = Sj[idx2]
            g[idx_h, i] = F1s * ((1-w2)*1.0 + w2*F2s)
    return g


# ── PART 1: full per-element comparison ───────────────────────────────────────
print("Computing v4 (primary softmax only)...")
g_v4 = compute_gain(headings, soft_secondary=False)
print("Computing v5 (both softmax)...")
g_v5 = compute_gain(headings, soft_secondary=True)

mpe_v4 = np.abs(np.diff(20*np.log10(np.abs(g_v4)+1e-30), axis=0)).max(axis=0)
mpe_v5 = np.abs(np.diff(20*np.log10(np.abs(g_v5)+1e-30), axis=0)).max(axis=0)

print()
print("="*70)
print(f"{'Ant':>4} | {'v4 max step':>12} | {'v5 max step':>12} | {'delta':>10} | {'change'}")
print("-"*70)
for el in range(16):
    d = mpe_v5[el] - mpe_v4[el]
    flags = []
    if mpe_v4[el] > 0.30 and mpe_v5[el] <= 0.30: flags.append("IMPROVED below 0.30")
    if mpe_v4[el] <= 0.30 and mpe_v5[el] > 0.30: flags.append("*** GAINED step, crossed 0.30 threshold")
    if mpe_v4[el] > 0.50 and mpe_v5[el] <= 0.50: flags.append("IMPROVED below 0.50")
    if mpe_v4[el] <= 0.50 and mpe_v5[el] > 0.50: flags.append("*** GAINED step, crossed 0.50 threshold")
    print(f" {el:3d} | {mpe_v4[el]:12.4f} | {mpe_v5[el]:12.4f} | {d:+10.4f} | {' '.join(flags)}")

print()
for label, mpe in [("v4", mpe_v4), ("v5", mpe_v5)]:
    print(f"{label}: steps>0.50={( mpe>0.50).sum()}  >0.30={(mpe>0.30).sum()}  >0.10={(mpe>0.10).sum()}")


# ── PART 2: trace Ant 2 across its jump ───────────────────────────────────────
print()
print("="*70)
print("TRACE: Ant 2 across its worst step heading (v5 code)")
print("="*70)

# Find worst heading for ant 2 in v5
gdb_v5 = 20*np.log10(np.abs(g_v5)+1e-30)
steps_2 = np.abs(np.diff(gdb_v5[:,2]))
worst_idx = np.argmax(steps_2)
h_lo = headings[worst_idx] - 1.0
h_hi = headings[worst_idx] + 1.0
print(f"Worst step at heading {headings[worst_idx]:.2f} deg "
      f"({gdb_v5[worst_idx,2]:.4f} -> {gdb_v5[worst_idx+1,2]:.4f} dB, "
      f"step={steps_2[worst_idx]:.4f} dB)")
print(f"Tracing heading {h_lo:.1f} to {h_hi:.1f} at fine resolution...")
print()

trace_headings = np.linspace(h_lo, h_hi, 80)
print(f"{'h':>6} | {'e1_idx':>7} | {'top_hp_idx':>10} | {'e1d':>7} | {'e2d':>7} | "
      f"{'w2':>6} | {'F1s':>7} | {'F2s':>7} | {'gain_dB':>9} | JUMP?")
print("-"*90)
prev_gdb = None
for h_ang in trace_headings:
    q = euler_to_quaternion(np.deg2rad(30), 0, np.deg2rad(h_ang))
    jb = rotate_points(jam_world.reshape(1,3), q.conjugate())[0]
    ro = antennas + normals * 1e-4
    hd, d1, _ = compute_distances(ro, jb, V1, V2)
    i = 2
    hi = hd[i]; d1i = d1[i]
    lw = -beta*(hi - hi.min()); lw -= lw.max(); ws = np.exp(lw); ws /= ws.sum()
    e1d = ws @ d1i
    Sj = 1 - np.exp(-(d1i - e1d)**2 / (2*0.2**2))
    hp = hi / (Sj + 1e-8)
    # primary
    t3 = np.argsort(hi)[:3]; wt3 = ws[t3]; wt3 /= wt3.sum()
    F1s = 0.0
    for k, ek in enumerate(t3):
        sd = signed_distance(mesh, [ro[i]+d1i[ek]*jb])[0]
        F1s += wt3[k]*fresnel_diffraction_gain(sd*np.sqrt(2/(lam*d1i[ek])))
    # secondary
    lw2 = -beta*(hp - hp.min()); lw2 -= lw2.max(); ws2 = np.exp(lw2); ws2 /= ws2.sum()
    w2 = ws2 @ Sj
    t3b = np.argsort(hp)[:3]; wt3b = ws2[t3b]; wt3b /= wt3b.sum()
    F2s = 0.0
    for k, ek in enumerate(t3b):
        sd2 = signed_distance(mesh, [ro[i]+d1i[ek]*jb])[0]
        F2s += wt3b[k]*fresnel_diffraction_gain(sd2*np.sqrt(2/(lam*(e1d+d1i[ek]))))
    total = F1s * ((1-w2)*1.0 + w2*F2s)
    gdb = 20*np.log10(abs(total)+1e-30)
    jump = ""
    if prev_gdb is not None and abs(gdb - prev_gdb) > 0.15:
        jump = f"<< {gdb-prev_gdb:+.3f} dB"
    e1_idx = np.argmin(hi)
    hp_idx = np.argmin(hp)
    print(f"{h_ang:6.2f} | {e1_idx:7d} | {hp_idx:10d} | {e1d:7.4f} | {d1i[hp_idx]:7.4f} | "
          f"{w2:6.4f} | {F1s:7.4f} | {F2s:7.4f} | {gdb:9.4f} | {jump}")
    prev_gdb = gdb
