import numpy as np
import matplotlib.pyplot as plt
from mission_profile import MissionProfile
from sinr_models import baseline_channel, phase_b_channel, phase_bc_channel
from pmcs_engine import run_configuration

EIRP_SWEEP = np.linspace(-10, 60, 90)   # dBW

configurations = [
    ('Baseline (No EW)',         baseline_channel,  False, 'red',   '--'),
    ('Phase B: LCMV Nulling',    phase_b_channel,   False, 'royalblue', '-'),
    ('Phase B+C+D: Full Stack',  phase_bc_channel,  True,  'green',  '-'),
]

mission = MissionProfile()
results = np.zeros((3, 90))
collapse_eirps = []

for i, (name, fn, arq, color, ls) in enumerate(configurations):
    print(f"Running config: {name}")
    pmcs_list = []
    for j, eirp in enumerate(EIRP_SWEEP):
        pmcs = run_configuration(fn, eirp, mission, apply_arq=arq)
        results[i, j] = pmcs
        pmcs_list.append(pmcs)
        
    pmcs_array = np.array(pmcs_list)
    
    # Find collapse point
    idx = np.where(pmcs_array <= 0.5)[0]
    if len(idx) > 0:
        collapse_eirps.append(EIRP_SWEEP[idx[0]])
    else:
        collapse_eirps.append(np.inf)

if not (collapse_eirps[0] < collapse_eirps[1] < collapse_eirps[2]):
    raise ValueError("Channel model error: curves not strictly ordered — check NULL_DEPTH_DB and RNCO_REDUCTION.")

plt.figure(figsize=(10, 6))
for i, (name, fn, arq, color, ls) in enumerate(configurations):
    plt.plot(EIRP_SWEEP, results[i], color=color, linestyle=ls, label=name)
    if collapse_eirps[i] != np.inf:
        plt.axvline(collapse_eirps[i], color=color, linestyle=':')
        ax = plt.gca()
        y_offset = 0.10 if color == 'green' else 0.03
        ha_align = 'right' if color == 'red' else 'center'
        ax.annotate(f"{collapse_eirps[i]:.1f} dBW", 
                    xy=(collapse_eirps[i], y_offset), 
                    va='bottom', ha=ha_align, color=color, fontsize=9)

plt.axhline(0.5, color='grey', linestyle='--', label='50% success threshold')
plt.ylim([0, 1.05])
plt.xlim([-10, 60])
plt.ylabel('Mission Success Probability $P_{mcs}$')
plt.xlabel('Jammer EIRP (dBW)')
plt.title("Mission Resilience Collapse Curves — H-MRSM vs. Baseline\n(600 s recon mission, 90% critical packet threshold)")
plt.legend(loc='upper right')
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig('mission_resilience_collapse.png', dpi=150)
print("Saved mission_resilience_collapse.png")

print("\n===== Mission Resilience Summary =====")
print(f"{'Config':<30}| {'Collapse EIRP (50%)':<19} | {'P_mcs @ 40 dBW':<14}")
print("-" * 30 + "|" + "-" * 21 + "|" + "-" * 16)
for i, (name, fn, arq, color, ls) in enumerate(configurations):
    col = f"{collapse_eirps[i]:.1f} dBW" if collapse_eirps[i] != np.inf else ">60.0 dBW"
    # Find pmcs at 40 dBW
    idx_40 = np.abs(EIRP_SWEEP - 40.0).argmin()
    val_40 = results[i, idx_40]
    print(f"{name:<30}| {col:^19} | {val_40:>14.3f}")
print("======================================")
margin = collapse_eirps[2] - collapse_eirps[0]
print(f"Resilience margin (Full Stack vs Baseline): {margin:.1f} dB")
