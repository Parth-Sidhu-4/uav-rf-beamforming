import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
import task_b4_integration as tb4

print("="*60)
print("NUMERICAL CHECKS FOR N=4 ARRAY VS POINTING ERROR")
print("="*60)

for p_err in [0.0, 1.0, 2.0, 5.0, 10.0]:
    r_def_trials = [tb4.solve_defeat_range(4, np.radians(0.0), np.radians(30.0), p_err, 500) for _ in range(30)]
    mean_r_def = np.mean(r_def_trials)
    print(f"DOA Mismatch sigma_theta = {p_err:2.1f}° | Defeat Range = {mean_r_def:.2f} m")
print("="*60)
