import numpy as np

with open('stage5_grand_integration.py', 'r') as f:
    code = f.read()

# Inject bias
code = code.replace('ins.x[0:4] = [0, 0, 20, 0] # Init true', 'ins.x[0:4] = [0, 0, 20, 0]\n    ins.x[4] = 0.5\n    ins.x[5] = 0.5')

# Add debug prints
code = code.replace('        log_rtl[k] = (mission_state == "RTL")', '''        log_rtl[k] = (mission_state == "RTL")
        if k % 5000 == 0:
            drift = np.linalg.norm(true_state.pos - est_state.pos)
            print(f"t={t:.1f}s | Mission: {mission_state} | Health: {telemetry_healthy} | Drift: {drift:.2f}m | PER: {current_per:.2f} | d_GCS: {d_gcs:.1f}m")
''')

# Reduce duration for debug
code = code.replace('duration = 400.0', 'duration = 200.0')

with open('stage5_debug_2.py', 'w') as f:
    f.write(code)
