import numpy as np

with open('stage5_grand_integration.py', 'r') as f:
    code = f.read()

# Inject bias
code = code.replace('ins.x[0:4] = [0, 0, 20, 0] # Init true', 'ins.x[0:4] = [0, 0, 20, 0]\n    ins.x[4] = 0.5\n    ins.x[5] = 0.5')

# Add debug prints
code = code.replace('        log_rtl[k] = (mission_state == "RTL")', '''        log_rtl[k] = (mission_state == "RTL")
        if k % 2000 == 0:
            drift = np.linalg.norm(true_state.pos - est_state.pos)
            print(f"t={t:.1f}s | Mission: {mission_state} | Health: {telemetry_healthy} | Drift: {drift:.2f}m | PER: {current_per:.2f} | d_GCS: {d_gcs:.1f}m | True: {true_state.pos[0]:.1f},{true_state.pos[1]:.1f} | Est: {est_state.pos[0]:.1f},{est_state.pos[1]:.1f}")
''')

with open('stage5_debug_3.py', 'w') as f:
    f.write(code)
