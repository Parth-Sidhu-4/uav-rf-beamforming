import os
import shutil
import re
import subprocess
import time

STAGE2_DIR = r"d:\UAV Internship project\Stage 2"
REMEDIATED_DIR = r"d:\UAV Internship project\Stage 2 (Remediated)"
SCRIPTS_DIR = os.path.join(REMEDIATED_DIR, "Remediated_Scripts")
RESULTS_DIR = os.path.join(REMEDIATED_DIR, "Remediated_Results")

os.makedirs(SCRIPTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

TARGET_SCRIPTS = [
    "task_a5_fspl_vs_rician.py",
    "task_a6_outage_contour.py",
    "task_b3_music_doa.py",
    "task_b8_beam_patterns.py",
    "task_b9_array_sweep.py",
    "task_b11_amc_occupancy.py",
    "task_b12_policy_comparison.py",
    "task_b_jamming_threats.py",
    "task_c5_rnco.py",
    "task_c6_competition.py",
    "task_c7_ablation.py",
    "task_c8_sensitivity.py",
    "task_c9_diversity.py"
]

def port_script(script_name):
    orig_path = os.path.join(STAGE2_DIR, script_name)
    if not os.path.exists(orig_path):
        print(f"WARNING: {script_name} not found in original Stage 2 directory.")
        return None

    with open(orig_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Update imports
    # Handle "import module as X" vs "import module"
    content = re.sub(r'\bimport simulator_core\b(?!\s+as)', 'import simulator_core_remediated as simulator_core', content)
    content = re.sub(r'\bimport simulator_core as\b', 'import simulator_core_remediated as', content)
    
    content = re.sub(r'\bimport phase_b_beamforming\b(?!\s+as)', 'import phase_b_beamforming_remediated as phase_b_beamforming', content)
    content = re.sub(r'\bimport phase_b_beamforming as\b', 'import phase_b_beamforming_remediated as', content)
    
    content = re.sub(r'\bimport mission_resilience_sim\b(?!\s+as)', 'import mission_resilience_sim_remediated as mission_resilience_sim', content)
    content = re.sub(r'\bimport mission_resilience_sim as\b', 'import mission_resilience_sim_remediated as', content)
    
    # Handle "from module import ..."
    content = re.sub(r'\bfrom simulator_core import\b', 'from simulator_core_remediated import', content)
    content = re.sub(r'\bfrom phase_b_beamforming import\b', 'from phase_b_beamforming_remediated import', content)
    content = re.sub(r'\bfrom mission_resilience_sim import\b', 'from mission_resilience_sim_remediated import', content)

    # 2. Add python path injection so it can find the remediated modules which are in the parent dir
    header = "import sys\nimport os\nsys.path.insert(0, os.path.abspath('..'))\nsys.path.insert(0, r'" + STAGE2_DIR + "')\n\n"
    content = header + content

    # 3. Save to Remediated_Scripts
    new_path = os.path.join(SCRIPTS_DIR, script_name)
    with open(new_path, 'w', encoding='utf-8') as f:
        f.write(content)
        
    print(f"Ported {script_name} -> {new_path}")
    return new_path

def run_script(script_path):
    script_name = os.path.basename(script_path)
    print(f"\n[{time.strftime('%H:%M:%S')}] RUNNING: {script_name} ...")
    
    # Ensure any stray pngs/csvs are cleared from SCRIPTS_DIR before running
    for f in os.listdir(SCRIPTS_DIR):
        if f.endswith('.png') or f.endswith('.csv') or f.endswith('.json'):
            os.remove(os.path.join(SCRIPTS_DIR, f))
            
    # Run the script
    start_time = time.time()
    try:
        result = subprocess.run(
            ['python', '-X', 'utf8', script_name],
            cwd=SCRIPTS_DIR,
            capture_output=True,
            text=True,
            check=False
        )
        duration = time.time() - start_time
        if result.returncode == 0:
            print(f"[{time.strftime('%H:%M:%S')}] SUCCESS: {script_name} ({duration:.1f} sec)")
        else:
            print(f"[{time.strftime('%H:%M:%S')}] FAILED: {script_name} (Exit {result.returncode})")
            print("--- STDOUT ---")
            print(result.stdout)
            print("--- STDERR ---")
            print(result.stderr)
            print("--------------")
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] EXCEPTION running {script_name}: {e}")

    # Move any generated outputs to RESULTS_DIR
    moved = 0
    for f in os.listdir(SCRIPTS_DIR):
        if f.endswith('.png') or f.endswith('.csv') or f.endswith('.json'):
            src = os.path.join(SCRIPTS_DIR, f)
            dst = os.path.join(RESULTS_DIR, f)
            if os.path.exists(dst):
                os.remove(dst)
            shutil.move(src, dst)
            moved += 1
    
    print(f"  -> Moved {moved} output files to Remediated_Results/.")

if __name__ == "__main__":
    print("===========================================================")
    print(" STARTING PORT AND RUN OF ALL TARGET SIMULATION SCRIPTS")
    print("===========================================================")
    
    valid_scripts = []
    for script in TARGET_SCRIPTS:
        p = port_script(script)
        if p:
            valid_scripts.append(p)
            
    print(f"\nSuccessfully ported {len(valid_scripts)} scripts.")
    print("Beginning sequential execution. This will take some time.\n")
    
    for script_path in valid_scripts:
        run_script(script_path)
        
    print("\n===========================================================")
    print(" ALL TASKS COMPLETED.")
    print(f" Outputs are available in: {RESULTS_DIR}")
    print("===========================================================")
