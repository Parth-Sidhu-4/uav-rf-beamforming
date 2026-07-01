import sys
import os
sys.path.insert(0, r'd:\UAV Internship project\Stage 2')
sys.path.insert(0, r'd:\UAV Internship project\Stage 2 (Remediated)')

from Remediated_Scripts.task_c6_competition import MissionSimulator

def test_ablation():
    print("Testing Ablation Differentiation...")
    for ablation in ['RNCO', 'No_RF', 'No_EXP', 'No_INFO']:
        ab_flag = ablation if ablation != 'RNCO' else None
        sim = MissionSimulator(scenario='A', policy='RNCO', ablation=ab_flag)
        
        sim.Q_RF = 1000.0
        sim.Q_EXP = 1000.0
        sim.Q_INFO = 1000.0
        
        if ablation == 'No_RF':
            sim.Q_EXP = 50000.0 # Force different tradeoff
        if ablation == 'No_EXP':
            sim.Q_INFO = 50000.0
            
        sim.step(epoch=0)
        action = (sim.logs['ts'][-1], sim.logs['dy'][-1])
        print(f"{ablation}: ts={action[0]}, dy={action[1]}")

test_ablation()
