from dataclasses import dataclass, field
import numpy as np

@dataclass
class MissionProfile:
    T: int = 600
    dt: int = 1
    n_steps: int = 600
    fc_hz: float = 2.4e9
    uav_gcs_m: float = 1000.0
    uav_jmr_m: float = 500.0
    signal_eirp_dbw: float = 10.0
    noise_dbm: float = -100.0
    
    # Generate critical mask: True at every 5th step (steps 4, 9, 14...)
    # This gives exactly 120 critical communication windows
    critical_mask: np.ndarray = field(default_factory=lambda: np.array([i % 5 == 4 for i in range(600)]))
