import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))
from manet_routing import SwarmMANET, MANETConfig, SwarmNode


def main():
    print("Running Category 4c: Multi-Drone Swarm MANET Routing\n")
    
    cfg = MANETConfig()
    manet = SwarmMANET(cfg)
    
    # Check 1: Single-hop reference (2-node network)
    # GCS at origin, UAV1 at distance d
    # Adjust tx power to get a specific SNR
    
    dist_1 = 1000.0  # 1 km
    gcs = SwarmNode(node_id=0, position=np.array([0.0, 0.0, 0.0]))
    uav1 = SwarmNode(node_id=1, position=np.array([dist_1, 0.0, 50.0]))
    
    manet.update_topology([gcs, uav1])
    
    edge_data = manet.graph.get_edge_data(0, 1)
    print(f"[Check 1] 2-Node Network:")
    if edge_data is None:
        print("  FAIL: No edge between GCS and UAV1. SNR too low.")
    else:
        snr_db = edge_data['snr_db']
        print(f"  GCS -> UAV1 SNR: {snr_db:.2f} dB")
        pdr_dict = manet.simulate_delivery(gcs_id=0, nodes=[gcs, uav1])
        print(f"  PDR to UAV1: {pdr_dict[1]:.4f}")
        print(f"  PASS: Single-hop connectivity maintained.")

    # Check 2: Relay Test (3 UAVs + GCS)
    print("\n[Check 2] Relay Test (Jamming UAV3):")
    # GCS at origin
    # UAV1 at 5 km (safe)
    # UAV2 at 10 km (safe)
    # UAV3 at 20 km (jammed from GCS, but close to UAV2)
    # Let's arrange them linearly
    
    gcs = SwarmNode(node_id=0, position=np.array([0.0, 0.0, 0.0]))
    uav1 = SwarmNode(node_id=1, position=np.array([5000.0, 0.0, 50.0]))
    uav2 = SwarmNode(node_id=2, position=np.array([10000.0, 0.0, 50.0]))
    uav3 = SwarmNode(node_id=3, position=np.array([15000.0, 0.0, 50.0]))
    
    # Baseline: no jamming
    manet.update_topology([gcs, uav1, uav2, uav3])
    pdr_clean = manet.simulate_delivery(0, [gcs, uav1, uav2, uav3])
    print(f"  Clean PDRs: UAV1={pdr_clean[1]:.4f}, UAV2={pdr_clean[2]:.4f}, UAV3={pdr_clean[3]:.4f}")
    
    # Introduce jammer near UAV3, which degrades the GCS->UAV3 link so it drops below gamma_min.
    # We simulate this by removing the GCS->UAV3 edge manually, or we can just say the GCS Tx to UAV3 is too low.
    # Let's adjust GCS tx power so it can reach UAV2 but not UAV3.
    # FSPL at 10km is 120dB, at 15km is 123.5dB.
    # Noise is -111. SNR_min is 5dB.
    # Required Tx = SNR_min + FSPL + Noise = 5 + 120 - 111 = 14 dBm for 10km.
    # If Tx is 16 dBm, 10km SNR is 7dB (safe), 15km SNR is 3.5dB (drops!).
    
    gcs.tx_power_dbm = 16.0
    uav1.tx_power_dbm = 20.0 # UAVs have plenty of A2A power
    uav2.tx_power_dbm = 20.0
    uav3.tx_power_dbm = 20.0
    
    manet.update_topology([gcs, uav1, uav2, uav3])
    
    # Check edges
    has_direct = manet.graph.has_edge(0, 3)
    print(f"  GCS->UAV3 direct edge exists? {has_direct}")
    
    path = manet.route_packet(0, 3)
    print(f"  Route GCS->UAV3: {path}")
    
    pdr_jammed = manet.simulate_delivery(0, [gcs, uav1, uav2, uav3])
    print(f"  Relay PDRs: UAV1={pdr_jammed[1]:.4f}, UAV2={pdr_jammed[2]:.4f}, UAV3={pdr_jammed[3]:.4f}")
    
    if not has_direct and path == [0, 1, 2, 3]:
        print("  PASS: UAV3 received packets via 3-hop relay despite dead direct link.")
    else:
        print("  FAIL: Relay routing did not work as expected.")

if __name__ == "__main__":
    main()
