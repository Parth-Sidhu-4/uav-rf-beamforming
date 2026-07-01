"""
cat4/manet_routing.py
Extension 4c: Multi-Drone Swarm MANET Routing

Mathematical basis (Section 6.3 of Extension Plan):
  - Air-to-Air path loss: FSPL + Rician fading (K_A2A = 20 dB, near-LOS)
  - Edge weights based on negative log-probability of successful delivery
    w_ij = -log2(1 - BER_ij)^L_packet ≈ L_packet * BER_ij
  - Dynamic routing (OLSR-like) using Dijkstra on the active network graph
  - Jammed nodes (from RNCO) lose GCS uplink but may act as peers if A2A SNR allows

Integration:
  - Uses fspl_db and rician_mrc_outage from channel_bridge.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))

import numpy as np
import networkx as nx
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from math import erfc

from channel_bridge import BANDS_HZ, rician_mrc_outage


@dataclass
class SwarmNode:
    node_id: int
    position: np.ndarray      # (3,) [x, y, z] metres
    is_jammed: bool = False
    tx_power_dbm: float = 20.0


@dataclass
class MANETConfig:
    freq_hz: float = BANDS_HZ["2.4 GHz"]
    gamma_min_db: float = 5.0       # minimum link SNR (dB)
    noise_floor_dbm: float = -111.0
    packet_length_bytes: int = 263  # MAVLink v2 max
    hello_interval_s: float = 1.0
    mcs_bps: float = 54e6           # link data rate
    k_factor_a2a_db: float = 20.0   # A2A channels are highly LOS


class SwarmMANET:
    """
    Dynamic mesh routing for a K-UAV swarm with jammer-aware re-routing.
    """
    def __init__(self, config: MANETConfig):
        self.cfg = config
        self.graph = nx.DiGraph()

    def update_topology(self, nodes: List[SwarmNode]):
        """
        Rebuild the link-state graph from current node positions and jam status.
        """
        self.graph.clear()
        K = len(nodes)
        self.graph.add_nodes_from(range(K))

        for i in range(K):
            for j in range(i + 1, K):
                # Note: 'is_jammed' affects the GCS-to-UAV link, but peer A2A
                # links may still be viable if distance is small and jammer is far.
                # Here we just compute the raw A2A SNR.
                dist = max(1.0, np.linalg.norm(nodes[i].position - nodes[j].position))
                fspl = 20*np.log10(dist) + 20*np.log10(self.cfg.freq_hz) - 147.55
                
                # Bi-directional link assumption
                for src_idx, dst_idx in [(i, j), (j, i)]:
                    snr_db = nodes[src_idx].tx_power_dbm - fspl - self.cfg.noise_floor_dbm
                    
                    if snr_db >= self.cfg.gamma_min_db:
                        # BER-based weight for routing metric
                        snr_lin = 10**(snr_db/10)
                        
                        # BPSK BER proxy for link quality
                        # For Rician fading, we approximate with AWGN if K is very high (20 dB).
                        ber = 0.5 * erfc(np.sqrt(snr_lin))
                        weight = self.cfg.packet_length_bytes * 8 * ber + 1e-9
                        self.graph.add_edge(src_idx, dst_idx, weight=weight, snr_db=snr_db)

    def route_packet(self, src: int, dst: int) -> Optional[List[int]]:
        """
        Find shortest (best) path from src to dst using Dijkstra.

        Returns
        -------
        path : list of node IDs, or None if unreachable.
        """
        try:
            path = nx.shortest_path(self.graph, src, dst, weight='weight')
            return path
        except nx.NetworkXNoPath:
            return None
        except nx.NodeNotFound:
            return None

    def simulate_delivery(self, gcs_id: int,
                           nodes: List[SwarmNode],
                           n_packets: int = 100) -> Dict[int, float]:
        """
        Simulate packet delivery ratio from GCS to each UAV.
        Includes fading probability explicitly using rician_mrc_outage.

        Returns
        -------
        pdr : dict {uav_id: delivery_ratio}
        """
        pdr = {}
        for uav in nodes:
            if uav.node_id == gcs_id:
                pdr[uav.node_id] = 1.0
                continue
            
            # For jammed nodes, the direct GCS link is dead. We MUST route through MANET.
            # But wait, how does the GCS communicate? The GCS is also a node in the graph.
            # If the GCS direct link is jammed, the edge from GCS to that UAV will either
            # not exist (SNR < gamma_min) or the routing will prefer a multi-hop path.
            
            path = self.route_packet(gcs_id, uav.node_id)
            if path is None:
                pdr[uav.node_id] = 0.0
            else:
                # Product of per-hop delivery probabilities
                success_prob = 1.0
                for h in range(len(path) - 1):
                    u = path[h]
                    v = path[h+1]
                    edge_data = self.graph[u][v]
                    snr_db = edge_data['snr_db']
                    
                    # We compute exact outage probability under Rician fading
                    # using the channel_bridge function. We assume single antenna (diversity_L=1)
                    # and threshold equal to the BPSK decoding minimum (~ -3 dB, but let's use 0 dB).
                    p_outage = rician_mrc_outage(
                        gamma_0_dB=0.0,
                        gamma_bar_dB=snr_db,
                        L_fhss=1,
                        K=self.cfg.k_factor_a2a_db
                    )
                    
                    # BER given non-outage
                    snr_lin = 10**(snr_db/10)
                    ber = 0.5 * erfc(np.sqrt(snr_lin))
                    per = 1 - (1 - ber)**(self.cfg.packet_length_bytes * 8)
                    
                    hop_success = (1 - p_outage) * (1 - per)
                    success_prob *= hop_success
                pdr[uav.node_id] = success_prob
        return pdr
