"""
cat3/japp.py
Extension 3a: Jammer Anti-Path Planning (JAPP)

Given an estimated jammer position (with uncertainty covariance from Cat 1
WLS/CRLB), computes UAV waypoints that maximise terrain-shadow masking to
deny the jammer a line-of-sight to the UAV.

Mathematical basis (Section 5.1 of Extension Plan):
  - Jammer visibility check: for each candidate path cell, test whether the
    direct jammer→cell ray clears all terrain obstacles (using knife-edge ν < -0.7).
  - Risk cost: C(x,y) = P(jammer can see cell) × link-loss penalty
  - Planning: Dijkstra on a 2-D grid with risk-weighted edges

Integration:
  - Uses knife_edge.py for terrain-shadow check
  - Uses Cat 1 CRLB covariance for jammer position uncertainty
  - Uses channel_bridge rician_mrc_outage() to compute link gain from shadowing
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List, Optional
import heapq

from channel_bridge import rician_mrc_outage, SINR_THRESH_DB, RICIAN_K, L_MRC
from knife_edge import knife_edge_total_loss_db, gaussian_hill_profile


@dataclass
class JAPPConfig:
    # Mission geometry (metres)
    gcs_pos_m:       np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    target_pos_m:    np.ndarray = field(default_factory=lambda: np.array([2000.0, 0.0]))
    # Grid resolution
    grid_dx_m:       float = 50.0
    grid_dy_m:       float = 50.0
    grid_xlim:       Tuple = (-200, 2200)
    grid_ylim:       Tuple = (-600, 600)
    # UAV parameters
    uav_height_agl_m: float = 50.0
    uav_speed_ms:    float  = 15.0
    # Jammer
    jammer_pos_m:    np.ndarray = field(default_factory=lambda: np.array([1000.0, 400.0]))
    jammer_cov_m2:   np.ndarray = field(default_factory=lambda: np.eye(2) * 50.0**2)
    jammer_height_m: float = 2.0
    # SINR/cost parameters
    snr_base_db:     float = 15.0
    null_depth_db:   float = 45.0


class TerrainMap:
    """2-D terrain height map backed by a synthetic Gaussian profile."""

    def __init__(self, config: JAPPConfig):
        self.cfg = config
        c = config
        self.xs = np.arange(c.grid_xlim[0], c.grid_xlim[1], c.grid_dx_m)
        self.ys = np.arange(c.grid_ylim[0], c.grid_ylim[1], c.grid_dy_m)
        self.nx = len(self.xs)
        self.ny = len(self.ys)

        # Synthetic ridge across the mid-path
        sigma_m = 150.0
        ridge_x = (c.grid_xlim[1] + c.grid_xlim[0]) / 2
        self.height_map = np.zeros((self.nx, self.ny))
        for ix, x in enumerate(self.xs):
            hill_h = 80.0 * np.exp(-0.5 * ((x - ridge_x) / sigma_m) ** 2)
            self.height_map[ix, :] = hill_h

    def height_at(self, x_m: float, y_m: float) -> float:
        """Bilinear interpolation of terrain height."""
        ix = np.clip(np.searchsorted(self.xs, x_m) - 1, 0, self.nx - 2)
        iy = np.clip(np.searchsorted(self.ys, y_m) - 1, 0, self.ny - 2)
        return float(self.height_map[ix, iy])

    def terrain_profile_between(self, p1: np.ndarray, p2: np.ndarray,
                                 n_pts: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """Sample terrain height along a 2-D line segment."""
        t     = np.linspace(0, 1, n_pts)
        pts   = p1[None, :] + t[:, None] * (p2 - p1)[None, :]
        dists = np.linalg.norm(pts - p1[None, :], axis=1)
        heights = np.array([self.height_at(p[0], p[1]) for p in pts])
        return dists, heights


def jammer_visible(cell_pos: np.ndarray,
                   jammer_pos: np.ndarray,
                   terrain: TerrainMap,
                   uav_height_agl: float = 50.0,
                   jammer_height: float = 2.0,
                   freq_hz: float = 2.4e9) -> bool:
    """
    Check if jammer has line-of-sight to a UAV at cell_pos.
    A cell is 'visible' if knife-edge ν > -0.7 (no significant terrain shadow).
    """
    p_uav = cell_pos
    p_jam = jammer_pos
    x_arr, h_arr = terrain.terrain_profile_between(p_jam, p_uav)
    if len(x_arr) < 3:
        return True  # too short to shadow
    L_diff = knife_edge_total_loss_db(
        x_arr, h_arr,
        h_tx_m=jammer_height,
        h_rx_m=uav_height_agl,
        freq_hz=freq_hz
    )
    # Shadowed if diffraction loss > 6 dB (ν > ~0.0)
    return L_diff < 6.0


def japp_risk_grid(config: JAPPConfig,
                   terrain: TerrainMap,
                   n_jammer_samples: int = 100) -> np.ndarray:
    """
    Compute risk score for each grid cell using Monte Carlo jammer sampling.

    Risk = P(jammer can see cell) computed by sampling jammer positions
    from the Gaussian uncertainty N(jammer_pos, jammer_cov).

    Returns
    -------
    risk : ndarray (nx, ny)  — probability of visibility, in [0, 1]
    """
    cfg = config
    rng = np.random.default_rng(42)
    jammer_samples = rng.multivariate_normal(
        cfg.jammer_pos_m, cfg.jammer_cov_m2, size=n_jammer_samples)

    risk = np.zeros((terrain.nx, terrain.ny))

    for ix, x in enumerate(terrain.xs):
        for iy, y in enumerate(terrain.ys):
            cell = np.array([x, y])
            visible_count = sum(
                1 for j_pos in jammer_samples
                if jammer_visible(cell, j_pos, terrain,
                                   cfg.uav_height_agl_m, cfg.jammer_height_m)
            )
            risk[ix, iy] = visible_count / n_jammer_samples

    return risk


def dijkstra_path(risk: np.ndarray,
                   terrain: TerrainMap,
                   start_pos: np.ndarray,
                   goal_pos: np.ndarray,
                   risk_weight: float = 5.0) -> List[np.ndarray]:
    """
    Dijkstra shortest path on the risk-weighted grid.
    Edge cost = Euclidean distance + risk_weight * risk_at_node.

    Returns
    -------
    path : list of (x, y) waypoints in metres
    """
    def pos_to_idx(pos):
        ix = int(np.clip(np.searchsorted(terrain.xs, pos[0]) - 1, 0, terrain.nx - 1))
        iy = int(np.clip(np.searchsorted(terrain.ys, pos[1]) - 1, 0, terrain.ny - 1))
        return (ix, iy)

    def idx_to_pos(idx):
        return np.array([terrain.xs[idx[0]], terrain.ys[idx[1]]])

    start_idx = pos_to_idx(start_pos)
    goal_idx  = pos_to_idx(goal_pos)

    dist   = {start_idx: 0.0}
    parent = {start_idx: None}
    pq     = [(0.0, start_idx)]

    while pq:
        d, node = heapq.heappop(pq)
        if d > dist.get(node, float('inf')) + 1e-9:
            continue
        if node == goal_idx:
            break

        ix, iy = node
        for dix in [-1, 0, 1]:
            for diy in [-1, 0, 1]:
                if dix == 0 and diy == 0:
                    continue
                nix, niy = ix + dix, iy + diy
                if not (0 <= nix < terrain.nx and 0 <= niy < terrain.ny):
                    continue
                move_dist = np.sqrt(
                    (terrain.xs[nix] - terrain.xs[ix]) ** 2 +
                    (terrain.ys[niy] - terrain.ys[iy]) ** 2
                )
                r_cost = risk_weight * risk[nix, niy] * move_dist
                new_d  = d + move_dist + r_cost
                nbr    = (nix, niy)
                if new_d < dist.get(nbr, float('inf')):
                    dist[nbr]   = new_d
                    parent[nbr] = node
                    heapq.heappush(pq, (new_d, nbr))

    # Reconstruct path
    path = []
    node = goal_idx
    while node is not None:
        path.append(idx_to_pos(node))
        node = parent.get(node)
    path.reverse()
    return path


def direct_path(start_pos: np.ndarray, goal_pos: np.ndarray,
                n_waypoints: int = 20) -> List[np.ndarray]:
    """Straight-line direct path for comparison."""
    return [start_pos + t * (goal_pos - start_pos)
            for t in np.linspace(0, 1, n_waypoints)]


def path_risk(path: List[np.ndarray], risk: np.ndarray,
              terrain: TerrainMap) -> float:
    """Mean risk exposure along a path."""
    risks = []
    for pt in path:
        ix = int(np.clip(np.searchsorted(terrain.xs, pt[0]) - 1, 0, terrain.nx - 1))
        iy = int(np.clip(np.searchsorted(terrain.ys, pt[1]) - 1, 0, terrain.ny - 1))
        risks.append(risk[ix, iy])
    return float(np.mean(risks)) if risks else 0.0
