"""
cat4/urban_ray_tracer.py
Extension 4b: Urban Ray-Tracing Channel Model

Mathematical basis (Section 6.2 of Extension Plan):
  - 2-D urban canyon with rectangular buildings.
  - Image source method for reflections.
  - Path contribution: alpha_i = (lambda / 4*pi*d_i) * Product(|Gamma|) * Product(|T|)
  - Fresnel TE reflection coefficient for concrete.
  - Rician K-factor derived from Channel Impulse Response (CIR).

Integration:
  - Replaces empirical K-factor when urban flag is set.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

@dataclass
class Building:
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def walls(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Return 4 wall segments as ((x1,y1),(x2,y2))."""
        return [
            (np.array([self.x_min, self.y_min]), np.array([self.x_max, self.y_min])),  # south
            (np.array([self.x_max, self.y_min]), np.array([self.x_max, self.y_max])),  # east
            (np.array([self.x_max, self.y_max]), np.array([self.x_min, self.y_max])),  # north
            (np.array([self.x_min, self.y_max]), np.array([self.x_min, self.y_min])),  # west
        ]

@dataclass
class RayPath:
    vertices: List[np.ndarray]   # sequence of (x,y) points
    path_type: str               # "direct", "reflection", "diffraction"
    total_length_m: float
    reflection_coeffs: List[complex]
    alpha: complex               # complex path amplitude

@dataclass
class RayTracerConfig:
    freq_hz: float = 2.4e9
    n_rays: int = 360
    max_bounces: int = 2
    max_path_m: float = 2000.0
    capture_radius_m: float = 5.0
    eps_r_concrete: complex = 5.0 - 0.1j


class UrbanRayTracer2D:
    """
    2-D urban ray tracer using image source method for reflections.
    """
    def __init__(self, buildings: List[Building], config: RayTracerConfig):
        self.buildings = buildings
        self.cfg = config
        self.lam = 3e8 / config.freq_hz

    def _fresnel_te(self, theta_i: float) -> complex:
        """
        Compute TE Fresnel reflection coefficient.
        theta_i is incidence angle with the NORMAL of the wall.
        """
        eps = self.cfg.eps_r_concrete
        ct = np.cos(theta_i)
        sqrt_term = np.sqrt(eps - np.sin(theta_i)**2 + 0j)
        return (ct - sqrt_term) / (ct + sqrt_term + 1e-12)

    def _segment_intersect(self, p1: np.ndarray, p2: np.ndarray, 
                           w1: np.ndarray, w2: np.ndarray) -> Optional[Tuple[float, np.ndarray]]:
        """
        Find intersection of line segment [p1,p2] with wall [w1,w2].
        Returns (t, point) where t is fraction along [p1,p2], or None if no intersection.
        """
        d1 = p2 - p1
        d2 = w2 - w1
        denom = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(denom) < 1e-10:
            return None
            
        t = ((w1[0]-p1[0])*d2[1] - (w1[1]-p1[1])*d2[0]) / denom
        u = ((w1[0]-p1[0])*d1[1] - (w1[1]-p1[1])*d1[0]) / denom
        
        # We use a small epsilon to avoid self-intersection on bounces
        if 1e-6 < t < 1 - 1e-6 and 0 <= u <= 1:
            return t, p1 + t * d1
        return None

    def trace_direct(self, tx: np.ndarray, rx: np.ndarray) -> Optional[RayPath]:
        """Check if direct LOS path exists (no building obstruction)."""
        d = rx - tx
        dist = np.linalg.norm(d)
        
        blocked = False
        for bldg in self.buildings:
            for wall in bldg.walls:
                if self._segment_intersect(tx, rx, wall[0], wall[1]):
                    blocked = True
                    break
            if blocked:
                break
                
        if not blocked:
            alpha = (self.lam / (4 * np.pi * dist)) * np.exp(-1j * 2 * np.pi * dist / self.lam)
            return RayPath([tx, rx], "direct", dist, [], alpha)
        return None

    def get_mirror_point(self, pt: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
        """Reflect pt across the line defined by w1, w2."""
        wall_vec = w2 - w1
        wall_dir = wall_vec / np.linalg.norm(wall_vec)
        pt_vec = pt - w1
        proj_len = np.dot(pt_vec, wall_dir)
        proj_pt = w1 + proj_len * wall_dir
        return 2 * proj_pt - pt

    def trace_single_bounce(self, tx: np.ndarray, rx: np.ndarray) -> List[RayPath]:
        """Find 1st-order reflected paths using Image Source Method."""
        paths = []
        for bldg in self.buildings:
            for wall in bldg.walls:
                w1, w2 = wall
                # Compute image of tx
                tx_img = self.get_mirror_point(tx, w1, w2)
                
                # Check intersection of image->rx with the wall
                intersect = self._segment_intersect(tx_img, rx, w1, w2)
                if intersect is not None:
                    _, bounce_pt = intersect
                    
                    # Verify tx->bounce_pt is not blocked
                    blocked = False
                    for b2 in self.buildings:
                        for w_in in b2.walls:
                            if self._segment_intersect(tx, bounce_pt, w_in[0], w_in[1]):
                                blocked = True
                                break
                        if blocked: break
                    if blocked: continue
                    
                    # Verify bounce_pt->rx is not blocked
                    for b2 in self.buildings:
                        for w_in in b2.walls:
                            if self._segment_intersect(bounce_pt, rx, w_in[0], w_in[1]):
                                blocked = True
                                break
                        if blocked: break
                    if blocked: continue

                    # Compute reflection coefficient
                    d1 = bounce_pt - tx
                    d2 = rx - bounce_pt
                    dist = np.linalg.norm(d1) + np.linalg.norm(d2)
                    
                    wall_vec = w2 - w1
                    wall_norm = np.array([-wall_vec[1], wall_vec[0]])
                    wall_norm = wall_norm / np.linalg.norm(wall_norm)
                    
                    # Incidence angle with normal
                    d1_dir = d1 / np.linalg.norm(d1)
                    cos_theta_i = abs(np.dot(d1_dir, wall_norm))
                    theta_i = np.arccos(cos_theta_i)
                    
                    gamma = self._fresnel_te(theta_i)
                    
                    alpha = gamma * (self.lam / (4 * np.pi * dist)) * np.exp(-1j * 2 * np.pi * dist / self.lam)
                    paths.append(RayPath([tx, bounce_pt, rx], "reflection", dist, [gamma], alpha))
        return paths

    def compute_cir(self, tx: np.ndarray,
                    rx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute channel impulse response amplitudes and delays.
        """
        c = 3e8
        paths = []
        
        # Direct
        direct = self.trace_direct(tx, rx)
        if direct:
            paths.append(direct)
            
        # 1-bounce
        bounces = self.trace_single_bounce(tx, rx)
        paths.extend(bounces)
        
        if not paths:
            return np.array([]), np.array([], dtype=complex)
            
        delays = np.array([p.total_length_m / c for p in paths])
        amplitudes = np.array([p.alpha for p in paths])
        
        # Sort by delay
        idx = np.argsort(delays)
        return delays[idx], amplitudes[idx]

    def received_power_dbm(self, tx_power_dbm: float,
                            tx: np.ndarray, rx: np.ndarray) -> float:
        """Compute total received power in dBm from all ray paths."""
        delays, amps = self.compute_cir(tx, rx)
        if len(amps) == 0:
            return -np.inf
            
        # For narrowband received power (e.g. evaluating path loss including flat fading),
        # paths sum coherently. This is required to capture the d^-4 two-ray roll-off.
        P_sum = np.abs(np.sum(amps))**2
        P_tx_w = 10**((tx_power_dbm - 30) / 10)
        P_rx_w = P_sum * P_tx_w
        if P_rx_w == 0: return -np.inf
        return 10 * np.log10(P_rx_w) + 30

    def rician_k_from_cir(self, delays: np.ndarray,
                           amplitudes: np.ndarray) -> float:
        """Estimate K-factor from CIR (first path assumed LOS)."""
        if len(amplitudes) == 0:
            return 0.0
        if len(amplitudes) == 1:
            return 1e6 # Practically infinite K-factor
            
        P_los = np.abs(amplitudes[0])**2
        P_scatter = np.sum(np.abs(amplitudes[1:])**2)
        return P_los / (P_scatter + 1e-30)
