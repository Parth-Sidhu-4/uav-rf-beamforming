"""
cat4/conformal_array.py
Extension 4a: Cylindrical Conformal Array

Mathematical basis (Section 6.1 of Extension Plan):
  - Cylindrical geometry: elements at (R cos(phi), R sin(phi), 0)
  - Element factor: gain g_n = (u . n_n)^q for outward hemisphere, 0 otherwise
  - Steering vector: 3-D rotation from NED to body frame before element evaluation
  - Null steering: LCMV over the conformal element patterns

Integration:
  - Replaces 1D ULA models when conformal mode is selected.
  - Takes roll/pitch/yaw from INS to rotate the steering vectors dynamically.
"""

import numpy as np
from typing import Tuple, List, Optional

class CylindricalConformalArray:
    """
    N-element conformal array on a cylinder of radius R.
    Elements span a specified arc (e.g., 2pi for full cylinder, pi for half).
    """
    def __init__(self, N: int = 8, R: float = 0.5, arc_span_rad: float = 2*np.pi,
                 freq_hz: float = 2.4e9, q: float = 1.0):
        self.N = N
        self.R = R
        self.freq = freq_hz
        self.k = 2 * np.pi * freq_hz / 3e8
        self.q = q
        
        # Element angular positions
        # If arc is 2pi, distribute evenly over [0, 2pi - dphi]
        if abs(arc_span_rad - 2*np.pi) < 1e-5:
            self.phi = np.linspace(0, 2*np.pi, N, endpoint=False)
        else:
            self.phi = np.linspace(-arc_span_rad/2, arc_span_rad/2, N)
            
        # Positions (Nx3)
        self.positions = np.column_stack([
            R * np.cos(self.phi),
            R * np.sin(self.phi),
            np.zeros(N)
        ])
        
        # Outward normals (Nx3)
        self.normals = np.column_stack([
            np.cos(self.phi),
            np.sin(self.phi),
            np.zeros(N)
        ])

    def steering_vector(self, theta_rad: float, phi_rad: float,
                        rotation_matrix: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute 3-D conformal steering vector including element gain.

        Parameters
        ----------
        theta_rad : float — elevation angle (0 = broadside, pi/2 = endfire in NED).
                    Note: for airborne coordinates, broadside is often theta=pi/2 (horizontal).
        phi_rad : float — azimuth angle in NED.
        rotation_matrix : ndarray (3x3) — R_NED_to_body. Transforms a vector in NED 
                          into the body frame where the antenna is fixed.
                          If None, assumes identity.

        Returns
        -------
        a : ndarray (N,) complex
        """
        # u is the direction OF ARRIVAL in NED frame
        u_ned = np.array([
            np.sin(theta_rad) * np.cos(phi_rad),
            np.sin(theta_rad) * np.sin(phi_rad),
            np.cos(theta_rad)
        ])
        
        if rotation_matrix is not None:
            # Transform DOA vector into body frame
            u_body = rotation_matrix @ u_ned
        else:
            u_body = u_ned
            
        # Phases depend on dot product of element position and DOA vector
        phases = self.k * (self.positions @ u_body)
        
        # Element gain depends on dot product of outward normal and DOA vector
        dot_products = self.normals @ u_body
        gain = np.where(dot_products > 0, dot_products ** self.q, 0.0)
        
        a = gain * np.exp(1j * phases)
        return a

    def lcmv_weights(self, R_yy: np.ndarray,
                     desired: Tuple[float, float],
                     nulls: List[Tuple[float, float]],
                     rotation_matrix: Optional[np.ndarray] = None) -> np.ndarray:
        """
        LCMV weights for conformal array.
        desired : (theta_rad, phi_rad) — SOI direction in NED.
        nulls   : list of (theta_rad, phi_rad) — jammer directions in NED.
        """
        constraints = [desired] + nulls
        A = np.column_stack([self.steering_vector(th, ph, rotation_matrix)
                             for th, ph in constraints]).astype(complex)
                             
        f = np.zeros(len(constraints), dtype=complex)
        f[0] = 1.0  # Unity gain in desired direction, 0 in null directions
        
        # Regularized inversion
        R_inv = np.linalg.inv(R_yy + 1e-8 * np.eye(self.N))
        M = A.conj().T @ R_inv @ A
        
        weights = R_inv @ A @ np.linalg.solve(M, f)
        return weights

    def pattern(self, theta_range: np.ndarray,
                phi_range: np.ndarray,
                weights: Optional[np.ndarray] = None,
                rotation_matrix: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute beampattern |w^H a(θ,φ)|² over a 2-D angular grid.
        Returns P : ndarray (len(theta_range), len(phi_range))
        """
        if weights is None:
            weights = np.ones(self.N) / self.N
            
        P = np.zeros((len(theta_range), len(phi_range)))
        for i, th in enumerate(theta_range):
            for j, ph in enumerate(phi_range):
                a = self.steering_vector(th, ph, rotation_matrix)
                P[i, j] = np.abs(weights.conj() @ a) ** 2
        return P

def euler_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Constructs R_NED_to_body rotation matrix from Euler angles (in radians).
    Order: yaw (Z), pitch (Y), roll (X).
    R_NED_to_body = R_roll * R_pitch * R_yaw
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    
    R_z = np.array([
        [cy,  sy, 0],
        [-sy, cy, 0],
        [0,   0,  1]
    ])
    
    R_y = np.array([
        [cp,  0, -sp],
        [0,   1,  0],
        [sp,  0,  cp]
    ])
    
    R_x = np.array([
        [1,  0,   0],
        [0,  cr, sr],
        [0, -sr, cr]
    ])
    
    return R_x @ R_y @ R_z
