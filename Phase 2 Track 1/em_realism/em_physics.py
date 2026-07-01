import numpy as np
import scipy.special

def fresnel_diffraction_gain(nu: np.ndarray) -> np.ndarray:
    """
    Computes the complex Fresnel Knife-Edge diffraction gain F(nu).
    
    nu: Dimensionless Fresnel parameter.
        nu > 0 means SHADOWED (ray intersects obstacle)
        nu < 0 means ILLUMINATED (ray clears obstacle)
        
    Returns:
    F: Complex gain relative to the direct unobstructed path.
    """
    # scipy.special.fresnel(x) returns (S(x), C(x))
    S_nu, C_nu = scipy.special.fresnel(nu)
    
    # F(nu) = ((1+j)/2) * [ (0.5 - C(nu)) - j(0.5 - S(nu)) ]
    F = ((1 + 1j) / 2) * ((0.5 - C_nu) - 1j * (0.5 - S_nu))
    return F

def polarisation_loss_gain(phi_tilt: np.ndarray) -> np.ndarray:
    """
    Computes the amplitude attenuation due to polarisation mismatch.
    Assuming perfect infinite Cross-Polarisation Discrimination (XPD) for now,
    so Lp = |cos(phi_tilt)|.
    
    phi_tilt: Array of angles (in radians) between element polarisation axis 
              and the incoming wave polarisation axis.
              
    Returns:
    Lp: Amplitude attenuation factor [0.0, 1.0].
    """

def compute_distances(P_rays: np.ndarray, D_ray: np.ndarray, V1: np.ndarray, V2: np.ndarray):
    """
    Computes shortest distance h and d1 from N rays to M line segments.
    P_rays: (N, 3) origins
    D_ray: (3,) direction, assumed normalized
    V1: (M, 3) start points of segments
    V2: (M, 3) end points of segments
    
    Returns:
    h: (N, M) minimum distances
    d1: (N, M) distance from ray origin to the foot on the segment
    """
    N = P_rays.shape[0]
    M = V1.shape[0]
    
    E = V2 - V1
    D = D_ray.reshape(1, 3)
    
    P_exp = P_rays[:, np.newaxis, :]
    V1_exp = V1[np.newaxis, :, :]
    W0 = P_exp - V1_exp
    
    a = np.sum(D**2)
    b = np.sum(D * E, axis=1)[np.newaxis, :]
    c = np.sum(E**2, axis=1)[np.newaxis, :]
    
    d = np.sum(D[np.newaxis, :, :] * W0, axis=2)
    e = np.sum(E[np.newaxis, :, :] * W0, axis=2)
    
    denom = a * c - b**2
    denom = np.where(denom < 1e-8, 1e-8, denom)
    
    s_c = (a * e - b * d) / denom
    s_c_clamped = np.clip(s_c, 0.0, 1.0)
    
    t_c = (b * s_c_clamped - d) / a
    t_c_clamped = np.maximum(t_c, 0.0)
    
    s_c_recalc = np.where(t_c < 0.0, -e / c, s_c_clamped)
    s_c_recalc_clamped = np.clip(s_c_recalc, 0.0, 1.0)
    
    s_final = np.where(t_c < 0.0, s_c_recalc_clamped, s_c_clamped)
    t_final = t_c_clamped
    
    E_exp = E[np.newaxis, :, :]
    s_final_exp = s_final[:, :, np.newaxis]
    P_edge = V1_exp + s_final_exp * E_exp
    
    t_final_exp = t_final[:, :, np.newaxis]
    D_exp = D[np.newaxis, :, :]
    P_ray = P_exp + t_final_exp * D_exp
    
    W = P_ray - P_edge
    h = np.linalg.norm(W, axis=2)
    d1 = np.linalg.norm(P_edge - P_exp, axis=2)
    
    return h, d1, W
