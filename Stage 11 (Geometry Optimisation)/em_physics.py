import numpy as np
import quaternion
from numba import njit
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

def compute_plf(element_pol_body: np.ndarray, q_attitude: np.quaternion, jammer_pol_world: np.ndarray) -> np.ndarray:
    """
    Computes the Polarisation Loss Factor (PLF) as an amplitude attenuation factor.
    Lp = | h_jammer_world . h_element_world |
    
    element_pol_body: (N, 3) nominal polarisation unit vectors of the N elements in body frame
    q_attitude: 3D attitude quaternion (body to world rotation)
    jammer_pol_world: (3,) unit vector of jammer's E-field in world frame
    
    Returns:
    PLF_amp: (N,) amplitude attenuation factor [0.0, 1.0].
    """
    from attitude import rotate_points
    # Rotate element polarisations to world frame
    # Since q_attitude is typically euler_to_quaternion(phi, theta, psi) mapping body to world,
    # we just rotate the body vectors by q_attitude.
    element_pol_world = rotate_points(element_pol_body, q_attitude)
    
    # Dot product with jammer E-field
    # PLF is the power ratio, so amplitude factor is |cos(theta)| = | dot |
    dot_products = np.sum(element_pol_world * jammer_pol_world, axis=1)
    plf_amp = np.abs(dot_products)
    return plf_amp
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
    
    P_ray_origins_exp = P_rays[:, np.newaxis, :]
    D_exp2 = D[np.newaxis, :, :]
    d1 = np.sum((P_edge - P_ray_origins_exp) * D_exp2, axis=2)
    
    return h, d1, W

def precompute_h_W_Pedge(P_rays, V1, V2):
    N = P_rays.shape[0]
    M = V1.shape[0]
    
    E = V2 - V1
    
    P_exp = P_rays[:, np.newaxis, :]
    V1_exp = V1[np.newaxis, :, :]
    W0 = P_exp - V1_exp
    E_exp = E[np.newaxis, :, :]
    
    t_num = np.sum(W0 * E_exp, axis=2)
    t_den = np.sum(E_exp * E_exp, axis=2)
    t = t_num / t_den
    
    t_final = np.clip(t, 0.0, 1.0)
    
    t_final_exp = t_final[:, :, np.newaxis]
    P_edge = V1_exp + t_final_exp * E_exp
    
    W = P_exp - P_edge
    h = np.linalg.norm(W, axis=2)
    return h, W, P_edge
