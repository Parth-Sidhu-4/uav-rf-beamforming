import numpy as np

def compute_lcmv_weights(R_xx: np.ndarray, a_target: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    """
    Computes the LCMV beamformer weights.
    If a mask is provided, it incorporates it into the steering vector, 
    effectively nulling out the contribution of shadowed antennas.
    """
    if mask is not None:
        # Apply the mask. We can just multiply the steering vector by the mask.
        # This forces the weights for shadowed elements to zero out in effect.
        # Alternatively, we could remove those rows/cols from R_xx, but masking
        # a_target is standard for simulating element failure or blockage 
        # in a fixed physical array if the DSP isn't dynamically resizing the matrix.
        a_target = a_target * mask

    # R_xx_inv
    # Add a small diagonal loading to prevent singular matrix errors 
    # especially if mask zeroed out some elements
    N = R_xx.shape[0]
    R_xx_loaded = R_xx + 1e-6 * np.eye(N)
    
    try:
        R_inv = np.linalg.inv(R_xx_loaded)
    except np.linalg.LinAlgError:
        return np.zeros(N, dtype=complex)
    
    numerator = R_inv @ a_target
    denominator = np.conj(a_target).T @ R_inv @ a_target
    
    # Avoid division by zero if completely shadowed
    if np.abs(denominator) < 1e-12:
        return np.zeros(N, dtype=complex)
        
    w = numerator / denominator
    
    if mask is not None:
        # Explicitly zero out weights of blocked antennas
        w = w * mask
        
    return w

def get_steering_vector(positions: np.ndarray, k_vec: np.ndarray) -> np.ndarray:
    """
    Computes the standard narrowband steering vector.
    positions: Nx3
    k_vec: 3x1 wave vector
    """
    phases = np.dot(positions, k_vec)
    return np.exp(1j * phases)
