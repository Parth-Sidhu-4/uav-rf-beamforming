import numpy as np
import quaternion

def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> np.quaternion:
    """
    Converts aerospace Euler angles (ZYX convention) to a quaternion.
    Angles should be in radians.
    """
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return np.quaternion(qw, qx, qy, qz)

def rotate_vector(vec: np.ndarray, q: np.quaternion) -> np.ndarray:
    """
    Rotates a 3D vector using a quaternion.
    """
    vec_q = np.quaternion(0, vec[0], vec[1], vec[2])
    q_conj = q.conjugate()
    rotated = q * vec_q * q_conj
    return np.array([rotated.x, rotated.y, rotated.z])

def rotate_points(points: np.ndarray, q: np.quaternion) -> np.ndarray:
    """
    Rotates an Nx3 array of points using a quaternion.
    We convert the quaternion to a rotation matrix for vectorized operations.
    """
    rot_matrix = quaternion.as_rotation_matrix(q)
    return points @ rot_matrix.T
