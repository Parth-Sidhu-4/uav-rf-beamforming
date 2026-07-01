import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))
from conformal_array import CylindricalConformalArray, euler_to_matrix


def main():
    print("Running Category 4a: Cylindrical Conformal Array\n")
    
    freq_hz = 2.4e9
    lam = 3e8 / freq_hz
    N = 8
    
    # Check 1: Omnidirectional coverage test
    arr = CylindricalConformalArray(N=N, R=0.5, arc_span_rad=2*np.pi, freq_hz=freq_hz, q=1.0)
    # Target at azimuth 0 (x-axis)
    u_0 = arr.steering_vector(theta_rad=np.pi/2, phi_rad=0.0)
    # Check that element 0 (at phi=0) has gain 1.0, and element N//2 (at phi=pi) has gain 0.0
    print("[Check 1] Element Shadowing (Target at +X direction):")
    print(f"  Element 0 (facing +X) amplitude: {np.abs(u_0[0]):.3f}")
    print(f"  Element {N//2} (facing -X) amplitude: {np.abs(u_0[N//2]):.3f}")
    if np.abs(u_0[0]) > 0.99 and np.abs(u_0[N//2]) < 1e-5:
        print("  PASS: Shadowing correctly zeros out back-facing elements.\n")
    else:
        print("  FAIL: Shadowing logic failed.\n")

    # Check 2: LCMV Null Depth
    desired = (np.pi/2, 0.0)      # GCS at broadside horizontal, azimuth 0
    jammer = (np.pi/2, np.pi/4)   # Jammer at horizontal, azimuth 45 deg
    
    # Generate covariance matrix (signal + jammer + noise)
    a_s = arr.steering_vector(*desired)
    a_j = arr.steering_vector(*jammer)
    
    snr_db = 10.0
    jsr_db = 30.0
    
    P_s = 10**(snr_db/10)
    P_j = 10**(jsr_db/10)
    P_n = 1.0
    
    R_yy = P_s * np.outer(a_s, a_s.conj()) + P_j * np.outer(a_j, a_j.conj()) + P_n * np.eye(N)
    
    weights = arr.lcmv_weights(R_yy, desired, nulls=[jammer])
    
    # Check gain in desired direction
    gain_s = np.abs(weights.conj() @ a_s)**2
    # Check null depth in jammer direction
    gain_j = np.abs(weights.conj() @ a_j)**2
    null_depth_db = 10 * np.log10(gain_j / gain_s + 1e-12)
    
    print("[Check 2] LCMV Null Depth:")
    print(f"  Target Gain (linear): {gain_s:.3f}")
    print(f"  Jammer Null Depth: {null_depth_db:.1f} dBc")
    if null_depth_db < -40.0:
        print("  PASS: Null depth < -40 dBc.\n")
    else:
        print("  FAIL: Null depth insufficient.\n")

    # Check 3: Body Rotation
    # Assume UAV rolls by 45 degrees. The desired signal comes from NED azimuth 0, horizontal.
    roll = np.pi/4
    pitch = 0.0
    yaw = 0.0
    R_mat = euler_to_matrix(roll, pitch, yaw)
    
    weights_rot = arr.lcmv_weights(R_yy, desired, nulls=[jammer], rotation_matrix=R_mat)
    a_s_rot = arr.steering_vector(*desired, rotation_matrix=R_mat)
    a_j_rot = arr.steering_vector(*jammer, rotation_matrix=R_mat)
    
    gain_s_rot = np.abs(weights_rot.conj() @ a_s_rot)**2
    gain_j_rot = np.abs(weights_rot.conj() @ a_j_rot)**2
    null_depth_rot_db = 10 * np.log10(gain_j_rot / gain_s_rot + 1e-12)
    
    print("[Check 3] Body Rotation (Roll = 45 deg):")
    print(f"  Rotated Target Gain: {gain_s_rot:.3f}")
    print(f"  Rotated Jammer Null Depth: {null_depth_rot_db:.1f} dBc")
    if null_depth_rot_db < -40.0:
        print("  PASS: Rotation logic maintains nulls correctly.\n")
    else:
        print("  FAIL: Rotation logic failed.\n")

if __name__ == "__main__":
    main()
