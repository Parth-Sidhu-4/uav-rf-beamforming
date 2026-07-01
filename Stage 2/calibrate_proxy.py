import numpy as np

def get_steering_vector(N, theta_deg):
    theta_rad = np.radians(theta_deg)
    return np.exp(-1j * np.pi * np.arange(N) * np.sin(theta_rad)).reshape(N, 1)

def lcmv_weights(R_hat, C, f, loading_factor=0.10):
    delta = loading_factor * np.real(np.trace(R_hat)) / R_hat.shape[0]
    R_loaded = R_hat + delta * np.eye(R_hat.shape[0])
    R_inv = np.linalg.inv(R_loaded)
    w = R_inv @ C @ np.linalg.inv(C.conj().T @ R_inv @ C) @ f
    return w

N = 4
snr = 15.0
inr = 50.0

a_s = get_steering_vector(N, -45.0)
a_j_true = get_steering_vector(N, 0.0)

R = 10**(snr/10) * (a_s @ a_s.conj().T) + 10**(inr/10) * (a_j_true @ a_j_true.conj().T) + np.eye(N)

# Test zero error
C_zero = np.hstack([a_s, get_steering_vector(N, 0.0)])
f_zero = np.array([[1.0], [0.0]])
w_zero = lcmv_weights(R, C_zero, f_zero)
nd_zero = 10 * np.log10(np.abs(w_zero.conj().T @ a_j_true)[0,0]**2)
print(f"Zero Error ND: {nd_zero:.2f} dB")

# Test 0.5 error
C_half = np.hstack([a_s, get_steering_vector(N, 0.5)])
w_half = lcmv_weights(R, C_half, f_zero)
nd_half = 10 * np.log10(np.abs(w_half.conj().T @ a_j_true)[0,0]**2)
print(f"0.5 Error ND: {nd_half:.2f} dB")

# Sweep to fit
errors = np.linspace(0, 5, 50)
nds = []
for err in errors:
    C_err = np.hstack([a_s, get_steering_vector(N, err)])
    w_err = lcmv_weights(R, C_err, f_zero)
    nds.append(10 * np.log10(np.abs(w_err.conj().T @ a_j_true)[0,0]**2))

# Fit a polynomial or simple linear curve
from scipy.optimize import curve_fit

def proxy_func(err, base, slope):
    return np.maximum(-65.0, base + slope * err)

popt, _ = curve_fit(proxy_func, errors, nds, bounds=([-100, 0], [0, 50]))
print(f"Empirical fit: base={popt[0]:.2f}, slope={popt[1]:.2f}")
