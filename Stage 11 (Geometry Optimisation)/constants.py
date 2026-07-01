import numpy as np

# Stage 11: Array Geometry Optimisation Constants

# Active element threshold for the conformal array.
# Derivation: For LCMV to utilise an element's spatial degree of freedom, the jammed signal must remain
# strictly above the receiver's thermal noise floor. Assuming a high-power jammer (J/N approx +60 dB),
# an element can suffer up to -40 dB of power attenuation (-20 dB in voltage amplitude) before the 
# jammer signal vanishes into the noise floor and the element becomes mathematically useless for nulling.
NACT_THRESHOLD_DB = -20.0
NACT_THRESHOLD = 10**(NACT_THRESHOLD_DB / 20.0) # 0.1 amplitude

# Soft-surrogate parameters
SIGMA_TEMPERATURE = 0.05
SOFTMIN_BETA = 10.0

# Array spacing constraint
MIN_SPACING_M = 0.06  # ~ lambda/2 for 2.4 GHz
SPACING_PENALTY_WEIGHT = 1e4

# GA / DE Parameters
M_CANDIDATES = 2000
DE_POPSIZE = 15
DE_MAXITER = 50

# Wavelength
LAMBDA_RF = 0.125 # 2.4 GHz
