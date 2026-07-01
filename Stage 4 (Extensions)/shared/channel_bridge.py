"""
shared/channel_bridge.py
Central import hub that wires Stage 4 extensions to Stage 2 (Remediated) and
Stage 3 (Phase D) code. Nothing is duplicated; everything is re-exported.

Stage 4 modules do:
    from shared.channel_bridge import music_doa, find_music_peaks, ...
"""
import sys, os

# ── Path injection ────────────────────────────────────────────────────────────
# Resolve project root as the grandparent of this file's directory
_THIS_FILE = os.path.abspath(__file__)
_ROOT      = os.path.abspath(os.path.join(os.path.dirname(_THIS_FILE), '..', '..'))
_S2R       = os.path.join(_ROOT, "Stage 2 (Remediated)")
_S2        = os.path.join(_ROOT, "Stage 2")
_S3        = os.path.join(_ROOT, "Stage 3 (Phase D)")

for _p in [_S2R, _S2, _S3]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Stage 2 (Remediated) — Phase B beamforming ───────────────────────────────
from phase_b_beamforming_remediated import (
    music_doa,                        # (R_xx, num_sources, scan_resolution_deg) → (angles_deg, spectrum)
    find_music_peaks,                 # (scan_angles_deg, spectrum, num_sources)  → ndarray of est. angles (deg)
    lcmv_beamformer,                  # (R_xx, theta_s_rad, theta_j_list_rad)     → weight vector w
    generate_received_signal_rician,  # (N, theta_s_deg, theta_j_list_deg, SNR_dB, INR_dB_list, L, ...) → (X, R_sample, R_true)
    build_rician_covariance_matrix,   # (N, theta_s, sigma2_s, K_sig, theta_j, sigma2_j, K_jam, sigma2_n) → R_xx
    compute_crlb_doa_snr_aware,       # (N, L, SNR_dB, theta_deg, calib_err_deg) → sigma_deg (RSS)
    compute_output_sinr,              # (w, R_jn, SNR_dB, theta_s) → SINR_dB
    ula_steering_vector,              # (N, theta_rad) → (N,) complex
)

# ── Stage 2 (Remediated) — Simulator core ────────────────────────────────────
from simulator_core_remediated import (
    rician_mrc_outage,    # (gamma_0_dB, gamma_bar_dB, L_fhss, K=10.0) → P_out
    detection_prob,       # (t_s_sec, BW_hz, JNR_dB, P_fa=0.01)        → P_d
    lcmv_with_fallback,   # (R_hat, a_sig, theta_null_deg, sigma_theta_deg, N) → (w, cond, mode, null_deg)
    JammerUKF,            # UKF tracker: predict/update on bearing measurements
)

# ── Stage 3 (Phase D) — Channel model & parameters ───────────────────────────
from sinr_models import (
    fspl_db,           # (range_m, fc_hz) → dB
    baseline_channel,  # (eirp_dbw) → (sinr_db, p_out)  no beamforming
    phase_b_channel,   # (eirp_dbw) → (sinr_db, p_out)  with 45 dB LCMV null
    phase_bc_channel,  # (eirp_dbw) → (sinr_db, p_out)  null + RNCO (85% p_out reduction)
    RICIAN_K,          # = 12.0
    SINR_THRESH_DB,    # = 3.0 dB
    NULL_DEPTH_DB,     # = 45.0 dB (conservative static value from Phase B)
)

from ew_channel import (
    EWChannelBridge,   # .transmit(pkt, ChannelSnapshot) → bytes | None
    ChannelSnapshot,   # frozen dataclass: sinr_post_db, p_out, sigma_theta
)

# ── Derived constants (consistent with Stage 2/3) ────────────────────────────
N_ARRAY        = 4          # ULA elements (all stages)
NOISE_DBM      = -100.0     # thermal noise floor (sinr_models.py)
NOISE_W        = 10**((NOISE_DBM - 30) / 10)
FC_HZ          = 2.4e9      # carrier frequency
UAV_GCS_M      = 1000.0     # UAV–GCS range (m)
UAV_JAM_M      = 500.0      # UAV–Jammer range (m)
L_SNAPSHOTS    = 100        # MUSIC snapshot count (max, Stage 2 default)
L_MRC          = 4          # MRC diversity branches (MRC_L4 scenario)
LOADING_FACTOR = 0.10       # LCMV diagonal loading factor

BANDS_HZ: dict = {
    "900 MHz":  900e6,
    "2.4 GHz": 2.4e9,
    "5.8 GHz": 5.8e9,
}
