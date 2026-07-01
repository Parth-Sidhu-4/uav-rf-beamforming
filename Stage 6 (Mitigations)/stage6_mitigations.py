# -*- coding: utf-8 -*-
"""
stage6_mitigations.py
═══════════════════════════════════════════════════════════════════════════════
Stage 6 — UAV Defensive Architecture Hardening

Implements and compares three mitigations against the Stage 5 worst-case threat:
  Co-located jammer at GCS, activated at maximum range (t=100 s, 2000 m).

Mitigations:
  M1: Polarization Diversity   (RF/MAC layer) — dual-pol Kronecker LCMV
  M2: Loiter Fallback          (Control layer) — orbit last known position
  M3: Visual Odometry          (Navigation)    — velocity-domain Kalman update

Key physics:
  INS-only: δp(τ) = ½·b·τ²        [quadratic drift → Stage 5 catastrophe]
  With VO:  σ_p(τ) = σ_v·√τ       [random walk → bounded error → safe RTL]

Critical implementation note:
  The beamformer output metric is SINR (Signal-to-Interference-plus-Noise Ratio),
  NOT SNR. When the LCMV fails to null the jammer (co-located, single-pol), the
  jammer leaks through with gain ≈0.5, producing SINR ≈ P_GCS/P_JAM ≈ −23 dB.
  Using only thermal noise in the denominator incorrectly gives +42 dB — a critical
  bug that causes false link-recovery in the Stage 5 baseline.

Run:  python stage6_mitigations.py
Output: stage6_comparison.png + terminal summary table
═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Tuple

RNG = np.random.default_rng(seed=2025)

# ══════════════════════════════════════════════════════════════════════════════
# SIMULATION PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

# ── RF / Antenna ──────────────────────────────────────────────────────────────
CARRIER_HZ      = 2.4e9
LAMBDA_M        = 3e8 / CARRIER_HZ           # 0.125 m
N_ELEM          = 16
R_ARRAY_M       = 0.05                        # Cylinder radius (m)
P_GCS_W         = 10.0                        # GCS TX power (W)
P_JAM_W         = 2000.0                      # Jammer TX power (W)
P_NOISE_W       = 1e-12                       # Thermal noise floor (W)
XPOL_LEAKAGE    = 1e-3                        # Cross-pol isolation (−30 dB hardware floor)
LCMV_REG        = 1e-4                        # LCMV diagonal regularisation

# ── Navigation ────────────────────────────────────────────────────────────────
IMU_BIAS_MPS2   = 0.5                         # Uncalibrated accelerometer bias (m/s²)
IMU_NOISE_MPS2  = 0.05                        # IMU white noise 1-σ (m/s²)
EMA_ALPHA       = 0.15                        # EMA gain for GCS absolute position fix

# ── Visual Odometry ──────────────────────────────────────────────────────────
VO_SIGMA_MPS    = 0.15                        # VO velocity noise 1-σ (m/s)
VO_PERIOD_S     = 0.10                        # VO update period (10 Hz)

# ── Mission ───────────────────────────────────────────────────────────────────
DT_S            = 0.10
T_END_S         = 420.0
T_JAM_ON_S      = 100.0                       # Jammer activates here
CRUISE_MPS      = 20.0
LOITER_R_M      = 50.0
LOITER_V_MPS    = 12.0
LOITER_KR       = 0.30
GCS_ARRIVE_M    = 25.0                        # Arrival radius (uses ins_pos)
PER_WINDOW      = 100
PER_THRESH      = 0.90
GCS_POS         = np.array([0.0, 0.0])
WAYPOINT_POS    = np.array([2000.0, 0.0])

# ══════════════════════════════════════════════════════════════════════════════
# ENUMERATIONS
# ══════════════════════════════════════════════════════════════════════════════

class MState(Enum):
    OUTBOUND = auto()
    RTL      = auto()
    LOITER   = auto()
    LANDED   = auto()

class Mit(Enum):
    NONE     = "Stage 5 Baseline (No Mitigation)"
    LOITER   = "M2: Loiter Fallback"
    VIS_ODOM = "M3: Visual Odometry"
    DUAL_POL = "M1: Polarization Diversity"

# ══════════════════════════════════════════════════════════════════════════════
# RF LAYER — LCMV Beamformer with correct SINR output metric
# ══════════════════════════════════════════════════════════════════════════════

_phi   = np.linspace(0, 2 * np.pi, N_ELEM, endpoint=False)
APOS   = np.column_stack([R_ARRAY_M * np.cos(_phi),
                           R_ARRAY_M * np.sin(_phi),
                           np.zeros(N_ELEM)])
JONES_V = np.array([1.0, 0.0], dtype=complex)
JONES_H = np.array([0.0, 1.0], dtype=complex)


def sv_spatial(az: float, el: float = 0.0) -> np.ndarray:
    """N×1 spatial steering vector for plane wave from (az, el)."""
    k = (2 * np.pi / LAMBDA_M) * np.array([
        np.cos(el) * np.cos(az),
        np.cos(el) * np.sin(az),
        np.sin(el)])
    return np.exp(1j * (APOS @ k))


def sv_dual_pol(az: float, jones: np.ndarray, el: float = 0.0) -> np.ndarray:
    """
    2N×1 dual-polarised steering vector: a_pol = a_spatial ⊗ jones.

    Orthogonality proof (co-located, same az):
        a_pol_GCS^H · a_pol_JAM = ||a_s||² · (p_V^H · p_H) = N · 0 = 0
    → constraint matrix C is rank-2 even when θ_GCS = θ_JAM.     [M1 fix]
    """
    return np.kron(sv_spatial(az, el), jones)


def build_R(signals: List[Tuple[np.ndarray, float]],
            noise_pwr: float, N_dim: int) -> np.ndarray:
    """Analytical covariance: R = σ²_n·I + Σ Pᵢ·aᵢ·aᵢᴴ"""
    R = noise_pwr * np.eye(N_dim, dtype=complex)
    for a, p in signals:
        R += p * np.outer(a, a.conj())
    return R


def lcmv(R: np.ndarray, C: np.ndarray, f: np.ndarray) -> np.ndarray:
    """
    LCMV weights: w = R⁻¹C(CᴴR⁻¹C)⁻¹f.

    Uses lstsq for numerical robustness on rank-deficient CᴴR⁻¹C.

    Stage 5 failure mode:
      Single-pol + co-located jammer → a_gcs = a_jam → C=[a,a] rank-1
      → CᴴR⁻¹C is rank-1 → lstsq gives w^H a_gcs ≈ 0.5  (can't do better)
      → jammer leaks through with the SAME gain ≈ 0.5
      → SINR ≈ P_GCS/P_JAM = −23 dB → link dead               ← Stage 5

    M1 fix:
      Dual-pol → a_gcs_pol ⊥ a_jam_pol → C full-rank 2
      → LCMV satisfies both constraints exactly
      → w^H a_jam = 0 → jammer nulled → SINR recovered          ← M1
    """
    N    = R.shape[0]
    Rreg = R + LCMV_REG * (np.linalg.norm(R) / N) * np.eye(N, dtype=complex)
    Rinv = np.linalg.inv(Rreg)
    CRC  = C.conj().T @ Rinv @ C
    coeff, *_ = np.linalg.lstsq(CRC, f, rcond=None)
    return Rinv @ C @ coeff


def sinr_db(w: np.ndarray,
            a_sig: np.ndarray, p_sig: float,
            interferers: List[Tuple[np.ndarray, float]],
            noise_pwr: float) -> float:
    """
    Post-beamformer SINR (Signal-to-Interference-plus-Noise Ratio).

    SINR = |w^H a_sig|² · P_sig
           ─────────────────────────────────────────────
           Σᵢ |w^H aᵢ|² · Pᵢ   +   ||w||² · σ²_noise

    CRITICAL: The denominator must include jammer leakage, not just thermal
    noise. Omitting the jammer term gives a false +42 dB when the LCMV fails
    (co-located, single-pol), masking the Stage 5 catastrophic failure.
    """
    g_sig = abs(w.conj() @ a_sig) ** 2 * p_sig
    g_int = sum(abs(w.conj() @ a) ** 2 * p for a, p in interferers)
    g_nse = float(np.real(w.conj() @ w)) * noise_pwr
    denom = max(g_int + g_nse, 1e-50)
    return 10.0 * np.log10(max(g_sig / denom, 1e-30))


def compute_rf_sinr(uav: np.ndarray, gcs: np.ndarray,
                    jam_on: bool, jam_pos: np.ndarray,
                    dual_pol: bool) -> float:
    """
    Full LCMV pipeline → post-beamformer SINR for the GCS uplink.

    ┌─────────────────┬───────────────────────────────────────────────────┐
    │ Configuration   │ Result                                            │
    ├─────────────────┼───────────────────────────────────────────────────┤
    │ No jammer       │ SINR ≈ +30..+60 dB  (array gain, clean signal)   │
    │ Jammer, single-pol, different DOA  │ SINR good (spatial null works) │
    │ Jammer, single-pol, same DOA       │ SINR ≈ −23 dB  [Stage 5 FAIL] │
    │ Jammer, dual-pol, same DOA         │ SINR ≈ +20..+35 dB  [M1 FIX]  │
    └─────────────────┴───────────────────────────────────────────────────┘
    """
    def fspl(a, b, p_tx):
        r = max(float(np.linalg.norm(a - b)), 1.0)
        return p_tx * (LAMBDA_M / (4 * np.pi * r)) ** 2

    az_gcs = np.arctan2(gcs[1] - uav[1], gcs[0] - uav[0])
    p_gcs  = fspl(uav, gcs, P_GCS_W)
    N_dim  = 2 * N_ELEM if dual_pol else N_ELEM

    a_gcs = sv_dual_pol(az_gcs, JONES_V) if dual_pol else sv_spatial(az_gcs)

    if jam_on:
        az_jam = np.arctan2(jam_pos[1] - uav[1], jam_pos[0] - uav[0])
        p_jam  = fspl(uav, jam_pos, P_JAM_W)
        a_jam  = sv_dual_pol(az_jam, JONES_H) if dual_pol else sv_spatial(az_jam)

        if dual_pol:
            # Cross-pol leakage: tiny H-pol jammer energy bleeds into V-pol ports
            a_gcs_eff = a_gcs + XPOL_LEAKAGE * sv_dual_pol(az_gcs, JONES_H)
        else:
            a_gcs_eff = a_gcs

        R = build_R([(a_gcs_eff, p_gcs), (a_jam, p_jam)], P_NOISE_W, N_dim)
        C = np.column_stack([a_gcs, a_jam])
        f = np.array([1.0, 0.0], dtype=complex)
        w = lcmv(R, C, f)
        return sinr_db(w, a_gcs, p_gcs, [(a_jam, p_jam)], P_NOISE_W)
    else:
        R = build_R([(a_gcs, p_gcs)], P_NOISE_W, N_dim)
        C = a_gcs.reshape(-1, 1)
        f = np.array([1.0], dtype=complex)
        w = lcmv(R, C, f)
        return sinr_db(w, a_gcs, p_gcs, [], P_NOISE_W)


def sinr_to_per(s: float) -> float:
    """Logistic SINR→PER. PER=0.5 at SINR=5 dB; PER→1 below; PER→0 above."""
    return 1.0 / (1.0 + np.exp(0.8 * (s - 5.0)))


# ══════════════════════════════════════════════════════════════════════════════
# NAVIGATION FILTER — Error-State INS + EMA-Kalman + optional VO
# ══════════════════════════════════════════════════════════════════════════════

class NavFilter:
    """
    Error-state INS filter.

    We explicitly track the hidden velocity error δv = ins_vel − true_vel.
    The position error is its integral: δp = ∫ δv dt.

    Error dynamics (τ = time since last absolute position correction):
    ─────────────────────────────────────────────────────────────────
    Stage 5 (no corrections after jammer ON):
        δv(τ)  = b · τ              b = IMU_BIAS_MPS2 = 0.5 m/s²
        δp(τ)  = ½ · b · τ²        → 10 km at τ=200 s   [catastrophic]

    M3 (VO pseudo-Kalman velocity update at 10 Hz):
        After each update: δv ← (1−K)·δv + K·η    η ~ N(0, σ_v²)
        Steady-state:      δv ~ N(0, K²·σ_v²)      [bounded]
        Position:          δp(τ) ~ σ_v·√τ           → ~2 m at τ=200 s [safe]

    M1 (dual-pol restores link):
        GCS EMA updates continuously: δp ≈ 0         [always corrected]
    """

    def __init__(self, init_pos: np.ndarray, use_vo: bool = False):
        self.ins_pos   = init_pos.copy().astype(float)
        self.vel_error = np.zeros(2)   # δv: accumulated velocity bias error
        self.vel_var   = 1e-4          # P_v: velocity error variance (m/s)²
        self.use_vo    = use_vo
        self._vo_tmr   = 0.0

    def predict(self, vel_cmd: np.ndarray, imu_bias: np.ndarray, dt: float):
        """
        INS dead-reckoning step.

        True velocity  = vel_cmd  (perfect actuator).
        IMU measures   = true_accel + bias + noise.
        Simplified:      δv accumulates at rate ≈ bias per second.

            δv  ← δv  + (b + n) · dt       n ~ N(0, σ_imu²)
            P_v ← P_v + (|b| · dt)²
            ins_pos ← ins_pos + (vel_cmd + δv) · dt
        """
        noise = IMU_NOISE_MPS2 * RNG.standard_normal(2)
        self.vel_error += (imu_bias + noise) * dt
        self.vel_var   += (np.linalg.norm(imu_bias) * dt) ** 2 + 1e-10
        self.ins_pos   += (vel_cmd + self.vel_error) * dt

    def update_gcs(self, true_pos: np.ndarray):
        """
        EMA absolute-position correction (healthy GCS link).
        Squashes δp and partially resets δv.
        """
        z_gcs = true_pos + 0.5 * RNG.standard_normal(2)
        self.ins_pos   = (1 - EMA_ALPHA) * self.ins_pos + EMA_ALPHA * z_gcs
        self.vel_error *= (1 - EMA_ALPHA)
        self.vel_var    = max(self.vel_var * (1 - EMA_ALPHA), 1e-7)

    def update_vo(self, true_vel: np.ndarray, dt: float):
        """
        Velocity-domain VO correction (pseudo-Kalman).

        Measurement: z_VO = v_true + η,   η ~ N(0, σ_v²·I)

        Pseudo-Kalman gain:  K = P_v / (P_v + σ_v²)

        Update:
            innovation ν = z_VO − ins_vel = η − δv
            δv  ← δv + K·ν  =  (1−K)·δv + K·η    ← bias term suppressed
            P_v ← (1−K)·P_v

        Result: δv → K·η ~ N(0, K²·σ_v²)  →  BOUNDED.
        ∫ δv dt ~ σ_v·√τ  (random walk, not quadratic).         [M3 physics]
        """
        if not self.use_vo:
            return
        self._vo_tmr += dt
        if self._vo_tmr < VO_PERIOD_S:
            return
        self._vo_tmr = 0.0

        R_vo = VO_SIGMA_MPS ** 2
        K    = float(np.clip(self.vel_var / (self.vel_var + R_vo), 0.0, 0.95))

        z_vo  = true_vel + VO_SIGMA_MPS * RNG.standard_normal(2)
        innov = z_vo - (true_vel + self.vel_error)   # = noise − δv
        self.vel_error += K * innov
        self.vel_var    = max((1 - K) * self.vel_var, R_vo * 0.05)


# ══════════════════════════════════════════════════════════════════════════════
# FLIGHT CONTROLLER
# ══════════════════════════════════════════════════════════════════════════════

class FC:
    """
    Proportional guidance + loiter orbit controller.
    All navigation uses ins_pos — the drone has no direct access to true_pos.
    """

    def __init__(self):
        self.loiter_center: Optional[np.ndarray] = None

    def set_loiter(self, ins_pos: np.ndarray):
        """Latch loiter centre at the estimated position when LOITER is entered."""
        self.loiter_center = ins_pos.copy()

    def command(self, ins_pos: np.ndarray, state: MState) -> np.ndarray:
        if state == MState.OUTBOUND:
            return self._goto(ins_pos, WAYPOINT_POS, CRUISE_MPS)
        elif state == MState.RTL:
            return self._goto(ins_pos, GCS_POS, CRUISE_MPS)
        elif state == MState.LOITER:
            return self._loiter(ins_pos)
        else:
            return np.zeros(2)

    def _goto(self, pos: np.ndarray, tgt: np.ndarray, spd: float) -> np.ndarray:
        d = tgt - pos
        n = float(np.linalg.norm(d))
        return spd * d / n if n > 1.0 else np.zeros(2)

    def _loiter(self, ins_pos: np.ndarray) -> np.ndarray:
        """
        Proportional orbit controller around self.loiter_center.

        v = V_LOITER · t̂  −  K_r · (|r|−R_loiter) · r̂

        M2 failure physics:
          loiter_center is fixed at ins_pos(t₀). As δv accumulates after t₀,
          ins_pos drifts quadratically. The commanded orbit in TRUE space
          follows a trochoidal path whose translation axis is ½·b·τ². The
          drone never returns home — but it also avoids the straight runaway
          of Stage 5 RTL, enabling a clean visual comparison.
        """
        if self.loiter_center is None:
            return np.zeros(2)
        r_vec = ins_pos - self.loiter_center
        r_mag = float(np.linalg.norm(r_vec))
        r_hat = r_vec / max(r_mag, 1e-3)
        t_hat = np.array([-r_hat[1], r_hat[0]])
        return LOITER_V_MPS * t_hat - LOITER_KR * (r_mag - LOITER_R_M) * r_hat


# ══════════════════════════════════════════════════════════════════════════════
# H-MRSM
# ══════════════════════════════════════════════════════════════════════════════

class HMRSM:
    """
    Hybrid Mission Resilience State Machine.

    State transitions:
      OUTBOUND → RTL    : waypoint arrival (ins_pos)
      OUTBOUND → LOITER : PER > 90%  AND  use_loiter  (M2 only)
      RTL      → LANDED : ins_pos within GCS_ARRIVE_M of GCS
                          ← Stage 5: ins_pos drifts quadratically → NEVER lands
                          ← M3/M1:  ins_pos ≈ true_pos            → lands correctly
      RTL      → LOITER : PER > 90% AND use_loiter AND not already triggered (M2)
    """

    def __init__(self, use_loiter: bool = False):
        self.state      = MState.OUTBOUND
        self._per_buf: list = []
        self.use_loiter = use_loiter
        self._triggered = False

    @property
    def per(self) -> float:
        return float(np.mean(self._per_buf)) if self._per_buf else 0.0

    def step(self, sinr: float, ins_pos: np.ndarray,
             true_pos: np.ndarray, fc: FC) -> MState:
        # Rolling PER (AR(1) packet model)
        p_inst = sinr_to_per(sinr)
        rx     = int(RNG.random() > p_inst)
        self._per_buf.append(1.0 - rx)
        if len(self._per_buf) > PER_WINDOW:
            self._per_buf.pop(0)

        if self.state == MState.OUTBOUND:
            if np.linalg.norm(ins_pos - WAYPOINT_POS) < 50.0:
                self.state = MState.RTL
            elif self.per > PER_THRESH and not self._triggered:
                self._triggered = True
                if self.use_loiter:
                    self.state = MState.LOITER
                    fc.set_loiter(ins_pos)
                else:
                    self.state = MState.RTL

        elif self.state == MState.RTL:
            # Arrival check uses ins_pos — this is the key Stage 5 / M3 / M1 discriminator
            if np.linalg.norm(ins_pos - GCS_POS) < GCS_ARRIVE_M:
                self.state = MState.LANDED
            elif self.use_loiter and self.per > PER_THRESH and not self._triggered:
                # M2: RTL underway but link dead → switch to LOITER
                self._triggered = True
                self.state = MState.LOITER
                fc.set_loiter(ins_pos)

        # LOITER and LANDED are terminal states
        return self.state


# ══════════════════════════════════════════════════════════════════════════════
# MAIN SIMULATION LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_sim(mit: Mit) -> dict:
    """
    Run full UAV simulation for one mitigation configuration.

    Loop order (per timestep):
      1. Environment (jammer state, IMU bias activation)
      2. RF layer  → SINR  (uses true_pos for geometry)
      3. H-MRSM   → mission state update
      4. Flight controller → vel_cmd  (uses ins_pos, NOT true_pos)
      5. True dynamics: true_pos += vel_cmd · dt   (perfect actuator)
      6. Navigation filter: predict → (GCS update if link up) → (VO update if M3)
      7. Record
    """
    dual_pol = (mit == Mit.DUAL_POL)
    use_vo   = (mit == Mit.VIS_ODOM)
    use_loi  = (mit == Mit.LOITER)

    nav = NavFilter(GCS_POS.copy(), use_vo=use_vo)
    fc  = FC()
    hm  = HMRSM(use_loiter=use_loi)

    true_pos = GCS_POS.copy().astype(float)
    imu_bias = np.zeros(2)
    vel_cmd  = np.zeros(2)

    rec_t, rec_tr, rec_ins, rec_sinr, rec_per, rec_err, rec_ms = \
        [], [], [], [], [], [], []

    t = 0.0
    while t <= T_END_S:

        # ── 1. Environment ────────────────────────────────────────────────
        jam_on = (t >= T_JAM_ON_S)
        if jam_on:
            imu_bias = np.array([IMU_BIAS_MPS2, 0.0])

        # ── 2. RF Layer ───────────────────────────────────────────────────
        sinr = compute_rf_sinr(true_pos, GCS_POS, jam_on, GCS_POS, dual_pol)
        link_up = (sinr > 5.0)

        # ── 3. H-MRSM ────────────────────────────────────────────────────
        hm.step(sinr, nav.ins_pos, true_pos, fc)
        mission = hm.state

        # ── 4. Flight Controller (uses ins_pos, not true_pos) ─────────────
        vel_cmd = fc.command(nav.ins_pos, mission)

        # ── 5. True dynamics (perfect actuator) ───────────────────────────
        true_pos = true_pos + vel_cmd * DT_S

        # ── 6. Navigation Filter ──────────────────────────────────────────
        nav.predict(vel_cmd, imu_bias, DT_S)
        if link_up:
            nav.update_gcs(true_pos)
        if use_vo:
            nav.update_vo(vel_cmd, DT_S)   # true_vel ≈ vel_cmd (perfect actuator)

        # ── 7. Record ─────────────────────────────────────────────────────
        rec_t   .append(t)
        rec_tr  .append(true_pos.copy())
        rec_ins .append(nav.ins_pos.copy())
        rec_sinr.append(sinr)
        rec_per .append(hm.per)
        rec_err .append(float(np.linalg.norm(true_pos - nav.ins_pos)))
        rec_ms  .append(mission)

        if mission == MState.LANDED:
            break

        t = round(t + DT_S, 6)

    return dict(
        t        = np.array(rec_t),
        true_pos = np.array(rec_tr),
        ins_pos  = np.array(rec_ins),
        sinr     = np.array(rec_sinr),
        per      = np.array(rec_per),
        pos_err  = np.array(rec_err),
        mission  = rec_ms,
        mit      = mit,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ANALYTICAL ERROR BOUNDS
# ══════════════════════════════════════════════════════════════════════════════

def analytical_bounds(t_arr: np.ndarray) -> dict:
    """
    Theoretical position error growth curves for Panel 4.

    τ = t − T_JAM_ON_S  (time since last reliable absolute correction)

    INS-only (Stage 5):  δp(τ) = ½·b·τ²            deterministic, O(τ²)
    VO-aided  (M3)  1σ:  σ_p(τ) = σ_v·√τ            stochastic,   O(√τ)
    VO-aided  (M3)  3σ:  3·σ_p(τ)                   practical worst-case
    """
    tau = np.maximum(t_arr - T_JAM_ON_S, 0.0)
    return {
        "tau"      : tau,
        "ins_exact": 0.5 * IMU_BIAS_MPS2 * tau ** 2,
        "vo_1sig"  : VO_SIGMA_MPS * np.sqrt(tau),
        "vo_3sig"  : 3.0 * VO_SIGMA_MPS * np.sqrt(tau),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6-PANEL COMPARISON FIGURE
# ══════════════════════════════════════════════════════════════════════════════

STYLE = {
    Mit.NONE    : dict(color="#e74c3c", ls="-",  lw=2.2, label="Stage 5 Baseline"),
    Mit.LOITER  : dict(color="#f39c12", ls="--", lw=2.0, label="M2: Loiter Fallback"),
    Mit.VIS_ODOM: dict(color="#2ecc71", ls="-",  lw=2.2, label="M3: Visual Odometry"),
    Mit.DUAL_POL: dict(color="#3498db", ls="-.", lw=2.0, label="M1: Polarization Div."),
}


def plot_comparison(results: dict, save_path: str = "stage6_comparison.png"):
    fig, axes = plt.subplots(3, 2, figsize=(16, 17))
    fig.suptitle(
        "Stage 6 — UAV Architecture Hardening: Three-Mitigation Comparison\n"
        f"Worst-case threat: jammer co-located at GCS, activated t = {T_JAM_ON_S:.0f} s  "
        f"(drone at {int(np.linalg.norm(WAYPOINT_POS))} m range)",
        fontsize=13, fontweight="bold", y=0.985,
    )
    ax_sinr, ax_per = axes[0]
    ax_err,  ax_bnd = axes[1]
    ax_tr,   ax_es  = axes[2]

    bnd_done = False

    for mit, d in results.items():
        s  = STYLE[mit]
        t  = d["t"]
        tr = d["true_pos"]
        ip = d["ins_pos"]

        ax_sinr.plot(t, d["sinr"],        **s)
        ax_per .plot(t, d["per"] * 100,   **s)
        ax_err .plot(t, d["pos_err"],     **s)
        ax_tr  .plot(tr[:, 0], tr[:, 1],  **s)
        ax_es  .plot(ip[:, 0], ip[:, 1],  **s)

        # Start / end markers on true trajectory
        ax_tr.plot(tr[0, 0], tr[0, 1], "o", color=s["color"], ms=6, zorder=5)
        end_lbl = d["mission"][-1].name
        ax_tr.annotate(end_lbl, xy=(tr[-1, 0], tr[-1, 1]),
                       xytext=(6, 6), textcoords="offset points",
                       fontsize=7, color=s["color"])

        # Analytical bounds panel (drawn once)
        if not bnd_done:
            b = analytical_bounds(t)
            tau_200 = 200.0
            ins_200 = 0.5 * IMU_BIAS_MPS2 * tau_200 ** 2
            vo_200  = VO_SIGMA_MPS * np.sqrt(tau_200)

            ax_bnd.fill_between(t, b["vo_1sig"], b["vo_3sig"],
                                alpha=0.25, color="#2ecc71",
                                label=f"VO 1σ–3σ  (σ_v = {VO_SIGMA_MPS} m/s)")
            ax_bnd.plot(t, b["vo_1sig"],  "--", color="#2ecc71", lw=2.0,
                        label="VO 1σ = σ_v·√τ")
            ax_bnd.plot(t, b["ins_exact"], "-", color="#e74c3c", lw=2.5,
                        label="INS-only ½bτ²")

            ax_bnd.annotate(
                f"INS @ τ=200s:\n{ins_200/1000:.1f} km",
                xy=(T_JAM_ON_S + tau_200, ins_200),
                color="#e74c3c", fontsize=9, ha="right",
                xytext=(-12, -45), textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.3),
            )
            ax_bnd.annotate(
                f"VO @ τ=200s:\n{vo_200:.1f} m (1σ)",
                xy=(T_JAM_ON_S + tau_200, vo_200),
                color="#2ecc71", fontsize=9,
                xytext=(12, 20), textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color="#2ecc71", lw=1.3),
            )
            bnd_done = True

    # Jammer-ON vertical lines
    for ax in (ax_sinr, ax_per, ax_err, ax_bnd):
        ax.axvline(T_JAM_ON_S, color="grey", ls=":", lw=1.2, alpha=0.8)

    # ── Panel decorations ─────────────────────────────────────────────────
    ax_sinr.axhline(5, color="k", ls=":", lw=1.2, label="Link threshold (5 dB)")
    ax_sinr.set(xlabel="Time (s)", ylabel="SINR (dB)",
                title="Post-Beamformer GCS Link SINR  [correct metric: incl. jammer leakage]")
    ax_sinr.legend(fontsize=8); ax_sinr.grid(True, alpha=0.3)

    ax_per.axhline(PER_THRESH * 100, color="k", ls=":", lw=1.2,
                   label="RTL / LOITER trigger (90 %)")
    ax_per.set(xlabel="Time (s)", ylabel="PER (%)",
               title=f"Rolling Packet Error Rate ({PER_WINDOW}-packet window)",
               ylim=[-5, 110])
    ax_per.legend(fontsize=8); ax_per.grid(True, alpha=0.3)

    ax_err.set(xlabel="Time (s)", ylabel="|true_pos − ins_pos|  (m)",
               title="Simulated INS Navigation Position Error  (log scale)",
               yscale="log", ylim=[0.05, 8e4])
    ax_err.legend(fontsize=8); ax_err.grid(True, alpha=0.3, which="both")

    ax_bnd.set(xlabel="Time (s)", ylabel="Position error bound  (m)",
               title="Analytical Error Growth Models   [τ = t − t_jammer]",
               yscale="log", ylim=[0.05, 8e4])
    ax_bnd.legend(fontsize=8); ax_bnd.grid(True, alpha=0.3, which="both")

    # ── Trajectory panels ─────────────────────────────────────────────────
    for ax in (ax_tr, ax_es):
        ax.plot(*GCS_POS,      "k*", ms=18, zorder=10, label="GCS / Launch")
        ax.plot(*WAYPOINT_POS, "b^", ms=10, zorder=9,  label=f"Waypoint ({int(np.linalg.norm(WAYPOINT_POS))} m)")
        ax.add_patch(mpatches.Circle(GCS_POS, GCS_ARRIVE_M,
                     color="green", fill=False, lw=1.8, ls="--"))
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", "datalim")

    legend_lines = [plt.Line2D([0], [0], **STYLE[m]) for m in results]
    legend_lines += [
        plt.Line2D([0], [0], marker="*", color="k",  ls="none", ms=13, label="GCS"),
        plt.Line2D([0], [0], marker="^", color="b",  ls="none", ms=9,  label="Waypoint"),
        mpatches.Patch(fc="none", ec="green", lw=1.8,
                       label=f"Arrival zone (±{GCS_ARRIVE_M:.0f} m)"),
    ]
    ax_tr.legend(handles=legend_lines, fontsize=7, loc="upper right")
    ax_tr.set(xlabel="East (m)", ylabel="North (m)",
              title="True Physical Trajectories  (Ground Truth)")
    ax_es.legend(handles=[plt.Line2D([0],[0],**STYLE[m]) for m in results],
                 fontsize=7)
    ax_es.set(xlabel="East (m)", ylabel="North (m)",
              title="INS-Estimated Trajectories  (Drone's Self-Belief)")

    plt.tight_layout(rect=[0, 0, 1, 0.975])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  → Saved: {save_path}")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    SEP = "═" * 70

    print(SEP)
    print("  Stage 6 — UAV Defensive Architecture Hardening")
    print(SEP)
    print(f"  Threat:    co-located jammer at GCS {GCS_POS},  t_ON = {T_JAM_ON_S:.0f} s")
    print(f"  IMU bias:  {IMU_BIAS_MPS2} m/s²   |   VO σ_v: {VO_SIGMA_MPS} m/s")
    print(f"  Key fix:   SINR denominator includes jammer leakage  "
          f"(false SNR was +42 dB; correct SINR = −23 dB)\n")

    runs    = [Mit.NONE, Mit.LOITER, Mit.VIS_ODOM, Mit.DUAL_POL]
    results = {}
    for m in runs:
        print(f"  [{m.name}]  '{m.value}'")
        results[m] = run_sim(m)
        d  = results[m]
        ms = d["mission"][-1].name
        print(f"    t_end = {d['t'][-1]:.1f} s  |  "
              f"pos_error = {d['pos_err'][-1]:.1f} m  |  state = {ms}\n")

    # ── Summary table ──────────────────────────────────────────────────────
    print(SEP)
    print(f"  {'Mitigation':<40} {'Pos Err':>9} {'t_end':>8} {'Outcome':>10}")
    print(f"  {'─'*40} {'─'*9} {'─'*8} {'─'*10}")
    for m, d in results.items():
        ms      = d["mission"][-1].name
        outcome = "✓ LANDED" if ms == "LANDED" else "✗ LOST"
        print(f"  {m.value:<40} {d['pos_err'][-1]:>8.1f}m {d['t'][-1]:>7.1f}s {outcome:>10}")
    print(SEP)

    # ── Analytical comparison ──────────────────────────────────────────────
    tau = 200.0
    ins_e = 0.5 * IMU_BIAS_MPS2 * tau ** 2
    vo_e  = VO_SIGMA_MPS * np.sqrt(tau)
    print(f"\n  Analytical Error Bounds at τ = {tau:.0f} s post-jammer:")
    print(f"    Stage 5 (INS-only):   ½·b·τ²    = {ins_e/1000:.1f} km")
    print(f"    M3 (VO) 1σ:           σ_v·√τ   = {vo_e:.2f} m")
    print(f"    M3 (VO) 3σ:           3·σ_v·√τ = {3*vo_e:.2f} m")
    print(f"    Improvement factor:   {ins_e/vo_e:,.0f}×  "
          f"({ins_e/1000:.0f} km → {vo_e:.1f} m)\n")

    print("  Generating 6-panel comparison figure…")
    plot_comparison(results)
    plt.show()
