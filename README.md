# RF Communication Resilience Enhancement for UAV Missions

> **Summer Internship Project**  
> **Author:** Parth Sidhu, Final Year B.Tech (Electronics and Communication Engineering), Gati Shakti Vishwavidyalaya, Vadodara  
> **Organisation:** Encap Technologies India Private Limited (Encaptechno), Phase 8B, Mohali, Punjab  
> **Duration:** Summer 2026

---

## The Problem in Plain Terms

A drone is only as useful as its radio link. Break the link, and you own the drone — it either crashes, enters a blind failsafe, or circles indefinitely. In a contested electromagnetic environment, an adversary does exactly this: they point a jammer at the control frequency and deny the mission without firing a single kinetic round.

The engineering response is a phased array antenna — combine 16 elements, steer a beam toward the friendly base station, and carve a deep null toward the jammer. On paper this is straightforward. In the air, it is not.

**The problem this project investigates:** When a fixed-wing UAV banks into a turn, its own wing physically blocks some of its antenna elements. At 13° of bank — a completely routine manoeuvre — the rising wing eclipses enough elements that the beamformer's mathematics breaks down entirely. An array that can suppress a 10 W jammer at rest is geometrically defeated by its own airframe at normal flight bank angles.

This project builds a 15-stage Python simulation pipeline that tracks this problem from first principles through to a real-time AI solution, covering inertial navigation, Rician radio propagation, LCMV adaptive beamforming, 3D mesh ray-tracing, Differential Evolution array optimisation, and a neural surrogate that executes the full physics calculation in under 0.08 ms.

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    15-STAGE SIMULATION PIPELINE                      │
│                                                                       │
│  PHASE 1 — BASELINE RF MODEL                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │ Stage 1  │→│  Phase A  │→│  Phase B  │→│  Phase C  │            │
│  │ Baseline │  │  Rician  │  │Beamforming│  │Extensions│            │
│  │   Sim    │  │ Channel  │  │  LCMV/   │  │  EKF/   │            │
│  │  H-MRSM  │  │  Outage  │  │   MUSIC  │  │  RNCO   │            │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘            │
│        ↓                                                              │
│  PHASE 2 — FULL SYSTEM INTEGRATION (Stages 3–7)                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │Stages 3–5│  │ Stage 6  │  │ Stage 7  │  │          │            │
│  │ MAVLink  │  │Mitigation│  │   High   │  │          │            │
│  │Schuler   │  │ Dual-Pol │  │ Fidelity │  │          │            │
│  │INS + VO  │  │ Loiter   │  │ Stress   │  │          │            │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘            │
│        ↓                                                              │
│  PHASE 3 — STRUCTURAL EM (Stages 8–11)                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │ Stage 8  │→│ Stage 9  │→│ Stage 10 │→│ Stage 11 │            │
│  │BVH Mesh  │  │Cognitive │  │Continuous│  │Mesh-Aware│            │
│  │Ray-Trace │  │Autopilot │  │  Fresnel │  │  LCMV   │            │
│  │4 Sweeps  │  │   LUT    │  │ EM Model │  │Corrected │            │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘            │
│        ↓                                                              │
│  PHASE 4 — OPTIMISATION & AI (Stages 12–15)                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │ Stage 12 │→│ Stage 13 │→│ Stage 14 │→│ Stage 15 │            │
│  │    DE    │  │Operational│  │  FFN AI  │  │Adversarial│           │
│  │Geometry  │  │ Analysis │  │Surrogate │  │  Flight  │            │
│  │  Optim.  │  │Multi-Jam │  │ <0.08ms  │  │  Trial   │            │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Stage-by-Stage Technical Summary

### Stage 1 — Baseline Mission Simulation
Built a 100 Hz Monte Carlo simulation of a UAV returning home under INS navigation while jamming is active. Derived the **critical handover radius** R\*\_comm = 1,037 m from IMU error dynamics and Schuler oscillation theory — the minimum safe range at which the drone must have already initiated RTL if it is to land within 50 m of the base station using dead reckoning alone. The Bayesian EW Estimator (BEE) was implemented as a 5-state Hidden Markov Model over features including mean SINR, SINR variance, hop-hit rate, and spatial variance.

**What this means for the drone:** The drone now knows, before the link actually dies, that it should start coming home. The BEE gives the H-MRSM state machine enough lead time that the UAV has already cleared the jammer's defeat radius before the link fails entirely.

**Outcome:** H-MRSM raises mission success probability from 13% (timeout-based failsafe) to 63%.

---

### Stage 2A — Rician Fading Channel
Replaced the FSPL binary-cutoff model with a physically correct Rician fading channel with an elevation-angle-dependent K-factor. Derived outage probability via the Marcum Q-function.

**What this means for the drone:** The FSPL model predicted total, instantaneous link failure at the defeat range. Reality is different — the Rician channel shows a 700-metre probabilistic transition zone where the link degrades gradually. At the nominal defeat range the outage probability is 62%, not 100%. This transition zone is where the BEE operates; it forces the state machine to make a probabilistic decision rather than react to a binary threshold.

---

### Stage 2B — LCMV Beamforming and MUSIC DOA
Implemented LCMV (Linearly Constrained Minimum Variance) beamforming with the weight vector:

```
w_LCMV = R⁻¹C(CᴴR⁻¹C)⁻¹f
```

Implemented the MUSIC (Multiple Signal Classification) DOA estimator and benchmarked it against the Cramér–Rao Lower Bound. Identified the catastrophic co-located jammer failure mode: when the jammer occupies the same spatial direction as the GCS, `rank(C) = 1`, the Gramian is singular, SINR collapses to −23 dB, and the link is dead.

**Outcome:** SINR improvement of 15–25 dB over baseline; MUSIC DOA within 12% of the CRLB across evaluated SNR ranges.

---

### Stage 2C — Extended Signal Processing
Added EKF jammer position tracker, Recursive Noise Covariance Optimiser (RNCO) for adaptive diagonal loading, and Adaptive Modulation and Coding (AMC) state transitions from BPSK through 64-QAM.

**Outcome:** +8–12 dB null depth improvement; RNCO within 3% of oracle diagonal loading.

---

### Stages 3–5 — Full System Integration (Grand Integration)
Integrated the MAVLink communication bridge, Schuler-tuned INS, cylindrical conformal array, EKF jammer tracker, two-ray propagation, and Visual Odometry into a single 100 Hz Python simulation over a 400 s mission window.

**What VO means for the drone:** In the event of total link loss, the drone does not need GPS or radio — it navigates home optically, integrating optical flow to maintain positional accuracy. VO position error grows as √τ vs. τ² for INS-only. At τ = 200 s, VO holds position error to ~2.1 m versus ~10,000 m for unaided INS.

**Outcome:** Against a co-located worst-case jammer, the system delivers RTL within 3.2 m of the GCS.

---

### Stage 6 — Defensive Mitigation Architecture
Three independent mitigations for the co-located jammer rank failure:

| Mitigation | Mechanism | Outcome |
|---|---|---|
| **M1: Dual-Polarisation LCMV** | Expands constraint space to 2N=32 ports using Kronecker-product steering vectors; GCS (vertical pol.) and jammer (horizontal pol.) are now orthogonal regardless of spatial direction | +18 dB SINR recovery |
| **M2: Loiter** | Proportional guidance maintains station until jammer drifts; buys time for frequency reassignment | 12 m closest-approach error |
| **M3: Visual Odometry RTL** | Fully non-RF navigation home | 4.1 m landing error |

---

### Stage 7 — High-Fidelity Stress Test
All four physics upgrades applied simultaneously: moving GCS, turn-rate kinematic constraints, calibration phase noise (σ = 0.1 rad), and cardioid element radiation patterns. Phase noise limits the dual-pol null depth from the ideal −∞ to a realistic −19 dB asymptote. All three mitigations still prevent mission failure.

**Outcome:** M1 SINR recovery reduced from +18 dB (ideal) to +11 dB (realistic) — the 7 dB loss quantifies what hardware calibration precision is worth.

---

### Stage 8 — BVH Mesh Ray-Tracing
Integrated a Bounding Volume Hierarchy ray-tracer operating on the actual STL mesh of a ScanEagle-class airframe. For each antenna element and each jammer direction, a ray is cast through the mesh and the element is either marked active or shadowed. Four parametric sweeps:

1. **Critical roll angle:** φ\_crit = **13°** — at this bank angle the rising wing eclipses enough elements that N\_act drops below 3 and the LCMV solver fails matrix inversion. Standard tactical bank angles are 20°–30°; the baseline array cannot manoeuvre under active EW.
2. **Jammer elevation sensitivity:** Array only fails for jammers at < 5° elevation (near-horizon); handles all elevated threats cleanly at 30° bank.
3. **Geometry comparison:** No single element placement dominates across all roll angles — motivates the evolutionary optimiser.
4. **Hardware scaling:** φ\_crit ∝ log N. Doubling element count from 16→32 buys only 6° of additional safe bank angle at double the hardware cost.

**The log N scaling result is structurally significant:** It proves that the occlusion problem is fundamentally geometric, not aperture-limited. Throwing hardware at it is economically indefensible. The correct solution is software attitude management.

---

### Stage 9 — Cognitive Autopilot
Built a precomputed Lookup Table (LUT) over a 10°×10° grid of (roll, jammer azimuth) pairs, populated via the BVH ray-tracer offline. At runtime, the Cognitive Autopilot queries the LUT at 100 Hz and reduces the commanded bank angle in 1° decrements whenever N\_act is predicted to drop below N\_min = 6 (3 for rank feasibility + 3 for LUT interpolation margin, derived from 99.9th-percentile Monte Carlo discrepancy).

**What this means for the drone:** The autopilot sacrifices turn rate to preserve the antenna array's utility. A 90° heading change now takes 9.1 s instead of 6.9 s (+32%), but the communication link stays alive 100% of the time throughout the manoeuvre.

**Outcome:** 100% link uptime across all three evaluated trajectory types.

---

### Stage 10 — Continuous Electromagnetic Shadow Model
Replaced the binary shadow mask with a physically continuous complex element gain derived from knife-edge Fresnel diffraction over the actual mesh edges. A softmax primary-edge selection scheme smooths the gain transitions, reducing maximum element gain step changes from >2 dB (binary) to **0.54 dB**.

**Why this matters:** The binary model caused discontinuous jumps in the spatial covariance matrix that made the LCMV weight vector oscillate wildly as the jammer swept across a shadow boundary. The continuous model gives the solver a smooth surface to optimise over, which is a prerequisite for stable real-time operation.

---

### Stage 11 — Dynamic Mesh-Aware LCMV
Corrected a fundamental physics error in the covariance formulation: occlusion must apply equally to both the signal and jammer covariance terms. An element blocked by the fuselage receives zero jammer power at its port — the airframe cannot simultaneously shadow the signal path and be transparent to the jammer. After correction:

- Baseline (fixed 30° bank): N\_act,min = 4; minimum SINR = +17.2 dB
- Cognitive Autopilot: 1–4 dB SINR gain inside the occlusion window

---

### Stage 12 — Differential Evolution Array Geometry Optimisation
Deployed Differential Evolution (population 2,000; 3-stage refinement) on a pool of symmetrised candidate element positions on the airframe surface. The fitness function is the minimum active element count over the full 360° heading envelope at 30° bank — a geometric robustness metric rather than instantaneous SINR.

**Outcome:** Optimal topology raises worst-case N\_act,min from 10 (baseline conformal) to **11 elements** across the complete heading envelope, without adding hardware.

---

### Stage 13 — Operational Analysis

**13A — Random Element Failure (50 Monte Carlo trials):**
Swept random element failure fraction *f* from 0 to 100%. Median SINR holds above **+13 dB even at 50% element loss**. The array degrades gracefully — there is no cliff-edge failure; elements lost in directions away from the jammer contribute less to null depth than elements in the jammer's sector.

**13B — Multi-Jammer Spatial DoF Sweep:**
An N=16 element array has at most N−1 = 15 spatial degrees of freedom for nulling. With airframe occlusion reducing N\_act, the effective null budget is N\_act − 1. Tested n\_J = 1 through 12 simultaneous jammers at equal angular spacing. **Graceful degradation holds through n\_J = 7; catastrophic SINR collapse occurs at n\_J = 9.**

**What this means for the drone:** Against a coordinated multi-jammer attack, the UAV can maintain its link through 7 simultaneous jammers. Against 9+, it needs the Cognitive Autopilot to maximise active elements before engaging.

---

### Stage 14 — AI Beamforming Surrogate (Hybrid FFN)

The continuous ray-tracer requires ~1,500 ms per covariance query — incompatible with a 100 Hz control loop. A deep Feed-Forward Network (FFN) with width-512 hidden layers, Fourier feature encoding (σ = 10.0), and a custom Null-Response Loss function was trained to predict the optimal covariance matrix directly.

Inside the structural deadzone ([168°, 192°] × [−27°, 27°] body-frame), the FFN smoothly blends with a bilinearly-interpolated precomputed Oracle grid to handle the phase singularities the network cannot extrapolate.

| Configuration | Inference Time | SINR Failure Rate |
|---|---|---|
| Raw Physics Judge | ~1,500 ms | 0% (but unusable at 100 Hz) |
| FFN Surrogate (Arm C) | **< 0.08 ms** | **0%** |
| Delayed Oracle (Arm B) | 500 ms | 83.6% outage |

**The 0.08 ms result is the technical headline of the project.** The AI surrogate compresses 1.5 seconds of BVH ray-tracing into a single matrix multiplication chain that executes in under a tenth of a millisecond — 18,750× speedup with zero degradation in link availability.

---

### Stage 15 — Adversarial Closed-Loop Flight Trial
Executed a closed-loop dynamic simulation over a 58.5 s trial window against two adversarial jammer motion strategies:

- **Center-Seeking:** Jammer moves to maximise spatial overlap with the GCS direction
- **Seam-Hunting:** Jammer actively seeks the structural shadow boundaries on the airframe, probing for positions where it simultaneously evades the null and obscures multiple elements

Four arms compared:

| Arm | Configuration | Cumulative Outage |
|---|---|---|
| A | Fixed covariance (no adaptation) | 58.5 s (100%) |
| B | Delayed Oracle (exact physics, 500 ms latency) | 48.9 s (83.6%) |
| **C** | **Hybrid FFN + Oracle Fallback** | **1.49 s (2.5%)** |
| D | Oracle Instant (ideal reference) | 0.0 s (0%) |

**The Arm B result is the most instructive:** The exact physics engine, with full fidelity, fails 83.6% of the time — not because it is wrong, but because it is slow. Latency is as lethal as inaccuracy in a real-time control loop. The AI surrogate, which is slightly less accurate but 18,750× faster, achieves 2.5% outage — 33× better than the perfect-but-slow solver.

---

## Key Results at a Glance

| Finding | Number | Operational Significance |
|---|---|---|
| Critical bank angle for LCMV failure | **13°** | Routine tactical bank angles (20°–30°) are already in the failure zone for baseline arrays |
| VO navigation accuracy (200 s RTL) | **2.1 m** | Drone can land optically without GPS or radio after total link loss |
| Grand Integration RTL error | **3.2 m** | Full-system worst-case, against co-located jammer |
| H-MRSM mission success improvement | **13% → 63%** | Proactive threat awareness vs. passive timeout |
| LCMV SINR gain | **15–25 dB** | Array suppresses a 10 W jammer to below the noise floor |
| Dual-pol SINR recovery (co-located jammer) | **+18 dB** | Polarisation restores full matrix rank when spatial filtering is algebraically impossible |
| Cognitive Autopilot uptime | **100%** | No link interruptions during heading changes |
| Multi-jammer graceful degradation limit | **7 simultaneous jammers** | SINR holds above +13 dB |
| Array geometry optimisation gain | **+1 element worst-case** | DE finds geometrically superior layout without hardware changes |
| FFN AI surrogate inference time | **< 0.08 ms** | 18,750× faster than physics engine; enables 100 Hz real-time deployment |
| Adversarial flight trial outage (Hybrid AI) | **2.5%** | vs. 83.6% for exact-but-delayed physics solver |

---

## Technology Stack

```
Language:           Python 3.11
Numerical Core:     NumPy, SciPy
Machine Learning:   PyTorch (FFN surrogate, Fourier feature encoding)
3D Geometry:        Trimesh (STL mesh), Rtree (BVH spatial index)
Data / Plotting:    Pandas, Matplotlib, Seaborn
Progress:           tqdm
```

---

## Repository Structure

```
UAV Internship project/
│
├── Stage 1/                      # Baseline sim: H-MRSM, BEE, INS, R*_comm
├── Stage 2 (Remediated)/         # Rician fading, LCMV, MUSIC, EKF, RNCO, AMC
├── Stage 3 (Phase D)/            # MAVLink bridge, Schuler INS
├── Stage 4 (Extensions)/         # Cat 1–5 extension experiments
├── Stage 5 (Grand Integration)/  # Full 400 s system simulation + VO
├── Stage 6 (Mitigations)/        # Dual-pol, loiter, VO RTL
├── Stage 7 (High Fidelity)/      # Phase noise, cardioid patterns, moving GCS
├── Stage 8 (Mesh Ray-Tracing)/   # BVH ray-tracer on STL mesh, 4 sweeps
├── Phase 2 Track 1/              # Cognitive Autopilot (Stage 9)
├── Phase 2 Track 2/              # Continuous EM shadow model (Stage 10)
├── Phase 2 Track 3/              # Mesh-aware LCMV (Stage 11)
├── Stage 11 (Geometry Opt.)/     # Differential Evolution optimiser (Stage 12)
├── Stage 12 (Operational)/       # Monte Carlo failure + multi-jammer analysis (Stage 13)
├── Stage 13 (AI Beamforming)/    # FFN surrogate + Null-Response Loss (Stage 14)
├── Stage 21 (Adversarial Trial)/ # Closed-loop seam-hunting trial (Stage 15)
│
└── README.md                     # This file
```

---

## How to Run

Each stage is a self-contained Python script. Dependencies are listed in [`requirements.txt`](requirements.txt).

```bash
pip install numpy scipy torch trimesh rtree pandas matplotlib seaborn tqdm

# Example: run the Grand Integration simulation
cd "Stage 5 (Grand Integration)"
python stage5_grand_integration.py

# Example: run the adversarial flight trial
cd "Stage 21 (Adversarial Trial)"
python stage15_adversarial_trial.py
```

---

## Author and Acknowledgements

**Parth Sidhu**  
B.Tech, Electronics and Communication Engineering (Final Year)  
Gati Shakti Vishwavidyalaya, Vadodara, Gujarat

Completed as a Summer Internship at **Encap Technologies India Private Limited (Encaptechno)**, Phase 8B, Mohali, Punjab, under their R&D and advanced engineering division.

---

## 🇮🇳 Devotion and Offer of Assistance to the Republic of India

As an Indian national and engineering student, the Author recognises the strategic sensitivity of UAV electronic warfare and adaptive beamforming. While this repository strictly embargoes adversarial nations, active conflict zones, and foreign militaries, **a resolute and formal exemption is granted to the Government of India.**

If any competent authority—including the **Indian Armed Forces, DRDO, R&AW, or MoD**—deems this research useful for the defence of the Nation, the Author formally stands ready to provide full technical assistance, operational support, and codebase adaptation. It is the Author's highest patriotic duty and privilege to assist in safeguarding the sovereignty and integrity of India. 

*(Please refer to [`NOTICE_OFFER_OF_ASSISTANCE_INDIA.md`](legal/NOTICE_OFFER_OF_ASSISTANCE_INDIA.md) and [`NOTICE_GOI_NATIONAL_SECURITY.md`](legal/NOTICE_GOI_NATIONAL_SECURITY.md) below for the formal legal declarations.)*

---

## Legal and Repository Documents

| Document | Purpose |
|---|---|
| [`LICENSE`](LICENSE) | Ownership, permitted use by Encaptechno and GSV, prohibited activities |
| [`DISCLAIMER.md`](DISCLAIMER.md) | Simulation-only notice, 9-section liability disclaimer, modelling assumptions |
| [`NOTICE_RF_TRANSMISSION.md`](legal/NOTICE_RF_TRANSMISSION.md) | Absolute prohibition on SDR integration; liability waiver for FCC/WPC/ITU spectrum violations |
| [`CREDITS.md`](CREDITS.md) | Attribution for the ScanEagle STL mesh; Boeing/govt non-affiliation notice |
| [`EXPORT_CONTROL.md`](legal/EXPORT_CONTROL.md) | Dual-use notice covering Wassenaar (42 states), US ITAR/EAR, EU, India SCOMET |
| [`NOTICE_GOI_NATIONAL_SECURITY.md`](legal/NOTICE_GOI_NATIONAL_SECURITY.md) | Author's explicit declaration of allegiance to India and disavowal of adversarial misuse |
| [`NOTICE_OFFER_OF_ASSISTANCE_INDIA.md`](legal/NOTICE_OFFER_OF_ASSISTANCE_INDIA.md) | Official offer of resolute technical assistance and embargo exemption for Indian defence and intelligence agencies |
| [`NOTICE_SOVEREIGN_RESTRICTION.md`](legal/NOTICE_SOVEREIGN_RESTRICTION.md) | Absolute blanket denial of consent for all foreign state actors, and moral defensive-retaliation clause |
| [`NOTICE_CHINA.md`](legal/NOTICE_CHINA.md) | Bilingual (English + 中文) notice covering China's 5 applicable laws |
| [`NOTICE_TURKEY.md`](legal/NOTICE_TURKEY.md) | Bilingual (English + Türkçe) notice covering Turkish defence laws and UAV industry alignments |
| [`NOTICE_ACTIVE_CONFLICT.md`](legal/NOTICE_ACTIVE_CONFLICT.md) | Absolute embargo on the use of this repository by nations currently involved in armed conflicts |
| [`NOTICE_PAKISTAN_BANGLADESH.md`](legal/NOTICE_PAKISTAN_BANGLADESH.md) | Trilingual notice for Pakistan (اردو) and Bangladesh (বাংলা) |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Closed project — no external contributions accepted |
| [`SECURITY.md`](SECURITY.md) | No security patches; not for operational deployment |
| [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) | Expected conduct for issue tracker interactions |
| [`CITATION.cff`](CITATION.cff) | Machine-readable academic citation (update URL after upload) |
