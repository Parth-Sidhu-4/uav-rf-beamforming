# DISCLAIMER AND LIMITATION OF LIABILITY

---

**Project Title:** RF Communication Resilience Enhancement for UAV Missions
Using Adaptive Beamforming and Threat-Aware Communications

**Author and Rights Holder:** Parth Sidhu
B.Tech (Electronics and Communication Engineering), Final Year
Gati Shakti Vishwavidyalaya, Vadodara, Gujarat
Email: parthecstasy@gmail.com

**Host Organisation (Internship Placement Only):**
Encap Technologies India Private Limited (Encaptechno), Mohali, Punjab

**Document Classification:** Legal Disclaimer, Limitation of Liability,
and Scope-of-Work Notice

**Governing Jurisdiction:** Himachal Pradesh, India

**Effective Date:** July 2026

---

## PREAMBLE

This document constitutes a legally binding disclaimer governing the
interpretation, use, and reliance upon all materials, results, code,
algorithms, and documents associated with the above-referenced project
(collectively, **"the Work"**). Any person or entity accessing, reading,
executing, or otherwise engaging with the Work is deemed to have read,
understood, and unconditionally accepted the terms of this Disclaimer in
their entirety. If you do not accept these terms, you must immediately
cease all access to and use of the Work.

---

## SECTION 1 — NATURE AND SCOPE OF THE WORK

**1.1** The Work is an original academic and technical research project
completed by Parth Sidhu ("the Author") in his individual capacity as a
student researcher during a Summer Internship placement at Encap Technologies
India Private Limited. The Work constitutes a **purely theoretical,
algorithmic, and computational investigation** implemented entirely in
software on a general-purpose personal computer.

**1.2** THE WORK IS A SIMULATION-ONLY STUDY. NO PHYSICAL HARDWARE OF ANY
KIND WAS CONSTRUCTED, PROCURED, ASSEMBLED, CALIBRATED, OR OPERATED AT ANY
POINT DURING THIS PROJECT. This includes, without limitation: antenna arrays,
phased array modules, RF front-ends, low-noise amplifiers, inertial
measurement units, unmanned aerial vehicles, autopilot hardware, ground
control stations, electronic warfare jamming equipment, or any other
physical electronics or avionics system.

**1.3** All quantitative results, performance metrics, SINR values,
navigation accuracy figures, failure rates, probability estimates, and
system characterisations reported in the Work are outputs of mathematical
simulation software executing idealised models under controlled, deterministic,
or stochastic computational conditions. They do not represent, and shall not
be construed to represent, measured physical performance of any real system.

**1.4** The Author is a final-year undergraduate student. This Work was
produced under the resource constraints, time limitations, and access
restrictions inherent to an academic internship. It is not a peer-reviewed
publication, a defence certification document, a type-approval document,
or an airworthiness assessment.

---

## SECTION 2 — KNOWN MODELLING ASSUMPTIONS AND DEPARTURES FROM PHYSICAL REALITY

The simulations contained in the Work operate under the following
acknowledged assumptions. These are not oversights; they are engineering
simplifications made knowingly and in good faith to produce a tractable
research study. Any person seeking to apply findings from this Work to a
physical system must independently evaluate and address each of the following:

**2.1 RF Propagation Model**
The radio channel is modelled using Free-Space Path Loss (FSPL) and a
two-ray ground reflection approximation. Real low-altitude tactical
environments involve terrain-specific multipath, urban clutter, atmospheric
refraction, foliage attenuation, and non-stationary fading statistics that
are not captured by these models and that can alter link budget predictions
by tens of decibels.

**2.2 Rician Fading Parameterisation**
The Rician K-factor is modelled as a deterministic function of elevation
angle alone. Physical K-factor distributions are trajectory-dependent,
terrain-dependent, frequency-dependent, and time-varying. Accurate
parameterisation requires empirical measurement campaigns; the values
used herein are engineering estimates drawn from published literature.

**2.3 Antenna Array Phase Coherence**
All beamforming algorithms assume perfect, continuous phase and amplitude
calibration across all antenna elements. Real RF front-ends exhibit
manufacturing-induced phase offsets, inter-channel amplitude imbalance,
temperature-dependent LNA gain drift, connector insertion phase variation,
and trace-length propagation delays that the simulation does not model
beyond a Gaussian noise perturbation of σ = 0.1 radians. The null depths
achievable in simulation represent mathematical upper bounds that are
physically unattainable in hardware without continuous calibration loops.

**2.4 Static Airframe Geometry**
The airframe is represented by a static, rigid STL mesh. Physical UAV wings
flex dynamically under aerodynamic loads; fuel burn alters the aircraft's
centre of gravity and structural loading; thermal expansion shifts element
mounting positions. The structural shadow manifold modelled herein is valid
only for a rigid, unloaded, isothermal airframe at the specific geometry
of the ScanEagle-class mesh used as the reference geometry.

**2.5 Electromagnetic Diffraction Approximation**
The shadow model employs a scalar Fresnel knife-edge diffraction
approximation. Real conductive airframe structures support vector
electromagnetic surface currents, creeping waves propagating around
fuselage curvature, cavity resonances within structural bays, and
near-field mutual coupling between array elements. Exact electromagnetic
treatment requires a full-wave solver such as the Uniform Theory of
Diffraction (UTD), Method of Moments (MoM), or Finite-Difference
Time-Domain (FDTD) — none of which are employed in this Work. The
continuous shadow model is an engineering approximation, not a
physical ground truth.

**2.6 Jammer Threat Model**
Jamming waveforms are modelled as additive Gaussian noise, barrage
interference, narrowband tones, or simplified follower signals derived from
a frequency-hopping hit probability model. This does not capture:
Digital Radio Frequency Memory (DRFM) repeater jammers capable of
coherent deception; adversarial beamformers that actively steer toward
array nulls; cognitive EW systems that adapt their waveform in response
to observed beamformer outputs; or co-channel interference from legitimate
civilian emitters. The security margins derived in this Work are valid
against the specific, non-adaptive jammer models described herein only.

**2.7 AI Surrogate Generalisation**
The Fourier Feature Network (FFN) neural surrogate is trained, validated,
and evaluated exclusively on data generated by the same simulation
environment that produced its training set. Its predictive accuracy outside
this data distribution — including real-world analog hardware, airframe
deformation, thermal drift, or in-flight vibration — has not been evaluated
and cannot be guaranteed. Neural networks are function approximators; they
have no inherent understanding of electromagnetics and will produce
undetected, potentially catastrophic errors when exposed to out-of-distribution
inputs.

**2.8 Navigation and Sensor Models**
Inertial navigation errors are parameterised from published manufacturer
datasheets under benign laboratory conditions. Vibration-induced IMU
degradation, gyroscope saturation during aggressive manoeuvres, magnetic
interference from the airframe, and GPS spoofing are not modelled.
Visual Odometry accuracy figures assume adequate ambient illumination,
stable optical features, and a camera free from motion blur — conditions
that are not guaranteed in real flight operations.

**2.9 Simulation Fidelity and Numerical Precision**
All results are subject to the numerical precision of IEEE 754 double-
precision floating-point arithmetic, the accuracy of the third-party
Python libraries used (NumPy, SciPy, PyTorch, Trimesh, Rtree), and
the statistical confidence intervals of Monte Carlo sample sizes used.
Results should be interpreted as estimates with inherent sampling variance,
not as exact physical constants.

---

## SECTION 3 — LIMITATION OF LIABILITY

**3.1 AUTHOR'S LIMITATION OF LIABILITY**

TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, PARTH SIDHU (THE
"AUTHOR") SHALL NOT BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, CONSEQUENTIAL, PUNITIVE, OR EXEMPLARY DAMAGES, INCLUDING BUT NOT
LIMITED TO LOSS OF LIFE, PERSONAL INJURY, PROPERTY DAMAGE, DATA LOSS,
MISSION FAILURE, AIRCRAFT LOSS, REGULATORY SANCTIONS, FINANCIAL LOSS, OR
ANY OTHER HARM, HOWEVER CAUSED, ARISING OUT OF OR IN CONNECTION WITH:

  (i)   THE USE OF, RELIANCE UPON, OR APPLICATION OF ANY ALGORITHM,
        RESULT, FINDING, RECOMMENDATION, OR DESIGN DESCRIBED IN THE WORK;

  (ii)  THE INTEGRATION OF ANY CONCEPT FROM THE WORK INTO ANY PHYSICAL
        SYSTEM, UNMANNED AERIAL VEHICLE, AVIONICS PLATFORM, OR ELECTRONIC
        WARFARE SYSTEM;

  (iii) THE FAILURE OF ANY SIMULATION RESULT TO REPLICATE IN PHYSICAL
        HARDWARE DUE TO THE MODELLING ASSUMPTIONS SET OUT IN SECTION 2;

  (iv)  ANY DECISION — ENGINEERING, OPERATIONAL, COMMERCIAL, REGULATORY,
        OR OTHERWISE — MADE IN RELIANCE ON THE WORK.

THIS LIMITATION APPLIES REGARDLESS OF WHETHER THE AUTHOR WAS ADVISED OF
THE POSSIBILITY OF SUCH DAMAGES AND REGARDLESS OF THE THEORY OF LIABILITY
(CONTRACT, TORT, STRICT LIABILITY, OR OTHERWISE).

**3.2 HOST ORGANISATION'S LIMITATION OF LIABILITY**

Encap Technologies India Private Limited's involvement in this project was
limited to providing an internship placement. The Company did not direct,
supervise, validate, or review the technical content of the Work at any
engineering level. The Company shall not be liable for any consequence
arising from the use, misuse, or reliance upon the Work by any third party.
The Company is, however, a permitted user of the Work under the terms of
the LICENSE file; such permitted use is exercised entirely at its own risk.

**3.3 UNIVERSITY'S LIMITATION OF LIABILITY**

Gati Shakti Vishwavidyalaya ("the University"), a Central University under
the Ministry of Railways, Government of India, is hereby acknowledged as
the Author's degree-awarding institution. The University and its faculty
members, research staff, and academic supervisors are granted limited
research and academic use rights under the LICENSE file, subject to full
attribution to the Author. Such use is exercised entirely at the University's
own risk. Neither the University nor any of its faculty, employees, or
representatives shall bear any liability to the Author or to any third party
for any consequence arising from their use, study, citation, or referencing
of this Work in accordance with the terms of the LICENSE file.

Furthermore, the University and its faculty bear no responsibility for the
technical content, simulation assumptions, results, or conclusions of this
Work, which represent the Author's own independent research and do not
constitute an endorsement, certification, or validation by the University
or any of its academic departments.

**3.4 NO REPRESENTATIONS**

Neither the Author nor the Company makes any representation that:

  (a) The Work is accurate, complete, current, or free from errors;
  (b) The results are reproducible on any particular hardware configuration;
  (c) The algorithms are patent-free or free from third-party IP claims;
  (d) The Work meets any particular standard of care for engineering practice.

---

## SECTION 4 — PROHIBITION ON OPERATIONAL DEPLOYMENT

**4.1** THE ALGORITHMS, TRAINED MODELS, SYSTEM ARCHITECTURES, AND DESIGN
RECOMMENDATIONS CONTAINED IN THE WORK ARE NOT VALIDATED FOR INTEGRATION
INTO ANY REAL UNMANNED AERIAL VEHICLE, AIRBORNE SYSTEM, OR OPERATIONAL
ELECTRONIC WARFARE PLATFORM.

**4.2** Any organisation or individual seeking to use concepts from this Work
in a real system must, at a minimum and entirely at their own risk:

  (a) Conduct full hardware-in-the-loop testing in a controlled RF
      environment, including anechoic chamber validation;
  (b) Perform live flight testing under certified test conditions with
      appropriate safety mitigations;
  (c) Obtain all necessary approvals from the competent civil aviation
      authority (DGCA in India, FAA in the United States, EASA in Europe,
      or equivalent) prior to any flight operation;
  (d) Obtain any applicable defence or export control clearances where
      electronic warfare capabilities are involved;
  (e) Conduct an independent safety assessment by a qualified avionics
      engineer against applicable airworthiness standards.

**4.3** The Author expressly and irrevocably disclaims all responsibility
for any outcome, damage, or liability arising from the deployment of the
Work, or any concept derived from it, in any physical system, without
having completed the steps in Section 4.2 to the satisfaction of all
applicable regulatory bodies.

---

## SECTION 5 — ACADEMIC AND RESEARCH CONTEXT

This Work was produced in partial fulfilment of the requirements of a Summer
Internship placement forming part of the B.Tech programme in Electronics and
Communication Engineering at Gati Shakti Vishwavidyalaya, Vadodara. The
Author acted in good faith and to the best of his ability within the
constraints of undergraduate-level resource access, a fixed internship
timeline, and the absence of access to physical RF measurement infrastructure.

Quantitative results should be read as theoretical performance bounds and
comparative simulation metrics produced under specific, documented assumptions
— not as certified engineering specifications or guaranteed field-performance
figures.

---

## SECTION 6 — THIRD-PARTY SOFTWARE ACKNOWLEDGEMENT

This project makes use of the following open-source third-party Python
libraries, each governed by its own licence terms: NumPy, SciPy, PyTorch,
Trimesh, Rtree/libspatialindex, Pandas, Matplotlib, Seaborn, and tqdm.
The Author makes no warranty regarding the correctness, completeness, or
fitness for purpose of these packages for any application beyond their
documented intended use.

---

## SECTION 7 — GOVERNING LAW

This Disclaimer shall be governed by and construed in accordance with the
laws of India. Any dispute arising out of or relating to this Disclaimer,
the Work, or its use shall be subject to the exclusive jurisdiction of the
competent courts of Himachal Pradesh, India.

---

## SECTION 8 — SEVERABILITY

If any provision of this Disclaimer is found to be invalid, unlawful, or
unenforceable under applicable law, that provision shall be deemed severed
from this Disclaimer to the minimum extent necessary, and the remaining
provisions shall continue in full force and effect.

---

## SECTION 9 — ENTIRE AGREEMENT

This Disclaimer, together with the LICENSE file and all companion legal
documents listed in Section 10 below, constitutes the entire agreement
between the Author and any user of the Work regarding its use and the
limitations thereof, and supersedes all prior representations, warranties,
and understandings.

---

## SECTION 10 — COMPANION LEGAL AND REPOSITORY DOCUMENTS

The following documents form part of the complete legal framework governing
this Work and are incorporated into this Disclaimer by reference:

| Document | Scope |
|---|---|
| [`LICENSE`](LICENSE) | Ownership; permitted use by Encaptechno and GSV; prohibited activities |
| [`NOTICE_RF_TRANSMISSION.md`](NOTICE_RF_TRANSMISSION.md) | Strict prohibition on SDR/hardware interfacing; indemnification for FCC/WPC/ITU spectrum violations |
| [`CREDITS.md`](CREDITS.md) | Third-party STL asset attribution; Boeing/government non-affiliation; mesh quality disclaimer |
| [`EXPORT_CONTROL.md`](EXPORT_CONTROL.md) | Dual-use technology notice; Wassenaar Arrangement; US ITAR/EAR; EU Dual-Use Regulation; India SCOMET; sanctions |
| [`NOTICE_GOI_NATIONAL_SECURITY.md`](NOTICE_GOI_NATIONAL_SECURITY.md) | Author's declaration of allegiance to India and absolute prohibition of adversarial misuse |
| [`NOTICE_OFFER_OF_ASSISTANCE_INDIA.md`](NOTICE_OFFER_OF_ASSISTANCE_INDIA.md) | Absolute embargo exemption for the Government of India and formal offer of technical support |
| [`NOTICE_SOVEREIGN_RESTRICTION.md`](NOTICE_SOVEREIGN_RESTRICTION.md) | Absolute blanket denial of consent for all foreign state actors, and moral defensive-retaliation clause |
| [`NOTICE_CHINA.md`](NOTICE_CHINA.md) | Bilingual notice for PRC users under China's Export Control Law, Cryptography Law, Cybersecurity Law, Data Security Law, and National Security Law |
| [`NOTICE_TURKEY.md`](NOTICE_TURKEY.md) | Bilingual notice for users in Türkiye regarding Wassenaar compliance and national defence industry laws (Law No. 5201 / 5202) |
| [`NOTICE_ACTIVE_CONFLICT.md`](NOTICE_ACTIVE_CONFLICT.md) | Strict embargo explicitly forbidding the use of this repository by any actor currently involved in an active armed conflict |
| [`NOTICE_PAKISTAN_BANGLADESH.md`](NOTICE_PAKISTAN_BANGLADESH.md) | Trilingual notice for users in Pakistan (Export Control Act 2004, PECA, Official Secrets Act) and Bangladesh (DSA/CSA 2023, Customs Act, Official Secrets Act) |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Closed project policy; no external contributions accepted |
| [`SECURITY.md`](SECURITY.md) | No security patches; prohibition on operational deployment |
| [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) | Conduct standards for repository interactions |
| [`CITATION.cff`](CITATION.cff) | Academic citation information |

---

*By accessing, reading, executing, or otherwise using any component of
this Work, you acknowledge that you have read this Disclaimer in full,
that you understand its terms, and that you agree to be bound by them.*

---

**Parth Sidhu**
Author and Sole Rights Holder
parthecstasy@gmail.com
*July 2026*
