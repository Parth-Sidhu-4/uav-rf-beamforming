"""
Reorganise master.tex: integrate Stage 11 content into Stage 10.
- Move EM physics model + three-baseline design → Methodology Stage 10
- Move EM realism results → EDA Stage 10 (replacing thin old Conclusion + Stage 11 block)
- Remove Stage 11 subsection
- Update progression table
- Update Future Research references
"""
import re, shutil
from pathlib import Path

TEX = Path(r'D:\UAV Internship project\master.tex')
shutil.copy(TEX, TEX.with_suffix('.tex.bak2'))

with open(TEX, encoding='utf-8') as f:
    txt = f.read()

# ──────────────────────────────────────────────────────────────
# 1.  METHODOLOGY: append EM physics + 3-baseline design
#     into Stage 10, just before the EDA section header.
# ──────────────────────────────────────────────────────────────
METH_MARKER = (
    "% ================================================================\r\n"
    "%  3. EXPLORATORY DATA ANALYSIS\r\n"
    "% ================================================================\r\n"
    "\\section{Exploratory Data Analysis}"
)

NEW_METH = r"""\subsubsection{Continuous Electromagnetic Shadow Model}

Stage 10 extends the binary shadow mask of Stage 8 to a physically continuous, complex-valued element gain factor $\tilde{g}_i \in \mathbb{C}$, computed from knife-edge Fresnel diffraction over the actual STL mesh geometry.

For each element $i$ and jammer direction $\hat{\mathbf{d}}_j$ (in the UAV body frame), a ray is cast from $\mathbf{p}_i + \varepsilon\hat{\mathbf{n}}_i$ (offset by $\varepsilon = 10^{-4}\text{ m}$ along the surface normal to prevent self-intersection) toward the jammer. The ray intersects the set of mesh edges $\{e_k\}$, each characterised by the perpendicular miss distance $h_k$ and along-ray distance $d_{1,k}$. Only edges with physical length $>0.015\text{ m}$ are retained; shorter sliver edges produce spurious diffraction terms.

The \textbf{primary edge} is selected via a softmax-weighted average over the top-3 closest candidates (replacing the hard $\arg\min$ used in earlier iterations, which caused discrete gain jumps when two edges swap rank):
\begin{equation}
    w_k^{(1)} = \frac{\exp(-\beta\, h_k)}{\sum_{k'}\exp(-\beta\, h_{k'})}, \quad \beta = 500,
    \label{eq:softmax_primary}
\end{equation}
with weighted primary distance $\bar{d}_1 = \sum_k w_k^{(1)}\, d_{1,k}$ and smooth primary Fresnel gain:
\begin{equation}
    F_1^{\text{soft}} = \sum_{k \in \text{top-3}} w_k^{(1)}\, F\!\left(\nu_k^{(1)}\right),
    \quad \nu_k^{(1)} = s_k\sqrt{\frac{2}{\lambda d_{1,k}}},
    \label{eq:F1soft}
\end{equation}
where $s_k$ is the signed distance from the ray-edge closest point to the mesh surface (positive = inside the body), $\lambda = 0.125\text{ m}$ (2.4 GHz), and $F(\nu)$ is the complex Fresnel diffraction gain:
\begin{equation}
    F(\nu) = \frac{1+j}{2}\int_{\nu}^{\infty}\exp\!\left(-j\frac{\pi t^2}{2}\right)dt.
    \label{eq:fresnel}
\end{equation}
For $\nu<0$ the element is illuminated ($|F|\approx 1$); for $\nu\gg0$ it is in deep shadow ($|F|\to 0$).

A \textbf{secondary edge} captures double-diffraction. The smooth separation weight
\begin{equation}
    S_j^{(k)} = 1 - \exp\!\left(-\frac{(d_{1,k}-\bar{d}_1)^2}{2\sigma_s^2}\right), \quad \sigma_s = 0.2\text{ m},
    \label{eq:Sj}
\end{equation}
penalises edges co-located with the primary cluster and promotes genuinely separate features. The penalised score $\tilde{h}_k = h_k/(S_j^{(k)}+\varepsilon')$ is soft-minimised over the top-3 secondary candidates:
\begin{equation}
    w_k^{(2)} = \frac{\exp(-\beta\, \tilde{h}_k)}{\sum_{k'}\exp(-\beta\, \tilde{h}_{k'})},
    \quad \bar{w}_2 = \sum_k w_k^{(2)}\, S_j^{(k)},
    \label{eq:softmax_secondary}
\end{equation}
with secondary Fresnel gain:
\begin{equation}
    F_2^{\text{soft}} = \sum_{k \in \text{top-3}} w_k^{(2)}\, F\!\left(\nu_k^{(2)}\right),
    \quad \nu_k^{(2)} = s_k\sqrt{\frac{2}{\lambda(\bar{d}_1 + d_{1,k})}}.
    \label{eq:F2soft}
\end{equation}
The effective path length $\bar{d}_1 + d_{1,k}$ reflects the total propagation distance through the diffracting body. The blended total gain is:
\begin{equation}
    \tilde{g}_i = F_1^{\text{soft}} \cdot \bigl[(1-\bar{w}_2)\cdot 1 + \bar{w}_2 \cdot F_2^{\text{soft}}\bigr].
    \label{eq:totalGain}
\end{equation}
The $(1-\bar{w}_2)\cdot 1$ term ensures that in the absence of a genuine secondary edge, the gain is not spuriously attenuated.

To verify smoothness, a SHA-256 hash of the complete $200\times16$ complex gain matrix $G$ is computed at the end of each diagnostic run and embedded in the figure title. The system-level simulation (runtime $\approx\!9$ min) is cached to a compressed \texttt{.npz} file; a \texttt{--replot} flag regenerates all figures in $<5\text{ s}$ without re-running the solver.

\subsubsection{Three-Baseline Evaluation Design}

Three LCMV configurations are evaluated over a $0^\circ$--$90^\circ$ heading sweep at fixed bank angle $\phi=30^\circ$, with a far-field jammer at bearing $90^\circ$:

\begin{enumerate}
    \item \textbf{Baseline (Binary Mask):} Element gains are binary --- $\tilde{g}_i\in\{0,1\}$ based on ray-mesh intersection. Calibration phase noise $\sigma_{\text{phase}}=0.10\text{ rad}$ is not applied to jammer signal paths, so jammer leakage to masked elements is artificially zeroed. This is the idealised best-case reference.
    \item \textbf{Baseline (Continuous Mask):} Element gains follow Eq.~(\ref{eq:totalGain}). Calibration noise is applied consistently to all signal paths, including jammer leakage. This is the physically realistic baseline.
    \item \textbf{Cognitive Autopilot (Continuous):} Identical physics to Baseline (Continuous), but the bank angle is modulated by the Stage 9 LUT subject to $N_{\text{act}}(\phi,\psi_j)\geq N_{\min}=15$.
\end{enumerate}

% ================================================================
%  3. EXPLORATORY DATA ANALYSIS
% ================================================================
\section{Exploratory Data Analysis}"""

txt = txt.replace(METH_MARKER, NEW_METH, 1)
print("Methodology insertion:", "OK" if NEW_METH in txt else "FAILED")

# ──────────────────────────────────────────────────────────────
# 2.  EDA: replace old thin Stage 10 Conclusion + entire
#     Stage 11 subsection block with integrated EDA results.
# ──────────────────────────────────────────────────────────────

# --- find start and end anchors ---
EDA_OLD_START = (
    "\\subsubsection{Conclusion}\r\n"
    "\r\n"
    "The rigorous independent audit definitively closes the Stage 10 investigation."
    " The structural ceiling that originally motivated the Cognitive Autopilot was an"
    " artifact of a kinematic approximation bug. The 16-element conformal array"
    " inherently possesses the geometric diversity necessary to survive the entire"
    " banked maneuver with 8+ elements and a robust $>15\\text{ dB}$ median Net SINR."
    " STAP is not strictly mandated for this airframe; standard adaptive beamforming"
    " alone guarantees total mission resilience.\r\n"
)
# Stage 11 block ends just before the Outcomes comment
EDA_OLD_END = (
    "The Stage~11 investigation establishes that the physically continuous Fresnel shadow"
    " model (Eq.~\\ref{eq:totalGain}) is a valid and numerically well-behaved replacement"
    " for the binary occlusion mask, with maximum per-element gain discontinuities reduced"
    " from the initial $>2\\text{ dB}$ to $0.54\\text{ dB}$ peak after dual-edge softmax"
    " smoothing. The single remaining violator is caused by the Fresnel knife-edge physics"
    " itself and would require a higher-order diffraction model (UTD) to eliminate. This is"
    " documented as a known, bounded limitation; all other elements maintain gain smoothness"
    " below $0.43\\text{ dB}$ with 13 of 16 elements below $0.30\\text{ dB}$.\r\n"
    "\r\n"
    "% ================================================================"
)

NEW_EDA = r"""\subsubsection{Electromagnetic Realism: System-Level Results}

\begin{figure}[H]
\centering
\includegraphics[width=0.95\textwidth]{phase2_track1_system_results.png}
\caption{Stage 10 EM-realism results: four-panel sweep over heading $0^\circ$--$90^\circ$ at $\phi=30^\circ$ bank. \textit{Panel 1 (SINR):} Binary mask (red dashed) sits 3--8~dB above the continuous curves due to artificially zeroed jammer leakage; both Continuous Mask (blue) and Cognitive (green) coincide within numerical precision. \textit{Panel 2 (Active elements):} Orange dashed line marks the LCMV rank floor $M=3$; blue dotted line marks the Cognitive Autopilot search threshold $N_{\min}=15$, which the array never achieves, explaining why the autopilot falls back to $\phi=30^\circ$ throughout. \textit{Panel 3 (Null depth):} Binary mask shows a $-300\text{ dB}$ floor because jammer power is truncated to zero; continuous curves maintain $-45$ to $-55\text{ dB}$ throughout. \textit{Panel 4 (Bank angle):} Autopilot commands constant $30^\circ$, confirming non-intervention.}
\label{fig:stage10_system}
\end{figure}

The Binary Mask baseline (red dashed) shows a minimum SINR of 20.68~dB --- substantially higher than the physically realistic Continuous Mask (15.18~dB minimum). The gap originates from the binary mask's artificial treatment of jammer leakage: occluded elements are entirely excluded from the covariance estimate, so their contribution to jammer leakage is zeroed rather than attenuated by the actual Fresnel diffraction factor. Once jammer leakage is modelled consistently (Continuous Mask), the SINR floor falls by $\approx\!5\text{ dB}$ to a physically defensible value.

The active element count (Panel~2) reaches a minimum of 12 of 16 elements in both mask variants. This is comfortably above the LCMV rank floor of $M=3$, confirming the array has ample degrees of freedom throughout the entire sweep.

The null depth panel (Panel~3) explains the binary mask's $-300\text{ dB}$ floor: when elements are hard-gated to zero, the LCMV null is formed against a jammer vector whose leakage is numerically zero, producing an artefactually deep null. The continuous model maintains physically realistic null depths of $-45$ to $-55\text{ dB}$ throughout, consistent with the $\sigma_{\text{phase}}=0.10\text{ rad}$ calibration noise calibrated in the Phase Noise section above.

\subsubsection{Cognitive Autopilot Overlap Confirmation}

A numerical verification confirmed that the overlap between the Cognitive and Continuous Mask SINR traces is genuine physics, not a plotting artefact. The Cognitive heading trajectory was interpolated onto the Continuous heading grid and the pointwise difference computed:
\begin{equation}
    \Delta_{\text{SINR}} = \max_\psi\, \bigl|S^{\text{cog}}(\psi) - S^{\text{cont}}(\psi)\bigr| = 0.0000\text{ dB}.
\end{equation}
The cause is structural: the Cognitive Autopilot LUT was built with $N_{\min}=15$, but the worst-case active element count across the full bank-angle and heading envelope is $N_{\text{act,min}}=12$. Since no bank angle ever satisfies $N_{\text{act}}\geq 15$, the fallback branch fires unconditionally, returning $\phi_{\text{cmd}}=30^\circ$ at every step --- identical to the unconstrained baseline. The unique commanded bank angle set was verified: $\{p_{\text{cog}}\}=\{30^\circ\}$.

This is documented as expected behaviour: the Cognitive Autopilot's intervention is only non-zero when $N_{\text{act,min}} < N_{\min}$, which requires either a tighter threshold (e.g.\ $N_{\min}=13$) or a different array layout. Evaluating the autopilot under those conditions is the objective of the Array Geometry Optimisation future work direction.

\subsubsection{Per-Element Gain Continuity Analysis}\label{sec:stage10_softmax}

A key quality metric for the shadow model is the smoothness of each element's gain $\tilde{g}_i$ as a function of heading. Discontinuities arise when the $\arg\min$ edge selection swaps rank between two nearly equidistant edges. Two successive softmax fixes were applied and verified against measured per-step jump sizes.

\paragraph{Fix 1 --- Primary-edge softmax.}
Replacing $k_1 = \arg\min_k h_k$ with the softmax of Eq.~(\ref{eq:softmax_primary}) eliminated the largest observed discontinuity: a $\approx\!2\text{ dB}$ cliff on Ant~9 (cyan trace) at heading $\approx\!29^\circ$, traced to an instantaneous swap between two near-equidistant edges. After this fix, Ant~9's max step dropped from $0.52\text{ dB}$ to $0.43\text{ dB}$.

\paragraph{Fix 2 --- Secondary-edge softmax.}
Applying the same technique to the penalised score $\tilde{h}_k$ (Eq.~\ref{eq:softmax_secondary}) reduced violations above $0.50\text{ dB}$ from 3 elements to 1. The full per-element comparison is given in Table~\ref{tab:step_comparison}.

\begin{table}[H]
\centering
\caption{Per-element maximum gain step (dB) before (primary softmax only) and after (both softmax) secondary-edge smoothing, over the 22--32$^\circ$ heading diagnostic window (200 headings, 3184 element$\times$step pairs total).}
\label{tab:step_comparison}
\begin{tabular}{crrrl}
\toprule
\textbf{Ant} & \textbf{Primary only} & \textbf{Both softmax} & \textbf{$\Delta$} & \textbf{Note}\\
\midrule
0  & 0.047 dB & 0.053 dB & $+0.007$ & \\
1  & 0.118 dB & 0.023 dB & $-0.095$ & \\
2  & 0.540 dB & 0.541 dB & $+0.001$ & Fresnel singularity (see text) \\
3  & 0.194 dB & 0.193 dB & $-0.000$ & \\
4  & 0.289 dB & 0.289 dB & $-0.000$ & \\
5  & 0.272 dB & 0.330 dB & $+0.058$ & Crossed 0.30 dB threshold \\
6  & 0.142 dB & 0.142 dB & $+0.000$ & \\
7  & 0.004 dB & 0.004 dB & $+0.000$ & \\
8  & 0.224 dB & 0.224 dB & $-0.000$ & \\
9  & 0.525 dB & 0.433 dB & $-0.092$ & Improved below 0.50 dB \\
10 & 0.571 dB & 0.178 dB & $-0.394$ & Improved below 0.30 dB \\
11 & 0.005 dB & 0.025 dB & $+0.020$ & \\
12 & 0.099 dB & 0.099 dB & $+0.000$ & \\
13 & 0.019 dB & 0.004 dB & $-0.015$ & \\
14 & 0.007 dB & 0.007 dB & $-0.000$ & \\
15 & 0.007 dB & 0.005 dB & $-0.002$ & \\
\midrule
\textbf{Overall max} & \textbf{0.571 dB} & \textbf{0.541 dB} & & \\
\textbf{Elements $>0.50$ dB} & \textbf{3} & \textbf{1} & & \\
\textbf{Elements $>0.30$ dB} & \textbf{3} & \textbf{3} & & Ant 5 gained, Ant 10 dropped\\
\bottomrule
\end{tabular}
\end{table}

The $>0.30\text{ dB}$ element count remained at 3 despite Ant~10 dropping below the threshold, because Ant~5 grew by $+0.058\text{ dB}$ simultaneously, crossing it from below. This is the standard redistribution effect of smoothing one part of a coupled nonlinear system.

\paragraph{Root cause of Ant~2's residual 0.54 dB step (directly traced).}
A per-step intermediate-variable trace at 80-point resolution over the $30.5^\circ$--$32.5^\circ$ window confirmed that the primary edge index ($e_1=11911$) and the top secondary index are both constant throughout --- no edge handoff occurs. The jump originates from the complex primary Fresnel gain $F_1^{\text{soft}}$ rotating rapidly in the complex plane:
\[
    F_1^{\text{soft}}:\quad 0.491 - 0.009j \;\ \longrightarrow \;\ 0.459 - 0.040j \;\ \longrightarrow \;\ 0.510 + 0.010j
\]
This is a \textbf{Fresnel phase singularity}: the Fresnel parameter $\nu_k^{(1)}$ for edge~11911 passes through zero (the knife-edge grazing condition), and the complex Fresnel integral $F(\nu)$ rotates rapidly near $\nu=0$ regardless of how many edges are blended. The 0.54~dB excursion over $\approx\!0.2^\circ$ is correct Fresnel diffraction physics, not a numerical artefact. Eliminating it would require replacing the scalar knife-edge model with the Uniform Theory of Diffraction (UTD), which regularises the singularity via Fresnel transition functions.

\begin{figure}[H]
\centering
\includegraphics[width=0.95\textwidth]{phase2_track1_diagnose_v5.png}
\caption{Stage 10 per-element diagnostic: individual element gains $|\tilde{g}_i|$ (dB) vs.\ UAV heading over the 22--32$^\circ$ window (200 heading steps). The figure title embeds the SHA-256 hash of the gain array (\texttt{74c69493f2f6}) and run timestamp (2026-06-21 21:57:12), providing tamper-evident verification of a freshly computed result. \textit{Upper panel:} all 16 traces occupy a compact $-7.5$ to $-1.5\text{ dB}$ band with gradual trajectories; no trace exhibits a discrete cliff. \textit{Lower panel:} aggregate proxy SINR declines monotonically from 2.75 to 1.63, with only minor kinks consistent with the $<0.33\text{ dB}$ maximum step on most elements.}
\label{fig:stage10_diagnose}
\end{figure}

\subsubsection{Stage 10 Conclusion}

The Stage~10 investigation delivers two distinct results. The first is the definitive falsification of the structural collapse hypothesis: the kinematic approximation error is corrected, the active element count is confirmed at $\geq12$ of 16 throughout the full $30^\circ$ banked sweep, and the median SINR is verified at $>15\text{ dB}$ with 0\% dropout probability. The second is the replacement of the binary shadow mask with a physically continuous Fresnel diffraction model, with the following verified outcomes:

\begin{table}[H]
\centering
\caption{Stage 10 key quantitative outcomes (EM-realism results).}
\label{tab:stage10_summary}
\begin{tabular}{p{5.5cm}p{8cm}}
\toprule
\textbf{Metric} & \textbf{Result}\\
\midrule
Baseline (Binary) minimum SINR & 20.68 dB (artificially elevated: jammer leakage zeroed)\\
Baseline (Continuous) minimum SINR & 15.18 dB (physically realistic)\\
Cognitive vs.\ Continuous max SINR diff.\ & 0.0000 dB (numerically confirmed)\\
Active elements --- worst case & 12 of 16\\
Cognitive Autopilot engagement & 0\% ($N_{\min}=15$ threshold never achievable; $N_{\text{act,min}}=12$)\\
Max per-element gain step (final) & 0.54 dB (1 out of 3184 element$\times$step pairs above 0.50 dB)\\
Elements $>0.30$ dB step & 3 of 16\\
Root cause of worst residual step & Fresnel phase singularity at knife-edge grazing (traced)\\
Binary mask null depth & $-300$ dB floor (artefact: jammer truncated to zero)\\
Continuous null depth & $-45$ to $-55$ dB (at $\sigma_{\text{phase}}=0.10$ rad)\\
\bottomrule
\end{tabular}
\end{table}

The physically continuous Fresnel shadow model (Eq.~\ref{eq:totalGain}) is a valid and numerically well-behaved replacement for the binary occlusion mask. Maximum per-element gain discontinuities were reduced from the initial $>2\text{ dB}$ cliff to $0.54\text{ dB}$ peak through dual-edge softmax smoothing. The single remaining violator is caused by the Fresnel knife-edge physics itself and is documented as a known, bounded physical limitation --- not a numerical artefact. All other elements maintain gain smoothness below $0.43\text{ dB}$, with 13 of 16 elements below $0.30\text{ dB}$.

% ================================================================"""

# find and replace the combined block
idx_start = txt.find(EDA_OLD_START)
idx_end   = txt.find(EDA_OLD_END)
if idx_start == -1:
    print("EDA START anchor: NOT FOUND")
elif idx_end == -1:
    print("EDA END anchor: NOT FOUND")
else:
    end_pos = idx_end + len(EDA_OLD_END)
    txt = txt[:idx_start] + NEW_EDA + txt[end_pos:]
    print("EDA replacement: OK")

# ──────────────────────────────────────────────────────────────
# 3.  PROGRESSION TABLE: merge Stage 10 and Stage 11 rows
# ──────────────────────────────────────────────────────────────
OLD_TABLE_ROWS = (
    "10: Mesh LCMV (corrected) & Exact 3D quaternion kinematics, phase noise calibration"
    " & $\\geq8$ active elements throughout; median SINR $>15\\text{ dB}$\\\\\r\n"
    "11 (Track 1) & Fresnel shadow model, softmax edge selection, 3-baseline SINR sweep"
    " & Max element step reduced from $>2\\text{ dB}$ to $0.54\\text{ dB}$;"
    " cognitive/baseline overlap confirmed at 0.0000 dB\\\\"
)
NEW_TABLE_ROW = (
    "10: Mesh-Aware LCMV & Exact 3D kinematics; Fresnel shadow model; dual-softmax edge"
    " selection; 3-baseline SINR sweep & $\\geq12$ active elements; min SINR 15.18~dB;"
    " max element step $0.54\\text{ dB}$; overlap verified at 0.0000~dB\\\\"
)
if OLD_TABLE_ROWS in txt:
    txt = txt.replace(OLD_TABLE_ROWS, NEW_TABLE_ROW, 1)
    print("Progression table: OK")
else:
    print("Progression table rows: NOT FOUND — trying fallback")
    # fallback: just remove Stage 11 row
    txt = txt.replace(
        "11 (Track 1) & Fresnel shadow model, softmax edge selection, 3-baseline SINR sweep"
        " & Max element step reduced from $>2\\text{ dB}$ to $0.54\\text{ dB}$;"
        " cognitive/baseline overlap confirmed at 0.0000 dB\\\\",
        "", 1)
    print("  Removed Stage 11 row only")

# ──────────────────────────────────────────────────────────────
# 4.  FUTURE RESEARCH: Track 1 references Stage 10 not Stage 11
# ──────────────────────────────────────────────────────────────
txt = txt.replace(
    "\\subsection{Track 1: Completed as Stage 11}",
    "\\subsection{Track 1: Completed in Stage 10}", 1)
txt = txt.replace("documented as Stage~11 above", "documented in Stage~10 above", 1)
txt = txt.replace("Stage~11 above", "Stage~10 above")
txt = txt.replace("Stage 11 above", "Stage 10 above")
print("Future Research: OK")

# ──────────────────────────────────────────────────────────────
# 5.  CLEAN UP any residual "Stage 11" / "Stage~11" references
# ──────────────────────────────────────────────────────────────
# (caption/label references that still say Stage 11)
txt = txt.replace("Stage~11", "Stage~10")
txt = txt.replace("Stage 11", "Stage 10")
txt = txt.replace("stage11_", "stage10_")  # label prefixes in \ref/\label

with open(TEX, 'w', encoding='utf-8') as f:
    f.write(txt)

print("\nDone. Backup at master.tex.bak2")
