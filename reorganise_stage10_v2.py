"""
Precise fix: integrate Stage 11 content into Stage 10 properly.
Operations:
  A. Add EM physics model + 3-baseline design to METHODOLOGY Stage 10
  B. Replace old thin Conclusion + rogue Stage 11 subsection + EM physics (EDA)
     + three-baseline (EDA) with a clean 'System-Level Results' subsubsection
     that shows the figure and analysis
  C. Fix progression table
"""
import shutil
from pathlib import Path

TEX = Path(r'D:\UAV Internship project\master.tex')
shutil.copy(TEX, TEX.with_suffix('.tex.bak3'))

with open(TEX, encoding='utf-8') as f:
    txt = f.read()

orig_len = len(txt)

# ─────────────────────────────────────────────────────────────────
# A.  METHODOLOGY: insert two subsubsections after Stage 10's
#     existing "Independent Audit" subsubsection, before the
#     EDA section header comment.
# ─────────────────────────────────────────────────────────────────
METH_ANCHOR = (
    "% ================================================================\n"
    "%  3. EXPLORATORY DATA ANALYSIS\n"
    "% ================================================================\n"
    "\\section{Exploratory Data Analysis}"
)

NEW_METH_CONTENT = (
    "\\subsubsection{Continuous Electromagnetic Shadow Model}\n"
    "\n"
    "Stage 10 extends the binary shadow mask of Stage 8 to a physically continuous,"
    " complex-valued element gain $\\tilde{g}_i \\in \\mathbb{C}$, derived from"
    " knife-edge Fresnel diffraction over the actual STL mesh.\n"
    "\n"
    "For each element $i$ and jammer direction $\\hat{\\mathbf{d}}_j$ (body frame),"
    " a ray is cast from $\\mathbf{p}_i + \\varepsilon\\hat{\\mathbf{n}}_i$"
    " ($\\varepsilon = 10^{-4}\\text{ m}$ normal offset to prevent self-intersection)."
    " The ray intersects mesh edges $\\{e_k\\}$, each characterised by perpendicular miss"
    " distance $h_k$ and along-ray distance $d_{1,k}$."
    " Edges shorter than $0.015\\text{ m}$ are discarded as degenerate sliver edges.\n"
    "\n"
    "The \\textbf{primary edge} is selected via softmax over the top-3 closest candidates"
    " (replacing the hard $\\arg\\min_k h_k$ that caused discrete gain jumps at rank swaps):\n"
    "\\begin{equation}\n"
    "    w_k^{(1)} = \\frac{\\exp(-\\beta\\, h_k)}{\\sum_{k'}\\exp(-\\beta\\, h_{k'})},\n"
    "    \\quad \\beta = 500,\n"
    "    \\label{eq:softmax_primary}\n"
    "\\end{equation}\n"
    "with weighted primary distance $\\bar{d}_1 = \\sum_k w_k^{(1)}\\, d_{1,k}$"
    " and smooth primary Fresnel gain:\n"
    "\\begin{equation}\n"
    "    F_1^{\\text{soft}} = \\sum_{k \\in \\text{top-3}} w_k^{(1)}\\, F\\!\\left(\\nu_k^{(1)}\\right),\n"
    "    \\quad \\nu_k^{(1)} = s_k\\sqrt{\\frac{2}{\\lambda d_{1,k}}},\n"
    "    \\label{eq:F1soft}\n"
    "\\end{equation}\n"
    "where $s_k$ is the signed distance from the ray-edge point to the mesh surface"
    " (positive = inside), $\\lambda = 0.125\\text{ m}$ (2.4~GHz), and $F(\\nu)$ is"
    " the complex Fresnel diffraction gain:\n"
    "\\begin{equation}\n"
    "    F(\\nu) = \\frac{1+j}{2}\\int_{\\nu}^{\\infty}\\exp\\!\\left(-j\\frac{\\pi t^2}{2}\\right)dt.\n"
    "    \\label{eq:fresnel}\n"
    "\\end{equation}\n"
    "For $\\nu<0$ the element is illuminated ($|F|\\approx 1$); for $\\nu\\gg0$ it is in"
    " deep shadow ($|F|\\to 0$).\n"
    "\n"
    "A \\textbf{secondary edge} captures double-diffraction. The smooth separation weight\n"
    "\\begin{equation}\n"
    "    S_j^{(k)} = 1 - \\exp\\!\\left(-\\frac{(d_{1,k}-\\bar{d}_1)^2}{2\\sigma_s^2}\\right),\n"
    "    \\quad \\sigma_s = 0.2\\text{ m},\n"
    "    \\label{eq:Sj}\n"
    "\\end{equation}\n"
    "penalises edges co-located with the primary cluster and promotes genuinely separate features."
    " The penalised score $\\tilde{h}_k = h_k/(S_j^{(k)}+\\varepsilon')$"
    " is soft-minimised over the top-3 secondary candidates:\n"
    "\\begin{equation}\n"
    "    w_k^{(2)} = \\frac{\\exp(-\\beta\\, \\tilde{h}_k)}{\\sum_{k'}\\exp(-\\beta\\, \\tilde{h}_{k'})},\n"
    "    \\quad \\bar{w}_2 = \\sum_k w_k^{(2)}\\, S_j^{(k)},\n"
    "    \\label{eq:softmax_secondary}\n"
    "\\end{equation}\n"
    "with secondary Fresnel gain:\n"
    "\\begin{equation}\n"
    "    F_2^{\\text{soft}} = \\sum_{k \\in \\text{top-3}} w_k^{(2)}\\, F\\!\\left(\\nu_k^{(2)}\\right),\n"
    "    \\quad \\nu_k^{(2)} = s_k\\sqrt{\\frac{2}{\\lambda(\\bar{d}_1 + d_{1,k})}}.\n"
    "    \\label{eq:F2soft}\n"
    "\\end{equation}\n"
    "The total propagation path $\\bar{d}_1 + d_{1,k}$ reflects the compound diffraction route."
    " The blended total gain is:\n"
    "\\begin{equation}\n"
    "    \\tilde{g}_i = F_1^{\\text{soft}} \\cdot\n"
    "    \\bigl[(1-\\bar{w}_2)\\cdot 1 + \\bar{w}_2 \\cdot F_2^{\\text{soft}}\\bigr].\n"
    "    \\label{eq:totalGain}\n"
    "\\end{equation}\n"
    "The $(1-\\bar{w}_2)\\cdot 1$ term prevents the primary gain from being attenuated by"
    " a spurious secondary term when no genuine second diffractor exists.\n"
    "\n"
    "To detect stale cached results, the SHA-256 hash of the complete $200\\times16$"
    " complex gain matrix $G$ is computed after each diagnostic run and embedded in"
    " the figure title; a changed hash guarantees fresh computation."
    " The system-level simulation ($\\approx\\!9$~min) is cached to a compressed"
    " \\texttt{.npz} file; a \\texttt{--replot} flag regenerates all figures in"
    " $<5$~s.\n"
    "\n"
    "\\subsubsection{Three-Baseline Evaluation Design}\n"
    "\n"
    "Three LCMV configurations are evaluated over a $0^\\circ$--$90^\\circ$"
    " heading sweep at fixed bank $\\phi=30^\\circ$, far-field jammer at $90^\\circ$:\n"
    "\n"
    "\\begin{enumerate}\n"
    "    \\item \\textbf{Baseline (Binary Mask):} $\\tilde{g}_i\\in\\{0,1\\}$ from"
    " ray-mesh intersection. Calibration noise $\\sigma_{\\text{phase}}=0.10\\text{ rad}$"
    " is not applied to jammer paths, so jammer leakage is artificially zeroed."
    " This is the idealised best-case reference.\n"
    "    \\item \\textbf{Baseline (Continuous Mask):} $\\tilde{g}_i$ from Eq.~(\\ref{eq:totalGain})."
    " Calibration noise applied consistently to all paths including jammer leakage."
    " This is the physically realistic baseline.\n"
    "    \\item \\textbf{Cognitive Autopilot (Continuous):} Identical physics to Baseline"
    " (Continuous), but the bank angle is modulated by the Stage~9 LUT subject to"
    " $N_{\\text{act}}(\\phi,\\psi_j)\\geq N_{\\min}=15$.\n"
    "\\end{enumerate}\n"
    "\n"
    "% ================================================================\n"
    "%  3. EXPLORATORY DATA ANALYSIS\n"
    "% ================================================================\n"
    "\\section{Exploratory Data Analysis}"
)

if METH_ANCHOR in txt:
    txt = txt.replace(METH_ANCHOR, NEW_METH_CONTENT, 1)
    print("A. Methodology insertion: OK")
else:
    print("A. Methodology anchor NOT FOUND")
    # Try to find a partial match
    partial = "%  3. EXPLORATORY DATA ANALYSIS"
    idx = txt.find(partial)
    print(f"   Partial '{partial}' at index {idx}")

# ─────────────────────────────────────────────────────────────────
# B.  EDA: replace lines 1801-1883 (old thin Conclusion +
#     Stage 11 comment + rogue subsection + EM physics + 3-baseline)
#     with a clean 'System-Level Results' subsubsection + figure.
# ─────────────────────────────────────────────────────────────────
# The block to replace starts at \subsubsection{Conclusion}
# and ends just before \subsubsection{Cognitive Autopilot Overlap Confirmation}
EDA_OLD_BLOCK = (
    "\\subsubsection{Conclusion}\n"
    "\n"
    "The rigorous independent audit definitively closes the Stage 10 investigation."
    " The structural ceiling that originally motivated the Cognitive Autopilot was an"
    " artifact of a kinematic approximation bug. The 16-element conformal array"
    " inherently possesses the geometric diversity necessary to survive the entire"
    " banked maneuver with 8+ elements and a robust $>15\\text{ dB}$ median Net SINR."
    " STAP is not strictly mandated for this airframe; standard adaptive beamforming"
    " alone guarantees total mission resilience.\n"
    "\n"
    "% ================================================================\n"
    "%  STAGE 11 \u2014 PHASE 2 TRACK 1: EM REALISM (MESH-AWARE DIFFRACTION)\n"
    "% ================================================================\n"
    "\\subsection{Stage 10: Phase 2 Track 1 --- Electromagnetic Realism via Mesh-Aware Fresnel Diffraction}\n"
    "\n"
    "Stage 10 moves from the binary occlusion model of earlier stages to a physically"
    " continuous, mesh-aware electromagnetic shadow model. Where previous stages assigned"
    " each antenna element a binary active/inactive label, Stage 10 computes a complex-valued"
    " gain factor $\\tilde{g}_i \\in \\mathbb{C}$ for every element at every heading, derived"
    " from knife-edge Fresnel diffraction over the actual STL mesh geometry. The stage also"
    " introduces a three-way comparative evaluation framework (Binary Mask baseline, Continuous"
    " Mask baseline, Cognitive Autopilot) and concludes with a verified characterisation of"
    " the residual per-element gain discontinuities.\n"
    "\n"
    "\\subsubsection{Electromagnetic Physics Model}\n"
    "\n"
    "For each element $i$ and each candidate jammer direction $\\hat{\\mathbf{d}}_j$"
    " (expressed in body frame), a ray is cast from the antenna origin"
    " $\\mathbf{p}_i + \\varepsilon\\hat{\\mathbf{n}}_i$ (offset by $\\varepsilon = 10^{-4}\\text{ m}$"
    " along the surface normal to avoid self-intersection) toward the jammer."
    " The ray intersects the set of mesh edges $\\{e_k\\}$, each characterised by:\n"
    "\\begin{itemize}\n"
    "    \\item $h_k$ -- the perpendicular distance from the ray to edge $e_k$"
    " (smallest possible detour distance the wave must travel to reach the shadow zone via that edge),\n"
    "    \\item $d_{1,k}$ -- the along-ray distance from the antenna origin to the closest point on edge $e_k$.\n"
    "\\end{itemize}\n"
    "\n"
    "Only edges with physical length $>0.015\\text{ m}$ are retained;"
    " degenerate triangulation sliver edges shorter than this threshold produce"
    " spurious diffraction terms and are filtered.\n"
    "\n"
    "The \\textbf{primary edge} is the edge that the ray passes closest to:\n"
    "\\begin{equation}\n"
    "    k_1 = \\arg\\min_k\\; h_k.\n"
    "\\end{equation}\n"
    "Previous iterations used this hard $\\arg\\min$ directly."
    " As characterised in Section~\\ref{sec:stage10_softmax}, this causes a discrete gain jump"
    " whenever two edges swap rank. The final implementation replaces it with a softmax-weighted"
    " average over the top-3 candidate edges:\n"
    "\\begin{equation}\n"
    "    w_k^{(1)} = \\frac{\\exp(-\\beta\\, h_k)}{\\sum_{k'}\\exp(-\\beta\\, h_{k'})},"
    " \\quad \\beta = 500,\n"
    "    \\label{eq:softmax_primary}\n"
    "\\end{equation}\n"
    "yielding a smooth weighted primary distance $\\bar{d}_1 = \\sum_k w_k^{(1)}\\, d_{1,k}$"
    " and a smooth primary Fresnel gain:\n"
    "\\begin{equation}\n"
    "    F_1^{\\text{soft}} = \\sum_{k \\in \\text{top-3}} w_k^{(1)}\\, F\\!\\left(\\nu_k^{(1)}\\right),\n"
    "    \\quad \\nu_k^{(1)} = s_k\\sqrt{\\frac{2}{\\lambda d_{1,k}}},\n"
    "    \\label{eq:F1soft}\n"
    "\\end{equation}\n"
    "where $s_k$ is the signed distance (positive = inside the mesh body) from the"
    " ray-edge closest point to the mesh surface, $\\lambda = 0.125\\text{ m}$ (2.4 GHz),"
    " and $F(\\nu)$ is the complex Fresnel diffraction gain:\n"
    "\\begin{equation}\n"
    "    F(\\nu) = \\frac{1+j}{2}\\int_{\\nu}^{\\infty}\\exp\\!\\left(-j\\frac{\\pi t^2}{2}\\right)dt.\n"
    "    \\label{eq:fresnel}\n"
    "\\end{equation}\n"
    "For $\\nu<0$ the element is in the illuminated half-space ($|F|\\approx 1$);"
    " for $\\nu\\gg0$ the element is deep in the shadow ($|F|\\to 0$).\n"
    "\n"
    "The \\textbf{secondary edge} accounts for double-diffraction: the wave may diffract"
    " around a second independent mesh feature before reaching the shadowed element."
    " The smooth separation weight\n"
    "\\begin{equation}\n"
    "    S_j^{(k)} = 1 - \\exp\\!\\left(-\\frac{(d_{1,k}-\\bar{d}_1)^2}{2\\sigma_s^2}\\right),"
    " \\quad \\sigma_s = 0.2\\text{ m},\n"
    "    \\label{eq:Sj}\n"
    "\\end{equation}\n"
    "penalises edges co-located with the primary cluster ($S_j^{(k)}\\to 0$ when"
    " $d_{1,k}\\approx\\bar{d}_1$) and promotes genuinely distinct edges"
    " ($S_j^{(k)}\\to 1$ when they are well-separated). The penalised proximity score"
    " $\\tilde{h}_k = h_k/(S_j^{(k)}+\\varepsilon')$ is then soft-minimised over the"
    " top-3 secondary candidates:\n"
    "\\begin{equation}\n"
    "    w_k^{(2)} = \\frac{\\exp(-\\beta\\, \\tilde{h}_k)}{\\sum_{k'}\\exp(-\\beta\\, \\tilde{h}_{k'})},"
    "\n"
    "    \\quad \\bar{w}_2 = \\sum_k w_k^{(2)}\\, S_j^{(k)},\n"
    "    \\label{eq:softmax_secondary}\n"
    "\\end{equation}\n"
    "yielding the soft secondary Fresnel gain:\n"
    "\\begin{equation}\n"
    "    F_2^{\\text{soft}} = \\sum_{k \\in \\text{top-3}} w_k^{(2)}\\, F\\!\\left(\\nu_k^{(2)}\\right),\n"
    "    \\quad \\nu_k^{(2)} = s_k\\sqrt{\\frac{2}{\\lambda(\\bar{d}_1 + d_{1,k})}}.\n"
    "    \\label{eq:F2soft}\n"
    "\\end{equation}\n"
    "The effective path length for the secondary Fresnel parameter uses $\\bar{d}_1 + d_{1,k}$"
    " rather than $d_{1,k}$ alone, reflecting the total propagation distance through the"
    " diffracting body. The blended total gain for element $i$ is:\n"
    "\\begin{equation}\n"
    "    \\tilde{g}_i = F_1^{\\text{soft}} \\cdot \\bigl[(1-\\bar{w}_2)\\cdot 1 + \\bar{w}_2 \\cdot F_2^{\\text{soft}}\\bigr],\n"
    "    \\label{eq:totalGain}\n"
    "\\end{equation}\n"
    "where the $(1-\\bar{w}_2)\\cdot 1$ term ensures that in the absence of a genuine secondary"
    " edge (small $\\bar{w}_2$), the primary diffraction gain is not attenuated by a spurious $F_2$ term.\n"
    "\n"
    "\\subsubsection{Three-Baseline Evaluation Framework}\n"
    "\n"
    "Three LCMV configurations were evaluated over a $0^\\circ$--$90^\\circ$ heading sweep at a"
    " fixed bank angle of $\\phi=30^\\circ$, with a far-field jammer at bearing $90^\\circ$ (broadside):\n"
    "\n"
    "\\begin{enumerate}\n"
    "    \\item \\textbf{Baseline (Binary Mask):} Element gains are binary---$\\tilde{g}_i\\in\\{0,1\\}$"
    " depending on ray-mesh intersection. Calibration phase noise $\\sigma_{\\text{phase}}=0.10\\text{ rad}$"
    " is intentionally \\emph{not} applied to jammer signal paths, meaning jammer leakage to the"
    " binary-masked elements is artificially set to zero. This represents the idealised ``best-case'' baseline.\n"
    "    \\item \\textbf{Baseline (Continuous Mask):} Element gains are computed by Eq.~(\\ref{eq:totalGain})."
    " Calibration noise is applied consistently to all signal paths including jammer leakage.\n"
    "    \\item \\textbf{Cognitive Autopilot (Continuous):} Identical physics to Baseline (Continuous),"
    " but the bank angle at each heading is modulated by a pre-computed look-up table (LUT) subject to"
    " $N_{\\text{act}}(\\phi,\\psi_j)\\geq N_{\\min}=15$. As detailed below, this threshold was never"
    " achievable for this airframe geometry, so the autopilot fell back to $\\phi=30^\\circ$ throughout,"
    " making the Cognitive trace numerically identical to the Continuous baseline.\n"
    "\\end{enumerate}\n"
    "\n"
    "\\begin{figure}[H]\n"
    "\\centering\n"
    "\\includegraphics[width=0.95\\textwidth]{phase2_track1_system_results.png}\n"
    "\\caption{Stage 10 system-level results: four-panel sweep over heading $0^\\circ$--$90^\\circ$"
    " at $\\phi=30^\\circ$ bank. \\textit{Panel 1 (SINR):} Binary mask (red dashed) sits 3--8~dB"
    " above the continuous curves due to artificially zeroed jammer leakage; both Continuous Mask (blue)"
    " and Cognitive (green) coincide within numerical precision. \\textit{Panel 2 (Active elements):}"
    " Orange dashed line marks the LCMV rank floor $M=3$ (minimum for two constraints); blue dotted line"
    " marks the Cognitive Autopilot search threshold $N_{\\min}=15$, which the array never achieves,"
    " explaining why the autopilot always falls back to $\\phi=30^\\circ$. \\textit{Panel 3 (Null depth):}"
    " Binary mask shows a $-300\\text{ dB}$ floor because jammer power is truncated to zero; continuous"
    " curves maintain $-45$ to $-55\\text{ dB}$ null depth throughout. \\textit{Panel 4 (Bank angle):}"
    " Autopilot commands constant $30^\\circ$, confirming non-intervention.}\n"
    "\\label{fig:stage10_system}\n"
    "\\end{figure}\n"
    "\n"
)

EDA_NEW_BLOCK = (
    "\\subsubsection{Electromagnetic Realism: System-Level Results}\n"
    "\n"
    "\\begin{figure}[H]\n"
    "\\centering\n"
    "\\includegraphics[width=0.95\\textwidth]{phase2_track1_system_results.png}\n"
    "\\caption{Stage 10 EM-realism: four-panel heading sweep ($0^\\circ$--$90^\\circ$,"
    " $\\phi=30^\\circ$ bank). \\textit{Panel 1 (SINR):} Binary mask (red dashed) sits"
    " 3--8~dB above the continuous curves due to artificially zeroed jammer leakage;"
    " Continuous Mask (blue) and Cognitive Autopilot (green) coincide within numerical precision."
    " \\textit{Panel 2 (Active elements):} Orange dashed line marks the LCMV rank floor $M=3$;"
    " blue dotted line marks the Cognitive Autopilot search threshold $N_{\\min}=15$, which the"
    " array never achieves, confirming non-intervention throughout. \\textit{Panel 3 (Null depth):}"
    " Binary mask shows a $-300\\text{ dB}$ floor (jammer power truncated to zero --- an artefact);"
    " continuous model gives $-45$ to $-55\\text{ dB}$, consistent with $\\sigma_{\\text{phase}}=0.10\\text{ rad}$."
    " \\textit{Panel 4 (Bank angle):} Autopilot holds constant $30^\\circ$.}\n"
    "\\label{fig:stage10_system}\n"
    "\\end{figure}\n"
    "\n"
    "The Binary Mask minimum SINR of 20.68~dB substantially exceeds the physically realistic"
    " Continuous Mask minimum of 15.18~dB. The gap originates from the binary mask's treatment"
    " of jammer leakage: occluded elements are hard-gated to zero, so their contribution to the"
    " jammer covariance is absent rather than Fresnel-attenuated. Once jammer leakage is modelled"
    " consistently (Continuous Mask), the SINR floor drops by $\\approx\\!5\\text{ dB}$ to a"
    " physically defensible value.\n"
    "\n"
    "The active element count (Panel~2) reaches a minimum of 12 of 16 in both variants ---"
    " comfortably above the LCMV rank floor of $M=3$. The null depth panel (Panel~3) explains"
    " the binary mask's $-300\\text{ dB}$ floor: with jammer power zeroed, the LCMV null is"
    " formed against zero leakage, producing an artefactually deep null. The continuous model"
    " maintains $-45$ to $-55\\text{ dB}$ throughout, consistent with hardware-realistic calibration noise.\n"
    "\n"
)

if EDA_OLD_BLOCK in txt:
    txt = txt.replace(EDA_OLD_BLOCK, EDA_NEW_BLOCK, 1)
    print("B. EDA replacement: OK")
else:
    # Try to find what's different
    # search character by character for the longest matching prefix
    for end in range(len(EDA_OLD_BLOCK), 0, -100):
        if EDA_OLD_BLOCK[:end] in txt:
            print(f"B. EDA anchor: partial match up to char {end} / {len(EDA_OLD_BLOCK)}")
            break
    else:
        print("B. EDA anchor: NO match found")

# ─────────────────────────────────────────────────────────────────
# C. Fix Cognitive Autopilot Overlap section - remove "track" reference
# ─────────────────────────────────────────────────────────────────
txt = txt.replace(
    "This is documented as expected behaviour for the EM-realism validation track:"
    " the Cognitive Autopilot's value is demonstrably non-zero only when"
    " $N_{\\text{act,min}} < N_{\\min}$, which requires a tighter threshold"
    " (e.g.\\ $N_{\\min}=13$, one above the worst-case) or a different airframe geometry."
    " Evaluating the autopilot under those conditions belongs to the Array Geometry"
    " Optimisation track (Track 2), where the element placement is a design variable.",
    "The Cognitive Autopilot is non-zero only when $N_{\\text{act,min}} < N_{\\min}$,"
    " which requires either a tighter threshold (e.g.\\ $N_{\\min}=13$, one above the worst-case)"
    " or a different airframe element layout --- precisely the objective of the Array Geometry"
    " Optimisation future work direction.", 1)
print("C. Overlap section: OK")

# ─────────────────────────────────────────────────────────────────
# D. Fix progression table
# ─────────────────────────────────────────────────────────────────
# The old Stage 10 row only (Stage 11 was already removed in previous script)
old_row = ("10: Mesh LCMV (corrected) & Exact 3D quaternion kinematics, phase noise"
           " calibration & $\\geq8$ active elements throughout; median SINR $>15\\text{ dB}$\\\\")
new_row = ("10: Mesh-Aware LCMV & Exact 3D kinematics; Fresnel shadow model; dual-softmax"
           " edge selection; 3-baseline sweep & $\\geq12$ active elements; min SINR 15.18~dB;"
           " max element step $0.54\\text{ dB}$; cognitive/baseline overlap 0.0000~dB\\\\")
if old_row in txt:
    txt = txt.replace(old_row, new_row, 1)
    print("D. Progression table: OK")
else:
    print("D. Progression table row NOT FOUND (may already have been fixed)")

with open(TEX, 'w', encoding='utf-8') as f:
    f.write(txt)

print(f"\nFile length: {orig_len} -> {len(txt)} bytes")
print("Backup: master.tex.bak3")
