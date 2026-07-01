import re

unified_refs = r"""% ================================================================
%  REFERENCES
% ================================================================
\newpage
\phantomsection
\addcontentsline{toc}{section}{References}
\begin{thebibliography}{99}

\bibitem{VanTrees2002}
Van Trees, H.L., \textit{Optimum Array Processing: Part IV of Detection, Estimation, and Modulation Theory}, Wiley-Interscience, New York, 2002.

\bibitem{Simon2005}
Simon, M.K. and Alouini, M.-S., \textit{Digital Communication over Fading Channels}, 2nd ed., Wiley, Hoboken, 2005.

\bibitem{AlHourani2014}
Al-Hourani, A., Kandeepan, S., and Jamalipour, A., ``Modeling Air-to-Ground Path Loss for Low Altitude Platforms in Urban Environments,'' in \textit{Proc.\ IEEE WCNC}, Istanbul, Turkey, 2014, pp.\ 2898--2902.

\bibitem{Khawaja2019}
Khawaja, W., Guvenc, I., Matolak, D.W., Fiebig, U.-C., and Schneckenburger, N., ``A Survey of Air-to-Ground Propagation Channel Modeling for Unmanned Aerial Vehicles,'' \textit{IEEE Commun.\ Surveys Tuts.}, vol.\ 21, no.\ 3, pp.\ 2361--2391, 2019.

\bibitem{Schmidt1986}
Schmidt, R.O., ``Multiple Emitter Location and Signal Parameter Estimation,'' \textit{IEEE Trans.\ Antennas Propag.}, vol.\ 34, no.\ 3, pp.\ 276--280, Mar.\ 1986.

\bibitem{Capon1969}
Capon, J., ``High-Resolution Frequency-Wavenumber Spectrum Analysis,'' \textit{Proc.\ IEEE}, vol.\ 57, no.\ 8, pp.\ 1408--1418, Aug.\ 1969.

\bibitem{Frost1972}
Frost, O.L., ``An Algorithm for Linearly Constrained Adaptive Array Processing,'' \textit{Proc.\ IEEE}, vol.\ 60, no.\ 8, pp.\ 926--935, Aug.\ 1972.

\bibitem{Proakis2008}
Proakis, J.G. and Salehi, M., \textit{Digital Communications}, 5th ed., McGraw-Hill, New York, 2008.

\bibitem{Molisch2010}
Molisch, A.F., \textit{Wireless Communications}, 2nd ed. John Wiley \& Sons, 2010.

\bibitem{Goldsmith2005}
Goldsmith, A., \textit{Wireless Communications}. Cambridge University Press, 2005.

\bibitem{Skolnik2008}
Skolnik, M.I., \textit{Radar Handbook}, 3rd ed. McGraw-Hill Education, 2008.

\bibitem{Grewal2014}
Grewal, M.S. and Andrews, A.P., \textit{Kalman Filtering: Theory and Practice with MATLAB}, 4th ed. John Wiley \& Sons, 2014.

\bibitem{Groves2013}
Groves, P.D., \textit{Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems}, 2nd ed. Artech House, 2013.

\bibitem{Rabiner1989}
Rabiner, L.R., ``A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition,'' \textit{Proceedings of the IEEE}, vol.\ 77, no.\ 2, pp.\ 257--286, 1989.

\bibitem{Puterman2014}
Puterman, M.L., \textit{Markov Decision Processes: Discrete Stochastic Dynamic Programming}. John Wiley \& Sons, 2014.

\bibitem{Neely2010}
Neely, M.J., \textit{Stochastic Network Optimization with Application to Communication and Queueing Systems}, Morgan \& Claypool, 2010.

\bibitem{Haykin2005}
Haykin, S., ``Cognitive Radio: Brain-Empowered Wireless Communications,'' \textit{IEEE Journal on Selected Areas in Communications}, vol.\ 23, no.\ 2, pp.\ 201--220, 2005.

\bibitem{Kouyoumjian1974}
Kouyoumjian, R.G. and Pathak, P.H., ``A Uniform Geometrical Theory of Diffraction for an Edge in a Perfectly Conducting Surface,'' \textit{Proc.\ IEEE}, vol.\ 62, no.\ 11, pp.\ 1448--1461, Nov.\ 1974.

\bibitem{Keller1962}
Keller, J.B., ``Geometrical Theory of Diffraction,'' \textit{Journal of the Optical Society of America}, vol.\ 52, no.\ 2, pp.\ 116--130, 1962.

\bibitem{Boyd2004}
Boyd, S. and Vandenberghe, L., \textit{Convex Optimization}. Cambridge University Press, 2004.

\bibitem{Tse2005}
Tse, D. and Viswanath, P., \textit{Fundamentals of Wireless Communication}. Cambridge University Press, 2005.

\bibitem{Trimesh}
Trimesh Python library, \url{https://trimsh.org/}, accessed June 2026.

\bibitem{SciPy}
SciPy scientific computing library, \url{https://scipy.org/}, accessed June 2026.

\end{thebibliography}"""

with open('master.tex', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = lines[:1780] + [unified_refs + '\n'] + lines[1864:]

with open('master.tex', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("Unified references injected.")
