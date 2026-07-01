import re

bib_content = r"""
% ================================================================
%  REFERENCES
% ================================================================
\newpage
\phantomsection
\addcontentsline{toc}{section}{References}
\begin{thebibliography}{99}

\bibitem{VanTrees2002}
Van Trees, H.\ L., \textit{Optimum Array Processing: Part IV of Detection, Estimation, and Modulation Theory}. John Wiley \& Sons, 2002.

\bibitem{Capon1969}
Capon, J., ``High-Resolution Frequency-Wavenumber Spectrum Analysis,'' \textit{Proc. IEEE}, vol.\ 57, no.\ 8, pp.\ 1408--1418, 1969.

\bibitem{Schmidt1986}
Schmidt, R.\ O., ``Multiple Emitter Location and Signal Parameter Estimation,'' \textit{IEEE Transactions on Antennas and Propagation}, vol.\ 34, no.\ 3, pp.\ 276--280, 1986.

\bibitem{AlHourani2014}
Al-Hourani, A., Kandeepan, S., and Lardner, S., ``Optimal LAP Altitude for Maximum Coverage,'' \textit{IEEE Wireless Communications Letters}, vol.\ 3, no.\ 6, pp.\ 569--572, 2014.

\bibitem{Molisch2010}
Molisch, A.\ F., \textit{Wireless Communications}, 2nd ed. John Wiley \& Sons, 2010.

\bibitem{Goldsmith2005}
Goldsmith, A., \textit{Wireless Communications}. Cambridge University Press, 2005.

\bibitem{Skolnik2008}
Skolnik, M.\ I., \textit{Radar Handbook}, 3rd ed. McGraw-Hill Education, 2008.

\bibitem{Grewal2014}
Grewal, M.\ S.\ and Andrews, A.\ P., \textit{Kalman Filtering: Theory and Practice with MATLAB}, 4th ed. John Wiley \& Sons, 2014.

\bibitem{Groves2013}
Groves, P.\ D., \textit{Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems}, 2nd ed. Artech House, 2013.

\bibitem{Rabiner1989}
Rabiner, L.\ R., ``A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition,'' \textit{Proceedings of the IEEE}, vol.\ 77, no.\ 2, pp.\ 257--286, 1989.

\bibitem{Puterman2014}
Puterman, M.\ L., \textit{Markov Decision Processes: Discrete Stochastic Dynamic Programming}. John Wiley \& Sons, 2014.

\bibitem{Haykin2005}
Haykin, S., ``Cognitive Radio: Brain-Empowered Wireless Communications,'' \textit{IEEE Journal on Selected Areas in Communications}, vol.\ 23, no.\ 2, pp.\ 201--220, 2005.

\end{thebibliography}

"""

with open('master.tex', 'r', encoding='utf-8') as f:
    tex = f.read()

# Insert before Appendix A
marker = r"%  APPENDIX A"
if marker in tex:
    tex = tex.replace(marker, bib_content + "\n" + marker)
else:
    print("Marker not found!")

with open('master.tex', 'w', encoding='utf-8') as f:
    f.write(tex)

print("References added.")
