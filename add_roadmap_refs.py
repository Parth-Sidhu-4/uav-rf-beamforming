import re

with open('master.tex', 'r', encoding='utf-8') as f:
    tex = f.read()

new_refs = r"""
\bibitem{Kouyoumjian1974}
Kouyoumjian, R.\ G.\ and Pathak, P.\ H., ``A Uniform Geometrical Theory of Diffraction for an Edge in a Perfectly Conducting Surface,'' \textit{Proceedings of the IEEE}, vol.\ 62, no.\ 11, pp.\ 1448--1461, 1974.

\bibitem{Keller1962}
Keller, J.\ B., ``Geometrical Theory of Diffraction,'' \textit{Journal of the Optical Society of America}, vol.\ 52, no.\ 2, pp.\ 116--130, 1962.

\bibitem{Boyd2004}
Boyd, S.\ and Vandenberghe, L., \textit{Convex Optimization}. Cambridge University Press, 2004.

\bibitem{Tse2005}
Tse, D.\ and Viswanath, P., \textit{Fundamentals of Wireless Communication}. Cambridge University Press, 2005.
"""

tex = tex.replace(r"\end{thebibliography}", new_refs + "\n" + r"\end{thebibliography}")

with open('master.tex', 'w', encoding='utf-8') as f:
    f.write(tex)

print("More references added.")
