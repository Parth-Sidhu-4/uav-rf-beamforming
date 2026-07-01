# THIRD-PARTY ASSET CREDITS AND NOTICES

---

## ScanEagle UAV 3D Mesh (STL File)

**Asset Name:** Scan Eagle UAV  
**Creator:** Daniel (`@Daniel_196575`)  
**Source:** Printables.com  
**URL:** https://www.printables.com/model/562283-scan-eagle-uav  
**Downloaded:** August 2023 (model last updated August 24, 2023)  
**Asset Type:** 3D printable STL mesh (ScanEagle-class airframe geometry)

### How This Asset Was Used

The STL mesh was used exclusively as a **reference geometry** for computational
electromagnetic simulation. Specifically, it was:

- Loaded into a Bounding Volume Hierarchy (BVH) ray-tracing engine to evaluate
  line-of-sight occlusion of antenna array elements under varying UAV roll
  and heading angles.
- Used to compute a physically continuous Fresnel knife-edge diffraction shadow
  model over the airframe's structural edges.
- Used to define conformal array element placement zones on the fuselage surface
  for the Differential Evolution array geometry optimiser (Stage 12).

The mesh was **not physically fabricated**, **not modified for redistribution**,
and **not used for any 3D printing purpose**. Its role was purely as a
numerical geometry kernel for Python-based simulation scripts.

### Attribution Statement

This project gratefully acknowledges the work of Daniel (`@Daniel_196575`)
for making the ScanEagle UAV 3D model freely available on Printables.com.
The airframe geometry used in all structural electromagnetic simulations in
this research is derived from that model.

If you use this research and also need the underlying mesh, please download
it directly from the original source and credit the original creator:

> **"Scan Eagle UAV" by Daniel (@Daniel_196575), Printables.com**  
> https://www.printables.com/model/562283-scan-eagle-uav

---

## Disclaimers Regarding the Third-Party STL Asset

**1. Quality and Accuracy**  
The ScanEagle UAV STL mesh is a community-created 3D printable model, not
an engineering-certified CAD dataset. The creator themselves noted it is
a "WIP" (work in progress). While Parth Sidhu performed basic geometric
validation of the mesh (manifold check, face-normal consistency, BVH
tree construction) prior to using it in simulation, **no warranty is made
regarding the dimensional accuracy, geometric fidelity, or real-world
representativeness of this mesh relative to any actual Boeing Insitu
ScanEagle production airframe.**

The simulation results derived using this mesh are therefore bounded by the
accuracy of the underlying community geometry. Parth Sidhu and Encap
Technologies India Private Limited accept no liability for any inaccuracies
that arise from reliance on this third-party asset.

**2. No Affiliation with Boeing or Insitu**  
The Boeing Insitu ScanEagle is a real-world UAV platform manufactured by
Insitu, Inc., a subsidiary of The Boeing Company. This research project has
**no affiliation, endorsement, sponsorship, or contractual relationship**
with Boeing, Insitu, or any of their subsidiaries, affiliates, or
government customers. The use of a publicly available community-made 3D
model resembling this airframe class is for **academic simulation purposes
only** and does not constitute access to, or disclosure of, any proprietary
Boeing/Insitu technical data, intellectual property, or controlled defence
information.

**3. No Liability for Mesh Availability**  
The STL mesh used in this project was obtained from a publicly accessible
community platform (Printables.com) where it was listed as a free download.
Parth Sidhu and Encap Technologies India Private Limited bear **no
responsibility for the original creator's decision to publish this model
publicly**, for the platform's distribution policies, or for any consequence
arising from the fact that this geometry is openly accessible to any party,
including but not limited to:

- The manufacturer of the actual ScanEagle airframe (Boeing/Insitu);
- Any national or international government, military, or defence agency
  that may operate, procure, or regulate the ScanEagle UAV system;
- Any other individual or organisation that downloads or uses the original
  3D model from its source platform.

Any concern regarding the public availability of the 3D model itself should
be directed to the original creator (`@Daniel_196575`) or to Printables.com,
not to the Author of this research project.

**4. No Export Control Concern**  
This project uses a community-made visual/geometric approximation of a
publicly known airframe class for domestic academic simulation. No
controlled technical data, ITAR/EAR-regulated information, or classified
defence specifications were accessed or used at any stage of this research.

---

*This file is incorporated by reference into DISCLAIMER.md and governs the
use of all third-party assets in this project.*

*Parth Sidhu — July 2026*
