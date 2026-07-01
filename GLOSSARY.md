# GLOSSARY OF TERMS

This document provides plain-English definitions for the technical acronyms and domain-specific terminology used throughout this simulation framework. It is intended to assist software engineers, recruiters, and non-specialists in understanding the physics and engineering concepts modeled in this repository.

---

### AWGN
**Additive White Gaussian Noise.** A basic noise model used in information theory to mimic the effect of many random natural processes (like thermal noise in a radio receiver) that distort a signal.

### BVH
**Bounding Volume Hierarchy.** A highly efficient spatial data structure (like a tree) used in computer graphics and ray-tracing. In this project, it is used to instantly calculate whether a radio wave collides with the 3D 37,000-polygon mesh of the drone's airframe.

### DE
**Differential Evolution.** A type of evolutionary algorithm used for complex optimization. In Stage 11, it is used to figure out the absolute best places to stick antennas on the drone's curved wings to maximize signal coverage.

### DRFM
**Digital Radio Frequency Memory.** An advanced electronic warfare technique where an enemy radar or jammer records an incoming signal, alters it slightly, and blasts it back to confuse the target. 

### EW
**Electronic Warfare.** Military action involving the use of electromagnetic energy (radio waves, infrared, etc.) to control the spectrum, attack an enemy, or impede enemy assaults. Jamming a drone's communication link is a form of EW.

### FFN
**Fourier Feature Network.** A type of neural network that is exceptionally good at learning high-frequency, complex functions. In Stage 14, this is used as an AI "surrogate" to replace slow, heavy mathematics with a lightning-fast neural approximation.

### LCMV
**Linearly Constrained Minimum Variance.** A classic mathematical algorithm used in adaptive beamforming. It figures out how to combine signals from multiple antennas so that you listen clearly in one direction (the ground station) while completely "tuning out" noise from another direction (the jammer).

### LOS / NLOS
**Line-Of-Sight / Non-Line-Of-Sight.** 
- *LOS* means there is a clear, unobstructed path between the transmitter and the receiver. 
- *NLOS* means the radio waves have to bounce off buildings, mountains, or (in this project) the drone's own wings to reach the receiver.

### RF
**Radio Frequency.** The portion of the electromagnetic spectrum used for wireless telecommunications (e.g., Wi-Fi, 5G, drone telemetry, radar).

### SINR
**Signal-to-Interference-plus-Noise Ratio.** A metric used to measure the quality of a wireless connection. It's the ratio of the "good" signal you want to hear, divided by the sum of the enemy jammer's interference plus background noise. If SINR drops too low, the drone loses connection.

### SNR
**Signal-to-Noise Ratio.** Similar to SINR, but it only looks at background noise (ignoring active enemy jammers).

### UAV
**Unmanned Aerial Vehicle.** The technical engineering term for a drone (e.g., the ScanEagle model used in this simulation).
