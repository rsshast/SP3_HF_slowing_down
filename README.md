# Hyperfine $SP_3$ Calculation in Slowing Down Region with Imposed Leakage $B^2$
Paper available upon request, please email rsshast@umich.edu

Ravi Shastri, Brian Kiedrowski

Discritization over angle and then energy to find the slowing down flux spectrum. 
Then, calculate the fission source, create new flux weighted cross-sections (xs's) and diffusion coefficients. 

# Requirements:
  * Python=3.11.0
  * numpy=1.26.2
  * pandas=2.1.4
  * matplotlib
  * h5py=3.12.1
  * numba=0.59.0
  * PyTorch, torch=2.5.1+cu124
  * psutil=5.9.0

# Usage:
## batch_sp3.sh contains:
  * Slurm cluster submission script
    * CPU/GPU compatibility
  * Edit with your cluster information

## HF_SD_Sp3.py contains:
  * $SP_3$ solver class
  * Initial Conditions
    * Energy ranges
    * Materials and number densities
    * Cross-section data processing and generation to order L=3
  * Reference solution from Scattering Source Solution, sp3.p0
  * Parametric Leakage Calculation
  * $\phi_0, \phi_2, \Phi_0, \Phi_2$ Calculations using the Loss Operator Driven Method
  * Tensorized and GPU Compatible Methods for Solving the Slowing-Down Spectrum
  * Result Analysis
  * Flux Weighted Cross-Section and Diffusion Coefficient Generation (Group Constants)

Completed Work:
  * Transformed from Energy to Lethargy Discretization in Energy
  * Generated group to group xs's for Legendre expansions [0,3] for H1 and U235
  * Integrated those cross sections to find the Loss Operator on the same range
  * Found the slowing down flux spectrum $\phi_0$ and weighting function $\phi_2$
  * Used those fine grid fluxes to calculate flux weighted xs's and diffusion coefficients
  * Exported to torch.tensor and GPU

Current work: 
  * Incorporate Anisotropic Group-Coupled Upscatter to reference and $SP_3$ solutions

Future Work:
  * Continuous Energy Monte-Carlo to find buckling parameter $B^2$
  * Submit research as a journal publication

Assumptions:
  * Constant gridspacing in lethargy
  * Group constants are constant within the energy group
  * Homogenous media weighted by number densities
