# Sp3 Hyper-fine Group Slowing Down Calculation
Paper available upon request, please email Ravi at rsshast@umich.edu

Contributors: Ravi Shastri, Brian Kiedrowski

Discritization over angle and then energy to find the slowing down flux spectrum. 
Then, calculate the fission source, create new flux weighted cross-sections (xs's) and diffusion coefficients. 

Usage:

The user must create a *results* directory with subdirectories *charts* and *data*. 
The user may change file paths within **Sp3.__init__** to match their desired directory structure. 

Data contains:
  * Cross section data for
    * Hydrogen
    * Uranium
  * Fission Spectrum, $\chi(E)$

HF_SD_Sp3.py contains:
  * Required modules (numpy, pandas, matplotlib, hdf5)
  * Reading data from Data
  * Flux calculation from Scattering Source
  * Linearly Interpolate (in lethargy) data on hyperfine grid. User sets the number of gridpoints.
  * Generate $\Sigma_{gtg}^{sl}$ for l = [0,1,2,3]
  * Solve for $\phi_0$ and $\phi_2$
  * Update cross sections with accurate flux weighting
  * Calculate Diffusion Coefficients

Completed Work:
  * Transformed from Energy to Lethargy Discretization in Energy
  * Generated group to group xs's for Legendre expansions [0,3] for H1 and U235
  * Integrated those cross sections to find the Loss Operator on the same range
  * Found the slowing down flux spectrum $\phi_0$ and weighting function $\phi_2$
  * Used those fine grid fluxes to calculate flux weighted xs's and diffusion coefficients
  * Data saved to .h5 files for superior compression
  * Loaded data from previous runs
  * N material compatibility

Current work: 
  * Flux weighted cross-sections and diffusion coefficients
  * Continuous Energy Monte-Carlo to find buckling parameter $B^2$

Assumptions:
  * Constant gridspacing in lethargy
  * Group constants are constant within the energy group
  * Homogenous media weighted by number densities
