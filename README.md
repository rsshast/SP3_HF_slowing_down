# Continuous Multigroup SpN Calculation in Slowing Down Region
Paper available upon request, 

rsshast@umich.edu

Ravi Shastri, Brian Kiedrowski

Discritization over angle and then energy to find the slowing down flux spectrum. 
Then, calculate the fission source, create new flux weighted cross-sections (xs's) and diffusion coefficients. 

Usage:

setup.py contains:
  * energy group discritization and corresponding interpolated cross sections
  * Number Densities
  * Option to generate fresh data or load from .h5 files. 
  
Sp3.py Method contains:
  methods for calculating scattering and loss operators
  methods for calculating flux moments
  methods for calculating new group cross sections and diffusion coefficients
  
run_sp3.py contains:
  gets the data
  plots the cross-sections
  sp3.run()

plot_scat_ranges.py contains: 
  function to visualize the group bound for the scattering integral

Completed Work:
  Transformed from Energy to Lethargy Discretization in Energy
  Generated group to group xs's for Legendre expansions [0,3] for H1 and U235
  Integrated those cross sections to find the Loss Operator on the same range
  Found the slowing down flux spectrum $\phi_0$ and weighting function $\phi_2$
  Used those fine grid fluxes to calculate flux weighted xs's and diffusion coefficients
  Data saved to .h5 files for superior compression

Current work: 
  Iteratively sove the Sp3 transport problem with updated xs's
  Compress the matrix operators to TT format OR CSR sparse matrix conversion
  Use flux spectrum to find resonant parameters $\lambda$

Future Work:
  Continuous Energy Monte-Carlo to find buckling parameter $B^2$
  Expansion of 2 material to N-region

Assumptions:
  Constant gridspacing in lethargy
  Group constants are constant within the energy group
  Homogenouse media weighted by number densities
  
HW2: Slowing down calculation for fine grid with discritization over energy then angle. This works. 
