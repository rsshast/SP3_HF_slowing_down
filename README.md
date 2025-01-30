# Continuous Multigroup SpN Calculation in Slowing Down Region
Paper available upon request, rsshast@umich.edu
Ravi Shastri, Brian Kiedrowski

Discritization over angle and then energy. 

Usage:
setup.py contains:
  energy group discritization and corresponding interpolated cross sections
  Number Densities
  
Sp3_method.py (class) contains:
  methods for calculating scattering and loss operators
  methods for calculating flux moments
  methods for calculating new group cross sections and diffusion coefficients
  
run_sp3.py contains:
  gets the data
  plots the cross-sections
  sp3.run()

Current Work:
  Fix loss operator function to get the right flux spectrum
  Mixture calculations
  Use those fine grid fluxes to calculate xs's and coefficients

Future Work:
  Tensor Network Compression on Hyperfine Grid for Loss Operators
  Continuous Energy Monte-Carlo to find buckling parameter $B^2$

HW2: Slowing down calculation for fine grid with discritization over energy then angle. This works. 
