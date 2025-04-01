'''
setup.py
'''
######################################
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import math as math
import pandas as pd
import time
import os
from functions_hw2 import *
#######################Read Data ##################################
q1plot = False
sigma_p_H = 20
sigma_p_O = 4
sigma_p_U = 9
data_dir = '../data/'
wims69 = pd.read_csv(f'{data_dir}wims69.txt',sep='\t',header=0)
#wims = np.array(wims69['E'],wims69['g'])
Ewims = np.array(wims69['E'])
Ewims = Ewims[0:39]
Gwims = np.array(wims69['g'])
Gwims = Gwims[0:39]
###
chi35 = pd.read_csv(f'{data_dir}chi_u235.txt', sep = '\t',header = 0)
chi = np.array([chi35['E'],chi35['chi']])
chi = np.transpose(chi)
###
HXs = pd.read_csv(f'{data_dir}xs_h1_T293k.txt', sep  = '\t', header = 0)
H = np.array([HXs['E'],HXs['sigma_t'],HXs['sigma_s']])
H = np.transpose(H)
###
xs38 = pd.read_csv(f'{data_dir}xs_u238_T293k.txt',sep  = '\t', header = 0)
XS38 = np.array([xs38['E'],xs38['sigma_t'],xs38['sigma_s']])
XS38 = XS38.T
# create lethatgy grid
E0 = 1e7 #this is set to ensure the first lethargy point is 0 for ease of physics comprehension
deltaU = .0001 #grid spacing
Min = np.log(E0/1e7)
Max = np.log(E0/1.0)
n = int((Max - Min)/deltaU) #creates gridpoints
gridSpace = np.linspace(Min, Max, n)
E = E0 / np.exp(gridSpace) # for setting up plotting

#set alphaU for U and O
AU = 238
alphaU = ((AU-1)/(AU+1))**2 # for U-238
NH = 5
NU = 1
AO = 16
alphaP = ((AO-1)/(AO+1))**2
###########################################################
#Linear Interpolations
chi = truncMatrix(chi)
H = truncMatrix(H)
XS38 = truncMatrix(XS38)
chi[:,0] = np.log(E0/chi[:,0])
H[:,0] = np.log(E0/H[:,0])
XS38[:,0] = np.log(E0/XS38[:,0])
#the calcs will go from min lethargy to max lethargy gain. flip the matricies
XS38 = flipCol(XS38,0)
XS38 = flipCol(XS38,1)
XS38 = flipCol(XS38,2)
H =  flipCol(H,0)
H = flipCol(H,1)
H = flipCol(H,2)
chi = flipCol(chi,0)
chi = flipCol(chi,1)
#interpolate the data
chi = interpolate_matrix_chi(chi,len(gridSpace))
H = interpolate_matrix(H,len(gridSpace))
XS38 = interpolate_matrix(XS38,len(gridSpace))

#account for number densities
H[:,1]*=NH
H[:,2]*=NH
phi = np.zeros_like(XS38[:,0])
sigma_b1 = 50 #barns
sigma_b2 = 5 #barns
