'''
setup.py
'''
###################################################################
# basic imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
####################################################################
# user inputs
E0 = 1e7 #this is set to ensure the first lethargy point is 0 for ease of physics comprehension
gridpoints = 200
B2 = 1
NH = 5
NU = 1

#######################get data ##################################
def get_data(data, gridpoints, N):
    data = data[(data[:, 0] <= E0) & (data[:, 0] >= 1)]  
    data[:, 0] = np.log(data[:, 0])

    # Define new grid
    new_grid = np.linspace(np.log(1), np.log(E0), gridpoints)
    new_data = np.vstack([np.interp(new_grid, data[:,0], data[:, i]) for i in range(data.shape[1])]).T

    return new_data

def get_fission_data(x1,M2):
    M2 = M2[(M2[:, 0] <= E0) & (M2[:, 0] >= 1)]
    M2[:,0] = np.flip(np.log(M2[:,0]))

    return np.interp(x1, M2[:,0], M2[:,1])

#######################Constants #########################################
#set alphaU for U and O
AH = 1
AU = 238
AO = 16
alphaU = ((AU-1)/(AU+1))**2 
alphaO = ((AO-1)/(AO+1))**2
###
data_dir = '~/WN25/SP3/data/'
chi35 = pd.read_csv(f'{data_dir}chi_u235.txt', sep = '\t',header = 0)
H1 = pd.read_csv(f'{data_dir}xs_h1_T293k.txt', sep  = '\t', header = 0)
U238 = pd.read_csv(f'{data_dir}xs_u238_T293k.txt',sep  = '\t', header = 0)
sigma_F = pd.read_csv(f'{data_dir}xs_f.csv', sep = ',',header=0)
###
chi = np.array([chi35['E'],chi35['chi']]).T
H = np.array([H1['E'],H1['sigma_t'],H1['sigma_s']]).T
XS38 = np.array([U238['E'],U238['sigma_t'],U238['sigma_s']]).T
sigma_f = np.array([sigma_F['E'],sigma_F['sigma_f']]).T

#############################################################
