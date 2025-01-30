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
gridpoints = 1000
B2 = 1
NH = 5
NU = 1
plotting = True
#######################get data ##################################
def get_data(data, m, N):
    # data for everything but fission xs's
    if data.ndim > 1: 
        data = np.delete(data,np.where(data[:,0]>E0)[0],axis = 0)
        data = np.delete(data,np.where(data[:,0]<1 )[0],axis = 0)
        for i in range(data.shape[1]): data[:,i] = np.flip(data[:,i])
        n = data.shape[0]
        return N * np.array([
            np.interp(np.linspace(0, n - 1, m), np.linspace(0, n - 1, n), data[:, i])
            for i in range(data.shape[1])
        ]).T  

    # data for fission xs's
    else: 
        data = np.delete(data,np.where(data>E0)[0])
        data = np.delete(data,np.where(data<1 )[0])
        data = np.flip(data)
        n = data.size
        return N * np.array([
            np.interp(np.linspace(0, n - 1, m), np.linspace(0, n - 1, n), data)]).T  


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
sigma_f = pd.read_csv(f'{data_dir}xs_fuel.csv', sep = ',',header=0).to_numpy()
###
chi = np.array([chi35['E'],chi35['chi']]).T
H = np.array([H1['E'],H1['sigma_t'],H1['sigma_s']]).T
XS38 = np.array([U238['E'],U238['sigma_t'],U238['sigma_s']]).T
sigma_f = sigma_f[:,4]

#############################################################
