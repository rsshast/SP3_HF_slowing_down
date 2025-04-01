import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt

E0 = 1e7
def set_width(gridSpace,alphaU):
    i = 1680
    counterP = 0
    maxLethGain = gridSpace[i] - np.log(1/alphaU)
    h = gridSpace[gridSpace < maxLethGain].size - 1
    f = (gridSpace[h+1] - maxLethGain) / (gridSpace[h+1] - gridSpace[h])
    counter = i - h
    return h,f,counter

#get the data in a form I can use
def truncMatrix(matrix):
    rowMax = np.where(matrix[:, 0] > E0)[0]
    rowMin = np.where(matrix[:, 0] < 1)[0]
    newMatrix = np.delete(matrix,rowMax,axis = 0)
    newMatrix = np.delete(newMatrix,rowMin,axis = 0)
    return newMatrix

def flipCol(matrix,index):
    flipCol = np.flip(matrix[:,index])
    matrix[:,index] = flipCol
    return matrix

def interpolate_matrix(data, m):
    # Extract the first column as x values
    x = data[:, 0]
    # Interpolate the other columns based on x values
    interpolated_data = np.zeros((m, 3))
    interpolated_data[:, 0] = np.linspace(x[0], x[-1], m)  # Evenly spaced values

    for i in range(1, 3):
        interpolated_data[:, i] = np.interp(interpolated_data[:, 0], x, data[:, i])

    return interpolated_data

def interpolate_matrix_chi(data, m):
    # Extract the first column as x values
    x = data[:, 0]

    # Interpolate the second column based on x values
    interpolated_data = np.zeros((m, 2))
    interpolated_data[:, 0] = np.linspace(x[0], x[-1], m)  # Evenly spaced values
    interpolated_data[:, 1] = np.interp(interpolated_data[:, 0], x, data[:, 1])

    return interpolated_data

def calc_phi(gridSpace,H,XS38,NP,sigma_b2,deltaU,alphaU,counter,f,h,chi,alphaP,oxy):
    scatH = 0
    scat38 = 0
    scatP = 0
    i = 1680
    phi = np.zeros_like(XS38[:,0])
    st = time.time()
    #set where scattering needs to be subtracted off, and grid size
    counterP = 25030 - h #corresponds to the max leth gain bin of O
    maxLethGainO = gridSpace[i] - np.log(1/alphaP)
    fp = (gridSpace[h+1] - maxLethGainO) / (gridSpace[h+1] - gridSpace[h])
    for i in range(1, len(gridSpace)):
        #removal xs w/ perturbation
        Sigma_R = H[i,1] + XS38[i,1] +NP*sigma_b2 - deltaU*(XS38[i,2]/(1-alphaU)) - deltaU*H[i,2] - \
            deltaU*NP*sigma_b2
        #Scattering Source for Hydrogen
        scatH += H[i-1,2] * phi[i-1] * np.exp(gridSpace[i-1]) * deltaU
        #Scattering Source for Uranium
        scat38 += XS38[i-1,2]/(1-alphaU) * phi[i-1] * np.exp(gridSpace[i-1]) * deltaU
        #subtract off the contribution from lesser lethargy bins
        if i >= counter:
            h = i - counter
            scat38 -= XS38[h,2]/(1-alphaU) * (1-f) * phi[h] * np.exp(gridSpace[h]) * deltaU
            if h > 0:
                scat38 -= XS38[h-1,2]/(1-alphaU) * f * phi[h-1] * np.exp(gridSpace[h-1]) * deltaU
        scatP += NP*sigma_b2*phi[i-1]*np.exp(gridSpace[i-1])*deltaU
        #scattering source for Oxygen (hydrogen like)
        if oxy == True:
            if i >= counterP:
                hp = i - counterP
                scatP -= NP*sigma_b2/(1-alphaP) * (1-fp) * phi[hp]*np.exp(gridSpace[hp]) * deltaU
                if h > 0:
                    scatP -= NP*sigma_b2/(1-alphaP) * fp * phi[hp-1] *  np.exp(gridSpace[hp-1])*deltaU

        #append to the array
        phi[i] = (float(chi[i,1]) + np.exp(-1*gridSpace[i])*(scatH + scat38 + scatP)) / Sigma_R
    et = time.time()
    print(f"time to calculate the slowing down spectrum: {et-st:.4f} seconds")
    return phi

def self_shielding_tables(H,XS38,gridSpace,alphaU,Ewims,NH,sigma_p_H,deltaU,chi,E,alphaP):
    st = time.time()
    #data remanipulation
    sigma_a_H = H[:,1] - H[:,2]
    sigma_s_H = H[:,2]
    sigma_a_U = XS38[:,1]- XS38[:,2]
    sigma_s_U = XS38[:,2]

    #defining bg xs's, and getting data from question 1
    sigma_b = np.array([1,10,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9,1e10])

    i=1000
    #set where scattering needs to be subtracted off, and grid size
    maxLethGain = gridSpace[i] - np.log(1/alphaU)
    h = gridSpace[gridSpace < maxLethGain].size - 1
    f = (gridSpace[h+1] - maxLethGain) / (gridSpace[h+1] - gridSpace[h])
    counter = i - h
    #set table definitions
    abs_table = np.zeros((len(Ewims),len(sigma_b)))
    scat_table = np.zeros((len(Ewims),len(sigma_b)))
    for i in range(len(sigma_b)):
        abs_table[0,i] = 10**i
        scat_table[0,i] = 10**i
    for k in range(len(sigma_b)):
        #redefine the macroscopic xs
        H[:,1]/=NH #resets the H xs array to original data
        H[:,2]/=NH
        NH = sigma_b[k]/sigma_p_H #num density for each bg xs
        H *= NH #multiply the xs's by this new number density.
        #recalculate phi, like done in question 1
        phi=np.zeros_like(XS38[:,0])
        oxy=False
        sigma_b2 = 0
        phi = (gridSpace,H,XS38,NH,sigma_b2,deltaU,alphaU,counter,f,h,chi,alphaP,oxy)
#        phi = calc_phi(gridSpace,H,XS38,deltaU,alphaU,phi,counter,f,chi)
        #sanity check
        if NH == 5:
            plt.plot(E,phi)
            plt.xscale('log')
            plt.xlabel('Energy (eV)')
            plt.ylabel('scalar flux')
            plt.title(f'Flux check, NH = {NH}')
            plt.grid(True,which ='both')
            plt.savefig(f'charts/flux_check_q2_{NH}.png')
            plt.close()
        #obtain group xs's given fluxes
        for j in range(1,len(Ewims)):
            #find bins between each energy in the wims file
            check = (Ewims[j] <= E) & (Ewims[j-1]>=E)
            #set group spacing and fluxes
            group = gridSpace[check]
            phi_g = phi[check]
            sigma_a_g = np.zeros(len(group))
            sigma_s_g = np.zeros_like(sigma_a_g)
            sigA_Temp = sigma_a_U[check]
            sigS_Temp = sigma_s_U[check]
            #calculate numerator of the flux. Easier to do this than to do it all at once
            for index in range(len(group)):
                sigma_a_g[index] = sigA_Temp[index]*phi_g[index]
                sigma_s_g[index] = sigS_Temp[index]*phi_g[index]
            #use the group xs definition using integrals
            abs_table[j,k]=((np.trapz(sigma_a_g,group)/np.trapz(phi_g,group)))
            scat_table[j,k]=((np.trapz(sigma_s_g,group)/np.trapz(phi_g,group)))

    #output tables to .csv
    scatTable = pd.DataFrame(scat_table)
    scatTable.to_csv("data/scat_table.csv")
    absTable = pd.DataFrame(abs_table)
    absTable.to_csv("data/abs_table.csv")
    et = time.time()
    diffTime2 =  et-st
    print(f'Time to Construct Resonance Table: {diffTime2:4f}')

def q3plot(E,p1,p2,p3):
    plt.plot(E,p1,label = 'phi1')
    plt.xscale('log')
    plt.plot(E,p2,label = 'phi2')
    plt.plot(E,p3,label = 'phi3')
    plt.xscale('log')
    plt.grid(True,which = 'both')
    plt.legend()
    plt.xlabel('energy (eV)')
    plt.ylabel('scalar flux')
    plt.title('sanity check for flux values')
    plt.savefig('charts/flux_check_3.png')
    plt.close()

