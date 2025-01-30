import numpy as np
import matplotlib.pyplot as plt
import scipy as sp 
import math as math
import pandas as pd
import time
import os
from hw2_setup import *
from functions_hw2 import *
from plotting_hw2 import *

#initializations
NP = 0
#set where scattering needs to be subtracted off, and grid size
h,f,counter = set_width(gridSpace,alphaU)
# calculate the slowing down spectrum
phi = calc_phi(gridSpace,H,XS38,NP,sigma_b2,deltaU,alphaU,counter,f,h,chi,alphaP,oxy=False)
if (q1plot == True): plots(E,phi)
print('end q1')
#df = pd.DataFrame({'E':E,'phi':phi})
#df.to_csv('hw2q1_phivsE.csv')
#
#construct self shielding tables and save to csv
#self_shielding_tables(H,XS38,gridSpace,alphaU,Ewims,NH,sigma_p_H,deltaU,chi,E)
print('end q2')

######################################################
'''
Question 3
IR Tables for Oxygen 16 and U238
a)Tabulate IR Params (lamdas)
b) generate group xs's
c)repeat the calc for i) WRA and ii)Carlvik
'''
'''
additional notes from OO
calculate the flux for the 3 cases, calculate lambda for each case. Use what you did in 1 to 
    calculate the xs at each wims group, then use lambda. 
USE A FINE GRID! 1e-5 should be a good number.
WRA and 2 term change the escape xs, which reduces the background xs, and it has an effect
    on the xs's
Don't factor the first element (-1) into the average when estimating dancoff factor. 
'''
#initializations
#Lambda_U = np.zeros(len(Gwims)) #length of the group structure
#Lambda_O = np.zeros(len(Gwims)) 

#perform 3+ flux calculations. 
#first flux calc, this is where NH = background/pot scattering.
#redefine the macroscopic xs
H[:,1] = H[:,1]/NH
H[:,2] = H[:,2]/NH
NH = sigma_b1/sigma_p_H
H[:,1] = NH*H[:,1]
H[:,2] = NH*H[:,2]

#recalculate phi, like done in question 1
NP = 0
h,f,counter = set_width(gridSpace,alphaU)
phi1 = calc_phi(gridSpace,H,XS38,NP,sigma_b2,deltaU,alphaU,counter,f,h,chi,alphaP,oxy=False)

#second flux calculation
#same moderator number density, add in a pseudo xs for oxygen, call it's number density NP
NP = sigma_b2/sigma_p_H
h,f,counter = set_width(gridSpace,alphaP)
phi2 = calc_phi(gridSpace,H,XS38,NP,sigma_b2,deltaU,alphaU,counter,f,h,chi,alphaP,oxy=False)

#exactly the same as this calculation, but account alpha for oxygen
h,f,counter = set_width(gridSpace,alphaU)
phi3 = calc_phi(gridSpace,H,XS38,NP,sigma_b2,deltaU,alphaU,counter,f,h,chi,alphaP,oxy=True)

#quick sanity check
plt.plot(E,phi1,label = 'phi1')
plt.xscale('log')
plt.plot(E,phi2,label = 'phi2')
plt.plot(E,phi3,label = 'phi3')
plt.xscale('log')
plt.grid(True,which = 'both')
plt.legend()
plt.xlabel('energy (eV)')
plt.ylabel('scalar flux')
plt.title('sanity check for flux values')
plt.savefig('charts/flux_check_3.png')
plt.close()

assert 0 == 1
#start IR parameter calculations for Oxygen
for i in range(1,len(Lambda_O)):
    #set group width
    check = (Ewims[i] <= E) & (Ewims[i-1]>=E)
    group = gridSpace[check]
    #initialize flux arrays
    xs_phi1 = np.zeros(len(group))
    xs_phi2 = np.zeros(len(group))
    xs_phi3 = np.zeros(len(group))
    phi1Temp = phi1[check]
    phi2Temp = phi2[check]
    phi3Temp = phi3[check]
    absXs = XS38[:,1] - XS38[:,2]
    sigA1 = absXs[check]
    #build numerator for trapz integration
    for j in range(len(group)):
        #calculate numerator of the xs's
        xs_phi1[j] = sigA1[j]*phi1Temp[j]
        xs_phi2[j] = sigA1[j]*phi2Temp[j]
        xs_phi3[j] = sigA1[j]*phi3Temp[j]
    # using the formula for lambda in the book, 1-196
    xs1 = np.trapz(xs_phi1,group)/np.trapz(phi1Temp,group)
    xs2 = np.trapz(xs_phi2,group)/np.trapz(phi2Temp,group)
    xs3 = np.trapz(xs_phi3,group)/np.trapz(phi3Temp,group)
    Lambda_O[i] = (xs3-xs1)/(xs2-xs1) 
Lambda_O[Lambda_O>1] = 1
Lambda_O[0] = 1
lambda_O_df = pd.DataFrame(Lambda_O)
lambda_O_df.to_csv('lambda_Ox.csv')    
#third flux calc, this time for Uranium, replace alphaP's with alphaU's
#exactly the same as the 2nd calculation, but account alpha for uranium
scatH = 0
scat38 = 0
scatP = 0
i=1680
#set where scattering needs to be subtracted off, and grid size
h = gridSpace[gridSpace < maxLethGain].size - 1
f = (gridSpace[h+1] - maxLethGainO) / (gridSpace[h+1] - gridSpace[h])
hp = gridSpace[gridSpace < maxLethGainO].size - 1
fp = (gridSpace[h+1] - maxLethGainO) / (gridSpace[h+1] - gridSpace[h])
counterP = i-h
counter = i - h
phi3 = np.zeros_like(XS38[:,0])
for i in range(1, len(gridSpace)):
    #removal xs w/ perturbation
    Sigma_R = H[i,1] + XS38[i,1] +NP*sigma_b2 - deltaU*(XS38[i,2]/(1-alphaU)) - deltaU*H[i,2] - \
        deltaU*NP*sigma_b2/(1-alphaU)
    #Scattering Source for Hydrogen
    scatH += H[i-1,2] * phi3[i-1] * np.exp(gridSpace[i-1]) * deltaU
    #Scattering Source for Uranium
    scat38 += XS38[i-1,2]/(1-alphaU) * phi3[i-1] * np.exp(gridSpace[i-1]) * deltaU
    #subtract off the contribution from lesser lethargy bins
    if i >= counter:
        h = i - counter
        scat38 -= XS38[h,2]/(1-alphaU) * (1-f) * phi3[h] * np.exp(gridSpace[h]) * deltaU
        if h > 0:
            scat38 -= XS38[h-1,2]/(1-alphaU) * f * phi3[h-1] * np.exp(gridSpace[h-1]) * deltaU
    #scattering source for Oxygen (hydrogen like)
    scatP += NP*sigma_b2/(1-alphaU)*phi3[i-1]*np.exp(gridSpace[i-1])*deltaU
    if i >= counter:
        hp = i - counter
        scatP -= NP*sigma_b2/(1-alphaU) * (1-fp) * phi3[hp]*np.exp(gridSpace[hp]) * deltaU
        if hp > 0:
            scatP -= NP*sigma_b2/(1-alphaU) * fp * phi3[hp-1] *  np.exp(gridSpace[hp-1])*deltaU
    #append to the array
    phi3[i] = (float(chi[i,1]) + np.exp(-1*gridSpace[i])*(scatH + scat38+scatP))/Sigma_R
#start IR parameter calculations for Oxygen
for i in range(1,len(Lambda_U)):
    #set group width
    check = (Ewims[i] <= E) & (Ewims[i-1]>=E)
    group = gridSpace[check]
    #initialize flux arrays
    xs_phi1 = np.zeros(len(group))
    xs_phi2 = np.zeros(len(group))
    xs_phi3 = np.zeros(len(group))
    phi1Temp = phi1[check]
    phi2Temp = phi2[check]
    phi3Temp = phi3[check]
    absXs = XS38[:,1] - XS38[:,2]
    sigA1 = absXs[check]
    #build numerator for trapz integration
    for j in range(len(group)):
        #calculate numerator of the xs's
        xs_phi1[j] = sigA1[j]*phi1Temp[j]
        xs_phi2[j] = sigA1[j]*phi2Temp[j]
        xs_phi3[j] = sigA1[j]*phi3Temp[j]
    xs1 = np.trapz(xs_phi1,group)/np.trapz(phi1Temp,group)
    xs2 = np.trapz(xs_phi2,group)/np.trapz(phi2Temp,group)
    xs3 = np.trapz(xs_phi3,group)/np.trapz(phi3Temp,group)
    Lambda_U[i] = (xs3-xs1)/(xs2-xs1) 
Lambda_U[Lambda_U>1] = 1
Lambda_U[0] = 1
lambda_U_df = pd.DataFrame(Lambda_U)
lambda_U_df.to_csv('lambda_U.csv')   
#############################################################
#part b), construct xs tables with given Lambdas
#############################################################
#generate group xs tables. 
#this should return two columns, with the rows being the group, and the value being the corrected xs at each group
#start w/ abs
sigma_b_interp = np.zeros(len(Gwims))
#fill a interpolated array with the corrected xs's. 
NP = sigma_b2/sigma_p_H
for i in range(len(Gwims)):
    sigma_b_interp[i] = NH*sigma_p_H + NP*Lambda_O[i]*sigma_p_O
abs_table_1 = abs_table[1:] #removes header from abs table
interpXsAbs = np.zeros(len(Gwims)-1)
#apply eqn 1-200 
#requires the performance of a search
for i in range(len(Ewims)-1):
    j =np.searchsorted(sigma_b,sigma_b_interp[i])
    #interpolated xs formula from the book
    interpXsAbs[i] = abs_table_1[i,j-1] + (np.log(sigma_b_interp[i])-np.log(sigma_b[j-1]))\
            /(np.log(sigma_b[j])-np.log(sigma_b[j-1]))*(abs_table_1[i,j]-abs_table_1[i,j-1])
#now for scat
scat_table_1 = scat_table[1:]
interpXsScat = np.zeros(len(Gwims)-1)
#apply eqn 1-200for scattering
for i in range(len(Ewims)-1):
    j =np.searchsorted(sigma_b,sigma_b_interp[i])
    if j == 0:
        print("fuck")
    else: #interpolated xs formula from the book
        interpXsScat[i] = scat_table_1[i,j-1] + (np.log(sigma_b_interp[i])-np.log(sigma_b[j-1]))\
                /(np.log(sigma_b[j])-np.log(sigma_b[j-1]))*(scat_table_1[i,j]-scat_table_1[i,j-1])
q3b_inf_hom_mix_xs = pd.DataFrame({'AbsXS':interpXsAbs,'ScatXS':interpXsScat})
q3b_inf_hom_mix_xs.to_csv('q3b_infHomMixXs.csv',index =  False)

######################################################
#part c, WRA and Carlvik Approx's
######################################################
#
#geometry initializations
AO = 16
AU = 238
density_UO2 = 10.4 #g/cm^3
density_H2O = .74
numAvo = .6022
fuel_radius = .6 #cm
pitch_to_diam = 1.3
Vf = np.pi * fuel_radius**2
Sf = 2*np.pi*fuel_radius
sbar_fuel = 4*Vf/Sf #mean chord length
#define esc xs, sigma_eq = sigma_esc
NF = density_UO2*numAvo/(AO*2+AU)
NM = density_H2O*numAvo/(2+AO)
sigma_esc = 1/(NF*sbar_fuel)
########################WRA############################
#use eqn 2-275, sigmab = sum(lambda_j)(NJ/NF)sigma_p^J + sigma_esc
#same as part b with added esc xs
sigma_b_wra = np.zeros(len(Gwims))
interpXsAbsWRA = np.zeros(len(Gwims)-1)
interpXsScatWRA = np.zeros(len(Gwims)-1)
for i in range(len(Gwims)):
    sigma_b_wra[i] = NH*sigma_p_H + NP*Lambda_O[i]*sigma_p_O + sigma_esc
#requires the performance of a search
for i in range(len(Ewims)-1):
    j =np.searchsorted(sigma_b,sigma_b_wra[i])
    interpXsAbsWRA[i] = abs_table_1[i,j-1] + (np.log(sigma_b_wra[i])-np.log(sigma_b[j-1]))\
            /(np.log(sigma_b[j])-np.log(sigma_b[j-1]))*(abs_table_1[i,j]-abs_table_1[i,j-1])
#now for scat
scat_table_1 = scat_table[1:]
#apply eqn 1-200for scattering
for i in range(len(Ewims)-1):
    j =np.searchsorted(sigma_b,sigma_b_wra[i])
    interpXsScatWRA[i] = scat_table_1[i,j-1] + (np.log(sigma_b_wra[i])-np.log(sigma_b[j-1]))\
        /(np.log(sigma_b[j])-np.log(sigma_b[j-1]))*(scat_table_1[i,j]-scat_table_1[i,j-1])
wra_fp_xs = pd.DataFrame({'AbsXS':interpXsAbsWRA,'ScatXS':interpXsScatWRA})
wra_fp_xs.to_csv('wra__fp_Xs.csv',index =  False)
#reminder that the WRA systematically underestimates pFM for a system. this means the flux is 
    #underestimated, and the cross sections are over estimated
########################################################
##################Carlvik###############################
########################################################
#use eqns 1-299 -> 1-300
#initializations
phi_g_1_carlv = np.zeros(len(Gwims)-1)
phi_g_2_carlv = np.zeros(len(Gwims)-1)
interpXsAbsCarlv = np.zeros(len(Gwims)-1)
interpXsScatCarlv = np.zeros(len(Gwims)-1)
sigma_b_carlv1 = np.zeros(len(Gwims)-1)
sigma_b_carlv2 = np.zeros(len(Gwims)-1)
interpXsAbscarlv1 = np.zeros(len(Gwims)-1)
interpXsAbscarlv2 = np.zeros(len(Gwims)-1)
interpXsScatcarlv1 = np.zeros(len(Gwims)-1)
interpXsScatcarlv2 = np.zeros(len(Gwims)-1)
## find the fuel xs's at the group energies. this may need to be adjusted for oxygen/Hydrogen
##absorption
for i in range(len(Gwims)-1): #equations 1.293a/b
    sigma_b_carlv1[i] = NH*sigma_p_H + NP*Lambda_O[i]*sigma_p_O + 2 * sigma_esc
    sigma_b_carlv2[i] = NH*sigma_p_H + NP*Lambda_O[i]*sigma_p_O + 3 * sigma_esc
for i in range(len(Ewims)-1): #interpolated xs's absorption
    j =np.searchsorted(sigma_b,sigma_b_carlv1[i])
    interpXsAbscarlv1[i] = abs_table_1[i,j-1] + (np.log(sigma_b_carlv1[i])-np.log(sigma_b[j-1]))\
            /(np.log(sigma_b[j])-np.log(sigma_b[j-1]))*(abs_table_1[i,j]-abs_table_1[i,j-1])
    j =np.searchsorted(sigma_b,sigma_b_carlv2[i])
    interpXsAbscarlv2[i] = abs_table_1[i,j-1] + (np.log(sigma_b_carlv2[i])-np.log(sigma_b[j-1]))\
            /(np.log(sigma_b[j])-np.log(sigma_b[j-1]))*(abs_table_1[i,j]-abs_table_1[i,j-1])
#partial fluxes
for i in range(len(Gwims)-1):#fluxes absorption
    phi_g_1_carlv[i] = deltaU*(1-interpXsAbscarlv1[i]/\
                        Lambda_U[i]*sigma_p_U + Lambda_O[i]*NP*sigma_p_O + NH*sigma_p_H + \
                         2*sigma_esc+interpXsAbscarlv1[i])
    phi_g_2_carlv[i] = deltaU*(1-interpXsAbscarlv2[i]/\
                        Lambda_U[i]*sigma_p_U + Lambda_O[i]*NP*sigma_p_O + NH*sigma_p_H + \
                         3*sigma_esc+interpXsAbscarlv2[i])
    interpXsAbsCarlv[i] = (2*interpXsAbscarlv1[i]*phi_g_1_carlv[i] - interpXsAbscarlv2[i]*phi_g_2_carlv[i])\
        /(2*phi_g_1_carlv[i] - phi_g_2_carlv[i])
#repeat for scatter#############################################################
phi_g_1_carlv = np.zeros(len(Gwims)-1)
phi_g_2_carlv = np.zeros(len(Gwims)-1)
for i in range(len(Ewims)-1):#interpolated xs's absorption
    j =np.searchsorted(sigma_b,sigma_b_carlv1[i])
    interpXsScatcarlv1[i] = scat_table_1[i,j-1] + (np.log(sigma_b_carlv1[i])-np.log(sigma_b[j-1]))\
            /(np.log(sigma_b[j])-np.log(sigma_b[j-1]))*(scat_table_1[i,j]-scat_table_1[i,j-1])
    j =np.searchsorted(sigma_b,sigma_b_carlv2[i])
    interpXsScatcarlv2[i] = scat_table_1[i,j-1] + (np.log(sigma_b_carlv2[i])-np.log(sigma_b[j-1]))\
            /(np.log(sigma_b[j])-np.log(sigma_b[j-1]))*(scat_table_1[i,j]-scat_table_1[i,j-1])
#partial fluxes
for i in range(len(Gwims)-1): #partial fluxes
    phi_g_1_carlv[i] = deltaU*(1-interpXsScatcarlv1[i]/\
                        Lambda_U[i]*sigma_p_U + Lambda_O[i]*NP*sigma_p_O + NH*sigma_p_H + \
                         2*sigma_esc+interpXsScatcarlv1[i])
    phi_g_2_carlv[i] = deltaU*(1-interpXsScatcarlv2[i]/\
                        Lambda_U[i]*sigma_p_U + Lambda_O[i]*NP*sigma_p_O + NH*sigma_p_H + \
                         3*sigma_esc+interpXsScatcarlv2[i])
    interpXsScatCarlv[i] = (2*interpXsScatcarlv1[i]*phi_g_1_carlv[i] - interpXsScatcarlv2[i]*phi_g_2_carlv[i])\
        /(2*phi_g_1_carlv[i] - phi_g_2_carlv[i])
carlv_fp_xs = pd.DataFrame({'AbsXS':interpXsAbsCarlv,'ScatXS':interpXsScatCarlv})
carlv_fp_xs.to_csv('carlv_fp_Xs.csv',index =  False)
#end timer
end_Time3 = time.time()
diffTime3 = end_Time3 - start_time3
print(f'Time Q3 = {diffTime3}')

######################################################
###########End Question 3 ############################
######################################################
'''
Question 4
Set a grid on cylinder, if discriminent is positive, then you hit the fuel pin, it contributes to the
    Dancoff Factor. Accellerates the code. 
#constants from the assignment, and declare random numbers
start_time_4 = time.time()
#geometry initializations
fuel_radius = .6 #cm
pitch_to_diam = 1.3
#find the xs for water in cm^-1
AO = 16
density_H2O = .74 #g/cm^3
numAvo = .6022 #avogadro's number
sigma_p_H = 20 #barns
sigma_p_O = 4 #barns
#define esc xs, sigma_eq = sigma_esc
NM = density_H2O*numAvo/(2+AO) #number density calculated using stoichiometry
sigma_t_m = NM * (2*sigma_p_H + sigma_p_O) #moderator xs
sigma_t_m *= 2
pitch = pitch_to_diam * 2 * fuel_radius 
H_offset = pitch/2 #offset of where the fuel elements are
n = 9 #9x9 grid
dancoff_factor = np.zeros(n * n)
################################################################
histories = int(5e4) 
#must be minimum 1e6
eps = 1e-4 
#tolerance factor, push the particle over the boundary a bit
################################################################
def mc_pos_dir(): #determines position and direction of the particle
    position = np.zeros(3)
    omega = np.zeros(3)
    #random numbers
    xi1 = np.random.rand()
    xi2 = np.random.rand()
    xi3 = np.random.rand()
    #start ray tracing algorithm
    #sample the first angle, find position
    fi = 2*np.pi*xi1
    x = fuel_radius*np.cos(fi)
    y = fuel_radius*np.sin(fi)
    z = 0
    #unit normals
    nx = x/fuel_radius
    ny = y/fuel_radius
    #because of the derivation done in the notes,
    mu_n = np.sqrt(xi2)
    gamma_n = 2*np.pi*xi3
    #coordinate transform, find diretion of particle
    theta_n = np.arccos(mu_n)
    omega_x = mu_n*nx -ny*np.sin(theta_n)*np.sin(gamma_n)
    omega_y = mu_n*ny +nx*np.sin(theta_n)*np.sin(gamma_n)
    omega_z = -1*np.sin(theta_n)*np.cos(gamma_n)
    position = np.array([x,y,z])
    omega = np.array([omega_x,omega_y,omega_z])
    return position,omega
#shift within the pins
def move_right(position, pin, n, H_offset):
    position[0] = -H_offset #reset the position to the center of the fuel pin
    if pin % n == (n - 1):
        pin -= (n - 1)#move to the pin
    else:
        pin += 1
    return position, pin
def move_left(position, pin, n, H_offset):
    position[0] = H_offset
    if pin % n:
        pin -= 1
    else:
        pin += (n - 1)
    return position, pin
def move_up(position, pin, n, H_offset):
    position[1] = -H_offset
    if pin < n:
        pin += n * (n - 1)
    else:
        pin -= n
    return position, pin
def move_down(position, pin, n, H_offset):
    position[1] = H_offset
    if pin >= n * (n - 1):
        pin -= n * (n - 1)
    else:
        pin += n
    return position, pin
def chordLength(position, omega, assembly, pin, n):
    #find the chord length of a history before it dies
    sbar = 0
    inFuelPin = False
    while not inFuelPin:
        if omega[0] != 0:
            #make sure the particle isn't going directly right, and calculate the distance to the planes
            SpxL = (H_offset - position[0]) / omega[0]
            SpxR = (-H_offset - position[0]) / omega[0]
        else:
            SpxL = SpxR = np.inf
        if omega[1] != 0:
            #same thing, but to the y planes going left and right
            SpyL = (H_offset - position[1]) / omega[1]
            SpyR = (-H_offset - position[1]) / omega[1]
        else:
            SpyL = SpyR = np.inf
        surfaces = np.array([SpxL,SpxR,SpyL,SpyR,np.inf])
        if assembly[pin] == 1: # if we are in a fuel pin
        #quadratic formula parameters from the notes
            c = position[0] ** 2 + position[1] ** 2 - fuel_radius** 2 
            if -eps < c < eps:  # push the particle over the edge
                surfaces[4] = np.inf
            else:
                a = omega[0] ** 2 + omega[1] ** 2
                b = 2 * (position[0] * omega[0] + position[1] * omega[1])
                det = b * b - 4*a*c
                if det < 0:
                    surfaces[4] = np.inf  # would never reach the fuel in the current pin
                else:
                    l1 = (-b + np.sqrt(det)) / (2 * a)
                    l2 = (-b - np.sqrt(det)) / (2 * a)
                    if l1 < l2:
                        return sbar + l1
                    else:
                        return sbar + l2
        else:
            surfaces[4] = np.inf

        for i in range(len(surfaces)):
            if surfaces[i] <= 0:
                surfaces[i] = np.inf

        l = min(surfaces) # find the minimum distance to the surface
        position += l * omega #incriment position along direction omega
        sbar += l #incriment chord length
        #move to the fuel pin based on which one is closest
        if l == surfaces[0]:
            position, pin = move_right(position, pin, n, H_offset)
        elif l == surfaces[1]:
            position, pin = move_left(position, pin, n, H_offset)
        elif l == surfaces[2]:
            position, pin = move_up(position, pin, n, H_offset)
        elif l == surfaces[3]:
            position, pin = move_down(position, pin, n, H_offset)
        else: #repeat loop
            inFuelPin = True
    return sbar
# fuel assemply as a list
assembly = [
            0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 1, 1, 1, 1, 1, 1, 1,
            0, 1, 1, 1, 1, 1, 1, 1, 1,
            0, 1, 1, 1, 1, 1, 1, 1, 1,
            0, 1, 1, 1, 0, 1, 1, 1, 1,
            0, 1, 1, 1, 1, 0, 1, 1, 1,
            0, 1, 1, 1, 1, 1, 1, 1, 1,
            0, 1, 1, 1, 1, 1, 1, 1, 1,
            0, 1, 1, 1, 1, 1, 1, 1, 1,
           ]
#calculate the dancoff factor in each pin
for i in range(n * n):
    position, omega = mc_pos_dir() #read in position and direction
    if assembly[i] == 1: #if in fuel pin
        for j in range(histories): #for all histories, calculate black dancoff factor
            dancoff_factor[i] += np.exp(-sigma_t_m *\
                                 chordLength(position.copy(), omega.copy(), assembly, i, n))
            print(i)
    dancoff_factor[i] /= histories #pin cell dancoff factor
#pin averaged dancoff factor
C_black_danF = np.mean(dancoff_factor[dancoff_factor != 0])
C_black_danF *= C_black_danF
print(C_black_danF)
###############################################################
##part b#######################################################
################corrected WRA and Carlvik Approxes#############
###############################################################
C = C_black_danF #better varialbe naming
A = (1-C)/C
########################WRA####################################
sigma_b_wra = np.zeros(len(Gwims))
interpXsAbsWRA = np.zeros(len(Gwims)-1)
interpXsScatWRA = np.zeros(len(Gwims)-1)
for i in range(len(Gwims)):
    sigma_b_wra[i] = NH*sigma_p_H + NP*Lambda_O[i]*sigma_p_O + sigma_esc*(1-C)
#requires the performance of a search
for i in range(len(Ewims)-1):
    j =np.searchsorted(sigma_b,sigma_b_wra[i])
    interpXsAbsWRA[i] = abs_table_1[i,j-1] + (np.log(sigma_b_wra[i])-np.log(sigma_b[j-1]))\
            /(np.log(sigma_b[j])-np.log(sigma_b[j-1]))*(abs_table_1[i,j]-abs_table_1[i,j-1])
#now for scat
scat_table_1 = scat_table[1:]
#apply eqn 1-200for scattering
for i in range(len(Ewims)-1):
    j =np.searchsorted(sigma_b,sigma_b_wra[i])
    interpXsScatWRA[i] = scat_table_1[i,j-1] + (np.log(sigma_b_wra[i])-np.log(sigma_b[j-1]))\
        /(np.log(sigma_b[j])-np.log(sigma_b[j-1]))*(scat_table_1[i,j]-scat_table_1[i,j-1])
wra_fp_dancoff_xs = pd.DataFrame({'AbsXS':interpXsAbsWRA,'ScatXS':interpXsScatWRA})
wra_fp_dancoff_xs.to_csv('wra_fp_dancoff_Xs.csv',index =  False)

##################Carlvik######################################
#get alpha and betas from the lecture notes
alpha1 = (5*A+6-np.sqrt(A*A+36*A+36))/(2*(A+1)) 
alpha2 = (5*A+6+np.sqrt(A*A+36*A+36))/(2*(A+1))
beta1 = ((4*A+6)/(A+1)-alpha1)/(alpha2-alpha1)
beta2 = 1-beta1
#initializations
phi_g_1_carlv = np.zeros(len(Gwims)-1)
phi_g_2_carlv = np.zeros(len(Gwims)-1)
interpXsAbsCarlv = np.zeros(len(Gwims)-1)
interpXsScatCarlv = np.zeros(len(Gwims)-1)
sigma_b_carlv1 = np.zeros(len(Gwims)-1)
sigma_b_carlv2 = np.zeros(len(Gwims)-1)
interpXsAbscarlv1 = np.zeros(len(Gwims)-1)
interpXsAbscarlv2 = np.zeros(len(Gwims)-1)
interpXsScatcarlv1 = np.zeros(len(Gwims)-1)
interpXsScatcarlv2 = np.zeros(len(Gwims)-1)
## find the fuel xs's at the group energies. this may need to be adjusted for oxygen/Hydrogen
##absorption
for i in range(len(Gwims)-1): #equations 1.293a/b
    sigma_b_carlv1[i] = NH*sigma_p_H + NP*Lambda_O[i]*sigma_p_O + alpha1 * sigma_esc
    sigma_b_carlv2[i] = NH*sigma_p_H + NP*Lambda_O[i]*sigma_p_O + alpha2 * sigma_esc
for i in range(len(Ewims)-1):
    j =np.searchsorted(sigma_b,sigma_b_carlv1[i])
    interpXsAbscarlv1[i] = abs_table_1[i,j-1] + (np.log(sigma_b_carlv1[i])-np.log(sigma_b[j-1]))\
            /(np.log(sigma_b[j])-np.log(sigma_b[j-1]))*(abs_table_1[i,j]-abs_table_1[i,j-1])
    j =np.searchsorted(sigma_b,sigma_b_carlv2[i])
    interpXsAbscarlv2[i] = abs_table_1[i,j-1] + (np.log(sigma_b_carlv2[i])-np.log(sigma_b[j-1]))\
            /(np.log(sigma_b[j])-np.log(sigma_b[j-1]))*(abs_table_1[i,j]-abs_table_1[i,j-1])
#partial fluxes
for i in range(len(Gwims)-1):
    phi_g_1_carlv[i] = deltaU*(1-interpXsAbscarlv1[i]/\
                        Lambda_U[i]*sigma_p_U + Lambda_O[i]*NP*sigma_p_O + NH*sigma_p_H + \
                         alpha2*sigma_esc+interpXsAbscarlv1[i])
    phi_g_2_carlv[i] = deltaU*(1-interpXsAbscarlv2[i]/\
                        Lambda_U[i]*sigma_p_U + Lambda_O[i]*NP*sigma_p_O + NH*sigma_p_H + \
                         alpha2*sigma_esc+interpXsAbscarlv2[i])
    interpXsAbsCarlv[i] = (beta1*interpXsAbscarlv1[i]*phi_g_1_carlv[i] + beta2*interpXsAbscarlv2[i]*phi_g_2_carlv[i])\
        /(beta1*phi_g_1_carlv[i] + beta2*phi_g_2_carlv[i])
#repeat for scatter#############################################################
phi_g_1_carlv = np.zeros(len(Gwims)-1)
phi_g_2_carlv = np.zeros(len(Gwims)-1)
for i in range(len(Ewims)-1):
    j =np.searchsorted(sigma_b,sigma_b_carlv1[i])
    interpXsScatcarlv1[i] = scat_table_1[i,j-1] + (np.log(sigma_b_carlv1[i])-np.log(sigma_b[j-1]))\
            /(np.log(sigma_b[j])-np.log(sigma_b[j-1]))*(scat_table_1[i,j]-scat_table_1[i,j-1])
    j =np.searchsorted(sigma_b,sigma_b_carlv2[i])
    interpXsScatcarlv2[i] = scat_table_1[i,j-1] + (np.log(sigma_b_carlv2[i])-np.log(sigma_b[j-1]))\
            /(np.log(sigma_b[j])-np.log(sigma_b[j-1]))*(scat_table_1[i,j]-scat_table_1[i,j-1])
#partial fluxes
for i in range(len(Gwims)-1):
    phi_g_1_carlv[i] = deltaU*(1-interpXsScatcarlv1[i]/\
                        Lambda_U[i]*sigma_p_U + Lambda_O[i]*NP*sigma_p_O + NH*sigma_p_H + \
                         alpha1*sigma_esc+interpXsScatcarlv1[i])
    phi_g_2_carlv[i] = deltaU*(1-interpXsScatcarlv2[i]/\
                        Lambda_U[i]*sigma_p_U + Lambda_O[i]*NP*sigma_p_O + NH*sigma_p_H + \
                         alpha2*sigma_esc+interpXsScatcarlv2[i])
    interpXsScatCarlv[i] = (beta1*interpXsScatcarlv1[i]*phi_g_1_carlv[i] + beta2*interpXsScatcarlv2[i]*phi_g_2_carlv[i])\
        /(beta1*phi_g_1_carlv[i] + beta2* phi_g_2_carlv[i])
carlv_fp_dancoff_xs = pd.DataFrame({'AbsXS':interpXsAbsCarlv,'ScatXS':interpXsScatCarlv})
carlv_fp_dancoff_xs.to_csv('carlv_fp_Dancoff_Xs.csv',index =  False)

end_time_4 = time.time()
elapsed_time_4 = end_time_4 - start_time_4
print(f'Time Q4 ={elapsed_time_4}')
################################################################
##########STD DEV of Dancoff Factor#############################
################################################################
histories = int(1e5)
dancoff_factor = np.zeros(n * n)
#calculate Dancoff Factor Normally
for i in range(n * n):
    position, omega = mc_pos_dir() #read in position and direction
    if assembly[i] == 1: #if in fuel pin
        for j in range(histories): #for all histories, calculate black dancoff factor
            dancoff_factor[i] += np.exp(-sigma_t_m *\
                                 chordLength(position.copy(), omega.copy(), assembly, i, n))
            print(i)
    dancoff_factor[i] /= histories
C_hat_quant = np.mean(dancoff_factor[dancoff_factor != 0])
C_hat_quant_squared = C_hat_quant*C_hat_quant
print(C_hat_quant_squared)
#calculate with squared baked in
dancoff_factor = np.zeros(n * n)
sigma_t_m *= 2
for i in range(n * n):
    position, omega = mc_pos_dir() #read in position and direction
    if assembly[i] == 1: #if in fuel pin
        for j in range(histories): #for all histories, calculate black dancoff factor
            dancoff_factor[i] += np.exp(-sigma_t_m *\
                                 chordLength(position.copy(), omega.copy(), assembly, i, n))
            print(i)
    dancoff_factor[i] /= histories
C_hat_std = np.mean(dancoff_factor[dancoff_factor != 0])
C_hat_std_squared = C_hat_std * C_hat_std
#Compute std dev
std_dev = (1/np.sqrt(histories))*np.sqrt(C_hat_std_squared - C_hat_quant_squared)
print(std_dev)
#######################################################
##################End Question 4#######################
#######################################################
'''
