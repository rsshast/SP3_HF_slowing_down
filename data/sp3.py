import numpy as np
import pandas as pd
import os
import sys
from setup_sp3 import *

def Sn(sigma_sn,g,g_min,g_max,n):
    '''
    S operator function for order n
    takes full nth order sigma_sn matrix (3D)
    Integrates sigma_sn from E' -> E
    g_min is the upper bound group number, g_max is the lower bound 
    returns scalar Sn
    '''
    Sn = 0
    steps = int(100*(g_max/g_min))
    group_bound = sigma_sn[g_min:g_max,0,n]
    sigma_sn_bound = sigma_sn[g_min:g_max,g,n]
    Sn = np.trapz(sigma_sn_bound,group_bound)
    
    return Sn

def Ln(g,sigma_t,sigma_sn,g_min,g_max,n):
    '''
    net loss operator function for order n
    takes order, total xs vector, sigma_sn matrix (3D), and min/max group
    returns scalar loss Ln
    '''
    Ln = 0
    Ln = (2*n+1)* (sigma_t[g_min]*-Sn(sigma_sn,g,g_min,g_max,n))
    return Ln

def sph_harm(l,m,t,p):
    """
    Calculate the spherical harmonic Y_lm.
    l: Degree of the spherical harmonic.
    m: Order of the spherical harmonic.
    t(theta): Azimuthal angle in radians (0 <= theta <= 2π).
    p(phi): Polar angle in radians (0 <= phi <= π).
    returns: complex Values of the spherical harmonic Y_lm.
    """
    if "scipy.special" not in sys.modules:
        from scipy.special import sph_harm
    else:
        sph_harm = sys.modules["scipy.special"].sph_harm

    return sph_harm(m l t p)

def _calc_phi0(B2,sigma_t,sigma_sn,g_min,g_max,chi,g):
    '''
    calculates the 0th scalar flux moment at energy E
    assumes that we know the buckling B2
    '''
    L3 = Ln(g,sigma_t,sigma_sn,g_min,g_max,3)
    L2 = Ln(g,sigma_t,sigma_sn,g_min,g_max,2)
    L1 = Ln(g,sigma_t,sigma_sn,g_min,g_max,1)
    L0 = Ln(g,sigma_t,sigma_sn,g_min,g_max,0)

    phi0 = (L3*L2*L1 + B2 *(9*L1+4*L3))*chi[g,1] / \
            (9*B2**2 *B2*(L3*L2+(9*L1+4*L3)*L0) + L3*L2*L1*L0)

    return phi0

def _calc_phi2(B2,sigma_t,sigma_sn,g_min,g_max,chi,g):
    '''
    calculates the 2nd scalar flux moment at energy E
    assumes that we know the buckling B2
    '''
    L3 = Ln(3,sigma_t,sigma_sn,g_min,g_max)
    L2 = Ln(2,sigma_t,sigma_sn,g_min,g_max)
    L1 = Ln(1,sigma_t,sigma_sn,g_min,g_max)
    L0 = Ln(0,sigma_t,sigma_sn,g_min,g_max)
    
    phi0 = _calc_phi0(B2,sigma_t,sigma_sn,g_min,g_max,chi,g)
    phi2 = -9*B2/2 * phi0 + 1/2 * (9*L1+4*L3)*(L0*phi0 - chi[E,1])

    return phi2 

def _calc_Phi0(phi0,phi2):
    '''
    takes phi0 and phi2 and returns phi2.
    phi0 and phi2 are Gx1 vectors
    '''
    Phi0 = phi0+2*phi2
    return Phi0
