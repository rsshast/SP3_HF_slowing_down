import numpy as np
import pandas as pd
import time
import os
import sys
from concurrent.futures import ProcessPoolExecutor


'''
What I need to check is if mu is a function of energy group and can be calculated once or has to be recalculated many times
'''

class Sp3:
    def __init__(self, A, xs, sigma_f, chi, B2):
        """
        Initialize the calculator with necessary parameters.
        
        Parameters:
        A (int): Atomic number of the isotope
        xs (array): Array of groups and xs's. col 0 is the group, col 1 is the total xs, col 2 is the scatter xs
        sigma_f (array): Fission cross-section vector.
        chi (array): Fission spectrum matrix.
        B2: buckling, solved using another method 
        phi0,phi2,Phi0,Phi2,Q: will be calculated from scattering and loss operators. 
            initialized here to be used later
        L0 - L3: loss operators for order l
        """
        self.A        = A
        self.sigma_s  = xs[:,2]
        self.sigma_t  = xs[:,1]
        self.sigma_f  = sigma_f
        self.groups   = chi[:,0]
        self.chi      = chi[:,1]
        self.B2       = B2
        self.mu_s     = None
        self.phi0     = np.zeros_like(self.groups)
        self.phi2     = np.zeros_like(self.groups)
        self.Phi0     = np.zeros_like(self.groups)
        self.Phi2     = np.zeros_like(self.groups)
        self.Q        = np.zeros_like(self.groups)
        self.L0       = np.zeros_like(self.groups)
        self.L1       = np.zeros_like(self.groups)
        self.L2       = np.zeros_like(self.groups)
        self.L3       = np.zeros_like(self.groups)
        self.alpha    = ((A-1)/(A+1))**2
        self.tol = 1e-8 # if a xs is really small, it's zero

    def group_bound(self,g):
        """
        Calculates the range of downscattering from g' to g
        Parameters:
        g (int): initial energy group

        returns g_min, the minimum group a neutron can downscatter to
        """ 
        return np.argmin(np.abs(self.groups - self.groups[g]*self.alpha)) 
        
    def mu_s_loop(self,A,g,g_min):
        mu_s = 0
        if A == 1:
            for i in range(g,g_min):
                mu_s += ((A+1)*np.sqrt(self.groups[g]/self.groups[i]))/2
        else: 
            for i in range(g,g_min):
                mu_s += ((A+1)*np.sqrt(self.groups[g]/self.groups[i]) - (A-1)*np.sqrt(self.groups[i]/self.groups[g]))/2
        return mu_s

    def scat_order(self,l,g):
        """
        Calculates new scattering xs's for order l
        Parameters:
        l (int): legendre expansion order
        g (int): group index

        Returns:
        vector: sigma_sn for order l and incoming energy g
        """
        if l == 0: sigma_sn = self.sigma_s
        if l == 1:
            g_min = self.group_bound(g)
            self.mu = self.mu_s_loop(self.A,g,g_min)
            sigma_sn = self.sigma_s*self.mu

        if l == 2: pl = (3*self.mu*self.mu - 1)/2

        if l == 3: pl = (5*self.mu**3 - 3*self.mu)/2
        
        if l > 3 or l < 0: raise ValueError("Legendre Order Not supported. Only supported to order 3")

        sigma_sn = self.sigma_s * pl 

        return sigma_sn

    def Sn(self, l, g):
        """
        Calculate the Sn operator for order n.
        
        Parameters:
        g: Current energy group.
        l (int): Order of the operator.
        
        Returns:
        float: Scalar Sn.
        """
        sigma_sn = self.scat_order(l,g)
        g_min = self.group_bound(g)
        return np.trapz(sigma_sn[g:g_min], self.groups[g:g_min])

    def Ln(self, l, g):
        """
        Calculate the net loss operator Ln for order n.
        
        Parameters:
        l (int): Order of the operator.
        g (int): Energy group
        
        Returns:
        float: Scalar Ln.
        """
        return (2 * l + 1) * (self.sigma_t[g] - self.Sn(l,g))

#    def _phi0(self):
#        """
#        Calculate the 0th scalar flux moment at energy E in parallel.
#
#        Returns:
#        numpy.ndarray: Updated self.phi0 values.
#        """
#        def compute_phi0(i):
#            """
#            Compute phi0 for a single group index.
#            """
#            # Compute L values
#            L_values = [self.Ln(l, i) for l in range(4)]
#            self.L0[i], self.L1[i], self.L2[i], self.L3[i] = L_values
#            L3, L2, L1, L0 = self.L0[i], self.L1[i], self.L2[i], self.L3[i]
#
#            # Calculate numerator and denominator
#            numerator = (L3 * L2 * L1 + self.B2 * (9 * L1 + 4 * L3)) * self.chi[i]
#            denominator = (9 * self.B2**3 * (L3 * L2 + (9 * L1 + 4 * L3) * L0) +
#                           L3 * L2 * L1 * L0)
#
#            # Update phi0 value
#            self.phi0[i] = numerator / denominator
#            return i  # Optional, just to track progress if needed
#
#        # Parallel processing using ThreadPoolExecutor
#        with concurrent.futures.ThreadPoolExecutor() as executor:
#            futures = {
#                executor.submit(compute_phi0, i): i for i in range(self.groups.size - 1)
#            }
#
#            # Track progress
#            for future in concurrent.futures.as_completed(futures):
#                i = future.result()
#                if i % 1000 == 0: print(i)
#
#        return self.phi0

    def _phi0(self):
        """
        Calculate the 0th scalar flux moment at energy E.
        
        Returns:
        float: 0th scalar flux moment.
        """
        t1 = time.time()
        for i in range(self.groups.size -1):
            L_values = [self.Ln(l, i) for l in range(4)]
            self.L0[i], self.L1[i], self.L2[i], self.L3[i] = L_values
            L3, L2, L1, L0 = self.L0[i], self.L1[i], self.L2[i], self.L3[i]
            self.phi0[i] = ((L3 * L2 * L1 + self.B2 * (9 * L1 + 4 * L3)) * self.chi[i]) / \
                         (9 * self.B2**2 + self.B2 * (L3 * L2 + (9 * L1 + 4 * L3) * L0) + L3 * L2 * L1 * L0)
            if (i + 1) % 1000 == 0: 
                print(i+1)
                t2 = time.time()
                print(f'integration time per 1000 groups = {np.round(t2-t1,5)} s')
                t1 = time.time()
        
        return self.phi0

    def _phi2(self):
        """
        Calculate the 2nd scalar flux moment at energy E.
        
        Returns:
        float: 2nd scalar flux moment.
        """
        L0, L1, L2, L3 = self.L0, self.L1, self.L2, self.L3  
        self.phi2 = (-9 * self.B2 / 2 * self.phi0) + (0.5 * (9 * L1 + 4 * L3) * (L0 * self.phi0 - self.chi))
        self.calc_Phi()

        return self.phi2
    
#    def playground(self):
#        phi_test = np.zeros_like(self.sigma_t)
#        phi_test[:] = 2
#        result = np.trapz(self.sigma_t*phi_test)/np.trapz(phi_test)
#        print(result)
#        assert 0 == 1

    def calc_Phi(self):
        """
        Combine phi0 and phi2 to calculate the scalar flux moment Phi0.
        """
        self.Phi2 = phi2
        self.Phi0 = phi0 + 2 * phi2

    def scattering_source(self):
        """
        Calculates the Scattering Source, Q

        Returns:
        float: Scattering Source
        """
        for g in range(self.Q.size):
            g_min = self.group_bound(self,g)
            self.Q[g] = self.chi[g] * np.trapz(self.sigma_f[g:g_min]*self.phi0[g:g_min],self.groups[g:g_min])

        return self.Q

    def diffusion_coef(self):
        """
        Calculates the diffusion coefficient for each group

        Returns:
        matrix: diffusion coef for each fine group (gxg)
        """
        D0 = np.zeros_like(self.phi0,self.phi0)
        D2 = np.zeros_like(D0)
        
        for i in range(D0.shape[0]):
            g_min = self.group_bound(i)
            for j in range(i,g_min):
                L1 = self.L1[j:g_min]**(-1)
                L3 = self.L3[j:g_min]**(-1)
                D0[i,j] = np.trapz(L1*self.Phi0[j:g_min]) / np.trapz(self.Phi0[i:g_min])
                D2[i,j] = np.trapz(L3*self.Phi2[j:g_min]) / np.trapz(self.Phi2[i:g_min])
                if D0[i,j] < self.tol and D2[i,j] < self.tol:
                    D0[i,j:] = 0.
                    D2[i,j:] = 0.
                    break
                    

        Dn = [D0,D2] 

        return Dn
        
    def sigma_update(self,sigma_x):
        """
        Updates xs's for moment n and group g
        corresponds to eqns 52a) and 52c)
        
        Parameters:
        sigma_x (vector): total or fission xs's. 

        Returns:
        vector: updated fission or total xs's for each moment
        """
        Phi0 = self.Phi0
        Phi2 = self.Phi2
        
        sigma_0_g = np.zeros_like(self.sigma_t)
        sigma_2_g = np.zeros_like(self.sigma_t)

        for g in range(sigma_0_g.size):
            g_min = self.group_bound(i)
            sigma_0_g[g] = np.trapz(sigma_x[g:g_min]*Phi0[g:g_min])/np.trapz(Phi0[g:g_min])
            sigma_2_g[g] = np.trapz(sigma_x[g:g_min]*Phi2[g:g_min])/np.trapz(Phi2[g:g_min])

        # write updated xs's as a list, return the list
        sigma_g = [sigma_0_g, sigma_2_g]

        return sigma_g

    def sigma_s_update(self,sigma_s):
        """
        Updates gtg xs's for moment n 
        ASSUMES THAT WE HAVE VECTORS FOR EACH LEGENDRE MOMENT (4 FOR THEM)
        
        Parameters:
        sigma_s: vector of P0 scattering xs's

        Returns:
        3D matrix: updated g to g scattering xs's for each moment l
        """
        # build xs library, sigma_sn
        sigma_sn = np.zeros(self.phi0.size,self.phi0.size,4) # only supported up to order 3!
        
        for i in range(sigma_s.shape[0]): # rows
            g_min = self.group_bound(i)
            for l in range(4):
                xs_s = self.scat_order
                for j in range(sigma_s.shape[1]): # columns
                    den = np.trapz(Phi[i:g_min],self.groups[i:g_min])
                    Eprime = np.trapz(xs_s[i:g_min]*Phi[i:g_min,None],self.groups[i:j],axis=0)
                    num = np.trapz(Eprime[j:g_min],self.groups[j:g_min])
                    sigma_sn[i,j,l] = num/den
        
        return sigma_sn
            
