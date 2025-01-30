import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import os

class Sp3:
    def __init__(self, xs_H, xs_U, sigma_f, chi, B2):
        """
        Initialize the calculator with necessary parameters.

        Parameters:
        xs (array): Cross-section data. Column 0: group, Column 1: total xs, Column 2: scatter xs.
        sigma_f (array): Fission cross-section vector.
        chi (array): Fission spectrum matrix. Column 0: groups, Column 1: chi values.
        B2 (float): Buckling value.
        """
        self.AH        = 1 # Atomic number of hydrogen
        self.AU        = 238 # Atomic number of U-238
        self.E0        = 1e7 # maximum energy
        self.sigma_s_H = xs_H[:, 2] # Hydrogen xs's
        self.sigma_t_H = xs_H[:, 1]
        self.sigma_s_U = xs_U[:, 2] # U-238 xs's
        self.sigma_t_U = xs_U[:, 1]
        self.sigma_f   = sigma_f # fission xs's
        self.EU        = xs_U[:,0] # energy groups
        self.EH        = xs_H[:,0] # energy groups
        self.groups    = np.flip(np.log(self.E0/chi[:, 0])) # lethargy groups
        self.chi       = chi[:, 1] # fission spectrum
        self.B2        = B2 # geometric buckling
        self.tol       = 1e-12 # Small value threshold

        # Initialize other attributes
        self.phi0 = np.zeros_like(self.groups)
        self.phi2 = np.zeros_like(self.groups)
        self.Phi0 = np.zeros_like(self.groups)
        self.Phi2 = np.zeros_like(self.groups)
        self.Q = np.zeros_like(self.groups)
        self.L0 = np.zeros((self.groups.size,self.groups.size))
        self.L1 = np.zeros_like(self.L0)
        self.L2 = np.zeros_like(self.L0)
        self.L3 = np.zeros_like(self.L0)
        self.mu_s = None

    @staticmethod
    def alpha(A):
        return ((A - 1) / (A + 1)) ** 2

    def group_bound(self, A, g):
        """
        Calculate the minimum group a neutron can downscatter to.
        Do not use for hydrogen! alpha not passed

        Parameters:
        g (int): Current energy group index.
        A (int): Atomic Number:

        Returns:
        int: Minimum group index for downscattering.
        """
        return np.argmin(np.abs(self.groups - self.groups[g] * self.alpha(A)))

    def scat_order(self, A, sigma_s, l, g):
        """
        Calculate the scattering cross-sections for a given Legendre order.

        Parameters:
        sigma_s (vector): Scattering cross sections
        A (int): Atomic Number
        sigma_s (vector): 0th order scattering xs for material with atomic number A
        l (int): Legendre expansion order
        g (int): Current energy group index

        Returns:
        vector: Scattering cross-section for the specified order.
        """
        g_min = self.group_bound(A,g)
        if l == 0:
            sigma_s[g:g_min] *= 1

        elif l > 0 and l < 4:
            self.mu_s = np.zeros(g_min - g)
            for i in range(g,g_min):
                #FIXME
                self.mu_s[i-g] = ((A + 1) * np.sqrt(self.groups[g] / self.groups[i]) -
                                   (A - 1) * np.sqrt(self.groups[i] / self.groups[g])) / 2
#                if self.mu_s[i-g] > 1 or self.mu_s[i-g] < 1:
#                    raise ValueError("Mu_s not in range!")

            if l == 1: sigma_s[g:g_min] *= self.mu_s

            elif l == 2: sigma_s[g:g_min] *= (3 * self.mu_s ** 2 - 1) / 2

            elif l == 3: sigma_s[g:g_min] *= (5 * self.mu_s ** 3 - 3 * self.mu_s) / 2

        else:
            raise ValueError("Legendre order not supported (only 0-3).")

        return sigma_s

    def Ln(self, l, g):
        """
        Calculate the net loss operator Ln for a given Legendre order.

        Parameters:
        l (int): Legendre expansion order.
        g (int): Current energy group index.

        Returns:
        vector: Loss operator for the specified order.
        """
        g_min = self.group_bound(self.AU,g)
        Sn = np.zeros(self.groups.size)

        for i in range(g,g_min):
            #FIXME: many materials
#            scat_bound_H = self.scat_order(self.AH,self.sigma_s_H, l, g)[g:]
            scat_bound_U = self.scat_order(self.AU,self.sigma_s_U, l, g)[g:g_min-i]
#            Sn[i]        = np.trapz(scat_bound_H, self.groups[g:])
            Sn[i]        = np.trapz(scat_bound_U, self.groups[g:g_min-i])
#            Sn[i]       += np.trapz(scat_bound_U, self.groups[g:g_min-i])
#            if Sn[i] < self.tol: break

#        xs_t_H = self.sigma_t_H[(self.sigma_t_H)< g] = 0
        xs_t_U = self.sigma_t_U[(self.sigma_t_U < g | ( self.sigma_t_U> g_min))] = 0
            
#        return (2 * l + 1) * (xs_t_H + xs_t_U - Sn)
        return (2 * l + 1) * (xs_t_U - Sn)

    def calc_phi0(self):
        """
        Calculate the 0th scalar flux moment (phi0).

        Returns:
        np.ndarray: Updated phi0 values.
        """
        t1 = time.time()
        for i in range(self.groups.size):
            L0, L1, L2, L3 = [self.Ln(l, i) for l in range(4)]
            self.L0[:, i] = L0
            self.L1[:, i] = L1
            self.L2[:, i] = L2
            self.L3[:, i] = L3
            
            RHS = ((L3 * L2 * L1 + self.B2 * (9 * L1 + 4 * L3)) * self.chi)
            LHS = (9 * self.B2 ** 2 + self.B2 * (L3 * L2 + (9 * L1 + 4 * L3) * L0) + L3 * L2 * L1 * L0)
            phi = np.dot(RHS, LHS**(-1))
            self.phi0[i] = phi

            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1} groups. Time elapsed: {np.round(time.time() - t1, 5)}s.")
                t1 = time.time()

        return self.phi0

    def calc_phi2(self):
        '''
        calculate the 2nd scalar flux moment. Eq 28 in paper
        big matmul
        
        Returns:
        vector: phi2
        '''
        # check for NAN's
        Ln = [self.L0,self.L1,self.L2,self.L3]
        for ind, arr in enumerate(Ln):
            assert not np.isnan(arr).any(), f"Array {ind} contains NaN values"

        LHS = self.L3 @ self.L2
        print('LHS Done')
        RHS = (-9*self.B2*self.phi0 + (9*self.L1 + 4*self.L3) @ (self.L0 @ self.phi0 - self.chi))/2
        print('RHS Done')
        self.phi2 = np.linalg.pinv(LHS) @ RHS
        print('inv Done')

        return self.phi2

    def calc_Phi(self):
        """
        Combine phi0 and phi2 to calculate the scalar flux moments Phi0 and Phi2.
        """
        self.Phi2 = self.phi2
        self.Phi0 = self.phi0 + 2 * self.phi2

    def fission_source(self):
        """
        Calculates the Fission Source, Q
        Eq. 47

        Returns:
        vector: Fission Source
        """
        for g in range(self.Q.size):
            g_min = self.group_bound(self.AU,g)
            self.Q[g] = self.chi[g] 
            self.Q[g]*= np.trapz(self.sigma_f[g:g_min].flatten()*self.phi0[g:g_min].flatten(),self.groups[g:g_min])

        return self.Q

    def plot_xs_t(self):
        # plot xs's for H and U vs E
        plt.figure()
        plt.plot(self.EU,self.sigma_t_U,label=r'$\Sigma_t^U$')
        plt.plot(self.EH,self.sigma_t_H,label=r'$\Sigma_t^H$')
        plt.xlabel("E")
        plt.ylabel(r"$\Sigma_t$")
        plt.title(f"XS's for {self.EU.size} groups")
        plt.yscale("log")
        plt.xscale("log")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"results/charts/xs_t_{self.EU.size}.png")

    def plot_fluxes(self):
        # pretty self explanitory
        plt.figure()
        plt.plot(self.EU,self.phi0,label=r'$\phi_0$')
        plt.title(r'$\phi_0(E)$')
        plt.xlabel('E')
        plt.ylabel(r'$\phi_0$')
        plt.xscale('log')
        plt.grid(True)
        plt.legend()
        plt.savefig(f'results/charts/phi0.png')

        plt.figure()
        plt.plot(self.EU,self.phi2,label=r'$\phi_2$')
        plt.title(r'$\phi_2(E)$')
        plt.xlabel('E')
        plt.ylabel(r'$\phi$')
        plt.xscale('log')
        plt.grid(True)
        plt.legend()
        plt.savefig(f'results/charts/phi2.png')

    def diffusion_coef(self,A):
        """
        Calculates the diffusion coefficient for each group

        Returns:
        matrix: diffusion coef for each fine group (gxg)
        """
        D0 = np.zeros((self.phi0.size,self.phi0.size))
        D2 = np.zeros_like(D0)

        for i in range(D0.shape[0]):
            g_min = self.group_bound(A,i)
            for j in range(i,g_min):
                L1 = self.L1[j:g_min,j:g_min]**(-1)
                L3 = self.L3[j:g_min,j:g_min]**(-1)
                D0[i,j] = np.trapz((L1@self.Phi0[j:g_min]).squeeze(),self.groups[j:g_min]) / np.trapz(self.Phi0[i:g_min].flatten(),self.groups[i:g_min])
                D2[i,j] = np.trapz((L3@self.Phi0[j:g_min]).squeeze(),self.groups[j:g_min]) / np.trapz(self.Phi0[i:g_min].flatten(),self.groups[i:g_min])
                if D0[i,j] < self.tol and D2[i,j] < self.tol:
                    D0[i,j:] = 0.
                    D2[i,j:] = 0.
                    break

        Dn = [D0,D2]

        return Dn

    def sigma_update(self,A,sigma_x):
        """
        Updates xs's for moment n and group g
        corresponds to eqns 52a) and 52c)

        Parameters:
        A (int): Atomic Number
        sigma_x (vector): total or fission xs's.

        Returns:
        matrix: updated fission or total xs's for each moment (gx2)
        """
        Phi0 = self.Phi0
        Phi2 = self.Phi2

        sigma_0_g = np.zeros_like(self.sigma_f)
        sigma_2_g = np.zeros_like(self.sigma_f)

        for g in range(sigma_0_g.size):
            g_min = self.group_bound(A,g)
            sigma_0_g[g] = np.trapz(sigma_x[g:g_min].flatten()*Phi0[g:g_min].flatten(),self.groups[g:g_min])/np.trapz(Phi0[g:g_min])
            sigma_2_g[g] = np.trapz(sigma_x[g:g_min].flatten()*Phi2[g:g_min].flatten(),self.groups[g:g_min])/np.trapz(Phi2[g:g_min])

        # write updated xs's as a list, return the list
        sigma_g = np.column_stack((sigma_0_g,sigma_2_g))

        return sigma_g

    def sigma_s_update(self,A,sigma_s):
        """
        Updates gtg xs's for moment n
        ASSUMES THAT WE HAVE VECTORS FOR EACH LEGENDRE MOMENT (4 FOR THEM)

        Parameters:
        A(int): Atomic Number
        sigma_s: vector of P0 scattering xs's

        Returns:
        3D matrix: updated g to g scattering xs's for each moment l
        """
        #FIXME
        # build xs library, sigma_sn
        sigma_sn = np.zeros((self.phi0.size,self.phi0.size,4)) # only supported up to order 3!

        for i in range(sigma_s.shape[0]): # rows
            g_min = self.group_bound(A,i)
            for l in range(4):
                xs_s = self.scat_order
                for j in range(sigma_s.shape[1]): # columns
                    den = np.trapz(Phi[i:g_min],self.groups[i:g_min])
                    Eprime = np.trapz(xs_s[i:g_min]*Phi[i:g_min,None],self.groups[i:j],axis=0)
                    num = np.trapz(Eprime[j:g_min],self.groups[j:g_min])
                    sigma_sn[i,j,l] = num/den

        return sigma_sn

    def run(self):
        """
        Run the complete SP3 calculation process.
        """
        print("Starting phi0 calculation...")
        self.calc_phi0()

        print("Starting phi2 calculation...")
        self.calc_phi2()

        print("Starting Phi0 and Phi2 calculation...")
        self.calc_Phi()

        print("New cross sections and Diffusion Coefs...")
        # fission source Q
        Q = self.fission_source()

        # generate diffusion coefs
#        coefs_DU = self.diffusion_coef(self.AU)
#        coefs_DH = self.diffusion_coef(self.AH)
       
        # group fission and total xs's for moments 0 and 2
        update_xs_t_H = self.sigma_update(self.AH,self.sigma_t_H) 
        update_xs_t_U = self.sigma_update(self.AU,self.sigma_t_U) 
        update_xs_f_U = self.sigma_update(self.AU,self.sigma_f) 

        #FIXME scattering...
#        update_xs_s_H = self.sigma_s_update(self.AH,self.sigma_t_H)
#        update_xs_s_U = self.sigma_s_update(self.AU,self.sigma_t_U)
        
        print("Calculation completed.\nSaving data and plotting...")
        self.plot_fluxes()
        df = pd.DataFrame({'phi0': self.phi0, 'phi2': self.phi2, 'Phi0': self.Phi0, 'Phi2': self.Phi2})
        df.to_csv('results/csvs/fluxes.csv', index = False)
        print("Data Saved")
        
        return df

