import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import os
import concurrent.futures

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
        self.sigma_t_H = xs_H[:, 1]
        self.sigma_s_H = xs_H[:, 2] # Hydrogen xs's
        self.sigma_t_U = xs_U[:, 1]
        self.sigma_s_U = xs_U[:, 2] # U-238 xs's
        self.sigma_f   = sigma_f # fission xs's
        self.E         = np.exp(xs_U[:,0]) # energy groups
        self.chi       = chi[:, 1] # fission spectrum
        self.groups    = xs_U[:,0] # energy groups
#        self.groups    = np.log(self.E0/self.E) # lethargy groups
        self.B2        = B2 # geometric buckling
        self.tol       = 1e-14 # Small value threshold

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

    @staticmethod
    def alpha(A):
        return ((A - 1) / (A + 1)) ** 2

    def group_bound(self, A, g):
        """
        Calculate the minimum group a neutron can downscatter to.

        Parameters:
        g (int): Current energy group index.
        A (int): Atomic Number:

        Returns:
        int: Minimum group index for downscattering.
        """
        target = self.groups[-1] if A == 1 else self.groups[g] + np.log(1 / self.alpha(A))
        return np.searchsorted(self.groups, target)

    def build_Ln(self, A, sigma_s, sigma_t):
        """
        build the Ln operators. L0, L1, L2, L3.
        start by calculating the scattering cosine for the lth moment to get the lth moment xs's
        
        Parameters:
        A (int) : atomic number
        sigma_s (vector): corresponding P0 scattering xs's
        sigma_t (vector): corresponding total xs's

        Returns:
        List of gxg matrices: L0, L1, L2, and L3 for a given material
        """
        # build gtg xs libraries
        L0 = np.zeros((self.groups.size,self.groups.size))
        L1 = np.zeros_like(L0)
        L2 = np.zeros_like(L0)
        L3 = np.zeros_like(L0)
        gridwidth = np.diff(self.groups[:-1])
        sigma_t = np.diag(sigma_t)

        # loop over all groups
        for g in range(self.groups.size):
            g_min = self.group_bound(A,g)
            print(g,g_min)

            # start with finding the scattering cosine for L0 -> L3
            sigma_s0 = sigma_s #s0
            mu_s = np.zeros(g_min - g)
            for i in range(g,g_min):
                mu_s[i-g] = (((A + 1) * np.exp((self.groups[g] - self.groups[i]) / 2)) 
                                    - ((A - 1) * np.exp((self.groups[i] - self.groups[g]) / 2))) / 2
                # protection
                if np.abs(mu_s[i-g]) > 1: raise ValueError("Mu not in range")

            sigma_s1 = sigma_s[g:g_min]*mu_s #s1
    
            sigma_s2 = sigma_s[g:g_min] * (3 * mu_s ** 2 - 1) / 2 #s2
                
            sigma_s3 = sigma_s[g:g_min] * (5 * mu_s ** 3 - 3 * mu_s) / 2 #s3

            for i in range(g,g_min): # for each incident group
                for j in range(i,g_min): # for the range it can scatter into
                    L0[j,i] = self.calc_Ln(sigma_t,sigma_s0,sigma_s1,sigma_s2,sigma_s3,i,j,gridwidth,g,l=0,tol=self.tol)
                    L1[j,i] = self.calc_Ln(sigma_t,sigma_s0,sigma_s1,sigma_s2,sigma_s3,i,j,gridwidth,g,l=1,tol=self.tol)
                    L2[j,i] = self.calc_Ln(sigma_t,sigma_s0,sigma_s1,sigma_s2,sigma_s3,i,j,gridwidth,g,l=2,tol=self.tol)
                    L3[j,i] = self.calc_Ln(sigma_t,sigma_s0,sigma_s1,sigma_s2,sigma_s3,i,j,gridwidth,g,l=3,tol=self.tol)
        #            if L0[j,i] == 0 and  L1[j,i] == 0 and L2[j,i] == 0 and L3[j,i] == 0: break
            
        # normalizations
#        sum_col_0 = L0.sum(axis=0)
#        sum_col_1 = L1.sum(axis=0)
#        sum_col_2 = L2.sum(axis=0)
#        sum_col_3 = L3.sum(axis=0)
#
#        for col in range(self.groups.size):
#            if sum_col_0[col] > 0: L0[col,:] /= sum_col_0[col]
#            if sum_col_1[col] > 0: L1[col,:] /= sum_col_1[col]
#            if sum_col_2[col] > 0: L2[col,:] /= sum_col_2[col]
#            if sum_col_3[col] > 0: L3[col,:] /= sum_col_3[col]

        # plot
        '''
        plt.figure()
        plt.imshow(L0, cmap='plasma', interpolation='none')
        plt.colorbar()  
        plt.savefig("L0.png")
        plt.figure()
        plt.imshow(L1, cmap='plasma', interpolation='none')
        plt.colorbar()  
        plt.savefig("L1.png")
        plt.figure()
        plt.imshow(L1, cmap='plasma', interpolation='none')
        plt.colorbar()  
        plt.savefig("L2.png")
        plt.figure()
        plt.imshow(L3, cmap='plasma', interpolation='none')
        plt.colorbar()  
        plt.savefig("L3.png")
        '''

        return L0,L1,L2,L3

    @staticmethod
    def calc_Ln(sigma_t,sigma_s0,sigma_s1,sigma_s2,sigma_s3,i,j,gridwidth,g,l,tol):
        """
        method to calculate the Ln for element i,j. 

        Parameters: too many to list

        Returns:
        float: the j,i element of the Ln gtg matrix. 
        """
        gw = gridwidth[j] if j < gridwidth.size else gridwidth[-1]
        if i == j:
            bound = gridwidth[i] if i < gridwidth.size else gridwidth[-1]
        elif i < gridwidth.size:
            bound = np.sum(gridwidth[i:j]) if j < gridwidth.size else np.sum(gridwidth[i:-1])
        else: 
            bound = gridwidth[-1]

#        return max(0,(2 * l + 1) * (sigma_t[g,g] - (sigma_s0[j-g] * (bound / gw)
#                    + sigma_s1[j-g] * (1/3)* (bound / gw)
#                        + sigma_s2[j-g] * (1/5)* (bound / gw)
#                            + sigma_s3[j-g] * (1/7)* (bound / gw))))
        return (2 * l + 1) * (sigma_t[g,g] - (sigma_s0[j-g] * (bound / gw)
                    + sigma_s1[j-g] * (1/3)* (bound / gw)
                        + sigma_s2[j-g] * (1/5)* (bound / gw)
                            + sigma_s3[j-g] * (1/7)* (bound / gw)))

    def calc_phi0(self):
        """
        Calculate the 0th scalar flux moment (phi0).

        Returns:
        np.ndarray: Updated phi0 values.
        """
        #t1 = time.time()
        L0U, L1U, L2U, L3U = self.build_Ln(self.AU,self.sigma_t_U,self.sigma_s_U)
#        L0H, L1H, L2H, L3H = self.build_Ln(self.AH,self.sigma_t_H,self.sigma_s_H)
#        self.L0 = L0U + L0H
#        self.L1 = L1U + L1H
#        self.L2 = L2U + L2H
#        self.L3 = L3U + L3H
        self.L0 = L0U 
        self.L1 = L1U 
        self.L2 = L2U 
        self.L3 = L3U 

        RHS = ((self.L3 @ self.L2 @ self.L1 + self.B2 * (9 * self.L1 + 4 * self.L3)) @ self.chi)
        LHS = (9 * self.B2 ** 2 + self.B2 * 
                (self.L3 @ self.L2 + (9 * self.L1 + 4 * self.L3) * self.L0) + self.L3 @ self.L2 @ self.L1 @ self.L0)
        print("Rank of LHS:", np.linalg.matrix_rank(LHS))
        print("Determinant of LHS:", np.linalg.det(LHS))
#        print("Any NaNs or Infs in LHS?", np.any(np.isnan(LHS)) or np.any(np.isinf(LHS)))
        print("Diag dominant: ", self.is_diagonally_dominant(LHS))
        self.phi0 = np.linalg.solve(LHS,RHS)
#        self.phi0 = np.linalg.lstsq(LHS,RHS,rcond=None)[0]
#        plt.figure()
#        plt.plot(self.E,abs(self.phi0))
#        plt.xscale('log')
#        plt.show()

        return self.phi0

    @staticmethod
    def is_diagonally_dominant(A):
        # Get the number of rows (or columns) in the matrix
        n = A.shape[0]
    
        # Check each row for diagonal dominance
        for i in range(n):
            row_sum = np.sum(np.abs(A[i])) - np.abs(A[i, i])  # Sum of non-diagonal elements
            if np.abs(A[i, i]) < row_sum:
                return False

        return True

    def calc_phi2(self):
        '''
        calculate the 2nd scalar flux moment. Eq 28 in paper
        big matmul
        
        Returns:
        vector: phi2
        '''
        # check for NAN's
        Ln = [self.L0, self.L1, self.L2, self.L3]
        for ind, arr in enumerate(Ln):
            assert not np.isnan(arr).any(), f"Array {ind} contains NaN values"

#        plt.imshow(self.L1, cmap='plasma', interpolation='nearest')
#        plt.colorbar()  
#        plt.show()

        LHS = self.L3 @ self.L2
        print('LHS Done')
        RHS = (-9 * self.B2 * self.phi0 + (9 * self.L1 + 4 * self.L3) @ (self.L0 @ self.phi0 - self.chi)) / 2
        print('RHS Done')

        # if diagonally dominant, perform a jacobi iteration in parallel
        if self.is_diagonally_dominant(LHS):
            print("Diagonally Dominant")
            self.phi2 = self.jacobi_parallel(LHS,RHS,self.phi2)

        else: # solve system directly
            print("Not Diagonally Dominant")
            self.phi2 = np.linalg.solve(LHS,RHS)
#            self.phi2 = np.linalg.lstsq(LHS,RHS,rcond=None)[0]

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
            self.Q[g] = self.chi[g] * np.trapz(self.sigma_f[g:g_min].flatten() * self.phi0[g:g_min].flatten(),self.groups[g:g_min])

        return self.Q

    def plot_xs_t(self):
        # plot xs's for H and U vs E
        plt.figure()
        plt.plot(self.E,self.sigma_t_U,label=r'$\Sigma_t^U$')
        plt.plot(self.E,self.sigma_t_H,label=r'$\Sigma_t^H$')
        plt.xlabel("E")
        plt.ylabel(r"$\Sigma_t$")
        plt.title(f"XS's for {self.groups.size} groups")
        plt.yscale("log")
        plt.xscale("log")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"results/charts/xs_t_{self.groups.size}.png")

    def plot_fluxes(self):
        # pretty self explanitory
        plt.figure()
        plt.plot(self.E,self.phi0,label=r'$\phi_0$')
        plt.title(r'$\phi_0(E)$')
        plt.xlabel('E')
        plt.ylabel(r'$\phi_0$')
        plt.xscale('log')
        plt.grid(True)
        plt.legend()
        plt.savefig(f'results/charts/phi0.png')

        plt.figure()
        plt.plot(self.E,self.phi2,label=r'$\phi_2$')
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
                D0[i,j] = np.trapz((L1 @ self.Phi0[j:g_min]).squeeze(),self.groups[j:g_min]) / np.trapz(self.Phi0[i:g_min],self.groups[i:g_min])
                D2[i,j] = np.trapz((L3 @ self.Phi0[j:g_min]).squeeze(),self.groups[j:g_min]) / np.trapz(self.Phi0[i:g_min],self.groups[i:g_min])
#                if D0[i,j] < self.tol and D2[i,j] < self.tol:
#                    D0[i,j:] = 0.
#                    D2[i,j:] = 0.
#                    break

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
            sigma_0_g[g] = np.trapz(sigma_x[g:g_min].flatten() * Phi0[g:g_min],self.groups[g:g_min]) / np.trapz(Phi0[g:g_min])
            sigma_2_g[g] = np.trapz(sigma_x[g:g_min].flatten() * Phi2[g:g_min],self.groups[g:g_min]) / np.trapz(Phi2[g:g_min])

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
#        update_xs_t_H = self.sigma_update(self.AH,self.sigma_t_H) 
#        update_xs_t_U = self.sigma_update(self.AU,self.sigma_t_U) 
#        update_xs_f_U = self.sigma_update(self.AU,self.sigma_f) 

        #FIXME scattering...
#        update_xs_s_H = self.sigma_s_update(self.AH,self.sigma_t_H)
#        update_xs_s_U = self.sigma_s_update(self.AU,self.sigma_t_U)
        
        print("Calculation completed.\nSaving data and plotting...")
        self.plot_fluxes()
        df = pd.DataFrame({'phi0': self.phi0, 'phi2': self.phi2, 'Phi0': self.Phi0, 'Phi2': self.Phi2})
        df.to_csv('results/csvs/fluxes.csv', index = False)
        print("Data Saved")
        
        return df

    def jacobi_parallel(self,A, b, x0, eps=1e-6, max_iter=1000):
        n = len(A)
        x = x0.copy()
        x_new = np.zeros_like(x)

        for iteration in range(max_iter):
            # Parallel computation of new values of x
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(self.jacobi_update, A, b, x, x_new, i) for i in range(n)
                ]
                # Wait for all threads to complete
                concurrent.futures.wait(futures)

            # Check for convergence (using norm of difference)
            norm = np.linalg.norm(x_new - x, ord=np.inf)
            if norm < eps:
                print(f"Converged in {iteration + 1} iterations")
                return x_new

            # Update x for the next iteration
            x[:] = x_new

        print(f"Reached max iterations ({max_iter})")

        return x_new

    @staticmethod
    def jacobi_update(A, b, x, x_new, i):
        row_sum = np.dot(A[i, :], x)  # Compute the row sum
        x_new[i] = (b[i] - (row_sum - A[i, i] * x[i])) / A[i, i]  # Update x[i]
    
    @staticmethod
    def pad_vector(v1, v2):
        size_diff = len(v2) - len(v1)
        if size_diff > 0:
            v1 = np.pad(v1, (0, size_diff), mode='constant', constant_values=0)
        return v1
