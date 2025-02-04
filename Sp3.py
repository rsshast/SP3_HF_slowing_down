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
        self.B2        = B2 # geometric buckling
        self.tol       = 1e-6 # Small value threshold

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
        Similar to question 4 on final exam. 
        double integral. first one over u can be done explicitly, second one over u' must be done numerically
        """
        # build gtg xs libraries
        sigma_t = np.diag(sigma_t) # diagonal of the total xs's

        # integration matrix corresponding to eqn 4
        I0, I1, I2, I3 = self.integrate_Sn(A,sigma_s)

        # loss operator corresponding to eqn 5
        L0 = self._Ln(0,sigma_t,I0)
        L1 = self._Ln(1,sigma_t,I1)
        L2 = self._Ln(2,sigma_t,I2)
        L3 = self._Ln(3,sigma_t,I3)
            
        return L0,L1,L2,L3

    def xs_sl(self,A,g,sigma_s):
        """
        Calculate the lth moment scattering xs for a given material for a given group

        Parameters:
        A (int): Atomic Mass Ratio
        g (int): incident group
        sigma_s: 0th moment scattering xs for a given material

        Returns:
        vectors: scattering xs's for moments [0,3]
        """
        # start with finding the scattering cosine for L0 -> L3
        g_min = self.group_bound(A,g)
        mu_s = np.zeros(g_min - g)

        sigma_s0 = sigma_s #s0

        for i in range(g,g_min):
            mu_s[i-g] = (((A + 1) * np.exp((self.groups[g] - self.groups[i]) / 2)) 
                                - ((A - 1) * np.exp((self.groups[i] - self.groups[g]) / 2))) / 2
            # protection
            if np.abs(mu_s[i-g]) > 1: raise ValueError("Mu not in range")

        sigma_s1 = sigma_s[g:g_min]*mu_s #s1

        sigma_s2 = sigma_s[g:g_min] * (3 * mu_s ** 2 - 1) / 2 #s2
            
        sigma_s3 = sigma_s[g:g_min] * (5 * mu_s ** 3 - 3 * mu_s) / 2 #s3

        return sigma_s0, sigma_s1, sigma_s2, sigma_s3

    def xs_pl(self,A,sigma_s):
        """
        calculates the full gtg scattering matrix for the lth order 

        Parameters:
        A (int): atomic number
        sigma_s (vector): 0th order scattering xs

        Returns: 
        4 matrices: gtg scattering matrices for l on the range [0,3]
        """
        # scattering xs's for the lth moment
        S0 = np.zeros_like(self.L0)
        S1 = np.zeros_like(S0)
        S2 = np.zeros_like(S0)
        S3 = np.zeros_like(S0)
        gridwidth = np.diff(self.groups[:-1])
        tol = self.tol

        # loop over all groups
        for g in range(self.groups.size):
            g_min = self.group_bound(A,g)
            sigma_s0, sigma_s1, sigma_s2, sigma_s3 = self.xs_sl(A,g,sigma_s) 
            # Compute group-to-group scattering in lethargy space
            for i in range(g,g_min):  # Initial group
                divisor = 1
                for j in range(g,g_min):  # Scattered-to group
                    # assumes evenly spaced grid
                    S0[j,g] = self.SN_mat(sigma_s0[i-g],gridwidth,divisor) 
                    S1[j,g] = self.SN_mat(sigma_s1[i-g],gridwidth,divisor) 
                    S2[j,g] = self.SN_mat(sigma_s2[i-g],gridwidth,divisor) 
                    S3[j,g] = self.SN_mat(sigma_s3[i-g],gridwidth,divisor) 
                    divisor += 1

                    if S0[j,g] < tol and S1[j,g] < tol and S2[j,g] < tol and S3[j,g] < tol: break
        
        return S0, S1, S2, S3

    def integrate_Sn(self,A,sigma_s):
        """
        calculates the scattering integral, eqn 4:

        Parameters:
        A (int): atomic number
        sigma_s (vector): 0th order scattering xs

        Returns: 
        4 matrices: gtg scattering integrals for l on the range [0,3]
        """
        # integration matrices init
        I0 = np.zeros_like(self.L0)
        I1 = np.zeros_like(I0)
        I2 = np.zeros_like(I0)
        I3 = np.zeros_like(I0)
        gridwidth = np.diff(self.groups[:-1])

        S0, S1, S2, S3 = self.xs_pl(A,sigma_s)
        
        for g in range(self.groups.size):
            g_min = self.group_bound(A,g)
            for gg in range(g,g_min):
                I0[gg,g] += S0[gg,g]*np.exp(self.groups[gg] - self.groups[g]) * gridwidth[-1]
                I1[gg,g] += S1[gg,g]*np.exp(self.groups[gg] - self.groups[g]) * gridwidth[-1]
                I2[gg,g] += S2[gg,g]*np.exp(self.groups[gg] - self.groups[g]) * gridwidth[-1]
                I3[gg,g] += S3[gg,g]*np.exp(self.groups[gg] - self.groups[g]) * gridwidth[-1]

        return I0, I1, I2, I3

    def calc_phi0(self):
        """
        Calculate the 0th scalar flux moment (phi0).

        Returns:
        np.ndarray: Updated phi0 values.
        """
        t1 = time.time()
        print("Uranium - uh")
        L0U, L1U, L2U, L3U = self.build_Ln(self.AU,self.sigma_t_U,self.sigma_s_U)
        print("Hydrogen")
        L0H, L1H, L2H, L3H = self.build_Ln(self.AH,self.sigma_t_H,self.sigma_s_H)
        print(f"Loss Matrices Computed in {np.round(time.time() - t1, 5)}s")
        self.L0 = L0U + L0H
        self.L1 = L1U + L1H
        self.L2 = L2U + L2H
        self.L3 = L3U + L3H

        # matrix
        LHS = (9 * self.B2 ** 2 + self.B2 * 
                (self.L3 @ self.L2 + (9 * self.L1 + 4 * self.L3) * self.L0) 
                    + self.L3 @ self.L2 @ self.L1 @ self.L0)
        # vector
        RHS = ((self.L3 @ self.L2 @ self.L1 + self.B2 * (9 * self.L1 + 4 * self.L3)) @ self.chi)

        self.print_mat_properties(LHS)

        # Ax = b
        self.phi0 = (self.jacobi_parallel(LHS,RHS,self.phi0) 
                        if self.is_diagonally_dominant(LHS) 
                            else np.linalg.solve(LHS,RHS))

        return self.phi0

    def calc_phi2(self):
        '''
        calculate the 2nd scalar flux moment. Eq 28 in paper
        big matmul
        
        Returns:
        vector: phi2
        '''
        # compute LHS and RHS
        LHS = self.L3 @ self.L2 # matrix
        RHS = (-9 * self.B2 * self.phi0 + (9 * self.L1 + 4 * self.L3) @ (self.L0 @ self.phi0 - self.chi)) / 2 # vector

        self.print_mat_properties(LHS)

        # Ax = b
        self.phi2 = (self.jacobi_parallel(LHS,RHS,self.phi2) 
                        if self.is_diagonally_dominant(LHS) 
                            else np.linalg.solve(LHS,RHS))

        return self.phi2

    def calc_Phi(self):
        """
        Combine phi0 and phi2 to calculate the scalar flux moments Phi0 and Phi2.
        
        Returns: 
        None
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
            self.Q[g] = (self.chi[g] * np.trapz(self.sigma_f[g:g_min].flatten() 
                            * self.phi0[g:g_min].flatten(),self.groups[g:g_min]))

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
        # plot phi0 and phi2
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
        """
        Jacobi iteration for parallel computing
        """
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
    def alpha(A):
        """
        calculate the scattering parameter for a given element
        Parameters:
        A (int): Mass Ratio

        Returns
        float
        """
        return ((A - 1) / (A + 1)) ** 2

    def print_mat_properties(self,LHS):
        """
        prints important matrix properties

        Parameters:
        LHS (matrix): matrix of interest

        Returns:
        None
        """
        print("Rank of LHS:", np.linalg.matrix_rank(LHS))
        print("Determinant of LHS:", np.linalg.det(LHS))
        print("Any NaNs or Infs in LHS?", np.any(np.isnan(LHS)) or np.any(np.isinf(LHS)))
        print("Diag dominant: ", self.is_diagonally_dominant(LHS))

    @staticmethod
    def jacobi_update(A, b, x, x_new, i):
        """
        helper function for the jacobi loop. Not sure what this is doing
        """
        row_sum = np.dot(A[i, :], x)  # Compute the row sum
        x_new[i] = (b[i] - (row_sum - A[i, i] * x[i])) / A[i, i]  # Update x[i]
    
    @staticmethod
    def SN_mat(sigma,gridwidth,divisor):
        """
        calculate each parameter in the gtg scattering matrix

        Parameters:
        sigma (vector): lth order scattering xs
        gridwidth (float): lethargy grid spacing
        divisor (int): how many points exist between the lethargy grids

        Returns:
        float: scattering xs for group g' into group g
        """
        return sigma * gridwidth[-1] / divisor

    @staticmethod
    def _Ln(l,sigma,I): 
        """
        calculates the Loss operator for order l

        Parameters: 
        l (int): expansion order
        sigma (matrix): total xs
        I (matrix): integration matrix for order l

        Returns: 
        matrix: Loss operator for order l
        """
        return (2 * l + 1) * (sigma - I)

    @staticmethod
    def is_diagonally_dominant(A):
        """
        checks if a matrix is diagonally dominant

        Parameters:
        A (matrix): matrix in question

        Returns: 
        bool: True if matrix is diagonally dominant
        """
        # loop over rows
        for i in range(A.shape[0]):
            row_sum = np.sum(np.abs(A[i])) - np.abs(A[i, i])  # Sum of non-diagonals
            if np.abs(A[i, i]) < row_sum:
                return False

        return True
