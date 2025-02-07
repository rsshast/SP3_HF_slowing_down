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
        self.phi0 = None
        self.phi2 = None
        self.Phi0 = None
        self.Phi2 = None
        self.Q    = None
        self.L0   = np.zeros((self.groups.size,self.groups.size))
        self.L1   = None
        self.L2   = None
        self.L3   = None
        self.mu_s = None

    def group_bound(self, A, g):
        """
        Calculate the minimum group a neutron can downscatter to.
        CONSTANT GRIDSPACING
        For Hydrogen, lim alpha -> 0 1/alpha = \inf, therefore no top lethargy bound
        For heavier isotopes (Uranium) we have 2 cases (no clipping):
        Take the minimum of (u + ln (1/alpha)) and gridwidth of lethargy grid 
        you can pass a function in here instead of the value. 

        Parameters:
        A (int): Atomic Number:
        g (int): Current energy group index.

        Returns:
        int: Minimum group index for downscattering.
        """
        if A == 1:
            return self.groups.size
        else: # assumes constant grid spacing
            # when lethargy width is less than the gridwidth
            t1 = self.groups[g] + np.log(1 / self.alpha(A))
            return np.searchsorted(self.groups, t1)

    def xs_gtg(self,sigma_s0,A):
        """
        find the gtg scattering matrix for s0

        Parameters: 
        sigma_s0 (vector): l = 0 group xs's
        A (int): Mass Ratio

        returns:
        matrix: gtg xs's for l = 0
        """
        # Initialize
        sigma_gtg = np.zeros_like(self.L0)

        # Incident lethargy
        for g in range(self.groups.size):
            u_low = self.groups[g]
            u_high = self.group_bound(A,g)
            xs_s = sigma_s0[g]

            # Compute fractional contributions for scattering
            fractions = np.zeros(self.groups.size)  # Initialize array to match the size of sigma
            total_fraction = 0.0

            # Outgoing lethargy groups obeying scattering ranges (upper triangular)
            for gg in range(g, min(u_high, self.groups.size)):  # Scattering to higher lethargy
                u_target_low = self.groups[gg]
                u_target_high = self.group_bound(A, gg)

                # Lethargy bin overlap
                overlap = max(0, min(u_high, u_target_high) - max(u_low, u_target_low))
                fraction = overlap / (u_high - u_low) if (u_high - u_low) > 0 else 0

                # Append fraction for the target group
                fractions[gg] = fraction
                total_fraction += fraction

                # Exit early if the fraction is very small (for efficiency)
                if fraction < self.tol: break

            # Normalize contributions: normalize each group so total fraction sums to 1
            if total_fraction > 0:
                fractions /= total_fraction

            # Build matrix (only upper triangle)
            sigma_gtg[g, :] = xs_s * fractions

        # Normalize each row to sigma[g]
        row_sums = np.sum(sigma_gtg, axis=1)
        for g in range(self.groups.size):
            if row_sums[g] > 0:
                sigma_gtg[g, :] *= sigma_s0[g] / row_sums[g]

        return sigma_gtg


    def xs_sl(self,A,sigma_s0,l):
        """
        Calculate the lth moment scattering xs for a given material for a given group

        Parameters:
        A (int): Atomic Mass Ratio
        g (int): incident group
        sigma_s (matrix): 0th moment scattering xs for a given material

        Returns:
        vectors: scattering xs's for moments [0,3]
        """
        self.mu_s = np.zeros_like(self.L0)
        if l == 1:
            for g in range(self.groups.size):   
                print(g)
                for i in range(g,self.group_bound(A,g)):
                    self.mu_s[g,i]  = (((A + 1) * np.exp((self.groups[g] - self.groups[i]) / 2)) 
                                            - ((A - 1) * np.exp((self.groups[i] - self.groups[g]) / 2))) / 2
                    if np.abs(self.mu_s[g,i]) > 1: raise ValueError("Mu not in range")
                    self.mu_s[g,i] *= (np.exp(self.groups[g] - self.groups[i])/(1 - self.alpha(A)))
            
            return sigma_s0 @ self.mu_s


        elif l == 2: mu = ((3 * self.mu_s ** 2 - 1) / 2)

        elif l == 3: mu = ((5 * self.mu_s ** 3 - 3 * self.mu_s) / 2)

        return sigma_s0 @ mu

    def calc_xs_l(self,sigma_s_0,A):
        """
        build full gxg matrix of scattering xs's for l = [0,3]

        Parameters:
        sigma_s_0 (vector): l = 0 xs's for a given material
        A (int): Mass Ratio

        Returns: 
        4 matrices: gtg scattering xs's 
        """
        # get the 0th moment gtg xs's
        sigma_s0 = self.xs_gtg(sigma_s_0,A)
        S0 = sigma_s0

        # get the rest of the orders 
        S_vals = [self.xs_sl(A,sigma_s0,l) for l in range(1,4)]
        S1, S2, S3 = S_vals

        return S0, S1, S2, S3

    def integrate_Sn(self,sigma_s0,A):
        """
        eqn 4, integrate the cross sections to create the Sn matrix for order l

        Parameters:
        sigma_s0 (vector): s0 xs's for a material to calculate the lth order xs's
        A (int): Mass ratio

        Return:
        4 matrices: Sn operator matrices
        """
        # Initialize I_vals and compute S_vals
        I_vals = [np.zeros_like(self.L0) for _ in range(4)]
        S_vals = self.calc_xs_l(sigma_s0, A)
        
        # Perform integration
        gridwidth = self.groups[1] - self.groups[0]
        
        for i in range(self.groups.size):
            g_min = self.group_bound(A,i)
            for j in range(i, g_min):
                g_bound = min(g_min, self.groups[-1])
                factor = (g_bound - g_min) + (g_min - j)
                
                for idx in range(4):
                    I_vals[idx][i, i] += S_vals[idx][i, j] * factor * gridwidth
        
        # Deallocate S_vals and return results
        self.deallocate([S_vals])

        return I_vals

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
        I_vals = self.integrate_Sn(sigma_s,A)

        # loss operator corresponding to eqn 5
        L_vals = [self._Ln(n, sigma_t, I_vals[n]) for n in range(4)]

        return L_vals

    def calc_phi0(self):
        """
        Calculate the 0th scalar flux moment (phi0).

        Returns:
        np.ndarray: Updated phi0 values.
        """
        t1 = time.time()
        print("U-238")
        L_vals_U  = self.build_Ln(self.AU,self.sigma_t_U,self.sigma_s_U)
        print("H-1")
        L_vals_H = self.build_Ln(self.AH,self.sigma_t_H,self.sigma_s_H)

        self.L0 = L_vals_U[0] + L_vals_H[0]
        self.L1 = L_vals_U[1] + L_vals_H[1]
        self.L2 = L_vals_U[2] + L_vals_H[2]
        self.L3 = L_vals_U[3] + L_vals_H[3]
        
        print(f"Loss Matrices Computed in {np.round(time.time() - t1, 5)}s")

        # compute LHS and RHS
        LHS = (9 * self.B2 ** 2 + self.B2 * (self.L3 @ self.L2 + (9 * self.L1 + 4 * self.L3) * self.L0) 
                    + self.L3 @ self.L2 @ self.L1 @ self.L0)
        RHS = ((self.L3 @ self.L2 @ self.L1 + self.B2 * (9 * self.L1 + 4 * self.L3)) @ self.chi)

        self.print_mat_properties(LHS)

        # Ax = b
        self.phi0 = np.zeros_like(self.groups)
        self.phi0 = (self.jacobi_parallel(LHS,RHS,self.phi0) 
                        if self.is_diagonally_dominant(LHS) 
                            else np.linalg.solve(LHS,RHS))
        
        # deallocate
        self.deallocate([LHS,RHS])

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
        self.phi2 = np.zeros_like(self.phi0)
        self.phi2 = (self.jacobi_parallel(LHS,RHS,self.phi2) 
                        if self.is_diagonally_dominant(LHS) 
                            else np.linalg.solve(LHS,RHS))

        # deallocate
        del LHS
        del RHS

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

        #print("New cross sections and Diffusion Coefs...")
        # fission source Q
        #Q = self.fission_source()

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

    @staticmethod
    def deallocate(my_list):
        for obj in my_list:
            del obj 
        my_list.clear()  
