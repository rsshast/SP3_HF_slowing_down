from __init__ import *
from setup_sp3 import scratch_dir

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
        return self.groups.size if A == 1 else np.searchsorted(self.groups, self.groups[g] +  np.log(1/self.alpha(A)))

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
        Now works in parallel

        Parameters:
        A (int): Atomic Mass Ratio
        g (int): incident group
        sigma_s (matrix): 0th moment scattering xs for a given material

        Returns:
        vectors: scattering xs's for moments [0,3]
        """
        self.mu_s = np.zeros_like(self.L0)
        g_min_vec = np.zeros_like(self.groups)

        if A == 1: 
            g_min_vec[:] = self.groups[-1]
        else:
            for g in range(self.groups.size):
                g_min_vec[g] = self.group_bound(A,g)
        
        @njit(parallel=True)
        def fill_mu_s(A,groups,alpha,g_min_vec):
            mu_s = np.zeros((groups.size,groups.size))
            for g in prange(groups.size):
                for i in range(g, g_min_vec[g]):
                    mu_s[g, i] = (((A + 1) * np.exp((groups[g] - groups[i]) / 2))
                                   - ((A - 1) * np.exp((groups[i] - groups[g]) / 2))) / 2
                    mu_s[g, i] *= (np.exp(groups[g] - groups[i]) / (1 - alpha))

            return mu_s
        
        if l == 1: 
            self.mu_s = fill_mu_s(A,self.groups,self.alpha(A),g_min_vec)
            return sigma_s0 @ self.mu_s

        elif l == 2: mu = ((3 * np.linalg.matrix_power(self.mu_s,2) - 1) / 2)

        elif l == 3: 
            mu = ((5 * np.linalg.matrix_power(self.mu_s,3) - 3 * self.mu_s) / 2)

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
        S_vals = [self.xs_sl(A,S0,l) for l in range(1,4)]
        S1, S2, S3 = S_vals
        
        # save gtg scattering xs's
        my_str = "H" if A == 1 else "U"
        with h5py.File(f"{scratch_dir}/sigma_s_{my_str}.h5", "w") as f:
            f.create_dataset("sigma_s0", data=S0)
            f.create_dataset("sigma_s1", data=S1)
            f.create_dataset("sigma_s2", data=S2)
            f.create_dataset("sigma_s3", data=S3)

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
        gridwidth = self.groups[1] - self.groups[0]
        g_min_vec = np.zeros_like(self.groups)

        if A == 1: 
            g_min_vec[:] = self.groups[-1]
        else:
            for g in range(self.groups.size):
                g_min_vec[g] = self.group_bound(A,g)

        # Parallelize core computation
        self.parallel_integrate(self.groups, A, S_vals, I_vals, gridwidth, g_min_vec)

        # Deallocate S_vals and return results
        self.deallocate([S_vals,g_min_vec])

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

        self.deallocate(I_vals)
        self.deallocate([sigma_t])

        return L_vals

    def calc_phi0(self, properties):
        """
        Calculate the 0th scalar flux moment (phi0).

        Returns:
        vector: phi0.
        """
        t1 = time.time()
        # compute the loss operators
        print("U-238 Loss Operators")
        L_vals_U  = self.build_Ln(self.AU,self.sigma_t_U,self.sigma_s_U)
        # send to csr to save memory
        L_U_sparse = [csr_matrix(matrix) for matrix in L_vals_U]
        print("H-1 Loss Operators")
        L_vals_H = self.build_Ln(self.AH,self.sigma_t_H,self.sigma_s_H)
        # send to csr to save memory
        L_H_sparse = [csr_matrix(matrix) for matrix in L_vals_H]

        # do sum on csr
        sparse_sum = [L_U_sparse[i] + L_H_sparse[i] for i in range(len(L_U_sparse))]
        # reconstruct the full matrix
        self.L0, self.L1, self.L2, self.L3 = [matrix.toarray() for matrix in sparse_sum]

        # deallocate unnecessary memory
        self.deallocate(L_vals_U)
        self.deallocate(L_vals_H)
        self.deallocate(L_U_sparse)
        self.deallocate(L_H_sparse)
        
        print(f"Loss Matrices Computed in {np.round(time.time() - t1, 5)}s")

        # compute LHS and RHS
        print("phi0 LHS")
        LHS = (9 * self.B2 ** 2 + self.B2 * (self.L3 @ self.L2 + (9 * self.L1 + 4 * self.L3) * self.L0) 
                    + self.L3 @ self.L2 @ self.L1 @ self.L0)
        print("phi0 RHS")
        RHS = ((self.L3 @ self.L2 @ self.L1 + self.B2 * (9 * self.L1 + 4 * self.L3)) @ self.chi)

        if properties: self.print_mat_properties(LHS)

        # Ax = b
        self.phi0 = np.zeros_like(self.groups)
        self.phi0 = (self.jacobi_parallel(LHS,RHS,self.phi0) 
                        if self.is_diagonally_dominant(LHS) 
                            else np.linalg.solve(LHS,RHS))
        
        # deallocate
        self.deallocate([LHS,RHS])

        return self.phi0

    def calc_phi2(self, properties):
        '''
        calculate the 2nd scalar flux moment. Eq 28 in paper
        big matmul
        
        Returns:
        vector: phi2
        '''
        # compute LHS and RHS
        LHS = self.L3 @ self.L2 # matrix
        RHS = (-9 * self.B2 * self.phi0 + (9 * self.L1 + 4 * self.L3) @ (self.L0 @ self.phi0 - self.chi)) / 2 # vector

        if properties: self.print_mat_properties(LHS)

        # Ax = b
        self.phi2 = np.zeros_like(self.phi0)
        self.phi2 = (self.jacobi_parallel(LHS,RHS,self.phi2) 
                        if self.is_diagonally_dominant(LHS) 
                            else np.linalg.solve(LHS,RHS))

        # deallocate
        self.deallocate([LHS,RHS])

        return self.phi2

    def calc_Phi(self):
        """
        Combine phi0 and phi2 to calculate the scalar flux moments Phi0 and Phi2.
        
        Returns: 
        None
        """
        self.Phi2 = self.phi2
        self.Phi0 = self.phi0 + 2 * self.phi2

    def fission_source(self,A, sigma_f):
        """
        Calculates the Fission Source, Q
        Eq. 47
        
        Parameters: 
        A (int): Atomic Number
        sigma_f (vector): fission xs's

        Returns:
        vector: Fission Source
        """
        Q = np.zeros_like(self.phi0)
        gridwidth = self.groups[1] - self.groups[0]
        for i in range(self.phi0.size):
            g_min = self.group_bound(A,i)
            for j in range(i,g_min): Q[i] += sigma_f[j] * self.phi0[j] * np.exp(self.groups[i])
            Q[i] *= (self.chi[i] * self.E[i])

        return Q

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

    def diffusion_coef(self,A,n):
        """
        Calculates the diffusion coefficient for each group

        Parameters:
        A (int): Atomic Number
        n (int): Flux Order (0 or 2)

        Returns:
        matrix: diffusion coef for each fine group (gxg)
        """
        D = np.zeros((self.phi0.size,self.phi0.size))
        gridwidth = self.groups[1] - self.groups[0]

        if n == 0:
            Linv = sp.linalg.inv(self.L1)
            Phi  = self.Phi0

        else: 
            Linv = sp.linalg.inv(self.L3)
            Phi  = self.Phi2

        # operatre on Phi_n(u)
        M = np.matmul(Linv,Phi)
        for i in range(D.shape[0]): 
            g_min = self.group_bound(A,i)
            num = np.zeros((g_min - i))

            # numerator integral
            for j in range(i,g_min): 
                num[i-j] += (M[j] * np.exp(self.groups[j]) * gridwidth)

            # accumulation matrix flipped
            # not multiplying by self.E[i], it cancels out
            num = np.flip(num)
            # construct the diffusion coef matrix
            D[i,i:g_min] = num / (np.sum(Phi[i:g_min] * gridwidth)) 

        return D

    def sigma_update(self,A,sigma_x,n):
        """
        Updates xs's for moment n and group g
        corresponds to eqns 52a) and 52c)

        Parameters:
        A (int): Atomic Number
        sigma_x (vector): total or fission xs's.
        n (int): moment

        Returns:
        vector: updated fission or total xs's for order l
        """
        Phi = self.Phi0 if n == 0 else self.Phi2
        sigma_new = np.zeros_like(Phi)
        gridwidth = self.groups[1] - self.groups[0]

        for i in range(Phi.size):
            g_min = self.group_bound(A,i)
            den = np.zeros((g_min - i))
            for j in range(i,g_min):
                # note: E0 cancels out in division
                sigma_new[i] += sigma_x[j] * Phi[j] * np.exp(self.groups[i]) * gridwidth
                # denominator in lethargy space
                den[i-j]      = np.exp(self.groups[j]) * gridwidth
            sigma_new[i] /= np.sum(Phi[i:g_min] * den)

        return sigma_new

    def sigma_s_update(self,A,l,n):
        """
        Updates gtg xs's for moment n

        Parameters:
        A(int): Atomic Number
        l (int): scattering legendre moment
        n (int): flux moment

        Returns:
        matrix: updated g to g scattering xs's for expansion l and moment n
        """
        # build xs library, sigma_sn
        sigma_s = np.zeros_like(self.L0)
        Phi = self.Phi0 if n == 0 else self.Phi2
        gridwidth = self.groups[1] - self.groups[0]
        # read scattering xs's from .h5
        my_str = "H" if A == 1 else "U"
        sigma_sl = self.read_sigma_s(l,A)

        # in integetral, E0 dependence cancels out
        M = np.matmul(sigma_sl,Phi)
        for i in range(Phi.size):
            g_min = self.group_bound(A,i)
            num = np.zeros((g_min - i))
            # integrate
            for j in range(i,g_min):
                num[i-j] += (M[j] * np.exp(self.groups[j]) * gridwidth)
            num = np.flip(num)
            den = np.sum(Phi[i:g_min] * gridwidth)
            sigma_sl[i,i:g_min] = num/den
                
        return sigma_sl

    def run(self, properties, from_h5):
        """
        Run the complete SP3 calculation process.
        """
        if from_h5 == False:
            print("Starting phi0 calculation...")
            st = time.time()
            self.calc_phi0(properties)
            et = time.time()
            print(f"phi0 calculation: {np.round(et-st,5)}")

            print("Starting phi2 calculation...")
            self.calc_phi2(properties)

            print("Plotting")
            st = time.time()
            self.plot_fluxes()
            et = time.time()
            print(f"Plotting Time: {np.round(et-st,5)}")

            print("Starting Phi0 and Phi2 calculation...")
            self.calc_Phi()

            print("Saving Data...")
            df = pd.DataFrame({'phi0': self.phi0, 'phi2': self.phi2, 'Phi0': self.Phi0, 'Phi2': self.Phi2})
            df.to_hdf(f"{scratch_dir}/fluxes.h5", key="df", mode="w", format="table")
            with h5py.File(f"{scratch_dir}/Ln.h5", "w") as f:
                f.create_dataset("L0", data=self.L0)
                f.create_dataset("L1", data=self.L1)
                f.create_dataset("L2", data=self.L2)
                f.create_dataset("L3", data=self.L3)
            print("Data Saved")

        else:
            print("Reading Data From File...")
            df = pd.read_hdf(f"{scratch_dir}/fluxes.h5", key="df")
            df = df.to_numpy() 
            self.phi0, self.phi2, self.Phi0, self.Phi2 = df[:, 0], df[:, 1], df[:, 2], df[:, 3]
            with h5py.File(f"{scratch_dir}/Ln.h5", "r") as f:
                self.L0 = f["L0"][:]
                self.L1 = f["L1"][:]
                self.L2 = f["L2"][:]
                self.L3 = f["L3"][:]
            print("Data Read!")

        # Group fission and total cross-sections for moments 0 and 2
        print("Calculating Fission Source and Updated Total / Fission Cross-Sections...")
        st = time.time()
        new_xs = {
            "xs_t_H_0": self.sigma_update(self.AH, self.sigma_t_H, 0),
            "xs_t_H_2": self.sigma_update(self.AH, self.sigma_t_H, 2),
            "xs_t_U_0": self.sigma_update(self.AU, self.sigma_t_U, 0),
            "xs_t_U_2": self.sigma_update(self.AU, self.sigma_t_U, 2),
            "xs_f_U_0": self.sigma_update(self.AU, self.sigma_f, 0),
            "xs_f_U_2": self.sigma_update(self.AU, self.sigma_f, 2),
            "Q(E)"    : self.fission_source(self.AU,self.sigma_f),
        }
        print(f"Time XS Update: {np.round(time.time() - st, 5)}s")

        # Save xs's to hdf5
        with h5py.File(f"{scratch_dir}/cross_sections.h5", "w") as f:
            for key, value in new_xs.items():
                f.create_dataset(key, data=value, compression="gzip")

        # Generate diffusion coefficients
        st = time.time()
        print("Calculating Diffusion Coefficients for l = 0, l = 2...")
        with h5py.File(f"{scratch_dir}/diffusion_coefficients.h5", "w") as f:
            for prefix, A in [("U", self.AU), ("H", self.AH)]:
                for l in [0, 2]:
                    D = self.diffusion_coef(A, l)
                    f.create_dataset(f"D{prefix}_{l}", data=D, compression="gzip", compression_opts=9)
        print(f"Time D Coef: {np.round(time.time() - st, 5)}s")

        # generate gtg scattering xs's for all materials
        print("Calculting Updated Group to Group Scattering Cross-Sections...")
        #new_xs_A_l_n
        mat_indx = [self.AH,self.AU]
        mat_name = ["H","U"]
        indices  = [(0, 0), (0, 2), (1, 0), (1, 2), (2, 0), (2, 2), (3, 0), (3, 2)]

        st = time.time()
        with h5py.File(f"{scratch_dir}/gtg_cross_sections.h5", "w") as f:
            for A, name in zip(mat_indx, mat_name):
                for l, n in indices:
                    xs = self.sigma_s_update(A, l, n)
                    f.create_dataset(f"new_xs_{name}_{l}_{n}", data=xs, compression="gzip", compression_opts=9)
        print(f"Time gtg XS Update: {np.round(time.time() - st, 5)}s")

        print("Calculation completed.")

    def jacobi_parallel(self,A, b, x0, eps=1e-6, max_iter=1000):
        """
        Jacobi iteration for parallel computing

        Parameters:
        A (matrix): matrix
        b (vector): vector
        x0 (vector): solution guess

        Returns:
        vector: solution
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
    @njit(parallel=True)
    def parallel_integrate(groups, A, S_vals, I_vals, gridwidth, g_min_vec):
        """
        Helper function to perform the integration in parallel.
        """
        for i in prange(groups.size):
            g_min = g_min_vec[i]
            for j in range(i, g_min):
                g_bound = min(g_min, groups[-1])
                factor = (g_bound - g_min) + (g_min - j)

                for idx in range(4):
                    I_vals[idx][i, i] += S_vals[idx][i, j] * factor * gridwidth

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
    def read_sigma_s(l, A):
        """Read only the S{l} dataset from the HDF5 file."""
        my_str = "H" if A == 1 else "U"
        with h5py.File(f"{scratch_dir}/sigma_s_{my_str}.h5", "r") as f:
            return np.array(f[f"sigma_s{l}"])  

    @staticmethod
    def deallocate(my_list):
        """
        Free up memory because large matrices

        Parameters: 
        my_list (list): a list of n parameters

        Returns: 
        None
        """
        for obj in my_list:
            del obj 
        my_list.clear()  

    @staticmethod
    def den_to_csr(A):
        """
        convert a dense matrix to csr format

        Parameters:
        A (matrix): matrix in dense format

        Returns (triplet):
        vector: values
        vector: column indices
        vector: row pointer
        """
        # init
        vals = []
        col_ind = []
        row_ptr = []

        for row in A:
            for col, val in enumerate(row):
                if val != 0:
                    vals.append(val)
                    col_ind.append(col)
            row_ptr.append(len(vals))

        return np.array(vals), np.array(col_ind), np.array(row_ptr)

#    @staticmethod
#    def np_to_tl(A):
#        core, factors = tucker(A, rank=[n//2 for n in A.shape])
#        print(core.nbytes)
#        reconstructed = tucker_to_tensor((core, factors))
#        error = np.linalg.norm(A - reconstructed) / np.linalg.norm(A)
#        print(f"Reconstruction error: {error:.4f}")
#        return tl.tensor(A)
#
#    @staticmethod
#    def tl_to_np(A):
#        return tl.to_numpy(A)
#
#    @staticmethod
#    def find_rank(A):
#        ranks = range(1, 10)
#
#        for r in ranks:
#            factors = parafac(A, rank=r)
#            reconstructed = tl.cp_to_tensor(factors)
#            error = np.linalg.norm(A - reconstructed) / np.linalg.norm(A)
#            if error < .05: return r
#            
#
#        # Plot the error curve to find the "elbow" point
#        #import matplotlib.pyplot as plt
#        #plt.plot(ranks, errors, marker="o")
#        #plt.xlabel("Rank")
#        #plt.ylabel("Reconstruction Error")
#        #plt.title("Choosing Best Rank for CP Decomposition")
#        #plt.show()
#
#    @staticmethod
#    def estimate_tucker_rank(tensor, energy_threshold):
#        """ Estimate Tucker ranks based on SVD energy retention. """
#        ranks = []
#        for mode in range(tl.ndim(tensor)):
#            unfolding = tl.unfold(tensor, mode)  # Unfold tensor along mode
#            singular_values = svd(unfolding, compute_uv=False)  # SVD on mode matrix
#            total_energy = np.cumsum(singular_values**2) / np.sum(singular_values**2)
#            rank = np.searchsorted(total_energy, energy_threshold) + 1  # Find where energy exceeds threshold
#            ranks.append(rank)
#        return ranks
