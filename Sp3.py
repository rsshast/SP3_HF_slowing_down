from __init__ import *

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
        self.E0        = 1e7
        self.groups    = xs_U[:,0] # lethargy groups, low leth to high leth
        self.sigma_t_H = np.flip(xs_H[:, 1])
        self.sigma_s_H = np.flip(xs_H[:, 2]) # Hydrogen xs's
        self.sigma_t_U = np.flip(xs_U[:, 1])
        self.sigma_s_U = np.flip(xs_U[:, 2]) # U-238 xs's
        self.sigma_f   = np.flip(sigma_f) # fission xs's
        self.E         = np.exp(xs_U[:,0]) # energy groups, low to high
        self.chi       = np.flip(chi[:, 1]) # fission spectrum
        self.B2        = B2 # geometric buckling
        self.tol       = 1e-6 # Small value threshold
        self.gridspace = self.groups[1] - self.groups[0]
        self.leg_order = 4
        self.chi      /= np.trapz(self.chi,self.E) # normalize chi

        # Initialize other attributes
        self.L0   = np.zeros((self.groups.size,self.groups.size))
        self.L1   = None
        self.L2   = None
        self.L3   = None
        self.phi0 = None
        self.phi2 = None
        self.Phi0 = None
        self.Phi2 = None
        self.p0   = None

    def initial_flux(self):
        """
        calculate the initial flux as shown in Nuclear Engineering handbook

        Returns:
        vector: phi, initial s0 fluxes used to construct the gtg scattering xs's
        """
        scatH = 0
        scatU = 0
        alpha = self.alpha(self.AU)
        phi = np.zeros_like(self.groups)
        g_min = self.group_bound(self.AU,0)

        sigma_th = (self.sigma_t_H)
        sigma_tu = (self.sigma_t_U)
        sigma_sh = (self.sigma_s_H)
        sigma_su = (self.sigma_s_U)
        chi = (self.chi)

        @staticmethod
        def scat_source(sigma,phi,E,gridspace,i,alpha):
            return sigma[i-1] / (1 - alpha) * phi[i-1] * E[i-1] * gridspace

        for i in range(0, self.groups.size):
            #removal xs
            Sigma_R = (sigma_th[i] + sigma_tu[i] - 
                        self.gridspace * (sigma_su[i] / (1 - alpha)) 
                            - self.gridspace * sigma_sh[i])

            if i == 0:
                phi[i] = chi[i]/Sigma_R

            else:
                #Scattering Source for Hydrogen
                scatH += scat_source(sigma_sh,phi,self.E,self.gridspace,i,alpha = 0)
    
                #Scattering Source for Uranium
                scatU += scat_source(sigma_su,phi,self.E,self.gridspace,i,alpha = alpha)

                #subtract off the contribution from lesser lethargy bins
                if i > g_min:
                    h = i - g_min
                    scatU -= (sigma_su[h]   / (1 - alpha) * phi[h] 
                                * self.E[h]   * self.gridspace)

                #append to the vector
                phi[i] = (chi[i] + np.exp(-1 * self.groups[i]) * (scatH + scatU)) / Sigma_R

        self.p0 = phi
        
    def group_bound(self, A, g):
        """
        Calculate the minimum group a neutron can downscatter to.

        Parameters:
        A (int): Atomic Number:
        g (int): Current energy group index.

        Returns:
        int: Minimum group index for downscattering.
        """
        return (self.groups.size if A == 1 
                else np.searchsorted(self.groups, self.groups[g] + np.log(1 / self.alpha(A))))

    def compute_scattering(self, g, gp, sigma_s, A, l, mu):
        """
        Finds the g' -> g xs for moment l

        Parameters:
        g (int): incident lethargy group
        gp (int): outgoing lethargy group
        sigma_s (vector): 0th order scatter xs
        A (int): Atomic number
        l (int): legendre order
        mu(int): Leg poly of scattering cosine

        Return:
        float(int): integral value of gtg xs
        """
        u_min, up_min = self.groups[g], self.groups[gp]

        u_max  = self.groups[g+1]  if g < self.groups.size - 1  else self.groups[g]
        up_max = self.groups[gp+1] if gp < self.groups.size - 1 else self.groups[gp]
        sigma_s_gp = sigma_s[gp]

        def sigma_s_kernel(u, up, sigma_s_gp, alpha):
            #if alpha == 0: return sigma_s_gp * np.exp(up - u)
            if alpha == 0: return sigma_s_gp * np.exp(u - up)

            #elif up - np.log(1 / alpha) <= u <= up: return (sigma_s_gp / (1 - alpha)) * np.exp(up - u)
            elif up - np.log(1 / alpha) <= u <= up: return (sigma_s_gp / (1 - alpha)) * np.exp(u - up)

            return 0

        def numerator_integrand(u, up, mu, l, g, gp, alpha, phi_u):
            return sigma_s_kernel(u, up, sigma_s_gp, alpha) * mu[l,g,gp] * phi_u[gp] * np.exp(-up)
#            if (l > 0 and g != gp): return (integrand * np.sum(mu[l,g,g:gp]))
#            else: return integrand
            return integrand

        def inner_integral(up, mu, l, g, gp, alpha, phi_u):
            if alpha > 0:
                u_lower = max(u_min, up - np.log(1 / alpha))
                u_upper = min(u_max, up)

            #else: u_lower, u_upper = u_min, up
            else: u_lower, u_upper = u_min, u_max

            return quad(numerator_integrand, u_lower, u_upper, args=(up, mu, l, g, gp, alpha, phi_u))[0]
    
        numerator = quad(inner_integral, up_min, up_max, 
                        args=(mu, l, g, gp, self.alpha(A), self.p0))[0]
        denominator = quad(lambda up: self.p0[gp] * np.exp(-up), up_min, up_max)[0]
    
        return numerator / denominator if denominator > 0 else 0
    
    def g_min_vec_fn(self,A):
        g_min_vec = np.zeros_like(self.groups, dtype = int)
        for g in range(self.groups.size): g_min_vec[g] = self.group_bound(A,g)
        return g_min_vec

    def calc_xs_l(self,sigma_s0,A):
        """
        build full gxg matrix of scattering xs's for l = [0,3]

        Parameters:
        sigma_s_0 (vector): l = 0 xs's for a given material
        A (int): Mass Ratio

        Returns: 
        3D matrices (leg_order x G x G): gtg scattering xs's 
        """
        G = self.groups.size
        sigma_gtg = np.zeros((self.leg_order, G, G))

        mu = self.compute_mu_matrix(self.groups, A)  

        flip_E = np.flip(self.E)
        for l in range(self.leg_order):
            P_l_mu = legendre(l)(mu)
    
            for g in range(G):
                gmin = self.g_min_vec[g]
                for gp in range(g,self.group_bound(A,g)):  
                    sigma_gtg[l,g,gp] = P_l_mu[g, gp] * sigma_s0[gp] * self.p0[gp] * self.E[gp]
#                    sigma_gtg[l,g,gp] = (P_l_mu[g, gp] * sigma_s0[gp] * self.p0[gp] * self.E[gp] / 
#                                    np.sum(self.p0[g:gmin]) * np.sum(self.E[g:gmin]))
    
        for g in range(G):
            for l in range(self.leg_order):
                norm = np.sum(sigma_gtg[l, g, :])
                target = sigma_s0[g] * self.xs_fraction(flip_E[g], A)[l]
                if norm > self.tol:
                    sigma_gtg[l, g, :] *= target / norm
                else:
                    sigma_gtg[l, g, :] = 0.0
   
        sigma_gtg[sigma_gtg <= self.tol] = 0.0
    
        # Save to HDF5
        my_str = "H" if A == 1 else "U"
        with h5py.File(f"{scratch_dir}/sigma_s_{my_str}.h5", "w") as f:
            for l in range(self.leg_order):
                f.create_dataset(f"sigma_s{l}", data=sigma_gtg[l])
    
        return sigma_gtg
        """
        # init xs and get mu
        sigma_gtg = np.zeros((self.leg_order, self.groups.size, self.groups.size))
        mu = np.zeros_like(sigma_gtg)
        mu[0, :, :] = 1 
        flip_E = np.flip(self.E)

        mu[1,:,:] = self.calc_mu(A, sigma_s0, 1, None)
        for l in range(2, self.leg_order): mu[l,:,:] = self.calc_mu(A, sigma_s0, l, mu[1,:,:])

        for g in range(self.groups.size):
            print(g)
            for gg in range(g, self.group_bound(A,g)):
                for l in range(self.leg_order):
                    sigma_gtg[l,g,gg] = self.compute_scattering(g, gg, sigma_s0 
                                        * self.xs_fraction(flip_E[g],A)[l], A, l, mu)

        # last group
        for i in range(self.leg_order): sigma_gtg[i,:,-1] = sigma_gtg[i,:,-2]

        # normalize to sigma_s0 frac
        for g in range(self.groups.size - 1):
            for l in range(self.leg_order):
                #sigma_gtg[l,g,:] *= sigma_s0[g] * self.xs_fraction(flip_E[g],A)[l] / np.sum(sigma_gtg[l,g,:])
                sigma_gtg[l,g,:] *= self.xs_fraction(flip_E[g],A)[l] / np.sum(sigma_gtg[l,g,:])

        # last group
        for l in range( self.leg_order): sigma_gtg[l,-1,-1] = sigma_s0[-1] * self.xs_fraction(flip_E[-1],A)[l] 

        sigma_gtg[sigma_gtg <= self.tol] = 0

        # save gtg scattering xs's
        my_str = "H" if A == 1 else "U"

        with h5py.File(f"{scratch_dir}/sigma_s_{my_str}.h5", "w") as f:
            f.create_dataset("sigma_s0", data=sigma_gtg[0,:,:])
            f.create_dataset("sigma_s1", data=sigma_gtg[1,:,:])
            f.create_dataset("sigma_s2", data=sigma_gtg[2,:,:])
            f.create_dataset("sigma_s3", data=sigma_gtg[3,:,:])

        return sigma_gtg
        """

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
        I_vals = [np.zeros_like(self.groups) for _ in range(self.leg_order)]
#        I_vals = np.zeros((self.leg_order,self.groups,self.groups))
        S_vals = self.calc_xs_l(sigma_s0, A)

#        gridwidths = np.diff(np.flip(self.E))
#        for g in range(self.groups.size -1):
#            for l in range(self.leg_order):
#                I_vals[l,g,:] = S_vals[l,g,:] * gridwidths[g]
        self.parallel_integrate(self.groups, S_vals, I_vals, self.gridspace, self.g_min_vec, self.leg_order)

        # Deallocate S_vals
        self.deallocate([S_vals])

        #return self.calc_xs_l(sigma_s0, A)
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
        self.g_min_vec = self.g_min_vec_fn(A)
        # integration matrix corresponding to eqn 4
        I_vals = self.integrate_Sn(sigma_s,A)
        # loss operator corresponding to eqn 5
        L_vals = [self._Ln(n, sigma_t, I_vals[n]) for n in range(self.leg_order)]

        self.deallocate(I_vals)
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
        L_vals_U  = self.build_Ln(self.AU,self.sigma_s_U,self.sigma_t_U)
        # send to csr to save memory
        #L_U_sparse = [csr_matrix(matrix) for matrix in L_vals_U]
#        print("H-1 Loss Operators")
#        L_vals_H = self.build_Ln(self.AH,self.sigma_s_H,self.sigma_t_H)
        ## send to csr to save memory
        #L_H_sparse = [csr_matrix(matrix) for matrix in L_vals_H]
        print(f"Loss Matrices Computed in {np.round(time.time() - t1, 5)}s")

        # do sum on csr
        #sparse_sum = [L_U_sparse[i] + L_H_sparse[i] for i in range(len(L_U_sparse))]
        # reconstruct the full matrix
#        self.L0, self.L1, self.L2, self.L3 = [L_vals_U[i] + L_vals_H[i] for i in range(4)]
        self.L0, self.L1, self.L2, self.L3 = [L_vals_U[i] for i in range(4)]
#        plt.figure()
#        plt.imshow(np.log10(self.L0), origin='upper')
#        plt.colorbar()
#        plt.show()
#        plt.savefig("L0.png")
#        plt.figure()
#        plt.imshow(self.L1, origin='upper')
#        plt.colorbar()
#        plt.savefig("L1.png")
#        plt.figure()
#        plt.imshow(self.L2, origin='upper')
#        plt.colorbar()
#        plt.savefig("L2.png")
#        plt.figure()
#        plt.imshow(self.L3, origin='upper')
#        plt.colorbar()
#        plt.savefig("L3.png")
        #self.L0, self.L1, self.L2, self.L3 = [matrix.toarray() for matrix in sparse_sum]
        # deallocate unnecessary memory
        self.deallocate(L_vals_U)
        #self.deallocate(L_U_sparse)
#        self.deallocate(L_vals_H)
        #self.deallocate(L_H_sparse)

        # compute LHS and RHS
        print("phi0 LHS")
        I = np.eye(self.groups.size)
        B4 = self.B2 ** 2 * I
        LHS = (9 * B4 * I + self.B2 * (self.L3 @ self.L2 + (9 * self.L1 + 4 * self.L3) * self.L0) 
                    + self.L3 @ self.L2 @ self.L1 @ self.L0)
        print("phi0 RHS")
        RHS = ((self.L3 @ self.L2 @ self.L1 + self.B2 * (9 * self.L1 + 4 * self.L3)) @ self.chi)
        # account for bin widths
        #W = np.diag(np.gradient(self.groups))  # lethargy bin widths
        #LHS = W @ LHS @ W
        #RHS = W @ RHS

        # matrix properties
        if properties: self.print_mat_properties(LHS)

        # Ax = b
        self.phi0 = np.zeros_like(self.groups)
        self.phi0 = (self.jacobi_parallel(LHS,RHS,self.phi0) 
                        if self.is_diagonally_dominant(LHS) 
                            else np.linalg.solve(LHS,RHS))
        print(self.phi0)

#        self.phi0 /= np.trapz(self.phi0, x=self.E)

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
        RHS = (-9 * self.B2 * self.phi0 + (9 * self.L1 + 4 * self.L3) 
                @ (self.L0 @ self.phi0 - self.chi)) / 2 # vector

        W = np.diag(np.gradient(self.groups))  # lethargy bin widths
        LHS = W @ LHS @ W
        RHS = W @ RHS

        if properties: self.print_mat_properties(LHS)

        # Ax = b
        self.phi2 = np.zeros_like(self.groups)
        self.phi2 = (self.jacobi_parallel(LHS,RHS,self.phi2) 
                        if self.is_diagonally_dominant(LHS) 
                            else np.linalg.solve(LHS,RHS))
#        self.phi2 /= np.trapz(self.phi2, x=self.E)

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
        u (int): Atomic Number
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
        plt.plot(np.flip(self.E),self.sigma_t_U,label=r'$\Sigma_t^U$')
        plt.plot(np.flip(self.E),self.sigma_t_H,label=r'$\Sigma_t^H$')
        plt.xlabel("E")
        plt.ylabel(r"$\Sigma_t$")
        plt.title(f"XS's for {self.groups.size} groups")
        plt.yscale("log")
        plt.xscale("log")
        plt.legend()
        plt.grid(True, which='both')
        plt.savefig(f"results/charts/xs_t_{self.groups.size}.png")

    def plot_fluxes(self):
        # plot phi0 and phi2
        plt.figure()
        plt.plot(np.flip(self.E),self.phi0,label=r'$\phi_0$')
        plt.title(r'$\phi_0(E)$')
        plt.xlabel('E')
        plt.ylabel(r'$\phi_0$')
        plt.xscale('log')
        plt.grid(True, which='both')
        plt.legend()
        plt.savefig(f'results/charts/phi0.png')

        plt.figure()
        plt.plot(np.flip(self.E),np.abs(self.phi2),label=r'$\phi_2$')
        plt.title(r'$|\phi_2(E)|$')
        plt.xlabel('E')
        plt.yscale('log')
        plt.ylabel(r'$\phi$')
        plt.xscale('log')
        plt.grid(True, which='both')
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

        if n == 0:
            Linv = sp.linalg.inv(self.L1)
            Phi  = self.Phi0

        else: 
            Linv = sp.linalg.inv(self.L3)
            Phi  = self.Phi2

        # operator on Phi_n(u)
        M = np.matmul(Linv,Phi)

        @staticmethod
        @njit(parallel=True)
        def parallel_coef_d(D,g_min_vec,M,groups,gridwidth,Phi):
            for i in prange(D.shape[0]): 
                num = np.zeros((g_min_vec[i] - i))

                # numerator integral
                for j in range(i,g_min_vec[i]): 
                    num[i-j] += (M[j] * np.exp(groups[j]) * gridwidth)

                # accumulation matrix flipped
                # not multiplying by self.E[i], it cancels out
                num = np.flip(num)
                # construct the diffusion coef matrix
                D[i,i:g_min_vec[i]] = num / (np.sum(Phi[i:g_min_vec[i]] * gridwidth * np.exp(groups[i]))) 
            
            return D

        D = parallel_coef_d(D,self.g_min_vec,M,self.groups,self.gridspace,Phi)
    
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

        @staticmethod
        @njit(parallel=True)
        def parallel_sigma_upd(Phi,g_min_vec,sigma_new,sigma_x,groups,gridwidth):
            for i in prange(Phi.size):
                g_min = g_min_vec[i]
                den = np.zeros((g_min - i))
                for j in range(i,g_min):
                    # note: E0 cancels out in division
                    sigma_new[i] += sigma_x[j] * Phi[j] * np.exp(groups[i]) * gridwidth
                    # denominator in lethargy space
                    den[i-j]      = np.exp(groups[j]) * gridwidth
                sigma_new[i] /= np.sum(Phi[i:g_min] * den)

            return sigma_new

        sigma_new = parallel_sigma_upd(Phi,self.g_min_vec,sigma_new,sigma_x,self.groups,self.gridspace)

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
        # read scattering xs's from .h5
        my_str = "H" if A == 1 else "U"
        sigma_sl = self.read_sigma_s(l,A)

        # in integetral, E0 dependence cancels out
        M = np.matmul(sigma_sl,Phi)

        @staticmethod
        @njit(parallel=True)
        def parallel_sigma_s_upd(Phi,g_min_vec,M,groups,gridwidth,sigma_sl):
            for i in prange(Phi.size):
                g_min = int(g_min_vec[i])
                num = np.zeros((g_min - i))
                # integrate
                for j in range(i,g_min):
                    num[i-j] += (M[j] * np.exp(groups[j]) * gridwidth)
                num = np.flip(num)
                den = np.sum(Phi[i:g_min] * gridwidth)
                sigma_sl[i,i:g_min] = num/den
                
            return sigma_sl

        sigma_sl = parallel_sigma_s_upd(Phi,self.g_min_vec,M,self.groups,self.gridspace,sigma_sl)

        return sigma_sl

    def run(self, properties, from_h5):
        """
        Run the complete SP3 calculation process.
        """
        if from_h5 == False:
            print("Starting phi0 calculation...")
            st = time.time()
            self.initial_flux()
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
            assert 0 == 1

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
        print("Log-Condition Number:", np.linalg.cond(LHS))
        print("Any NaNs or Infs in LHS?", np.any(np.isnan(LHS)) or np.any(np.isinf(LHS)))
        print("Diagonally dominant: ", self.is_diagonally_dominant(LHS))

    # helper functions and staticmethods

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

    @staticmethod
    def xs_fraction(E, A):
        """
        Returns fractions for sigma_s0, sigma_s1, sigma_s2, sigma_s3 based on energy E (in eV) for U-238.
        Adjusted for hyperfine group structure with detailed resonance treatment.
        """
        if A > 1:
            if E > 1e6:  return np.array([0.5, 0.3, 0.15, 0.05])
            elif E > 2e4:  return np.array([0.8, 0.15, 0.04, 0.01])
            elif E > 1e3: return np.array([0.75, 0.18, 0.06, 0.01])
            elif E > 10: return np.array([0.85, 0.12, 0.02, 0.01])
            else: return np.array([0.92, 0.05, 0.02, 0.01])

        else:
            if E > 1e6: return np.array([0.6, 0.3, 0.08, 0.02])  
            elif E > 1e5: return np.array([0.7, 0.2, 0.07, 0.03])
            elif E > 1e3: return np.array([0.9, 0.08, 0.015, 0.005])  
            else: return np.array([0.98, 0.015, 0.004, 0.001])  
    
    @staticmethod
    def jacobi_update(A, b, x, x_new, i):
        """
        helper function for the jacobi loop. Not sure what this is doing
        """
        row_sum = np.dot(A[i, :], x)  # Compute the row sum
        x_new[i] = (b[i] - (row_sum - A[i, i] * x[i])) / A[i, i]  # Update x[i]
    
    @staticmethod
    @njit(parallel=True)
    def parallel_integrate(groups, S_vals, I_vals, gridwidth, g_min_vec, leg_order):
        """
        Helper function to perform the integration in parallel.
        """
        for i in prange(groups.size):
            g_min = g_min_vec[i]
    
            for j in range(i, g_min):
                for l in range(leg_order):
                    val = S_vals[l, i, j]
                    if np.isfinite(val):
                        I_vals[l][i] += val * gridwidth

#        for i in prange(groups.size):
#            g_min = g_min_vec[i]
#            for j in range(i, g_min):
##                g_bound = min(g_min, groups[-1])
##                factor = (g_bound - g_min) + (g_min - j)
#
#                for l in range(leg_order):
##                    I_vals[l][i] += S_vals[l, i, j] * factor * gridwidth
#                    I_vals[l][i] += S_vals[l, i, j] * gridwidth

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
        return np.diag((2 * l + 1) * (sigma - I))

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
    def compute_mu_matrix(lethargy, A):
        """
        Compute scattering cosine matrix mu(u, u') for all g, g' combinations.
        """
        u = lethargy.reshape(-1, 1)     # (G,1)
        up = lethargy.reshape(1, -1)    # (1,G)
        mu = ((A + 1) / 2) * np.exp((up - u) / 2) - ((A - 1) / 2) * np.exp((u - up) / 2)
        mu = np.clip(mu, -1, 1)

        return mu

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

