from __init__ import *

class Sp3:
    def __init__(self, xs_H, xs_U, sigma_f, chi, B2, NH):
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
        self.T         = 293 # degrees Kelvin
        self.k         = 8.617e-5  # eV/K (Boltzmann constant)
        self.kT        = self.k * self.T
        self.g_min_vec = None
        self.NH        = NH

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

            if i == 0: phi[i] = chi[i]/Sigma_R

            else:
                #Scattering Source for Hydrogen
                scatH += scat_source(sigma_sh,phi,self.E,self.gridspace,i,alpha = 0)
    
                #Scattering Source for Uranium
                scatU += scat_source(sigma_su,phi,self.E,self.gridspace,i,alpha = alpha)

                #subtract off the contribution from lesser lethargy bins
                if i > g_min:
                    h = i - g_min
                    scatU -= (sigma_su[h]   / (1 - alpha) * phi[h] * self.E[h]   * self.gridspace)

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
        Pl_mu = np.zeros((self.leg_order,self.groups.size,self.groups.size))
        sigma_gtg = np.zeros_like(Pl_mu)

        mu = self.compute_mu_matrix(self.groups, A)  
        for l in range(self.leg_order): Pl_mu[l,:,:] = legendre(l)(mu)

        for i in range(self.groups.size):
            for j in range(self.groups.size):
                if i == j: 
                    Pl_mu[1,i,j] = 0
                    Pl_mu[2,i,j] = -.5
                    Pl_mu[3,i,j] = 0

        @njit(parallel=True)
        def parallel_xs_gen(order, groups, gmax_vec, alpha, sigma_gtg, sigma_s0, Pl_mu):
            G = groups.size
            du = groups[1] - groups[0]   
            for l in range(order):      
                for gp in prange(G):    
                    for g in range(gp, gmax_vec[gp]):
                        sigma_gtg[l,gp,g] = (sigma_s0[gp] * np.exp(groups[gp] - groups[g]) * Pl_mu[l, gp, g] * du
                                                / (1 - alpha))

                sigma_gtg[l,-1,-1] = sigma_gtg[l,-2,-2] + sigma_gtg[l,-2,-1]

            return sigma_gtg

        sigma_gtg = parallel_xs_gen(self.leg_order, 
                        self.groups, 
                        self.g_min_vec_fn(A),
                        self.alpha(A),
                        sigma_gtg, 
                        sigma_s0, 
                        Pl_mu)

        rowsum = np.zeros_like(self.groups)
        l2 = 0
        def rowsum_norm(sigma_gtg,l,rowsum,sigma_s0,l2):
            xs = sigma_gtg[l,:,:]

            for i in range(rowsum.size):
                rowsum[i] += np.sum(xs[i,:])

            l2 += np.linalg.norm(rowsum - sigma_s0,ord = 2)
            return rowsum, l2

        for l in range(self.leg_order):
            rowsum, l2 = rowsum_norm(sigma_gtg,l,rowsum,sigma_s0,l2)
        print(f"L2 Norm on xs's: {l2}")

        """
        # checking mu
        g_min_vec = self.g_min_vec_fn(A)
        @njit(parallel=True)
        def check_mu(mu,g_min_vec):
            print("checking mu")
            for i in prange(mu.shape[0]):
                for j in range(i,g_min_vec[i]):
                    if np.abs(mu[i,j]) > 1: raise ValueError(f"Mu[{i,j}] is beyond range. {mu[i,j]}")
            print("mu checked")
        check_mu(mu,g_min_vec)
        def scattering_legendre_moments(A, sigma_s, npts=2000):
            #Numerically compute sigma_s0..sigma_s3 for isotropic scattering in CM.
            #Stable even for very heavy targets.
            #
            #Parameters
            #----------
            #A : float
            #    Target-to-neutron mass ratio
            #sigma_s : float or ndarray
            #    Scattering cross section [barns] (incident group or energy)
            #npts : int
            #    Number of quadrature points for integration (default 2000)
            #    
            #Returns
            -------
            sigma_s0, sigma_s1, sigma_s2, sigma_s3
            # Gauss-Legendre quadrature points over mu_cm in [-1,1]
            mu_cm, w = np.polynomial.legendre.leggauss(npts)
            
            # transform to lab cosine
            def mu_lab(mu): return mu if A == 1 else (mu + 1 / A) / (np.sqrt(1 + 1 / (A * A) + 2 * mu / A))
        
            muL = mu_lab(mu_cm)
            P0 = np.ones_like(muL)
            P1 = muL
            P2 = 0.5*(3*muL**2 - 1)
            P3 = 0.5*(5*muL**3 - 3*muL)
            
            # integrate each moment, factor 0.5 because CM distribution is isotropic: (1/2) over mu_cm
            mom0 = 0.5*np.sum(w*P0)
            mom1 = 0.5*np.sum(w*P1)
            mom2 = 0.5*np.sum(w*P2)
            mom3 = 0.5*np.sum(w*P3)

            # scale by (2l+1) to convert to sigma_sl
            sigma_s0 = sigma_s * mom0 * 1     
            sigma_s1 = sigma_s * mom1 * 3   
            sigma_s2 = sigma_s * mom2 * 5
            sigma_s3 = sigma_s * mom3 * 7

            sigma_s0 = np.clip(sigma_s0,0,np.max(sigma_s0))
            sigma_s1 = np.clip(sigma_s1,0,np.max(sigma_s0))
            sigma_s2 = np.clip(sigma_s2,0,np.max(sigma_s0))
            sigma_s3 = np.clip(sigma_s3,0,np.max(sigma_s0))

            return sigma_s0, sigma_s1, sigma_s2, sigma_s3

        xs_0, xs_1, xs_2, xs_3 = scattering_legendre_moments(A,sigma_s0)
        sigma_sl = np.column_stack((xs_0,xs_1,xs_2,xs_3))
        """
        """
        #@njit(parallel=True)
        def parallel_xs_gen(order, groups, sigma_gtg, alpha, sigma_sl, Pl_mu, flip_E, sigma_s0, kT, A):
            G = groups.size
            du = np.empty(G, dtype=float)
            du = groups[1:] - groups[:-1]
            max_jump = np.inf if alpha == 0 else np.log(1.0/alpha)
            tiny = 1e-30
            for l in range(order):                 
                for gp in range(G): # incident 
                    rowsum = 0.0

                    for g in range(G): # outgoing
                        tmp = 0.0

                        # Downscatter (g >= gp) 
                        if g >= gp and (alpha == 0 or g < gp + int(max_jump/du_mean) + 1):
                            w = np.exp(groups[gp] - groups[g])   
                            tmp += w * Pl_mu[l, gp, g] * du[g]
        
                        # Upscatter (below 50 eV)
                        if g < gp and flip_E[g] < 50.0:
                            E = flip_E[g]
                            arg = np.sqrt(A * E / kT)
                            kernel = ((1 + kT / (2 * A * E)) * math.erf(arg)
                                      + np.sqrt(kT / (np.pi * A * E)) * np.exp(-A * E / kT))
                            tmp += Pl_mu[l, gp, g] * sigma_s0[g] * kernel
        
                        sigma_gtg[l, gp, g] = tmp
                        rowsum += tmp
        
                    # Normalize row to sigma_sl[gp,l] 
                    scale = 0 if rowsum < tiny else sigma_sl[gp, l] / rowsum
                    for g in range(G): sigma_gtg[l, gp, g] *= scale
        
            return sigma_gtg
        sigma_gtg = parallel_xs_gen(self.leg_order,
                                    self.groups,
                                    sigma_gtg,
                                    self.alpha(A),
                                    sigma_sl,
                                    Pl_mu,
                                    np.flip(self.E),
                                    sigma_s0,
                                    self.kT,
                                    A,)
        print(sigma_gtg[0,:10,:10])
        print(sigma_gtg[1,:10,:10])
        print(sigma_gtg[2,:10,:10])
        print(sigma_gtg[3,:10,:10])
        @njit(parallel=True)
        def parallel_xs_gen(order, groups, sigma_gtg, alpha, sigma_sl, Pl_mu):
            G = groups.size
            du = groups[1:] - groups[:-1]           # lethargy bin widths (length G)
            du = np.append(du, du[-1])
            du_mean = np.mean(du)
            max_jump = np.inf if alpha == 0 else np.log(1.0/alpha)
            tiny = 1e-30
            for l in range(order):                    # l = 0..L
                for gp in prange(G):                  # incident bin
                    # find allowed outgoing band: u_g - u_gp in [0, ln(1/alpha)]
                    gmax = min(G, gp + int(np.floor(max_jump/du_mean)) + 1) if alpha != 0 else G 
        
                    # unnormalized row build
                    rowsum = 0.0
                    for g in range(gp, gmax):
                        w = np.exp((groups[gp] - groups[g]))      
                        #w = np.exp(-(groups[gp] - groups[g]))      
                        tmp = w * Pl_mu[l, gp, g] * du[g]          # midpoint rule
                        sigma_gtg[l, gp, g] = tmp
                        rowsum += tmp

                    scale = sigma_sl[gp,l] / rowsum if rowsum > tiny else 0
                    #scale = (2*l + 1) * sigma_sl[gp,l] / rowsum if rowsum > tiny else 0

                    for g in range(gp, gmax): sigma_gtg[l, gp, g] *= scale

                sigma_gtg[l,-1,-1] = sigma_gtg[l,-2,-2] + sigma_gtg[l,-2,-1]

            return sigma_gtg

        sigma_gtg = parallel_xs_gen(self.leg_order,
                                    self.groups,
                                    sigma_gtg,
                                    self.alpha(A),
                                    sigma_sl,
                                    Pl_mu,)
        @njit(parallel=True)
        def parallel_xs_gen(order,groups,g_min_vec,sigma_gtg,alpha,sigma_s0,Pl_mu):
            for l in range(order):
                for gp in prange(groups.size):
                    for g in range(gp,g_min_vec[gp]):
                        sigma_gtg[l,gp,g] = ((np.exp(groups[gp] - groups[g])) / (1 - alpha) * 
                                            (sigma_s0[gp] * Pl_mu[l,gp,g]))
        
            return sigma_gtg

        sigma_gtg = parallel_xs_gen(self.leg_order,
                                    self.groups,
                                    self.g_min_vec_fn(A),
                                    sigma_gtg,
                                    self.alpha(A),
                                    sigma_s0,
                                    Pl_mu,)
        # UPSCATTER INCLUDED
        for l in range(self.leg_order):
            P_l_mu = legendre(l)(mu)
    
            for gp in range(G):
                for g in range(gp, self.group_bound(A,gp)):  
                    # use maxwellian to get upscatter contributions
                    #if gp <= g and flip_E[g] < 50: # upscattering at 50 eV
#                    if flip_E[g] < 50: # upscattering at 50 eV
#                        kernel = ((1 + self.kT / (2 * A * flip_E[g])) * erf(np.sqrt(A * flip_E[g] / self.kT))
#                                    + np.sqrt(self.kT / (np.pi * A * flip_E[g])) * np.exp(-A*flip_E[g] / self.kT))
#                        sigma_gtg[l,gp,g] = P_l_mu[gp,g] * sigma_s0[g] * self.p0[g] * kernel

#                    elif gp > g and flip_E[g] > 50: sigma_gtg[l,gp,g] = 0 # no upscatter contribution

#                    else: sigma_gtg[l,gp,g] = P_l_mu[gp,g] * sigma_s0[g] * self.p0[g] * flip_E[g]
                    sigma_gtg[l,gp,g] = P_l_mu[gp,g] * sigma_s0[gp] * self.p0[g] * (flip_E[gp] - flip_E[g]) 

        for l in range(self.leg_order):
            P_l_mu = legendre(l)(mu)
            for g in range(G):
                for gg in range(G):
                    norm = np.sum(sigma_gtg[l, g, :])
                    sigma_gtg[l,g,gp] = (P_l_mu[g,gg] * sigma_s0[g]) / norm if norm > self.tol else 0
                target = sigma_s0[g] * self.xs_fraction(flip_E[g], A)[l]
                sigma_gtg[l,g,:] *= target/norm if norm > self.tol else 0

        row_sum = np.zeros_like(sigma_s0)
        for n in range(self.leg_order): row_sum += np.sum(sigma_gtg[n,:,:], axis = 1)
        print(row_sum-sigma_s0)
        norm = self.L2_norm(sigma_s0,row_sum)
        print(f'Norm {norm}')
        """
        plt.figure(figsize=(6, 5))
        im = plt.imshow(
            sigma_gtg[0, :, :],
            cmap='viridis',
            origin='upper',
            norm=LogNorm()
        )
        plt.colorbar(im, label=r'$\Sigma_{gtg}$') 
        plt.title("Scattering Xs's")
        #plt.show()
        plt.savefig(f'results/charts/sigma_gtg_{A}.png')
        plt.close()
        plt.clf()
        
        plt.figure()
        plt.plot(np.flip(self.E),sigma_s0,label = 'sigma_s0')
        plt.plot(np.flip(self.E),rowsum,label = 'sigma_gtg')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.savefig(f"results/charts/parallel_xs_gtg_{A}.png")
        #plt.show()
        plt.close()
        plt.clf()
#        sigma_gtg[sigma_gtg <= self.tol] = 0.0

        # Save to HDF5
        my_str = "H" if A == 1 else "U"
        with h5py.File(f"{scratch_dir}/sigma_s_{my_str}_{self.NH}.h5", "w") as f:
            for l in range(self.leg_order):
                f.create_dataset(f"sigma_s{l}", data=sigma_gtg[l])
    
        return sigma_gtg

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
        S_vals = self.calc_xs_l(sigma_s0, A)

        self.parallel_integrate(self.groups, S_vals, I_vals, self.gridspace, self.g_min_vec, self.leg_order)

        # Deallocate S_vals
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
        self.g_min_vec = self.g_min_vec_fn(A)
        # integration matrix corresponding to eqn 4
        I_vals = self.integrate_Sn(sigma_s,A)
        # loss operator corresponding to eqn 5
        L_vals = [self._Ln(n, sigma_t, I_vals[n]) for n in range(self.leg_order)]

        self.deallocate(I_vals)
        return L_vals

    def calc_Ln(self):
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
        L_U_sparse = [csr_matrix(matrix) for matrix in L_vals_U]

        print("H-1 Loss Operators")
        L_vals_H = self.build_Ln(self.AH,self.sigma_s_H,self.sigma_t_H)
        # send to csr to save memory
        L_H_sparse = [csr_matrix(matrix) for matrix in L_vals_H]

        print(f"Loss Matrices Computed in {np.round(time.time() - t1, 5)}s")

        # do sum on csr
        sparse_sum = [L_U_sparse[i] + L_H_sparse[i] for i in range(len(L_U_sparse))]
        # reconstruct the full matrix
        self.L0, self.L1, self.L2, self.L3 = [L_vals_U[i] + L_vals_H[i] for i in range(4)]

        self.deallocate(L_vals_U)
        self.deallocate(L_U_sparse)
        self.deallocate(L_vals_H)
        self.deallocate(L_H_sparse)

        return self.L0, self.L1, self.L2, self.L3

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
        for i in range(self.phi0.size):
            g_min = self.group_bound(A,i)
            for j in range(i,g_min): Q[i] += sigma_f[j] * self.phi0[j] * np.exp(self.groups[i])
            Q[i] *= (self.chi[i] * self.E[i])

        return Q

    def plot_flux_diff(self,phi1,phi2):
        # compare the traditional to new method
        plt.figure()
        plt.plot(np.flip(self.E),phi1,label='Scattering Source')
        plt.plot(np.flip(self.E),phi2,label='Sp3')
        plt.title("Hyperfine Slowing-Down Flux Comparison")
        plt.ylabel(r"$\phi (E) (n/cm^2)$")
        plt.xscale('log')
        plt.xlabel("Energy (eV)")
        plt.legend()
        plt.grid(True,which='both')
        plt.savefig("results/charts/order_comp.png")
        plt.clf()

        plt.figure()
        plt.plot(np.flip(self.E),phi1 - phi2)
        plt.title("Hyperfine Slowing-Down Flux Difference")
        plt.ylabel(r"$\phi (E) (n/cm^2)$")
        plt.xscale('log')
        plt.xlabel("Energy (eV)")
        plt.grid(True,which='both')
        plt.savefig("results/charts/flux_difference.png")

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
        plt.savefig(f'results/charts/phi0_{self.NH}.png')

        plt.figure()
        plt.plot(np.flip(self.E),np.abs(self.phi2),label=r'$\phi_2$')
        plt.title(r'$|\phi_2(E)|$')
        plt.xlabel('E')
        plt.yscale('log')
        plt.ylabel(r'$\phi_2$')
        plt.xscale('log')
        plt.grid(True, which='both')
        plt.legend()
        plt.savefig(f'results/charts/phi2_{self.NH}.png')

        plt.figure()
        plt.plot(np.flip(self.E),self.phi2,label=r'$\phi_2$')
        plt.title(r'$\phi_2(E)$')
        plt.xlabel('E')
        plt.ylabel(r'$\phi_2$')
        plt.xscale('log')
        plt.grid(True, which='both')
        plt.legend()
        plt.savefig(f'results/charts/phi2_{self.NH}_lin.png')

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
        g_min_vec = self.g_min_vec_fn(A)

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

        return parallel_coef_d(D,g_min_vec,M,self.groups,self.gridspace,Phi)

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

        g_min_vec = self.g_min_vec_fn(A)

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

        return parallel_sigma_upd(Phi,g_min_vec,sigma_new,sigma_x,self.groups,self.gridspace)

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
        sigma_sl = self.read_sigma_s(l,A,self.NH)

        # in integetral, E0 dependence cancels out
        M = np.matmul(sigma_sl,Phi)
        g_min_vec = self.g_min_vec_fn(A)

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

        return parallel_sigma_s_upd(Phi,g_min_vec,M,self.groups,self.gridspace,sigma_sl)

    def calc_phi_tt(self):
        """
        Calculate the flux using operators in TT format

        Returns:
        phi0 and phi2 in TT format
        """
        # convert Loss, chi, and scalers to Tensorly
        self.L0 = self.npy_to_tensor(self.L0)
        self.L1 = self.npy_to_tensor(self.L1)
        self.L2 = self.npy_to_tensor(self.L2)
        self.L3 = self.npy_to_tensor(self.L3)
        I = np.eye(self.groups.size)
        B4 = self.B2 ** 2 * I
        B4 = self.npy_to_tensor(B4)
        groups = self.npy_to_tensor(self.groups)

        # solve for phi0
        LHS = (9 * B4 * I + self.B2 * (self.L3 @ self.L2 + (9 * self.L1 + 4 * self.L3) * self.L0) 
                    + self.L3 @ self.L2 @ self.L1 @ self.L0)
        RHS = ((self.L3 @ self.L2 @ self.L1 + self.B2 * (9 * self.L1 + 4 * self.L3)) @ self.chi)
        self.phi0 = np.linalg.solve(LHS,RHS)
        self.phi0 /= np.sum(self.phi0) # normalize

        # solve for phi2
        LHS = self.L3 @ self.L2 
        RHS = .5 * (-9 * self.B2 * self.phi0 + (9 * self.L1 + 4 * self.L3) 
                @ (self.L0 @ self.phi0 - self.chi))  
        self.phi2 = np.linalg.solve(LHS,RHS)
        self.phi2 /= np.sum(self.phi2) # normalize

    def run(self, properties, from_h5):
        """
        Run the complete SP3 calculation process.
        """
        st = time.time()
        self.initial_flux()
        if from_h5 == False:
            print("Starting phi0 calculation...")
            self.calc_Ln()
            with h5py.File(f"{scratch_dir}/Ln_{self.NH}.h5", "w") as f:
                f.create_dataset("L0", data=self.L0)
                f.create_dataset("L1", data=self.L1)
                f.create_dataset("L2", data=self.L2)
                f.create_dataset("L3", data=self.L3)
            self.calc_phi_tt() 
            print(f"phi calculation: {np.round(time.time()-st,5)}")
            print("Plotting...")
            self.plot_fluxes()

            print("Starting Phi0 and Phi2 calculation...")
            self.calc_Phi()

            print("Saving Data...")
            df = pd.DataFrame({'phi0': self.tensor_to_npy(self.phi0), 'phi2': self.tensor_to_npy(self.phi2), 
                               'Phi0': self.tensor_to_npy(self.Phi0), 'Phi2': self.tensor_to_npy(self.Phi2)})
            df.to_hdf(f"{scratch_dir}/fluxes_{self.NH}.h5", key="df", mode="w", format="table")
            print("Data Saved")

        else:
            print("Reading Data From File...")
            with h5py.File(f"{scratch_dir}/Ln_{self.NH}.h5", "r") as f:
                self.L0 = f["L0"][:]
                self.L1 = f["L1"][:]
                self.L2 = f["L2"][:]
                self.L3 = f["L3"][:]
            print("Ln Read!")

            df = pd.read_hdf(f"{scratch_dir}/fluxes_{self.NH}.h5", key="df")
            self.phi0 = df['phi0'].to_numpy()
            self.phi2 = df['phi2'].to_numpy()
            self.Phi0 = df['Phi0'].to_numpy()
            self.Phi2 = df['Phi2'].to_numpy()
            print("Phi Read!")

        self.eval_fluxes()
        assert 0 == 1
        self.eval_xs_t()
        self.eval_diff_matrix()
        self.eval_xs_s()

        print("Calculation completed.")

    def eval_fluxes(self):
        """Compute L2 norm of fluxes"""
        print("Evaluate Fluxes..")
        phi0_norm = self.normalize(self.phi0)
        p0_norm   = self.normalize(self.p0)
        l2_norm   = self.L2_norm(phi0_norm,p0_norm)
        print(f"L2 norm: {l2_norm}")
        self.plot_flux_diff(p0_norm,phi0_norm)

    def eval_xs_t(self):
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

        print("Evaluating Updated XS's")
        xs_t_H = new_xs["xs_t_H_0"] + new_xs["xs_t_H_2"]
        xs_t_U = new_xs["xs_t_U_0"] + new_xs["xs_t_U_2"]
        plt.figure()
        plt.plot(np.flip(self.E),xs_t_H,label='Flux Weighted')
        plt.plot(np.flip(self.E),self.sigma_t_H,label='Original')
        plt.xlabel("E")
        plt.ylabel(r"$\Sigma_t$")
        plt.title(f"XS's for {self.groups.size} groups")
        plt.yscale("log")
        plt.xscale("log")
        plt.legend()
        plt.grid(True, which='both')
        plt.savefig("results/charts/updated_xs_t.png")
        l2_sigma_t_H = self.L2_norm(xs_t_H,self.sigma_t_H)
        l2_sigma_t_U = self.L2_norm(xs_t_U,self.sigma_t_U)
        print(f"L2 U: {l2_sigma_t_U}. L2 H: {l2_sigma_t_H}")

        # Save xs's to hdf5
        with h5py.File(f"{scratch_dir}/cross_sections_{self.NH}.h5", "w") as f:
            for key, value in new_xs.items(): f.create_dataset(key, data=value, compression="gzip")

    def eval_diff_matrix(self):
        """Calculate Diffusion coefficients and compare to reference solution"""
        # Generate diffusion coefficients
        st = time.time()
        print("Calculating Diffusion Coefficients for l = 0, l = 2...")
        with h5py.File(f"{scratch_dir}/diffusion_coefficients_{self.NH}.h5", "w") as f:
            for prefix, A in [("U", self.AU), ("H", self.AH)]:
                for l in [0, 2]:
                    D = self.diffusion_coef(A, l)
                    if np.any(np.isnan(D)): raise ValueError("D must be defined")
                    f.create_dataset(f"D{prefix}_{l}", data=D, compression="gzip", compression_opts=9)
        print(f"Time D Coef (new): {np.round(time.time() - st, 5)}s")

        st = time.time()
        print("Calculating Transport xs's and expected diffusion coefficients")
        sigma_s1_U = self.read_sigma_s(l=1, A=self.AU, NH=self.NH)
        sigma_s3_U = self.read_sigma_s(l=3, A=self.AU, NH=self.NH)
        D_U_tr = self.diff_matrix(self.sigma_t_U,sigma_s1_U,self.B2)
        D_U_tr+= self.diff_matrix(self.sigma_t_U,sigma_s3_U,self.B2)
        D_U = self.read_d_mat(self.NH,prefix = "U",l = 0)
        D_U+= self.read_d_mat(self.NH,prefix = "U",l = 2)
        l2_DU = self.L2_norm(D_U,D_U_tr)

        sigma_s1_H = self.read_sigma_s(l=1, A=self.AH, NH=self.NH)
        D_H_tr = self.diff_matrix(self.sigma_t_H,sigma_s1_H,self.B2)
        D_H = self.read_d_mat(self.NH,prefix = "H",l = 0)
        l2_DH = self.L2_norm(D_H,D_H_tr)
        print(f"Time D Coef (trad): {np.round(time.time() - st, 5)}s")
        print(f"L2 norm on diffusion coefs for U and H: {l2_DU},{l2_DH}")

    def eval_xs_s(self):
        # generate gtg scattering xs's for all materials
        print("Calculting Updated Group to Group Scattering Cross-Sections...")
        #new_xs_A_l_n
        mat_indx = [self.AH,self.AU]
        mat_name = ["H","U"]
        indices  = [(0, 0), (0, 2), (1, 0), (1, 2), (2, 0), (2, 2), (3, 0), (3, 2)]

        st = time.time()
        with h5py.File(f"{scratch_dir}/gtg_cross_sections_{self.NH}.h5", "w") as f:
            for A, name in zip(mat_indx, mat_name):
                for l, n in indices:
                    xs = self.sigma_s_update(A, l, n)
                    f.create_dataset(f"new_xs_{name}_{l}_{n}", data=xs, compression="gzip", compression_opts=9)
        print(f"Time gtg XS Update: {np.round(time.time() - st, 5)}s")

    # helper functions and static methods
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
    def tensor_to_npy(tensor): return tl.to_numpy(tensor)

    @staticmethod
    def npy_to_tensor(array): return tl.tensor(array)

    @staticmethod
    @njit(parallel=True)
    def parallel_integrate(groups, S_vals, I_vals, gridwidth, g_min_vec, leg_order):
        """
        Helper function to perform the integration in parallel.
        """
        for l in range(leg_order):
            for i in prange(groups.size):
                for j in range(i, g_min_vec[i]):
                    val = S_vals[l, i, j]
                    if np.isfinite(val):
                        I_vals[l][i] += val * gridwidth

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
    def read_sigma_s(l, A, NH):
        """Read only the S{l} dataset from the HDF5 file."""
        my_str = "H" if A == 1 else "U"
        with h5py.File(f"{scratch_dir}/sigma_s_{my_str}_{NH}.h5", "r") as f: 
            return np.array(f[f"sigma_s{l}"])  

    @staticmethod
    def read_d_mat(NH,prefix,l):
        """Read .h5 diffusion matrix for an isotope."""
        with h5py.File(f"{scratch_dir}/diffusion_coefficients_{NH}.h5", "r") as f: 
            return np.array(f[f"D{prefix}_{l}"])  

    @staticmethod
    def deallocate(my_list):
        """
        Free up memory because large matrices

        Parameters: 
        my_list (list): a list of n parameters

        Returns: 
        None
        """
        for obj in my_list: del obj 
        my_list.clear()  

    @staticmethod
    def print_mat_properties(LHS):
        """
        prints important matrix properties
        Parameters: LHS (matrix): matrix of interest
        """
        def is_diagonally_dominant(A):
            """
            checks if a matrix is diagonally dominant
            Parameters: A (matrix): matrix in question
    
            Returns: 
            bool: True if matrix is diagonally dominant
            """
            # loop over rows
            for i in range(A.shape[0]):
                row_sum = np.sum(np.abs(A[i])) - np.abs(A[i, i])  
                if np.abs(A[i, i]) < row_sum: return False
    
            return True

        print("Rank of LHS:", np.linalg.matrix_rank(LHS))
        print("Determinant of LHS:", np.linalg.det(LHS))
        print("Log-Condition Number:", np.linalg.cond(LHS))
        print("Any NaNs or Infs in LHS?", np.any(np.isnan(LHS)) or np.any(np.isinf(LHS)))
        print("Diagonally dominant: ", is_diagonally_dominant(LHS))

    @staticmethod
    def compute_mu_matrix(lethargy, A):
        """
        Compute scattering cosine matrix mu(u, u') for all g, g' combinations.
        """
        u = lethargy.reshape(1, -1)     
        up = lethargy.reshape(-1, 1)    
        return ((A + 1) / 2) * np.exp((up - u) / 2) - ((A - 1) / 2) * np.exp((u - up) / 2)

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

    @staticmethod
    def normalize(vec):
        assert vec.ndim == 1
        return vec / np.sum(vec) 

    @staticmethod
    def L2_norm(A,B):
        assert A.shape == B.shape
        return np.linalg.norm(A-B, ord=2)

    @staticmethod
    def diff_matrix(sigma_t,sigma_s1,B2): # sigma_s1 is a matrix
        """Traditional diffusion matrix calculation"""
        # gamma buckling correction factor
        def gamma_fn(B2,sigma_t): 
            # takes in the individual xs!
            if B2 == 0: return 1
    
            b = np.sqrt(np.abs(B2)) / sigma_t
    
            if B2 > 0: return b*np.arctan(b)/(3*(1-b**(-1)*np.arctan(b)))
    
            else:
                gamma = b*np.log((1+b)/(1-b))
                gamma /= 3*(-2+b**(-1)*np.log((1+b)/(1-b)))
                return gamma
    
        gamma_mat = np.zeros_like(sigma_t)
        for i in range(sigma_t.size): gamma_mat[i] = gamma_fn(B2,sigma_t[i])

        @njit(parallel=True)
        def parallel_sigma_tr(sigma_t,sigma_s1,B2,gamma_mat):
            # transport cross-section
            sigma_tr = np.zeros_like(sigma_s1)
            for i in prange(sigma_s1.shape[0]): # rows p0
                for j in range(sigma_s1.shape[1]): 
                    if i == j: sigma_tr[i,j] = gamma_mat[i] * sigma_t[i] - sigma_s1[i,j]
                    else: sigma_tr[i,j] = - sigma_s1[i,j]
        
            return sigma_tr

        # diffusion coef calculation
        sigma_tr = parallel_sigma_tr(sigma_t,sigma_s1,B2,gamma_mat)

        return np.linalg.inv(3*sigma_tr)

