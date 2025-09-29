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
        self.NH        = NH
        self.E0        = 1e7
        self.groups    = xs_U[:,0] 
        self.sigma_t_H = np.flip(xs_H[:, 1])
        self.sigma_s_H = np.flip(xs_H[:, 2]) # Hydrogen xs's
        self.sigma_t_U = np.flip(xs_U[:, 1])
        self.sigma_s_U = np.flip(xs_U[:, 2]) # U-238 xs's
        self.sigma_f   = np.flip(sigma_f) # fission xs's
        self.E         = np.flip(np.exp(self.groups))
        #self.E         = np.exp(-self.groups) * self.E0
        self.chi       = np.flip(chi[:, 1]) # fission spectrum
#        self.chi      /= np.trapz(self.chi,self.E) # normalize chi
        self.B2        = B2 # geometric buckling
        self.tol       = 1e-9 # Small value threshold
        self.gridspace = self.groups[1] - self.groups[0]
        self.leg_order = 4
        self.T         = 293 # degrees Kelvin
        self.k         = 8.617e-5  # eV/K (Boltzmann constant)
        self.kT        = self.k * self.T
        self.gmax_vec  = None
        self.ind       = (self.groups.size - 
                        (np.where(self.groups>0)[0][np.argmin(self.groups[self.groups>0])]))
        self.m         = 1.674923e-27

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
        
    def group_bound(self, A, g): return (self.groups.size if A == 1 
                else np.searchsorted(self.groups, self.groups[g] + np.log(1 / self.alpha(A))))

    def gmax_vec_fn(self,A):
        gmax_vec = np.zeros_like(self.groups, dtype = int)
        for g in range(self.groups.size): gmax_vec[g] = self.group_bound(A,g)
        return gmax_vec

    def sigma_sn_fn(self,A,mu,sigma_s0):
        """Calculate and return vectors 0 -> 3 for scattering xs's"""
        # look at ners 561 notes pages 11 and 12
        G = self.groups.size
        sigma_sn = np.zeros((self.leg_order,G))
        sigma_sn[0,:] = sigma_s0
        gmax_vec = self.gmax_vec_fn(A)

        @njit(parallel=True, fastmath=True)
        def get_mu_bar(groups,gmax_vec,mu = mu):
            mu_bar = np.zeros_like(groups)
            for i in prange(G):
                gmax = gmax_vec[i]
                counter = 0
                for j in range(i,gmax):
                    mu_bar[i] += mu[i,j]
                    counter += 1
                mu_bar[i] /= counter
            
            return mu_bar

        mu_bar = get_mu_bar(self.groups,self.gmax_vec_fn(A))

        sigma_sn[1,:] = sigma_s0 * mu_bar
        sigma_sn[2,:] = .5 * (3 * mu_bar * mu_bar - 1) * sigma_s0
        sigma_sn[3,:] = .5 * (5 * mu_bar**3 - 3 * mu_bar)  * sigma_s0

        @njit(parallel=True, fastmath=True)
        def thermal_xs(ind,G,leg_order,sigma_sn):
            thermal_weights = [.85,.1,.04,.01]
            for i in prange(ind,G):
                for l in range(leg_order):
                    sigma_sn[l,i] = thermal_weights[l] * sigma_s0[i]
                
        thermal_xs(self.ind,G,self.leg_order,sigma_sn)
        return sigma_sn

    def xs_generator(self, A, sigma_s0, nq = 64):
        G = self.groups.size
        sigma_gtg = np.zeros((self.leg_order,G,G))
        Pl_mu = np.zeros_like(sigma_gtg)
        sigma_s0 = self.sigma_s_H if (A == 1) else self.sigma_s_U
        du = self.groups[1] - self.groups[0]
        alpha = self.alpha(A)
        self.gmax_vec = self.gmax_vec_fn(A) 

        mu = self.compute_mu_matrix(self.groups,A,self.gmax_vec)
        sigma_sn = self.sigma_sn_fn(A,mu,sigma_s0)
#        for l in range(self.leg_order): Pl_mu[l] = legendre(l)(mu) # this works!

        # GL quadrature set
        t, w = leggauss(nq)

        @njit(inline='always')
        def clamp(x):
            if x < -1.0: return -1.0
            elif x > 1.0: return 1.0
            return x

#        du = self.gridspace
#        u_edges = np.empty(G + 1, dtype=self.groups.dtype)
#        u_edges = np.concatenate([[self.groups[0] - 0.5*du], self.groups[:-1] + 0.5*du])
#        u_edges = np.concatenate([[self.groups[0] - 0.5*du], self.groups[:-1] + 0.5*du, [self.groups[-1] + 0.5*du]])

        @njit(parallel=True, fastmath=True)
        def xs_gen(u, sigma_sn, sigma_s0, A, alpha, L, gmax_vec, t=t, w=w, tol = self.tol, mul = mu):
            G = u.size
            du = u[1] - u[0]
            sigma = np.zeros((L, G, G), dtype=float)
        
            def mu_from_R(R): return ((A + 1)**2 * R - (A * A + 1)) / (2 * A)
        
            def mu_lab(A,mu_cm): return (1 + A * mu_cm) / np.sqrt(A * A + 2 * A * mu_cm + 1)

            def legendre_single_parallel(mu_vec, l):
                nq = mu_vec.size
                out = np.ones((nq),dtype=float)
            
                if l == 0: return out
            
                if l == 1: return mu_vec
            
                for k in prange(nq):
                    x = mu_vec[k]
                    if x > 1.0: x = 1.0
                    elif x < -1.0: x = -1.0
                    Pm1 = 1.0     
                    P0  = x       
                    for n in range(1, l):
                        P1 = ((2.0*n + 1.0)*x*P0 - n*Pm1) / (n + 1.0)
                        Pm1, P0 = P0, P1
                    out[k] = P0

                return out
            
            for gp in prange(G):
                gmax = gmax_vec[gp]

                for g in range(gp, gmax):
                    """
                    # Outgoing energy bin g has edges [E_edges[g+1], E_edges[g]]
                    R_low  = E_edges[g+1] / E_in[gp] if (g+1 < G) else (min(E_edges) / E_in[gp])
                    R_high = min(E_edges[g] / E_in[gp], 1.0)
        
                    R1 = max(R_low,  alpha)
                    R2 = min(R_high, 1.0)
        
                    mu1 = clamp(mu_from_R(R1))
                    mu2 = clamp(mu_from_R(R2))
                    """
                    # GL quadrature
                    mu1 = mul[gp,g]
                    #mu2 = mul[gp,min(g,G-1)]
                    mu2 = mul[gp,min(g+1,G-1)]
                    mu = 0.5 * ((mu2 - mu1) * t + (mu2 + mu1))
                    dmu = 0.5 * (mu2 - mu1) * w
        
#                    mul = mu_lab(A,mu)
                    for l in range(L):
                        Pl = legendre_single_parallel(mu,l)
                        #Pl = legendre_single_parallel(mul,l)
                        sigma[l, gp, g] = (-1) * sigma_sn[l,gp] * .5 * np.sum(Pl * dmu)
                        #sigma[l, gp, g] = sigma_s0[gp] * .5 * np.sum(Pl * dmu)

#                    if sigma[0,gp,g] < tol: break

#            if A == 1: sigma[3,:,:] = 0
            # enforce exact conservation for l=0 (nice for tiny truncation errors)
            for gp in range(G):
                row_sum = sigma[0, gp, :].sum()
                if row_sum != 0:
                    scale = sigma_s0[gp] / row_sum
                    sigma[0, gp, :] *= scale

            return sigma
    
        sigma_gtg = xs_gen(self.groups, 
                            sigma_sn, 
                            sigma_s0, 
                            A, 
                            alpha,
                            self.leg_order, 
                            self.gmax_vec,)

        @njit(parallel=True, fastmath=True)
        def upscatter(A,u,sigma,sigma_s0,index, kT):
            G = u.size
            E = np.flip(np.exp(u))
            E0 = np.exp(u[-1])
            kT_u =  np.log(E0/kT)
            du = u[1] - u[0]
            theta = .5 * (np.sqrt(A) + A**(-.5))
            rho = .5 * (np.sqrt(A) - A**(-.5))

            for gp in prange(index,G):
#                xp = np.sqrt(E[gp] / kT)
                xp = np.sqrt(E0 / kT_u) * np.exp(-u[gp] / 2) 
                coef = (sigma_s0[gp] / (2 * kT_u)) * (theta * theta / (xp * xp)) * du
                for g in range(index,gp):
                    x = np.sqrt(E0 / kT_u) * np.exp(-u[g] / 2) 
#                    x = np.sqrt(E[g] / kT)
                    term1 = (np.exp(xp * xp - x *x) * 
                                (math.erf(theta * xp - rho * x) 
                                    + math.erf(theta * xp + rho * x)))
                    term2 = math.erf(theta * x - rho * xp) - math.erf(theta * x + rho * xp)
                    sigma[0,gp,g] += coef * (term1 + term2)
            
            return sigma

#        print('upscatter')
#        sigma = upscatter(A,self.groups,sigma_gtg,sigma_s0,self.ind,self.kT)

        rowsum = np.zeros_like(self.groups)
        for g in range(G): rowsum[g] = np.sum(sigma_gtg[:,g,:])
#        scale  = (sigma_s0 / rowsum)
#        for g in range(G): sigma_gtg[0,g,:] *= scale[g]

        self.gtg_and_line_plots(sigma_gtg,A,sigma_s0,rowsum)
#        for l in range(self.leg_order): print(sigma_gtg[l,:5,:5])
        for l in range(self.leg_order): print(sigma_gtg[l,-5:,-5:])

        return sigma_gtg

    def build_Ln(self, A, sigma_s, sigma_t):
        """build the Ln operators. L0, L1, L2, L3."""
        # Initialize I_vals and compute S_vals
        I_vals = [np.zeros_like(self.L0) for _ in range(self.leg_order)]
        S_vals = self.xs_generator(A,sigma_s)

        # integration matrix corresponding to eqn 4
        self.parallel_integrate(self.groups, S_vals, I_vals, self.gridspace, self.gmax_vec, self.leg_order)

        self.nan_list_check(I_vals)
#        for i in range(self.leg_order): print(I_vals[i])
        self.deallocate([S_vals]) # deallocate
        
#        self.groups = self.groups[:self.ind]
#        self.E = self.E[:self.ind]
#        for i in range(self.leg_order):
#            I_vals[i] = I_vals[i][:self.ind,:self.ind]

        # loss operator corresponding to eqn 5
#        L_vals = [self._Ln(n, sigma_t[:self.ind], I_vals[n]) for n in range(self.leg_order)]
        L_vals = [self._Ln(n, sigma_t, I_vals[n]) for n in range(self.leg_order)]
#        for i in range(self.leg_order): print(L_vals[i])
        self.deallocate(I_vals)

        return L_vals

    def calc_Ln(self):
        t1 = time.time()
        # compute the loss operators
        print("U-238 Loss Operators")
        L_vals_U  = self.build_Ln(self.AU,self.sigma_s_U,self.sigma_t_U)

        print("H-1 Loss Operators")
        L_vals_H = self.build_Ln(self.AH,self.sigma_s_H,self.sigma_t_H)

        print(f"Loss Matrices Computed in {np.round(time.time() - t1, 5)}s")

        self.L0, self.L1, self.L2, self.L3 = [(L_vals_U[i] + L_vals_H[i]) for i in range(4)]

        self.deallocate(L_vals_U)
        self.deallocate(L_vals_H)

        self.L0 = self.L0.T
        self.L1 = self.L1.T
        self.L2 = self.L2.T
        self.L3 = self.L3.T

        return self.L0, self.L1, self.L2, self.L3

    def calc_Phi(self):
        """Combine phi0 and phi2 to calculate the scalar flux moments Phi0 and Phi2."""
        self.Phi2 = self.phi2
        self.Phi0 = self.phi0 + 2 * self.phi2

    def calc_phi_np(self):
        B4 = self.B2 * self.B2
        LHS = (9 * B4 + self.B2 * (self.L3 @ self.L2 + (9 * self.L1 + 4 * self.L3) @ self.L0)
                + self.L3 @ self.L2 @ self.L1 @ self.L0)
        RHS = (self.L3 @ self.L2 @ self.L1 + self.B2 * (9 * self.L1 + 4 * self.L3)) @ self.chi
        self.phi0 = np.linalg.solve(LHS,RHS)

        LHS = self.L3 @ self.L2
        RHS = .5 * (-9 * self.B2 * self.phi0 + (9 * self.L1 + 4 * self.L3)
                @ (self.L0 @ self.phi0 - self.chi))
        self.phi2 = np.linalg.solve(LHS,RHS)

    def calc_phi_tt(self):
        """Calculate the flux using operators in TT format"""
        # convert Loss, chi, and scalers to Tensorly
        L0 = self.npy_to_tensor(self.L0)
        L1 = self.npy_to_tensor(self.L1)
        L2 = self.npy_to_tensor(self.L2)
        L3 = self.npy_to_tensor(self.L3)
        I = np.eye(self.groups.size)
        B4 = (self.B2 * self.B2) * I
        B4 = self.npy_to_tensor(B4)
        groups = self.npy_to_tensor(self.groups)
        chi = self.npy_to_tensor(self.chi)

        # solve for phi0
        LHS = (9 * B4 * I + self.B2 * (L3 @ L2 + (9 * L1 + 4 * L3) * L0)
                    + L3 @ L2 @ L1 @ L0)
        RHS = ((L3 @ L2 @ L1 + self.B2 * (9 * L1 + 4 * L3)) @ chi)
        self.phi0 = np.linalg.solve(LHS,RHS)

        # solve for phi2
        LHS = L3 @ L2
        RHS = .5 * (-9 * self.B2 * self.phi0 + (9 * L1 + 4 * L3)
                @ (L0 @ self.phi0 - chi))
        self.phi2 = np.linalg.solve(LHS,RHS)

    def plot_fluxes(self):
        # plot phi0 and phi2
        plt.figure()
        plt.plot(self.E,self.phi0,label=r'$\phi_0$')
        plt.title(r'$\phi_0(E)$')
        plt.xlabel('E')
        plt.ylabel(r'$\phi_0$')
        plt.xscale('log')
        plt.grid(True, which='both')
        plt.legend()
        plt.savefig(f'results/charts/phi0_{self.NH}.png')
        plt.clf()

        plt.figure()
        plt.plot(self.E,self.phi2,label=r'$\phi_2$')
        plt.title(r'$\phi_2(E)$')
        plt.xlabel('E')
        plt.ylabel(r'$\phi_2$')
        plt.xscale('log')
        plt.grid(True, which='both')
        plt.legend()
        plt.savefig(f'results/charts/phi2_{self.NH}_lin.png')
        plt.clf()

        plt.figure()
        plt.plot(self.E,np.abs(self.phi2),label=r'$\phi_2$')
        plt.title(r'$|\phi_2(E)|$')
        plt.xlabel('E')
        plt.yscale('log')
        plt.ylabel(r'$\phi_2$')
        plt.xscale('log')
        plt.grid(True, which='both')
        plt.legend()
        plt.savefig(f'results/charts/phi2_{self.NH}.png')
        plt.clf()

    def run(self, properties, from_h5):
        """Run the complete SP3 calculation process."""
        st = time.time()
        if from_h5 == False:
            print("Starting phi0 calculation...")
            self.calc_Ln()
            #with h5py.File(f"{scratch_dir}/Ln_{self.NH}.h5", "w") as f:
            #    f.create_dataset("L0", data=self.L0)
            #    f.create_dataset("L1", data=self.L1)
            #    f.create_dataset("L2", data=self.L2)
            #    f.create_dataset("L3", data=self.L3)

            self.calc_phi_np()
            #self.calc_phi_tt()
            print(f"phi calculation: {np.round(time.time()-st,5)}")
            print("Plotting...")
            self.plot_fluxes()
            print("Starting Phi0 and Phi2 calculation...")
            self.calc_Phi()

            print("Saving Data...")
            df = pd.DataFrame({'phi0': self.phi0, 'phi2': self.phi2,
                               'Phi0': self.Phi0, 'Phi2': self.Phi2})
            #df = pd.DataFrame({'phi0': self.tensor_to_npy(self.phi0), 'phi2': self.tensor_to_npy(self.phi2),
            #                   'Phi0': self.tensor_to_npy(self.Phi0), 'Phi2': self.tensor_to_npy(self.Phi2)})
            df.to_hdf(f"{scratch_dir}/fluxes_{self.NH}.h5", key="df", mode="w", format="table")
            print("Data Saved")

    def gtg_and_line_plots(self,sigma_gtg,A,sigma_s0,rowsum):
        print("xs plots")
        plt.figure(figsize=(6, 5))
        im = plt.imshow(
                sigma_gtg[0, :, :],
                cmap='viridis',
                origin='upper',
                norm=LogNorm()
            )
        plt.colorbar(im, label=r'$\Sigma_{gtg}$')
        plt.title("Scattering Xs's")
#        plt.show()
        plt.savefig(f'results/charts/sigma_gtg_{A}.png')
        plt.close()
        plt.clf()

        plt.figure()
        plt.plot(self.E,rowsum, label="GTG")
        plt.plot(self.E,sigma_s0,label="S0")
        plt.xscale("log")
        plt.yscale("log")
        plt.grid(which='Both')
        plt.legend()
        plt.savefig(f"results/charts/parallel_xs_gtg_{A}.png")
        plt.close()
        plt.clf()

    @staticmethod
    def tensor_to_npy(tensor): return tl.to_numpy(tensor)

    @staticmethod
    def npy_to_tensor(array): return tl.tensor(array)

    @staticmethod
    @njit(parallel=True, fastmath=True)
    def parallel_integrate(groups, S_vals, I_vals, du, gmax_vec, leg_order):
        """Helper function to perform the integration in parallel."""
        print("Integrating Sn")
        for l in range(leg_order):
            for gp in prange(groups.size):
                gmax = gmax_vec[gp]
                for g in range(gp,min(gmax,groups.size-1)): I_vals[l][gp,g] = .5*du*(S_vals[l,gp,g+1] + S_vals[l,gp,g])
#                for g in range(gp,min(gmax,groups.size - 1)): I_vals[l][gp,gp] += S_vals[l,gp,g] * du
#                for g in range(gp,gmax): I_vals[l][gp,g] = S_vals[l,gp,g]
                #for g in range(gp,gmax - 1): I_vals[l][gp,g] = .5 * du * (S_vals[l,gp,g+1] + S_vals[l,gp,g])
#                accum1 = 0
#                for g in range(gp,gmax - 1):    
#                    accum1 += du * ((S_vals[l,gp,g+1] + S_vals[l,gp,g]) / 2)
#                I_vals[l][gp,gp] = accum1
#                accum2 = 0
#                for g in range(gp+1,gmax - 1):
#                    accum2 += du * ((S_vals[l,gp,g+1] + S_vals[l,gp,g]) / 2)
#                    I_vals[l][gp,g] = accum1 - accum2
                #print(I_vals[l][gp,:gmax])

    @staticmethod
    @njit(parallel=True, fastmath=True)
    def _Ln(l,sigma,I): return (2 * l + 1) * (np.diag(sigma) - I)

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
        """Free up memory because large matrices"""
        for obj in my_list: del obj
        my_list.clear()

    @staticmethod
    def print_mat_properties(LHS):
        """prints important matrix properties"""
        def is_diagonally_dominant(A):
            """checks if a matrix A is diagonally dominant"""
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
    @njit(parallel=True, fastmath=True)
    def compute_mu_matrix(groups, A, gmax):
        """Compute scattering cosine matrix mu(u, u') for all g, g' combinations."""
        G = groups.size
        mu_cm = np.zeros((G,G))
        for gp in prange(G):
            for g in range(gp,gmax[gp]):
                mu_foo = (((A + 1) / 2) * np.exp((groups[gp] - groups[g]) / 2) 
                            - ((A - 1) / 2) * np.exp((groups[g] - groups[gp]) / 2))
                if np.abs(mu_foo) <= 1: mu_cm[gp,g] = mu_foo
#        u = lethargy.reshape(1, -1)
#        up = lethargy.reshape(-1, 1)
#        mu_cm = ((A + 1) / 2) * np.exp((up - u) / 2) - ((A - 1) / 2) * np.exp((u - up) / 2)
#        mu_cm = np.clip(mu_cm,-1,1)
        mu_lab = ((1 + A * mu_cm) / np.sqrt(A * A + 2 * A * mu_cm + 1))
        return mu_lab

    @staticmethod
    def normalize(vec):
        assert vec.ndim == 1
        return vec / np.sum(vec)

    @staticmethod
    def L2_norm(A,B):
        assert A.shape == B.shape
        return np.linalg.norm(A-B, ord=2)

    @staticmethod
    def nan_list_check(mylist):
        has_nan = any(np.isnan(arr).any() for arr in mylist)
        if has_nan: raise ValueError("I_vals has nans")

    @staticmethod
    def alpha(A): return ((A - 1) / (A + 1)) ** 2

