import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import time
import math
import h5py
from numba import njit, prange
import psutil
import torch # tensor decomps, gpu

class Sp3:
    def __init__(self, nbins, B2, NH, few_groups, fromH5, dx, adaptive, xs_tol):
        # directories
        self.data_dir = 'data/'
        self.save_dir = "/scratch/bckiedro_root/bckiedro0/rsshast/Sp3/results/"
        self.chart_dir = "results/charts/"

        # ranges
        self.B2 = B2
        self.E0 = 1e7
        self.Emin = 1
        #self.Emin = 1e-2
        self.leg_order = 4

        # number densities
        self.AH = 1
        self.NH = 2 / 3
        #self.NH = 1 / 9
        self.AU = 238
        #self.NU = 1
        self.NU = 1/3
        #self.NU = 10.97 / 270
        self.AO = 16
        #self.NO = 1
        #self.NO = self.NH / 2
        self.NO = self.NH * 2
        
        # grp constant parameters
        self.save_data = False
        self.fromH5 = fromH5
        self.few_groups = few_groups
        self.nu = 2.43

        # point-wise cross-sections and fission spectrum
        chi35 = pd.read_csv(f'{self.data_dir}chi_u235.txt', sep = '\t', header = 0)
        H1 = pd.read_csv(f'{self.data_dir}xs_h1_T293k.txt', sep = '\t', header = 0)
        U238 = pd.read_csv(f'{self.data_dir}xs_u238_T293k.txt',sep = '\t', header = 0)
        O16 = pd.read_csv(f'{self.data_dir}xs_o16_T293k.csv', sep = ';', dtype=float).to_numpy()
        sigma_f = pd.read_csv(f'{self.data_dir}xs_u238_fission.csv', sep = ',', dtype=float).to_numpy()
        sigma_fr_U = pd.read_csv(f"{self.data_dir}sigma_fr_U.csv",sep = ';', dtype=float).to_numpy()
        sigma_fr_H = pd.read_csv(f"{self.data_dir}sigma_fr_H.csv",sep = ';', dtype=float).to_numpy()

        # interpolations
        def get_adaptive_Ugrid(data, E0, Emin, tol):
            data = data[(data[:, 0] <= E0) & (data[:, 0] >= Emin)]
            data[:, 0] = np.log(E0 / data[:, 0])
            sort_idx = np.argsort(data[:, 0])
            data = data[sort_idx]
            new_grid = [data[0]]  # Start with the first row
        
            i = 0
            while i < data.shape[0] - 1:  # We'll iterate until the second last element
                found = False
                for j in range(i + 1, data.shape[0]):
                    diff = 100 * np.abs(data[i, 1] - data[j, 1]) / data[i, 1]
                    if diff >= tol:
                        new_grid.append(data[j])  # append entire row
                        i = j
                        found = True
                        break  # Break inner for-loop when condition is met
                if not found:
                    # If no suitable j is found, we are done
                    break

            return np.array(new_grid)

        def get_adaptive_H_grid(data, new_grid, E0, Emin):
            # data is hydrogen, new_grid is Uranium
            data = data[(data[:, 0] <= E0) & (data[:, 0] >= Emin)]
            data[:, 0] = np.log(E0 / data[:, 0])

            # Sort
            sort_idx = np.argsort(data[:, 0])
            data = data[sort_idx]
            out = np.empty((new_grid.size, data.shape[1]), dtype=float)
            out[:, 0] = new_grid
    
            xp = data[:, 0]
            for i in range(1, data.shape[1]): out[:, i] = np.interp(new_grid, xp, data[:, i])
            return out

        # interpolate xs's and chi to U grid. 
        XS38 = np.array([U238['E'],U238['sigma_t'],U238['sigma_s']]).T
        chi = np.array([chi35['E'],chi35['chi']]).T
        H = np.array([H1['E'],H1['sigma_t'],H1['sigma_s']]).T

        if adaptive == True:
            XS38 = get_adaptive_Ugrid(XS38,self.E0,self.Emin,xs_tol)
            self.nbins = XS38.shape[0]
            H = get_adaptive_H_grid(H,XS38[:,0], self.E0, self.Emin)
            chi = get_adaptive_H_grid(chi,XS38[:,0], self.E0, self.Emin)
            XS16 = get_adaptive_H_grid(O16, XS38[:,0], self.E0, self.Emin)
        else:
            self.nbins = nbins + 1
            chi = self.get_data(chi,self.nbins)
            H = self.get_data(H,self.nbins) * self.NH
            XS38 = self.get_data(XS38,self.nbins) 
            XS16 = self.get_data(O16, self.nbins)

        self.chi = chi[:,1]
        self.boundaries = XS38[:,0]
        self.Evec = self.E0 * np.exp(-self.boundaries)
        print(f"Initializing, {self.nbins - 1} Groups")
        assert (self.nbins - 1) % few_groups == 0, ("Number of fine bins must be multiple of number of coarse bins")

        # cross-sections and number densities
        self.sigma_fr_U = self.get_data(sigma_fr_U,self.nbins)[:,1] * self.NU
        self.sigma_fr_H = self.get_data(sigma_fr_H,self.nbins)[:,1] * self.NH
        self.sig_t_U   = self.NU * XS38[:,1]
        self.sig_s0_U  = self.NU * XS38[:,2]
        self.sig_t_H   = self.NH * H[:,1]
        self.sig_s0_H  = self.NH * H[:,2]
        self.sig_t_O   = self.NO * XS16[:,1]
        self.sig_s0_O  = self.NO * XS16[:,2]
        self.sigma_f   = self.nu * np.flip(self.get_data(sigma_f,self.nbins)[:,1])
        self.T         = 293.15    # degrees Kelvin
        self.k         = 8.617e-5  # eV/K (Boltzmann constant)
        self.kT        = self.k * self.T

        # declare operators
        G = self.boundaries.size
        self.phi0 = np.zeros((G-1))
        self.phi2 = np.zeros_like(self.phi0)
        self.Phi0 = np.zeros_like(self.phi0)
        self.Phi2 = np.zeros_like(self.phi0)
        self.p0 = np.zeros_like(self.phi0)
        self.Eplot = np.zeros_like(self.phi0)
        self.L0 = np.zeros((G-1,G-1))
        self.L1 = np.zeros_like(self.L0)
        self.L2 = np.zeros_like(self.L0)
        self.L3 = np.zeros_like(self.L0)
        # overwrite to enable cuda
        #self.L0 = torch.from_numpy(self.L0)
        #self.L1 = torch.from_numpy(self.L1)
        #self.L2 = torch.from_numpy(self.L2)
        #self.L3 = torch.from_numpy(self.L3)

        self.sigma_s_gtg_U = np.zeros((self.leg_order,G-1,G-1))
        self.sigma_s_gtg_H = np.zeros_like(self.sigma_s_gtg_U)
        self.sigma_s_gtg_O = np.zeros_like(self.sigma_s_gtg_U)

        # Sp3 Transport Solve Parameters
        self.L = 1
        self.dx = dx
        self.num_cells = int(self.L / self.dx)
        self.max_iters = 100000

    def get_data(self,data, gridpoints):
        data = data[(data[:, 0] <= self.E0) & (data[:, 0] >= self.Emin)]
        data[:, 0] = np.log(self.E0 / data[:, 0])

        # Sort
        sort_idx = np.argsort(data[:, 0])
        data = data[sort_idx]
        new_grid = np.linspace(np.log(self.E0/self.E0), np.log(self.E0 / self.Emin), gridpoints)
        out = np.empty((new_grid.size, data.shape[1]), dtype=float)
        out[:, 0] = new_grid

        xp = data[:, 0]
        for i in range(1, data.shape[1]): out[:, i] = np.interp(new_grid, xp, data[:, i])
        return out

    def initial_flux(self):
        """Compute the initial scalar flux using scattering source method nuclear engineering handbook method"""
        stt = time.time()
        # Problem setup
        u = np.zeros_like(self.p0)
        chi = np.zeros_like(self.p0)
        StH = np.zeros_like(self.p0)
        SsH = np.zeros_like(self.p0)
        StU = np.zeros_like(self.p0)
        SsU = np.zeros_like(self.p0)

        for i in range(self.p0.size):
            u[i] = .5 * (self.boundaries[i+1] + self.boundaries[i])
            chi[i] = np.trapz(self.chi[i:i+2],self.boundaries[i:i+2]) / (self.boundaries[i+1] - self.boundaries[i])
            self.Eplot[i] = .5 * (self.Evec[i+1] + self.Evec[i]) 
            StH[i] = np.trapz(self.sig_t_H[i:i+2],self.boundaries[i:i+2])  / (self.boundaries[i+1] - self.boundaries[i])
            SsH[i] = np.trapz(self.sig_s0_H[i:i+2],self.boundaries[i:i+2]) / (self.boundaries[i+1] - self.boundaries[i])
            StU[i] = np.trapz(self.sig_t_U[i:i+2],self.boundaries[i:i+2])  / (self.boundaries[i+1] - self.boundaries[i])
            SsU[i] = np.trapz(self.sig_s0_U[i:i+2],self.boundaries[i:i+2]) / (self.boundaries[i+1] - self.boundaries[i])

        du = np.diff(self.boundaries)
        #du = u[1] - u[0]
        alphaU = self.alpha_fn(self.AU)
        inv1ma = 1 / (1 - alphaU)
        lga = -np.log(alphaU)                

        expu = np.exp(u)
        expm = 1 / expu
        phi = np.zeros_like(u, dtype=float)
        scatH = 0
        scatU = 0    
        gmax = int(np.floor(lga / du[0]))

        for i in range(u.size):
            Sigma_R = (StH[i] + StU[i]) - du[i] * (SsU[i] * inv1ma) - du[i] * SsH[i]
            if i == 0: phi[i] = chi[i] / Sigma_R
    
            # Add scattering sources from previous bin
            scatH += SsH[i-1] * phi[i-1] * expu[i-1] * du[i]
            scatU += (SsU[i-1] * inv1ma) * phi[i-1] * expu[i-1] * du[i]
    
            # Subtract contributions outside uranium lethargy window
            if i > gmax:
                f = (lga / du[i]) - gmax  
                h = i - gmax
                scatU -= (SsU[h] * inv1ma) * (1 - f) * phi[h] * expu[h] * du[i]
                scatU -= (SsU[h-1] * inv1ma) * f * phi[h-1] * expu[h-1] * du[i]
    
            phi[i] = (chi[i] + expm[i] * (scatH + scatU)) / Sigma_R
    
        # set the spectra as class attributes
        self.p0 = phi
        self.chi = chi

        print(f"Initial Flux Time: {np.round(time.time()-stt,5)} s")
        """
        plt.figure(figsize=(8,6))
        plt.plot(self.Eplot, self.chi, label=r'$\chi$')
        plt.title(r'$^{235}$U Fission Spectrum $\chi (E)$')
        plt.xlabel('Energy [eV]')
        plt.ylabel(r'$\chi (E)$')
        plt.xscale('log')
        plt.grid(True, which='both')
        plt.legend()
        plt.savefig(f"{self.chart_dir}chi.png")   
        plt.clf()

        plt.figure()
        plt.plot(self.Eplot, self.p0, label=r'$\phi_0$')
        plt.title(r'Initial Scalar Flux $\phi_0(E)$')
        plt.xlabel('Energy [eV]')
        plt.ylabel(r'$\phi_0$')
        plt.xscale('log')
        plt.grid(True, which='both')
        plt.legend()
        plt.savefig(f"{self.chart_dir}initial_flux.png")   
        """

    def group_bound(self, A, g, lga): return (self.boundaries.size if A == 1
                else 1 + np.searchsorted(self.boundaries, self.boundaries[g] + lga))

    def gmax_vec_fn(self, A, lga):
        gmax_vec = np.zeros_like(self.boundaries, dtype = int)
        for g in range(self.boundaries.size): gmax_vec[g] = self.group_bound(A,g,lga)
        return gmax_vec

    def gtg_xs(self,A,sigma_s,verbose = False):
        # l x g-1 x g-1 matrix
        phi = np.ones_like(self.p0)
        sig_s = np.zeros_like(self.p0)

        for i in range(phi.size):
            sig_s[i] = np.trapz(sigma_s[i:i+2],self.boundaries[i:i+2]) / (self.boundaries[i+1] - self.boundaries[i])

        stt = time.time()
        du = np.diff(self.boundaries)
        sigma_gtg = self.gen_sig_sn_gtg(A,sig_s,self.leg_order, self.boundaries,
                                        self.gmax_vec_fn(A,self.lga_fn(self.alpha_fn(A))),
                                        self.alpha_fn(A), self.E0, phi, du)

        print(f"Sigma_gtg A={A} Time: {np.round(time.time()-stt,5)} s")
        # transpose with gp on x axis and g on y axis
        S = np.transpose(sigma_gtg, (0, 2, 1))

        if self.save_data: self.save_sig_s_gtg(S,A,self.save_dir,self.NH)

        if A == 1: self.sigma_s_gtg_H = S
        elif A == 16: self.sigma_s_gtg_O = S
        else: self.sigma_s_gtg_U = S
        
        if verbose:
            self.plot_sig_sn_gtg(sigma_gtg, sigma_s, A)
            self.plot_each_l(A,sigma_s,sigma_gtg)

        return S

    def calc_Ln(self,A,sigma_t, sigma_s,verbose):
        def _torch_Ln(order, sig_t, S, device="auto", dtype=torch.float64, return_numpy=True):
            if device == "auto":
                if isinstance(S, torch.Tensor) and S.is_cuda: dev = S.device
                elif isinstance(sig_t, torch.Tensor) and sig_t.is_cuda: dev = sig_t.device
                else: dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else: dev = torch.device(device)
        
            sig_t = sig_t.to(dev, dtype=dtype)
            S_tt = torch.from_numpy(S).to(dev, dtype=dtype)
        
            l_vals = torch.arange(order, device=dev, dtype=dtype)
            factors = (2 * l_vals + 1).view(order, 1, 1)
            Ln_tt = factors * (sig_t.unsqueeze(0) - S_tt)
        
            return Ln_tt.cpu().numpy()

        S = self.gtg_xs(A,sigma_s,verbose)
        sig_t = np.zeros_like(self.p0)
        for i in range(sig_t.size):
            sig_t[i] = np.trapz(sigma_t[i:i+2],self.boundaries[i:i+2]) / (self.boundaries[i+1] - self.boundaries[i])
        sig_t = torch.from_numpy(np.diag(sig_t)) # sig_t is now a torch tensor

        stt = time.time()
        Ln = _torch_Ln(self.leg_order,sig_t,S,device='auto')
        self.L0 += Ln[0,:,:]
        self.L1 += Ln[1,:,:]
        self.L2 += Ln[2,:,:]
        self.L3 += Ln[3,:,:]

        print(f"Ln A = {A} Time: {np.round(time.time()-stt,5)} s")

    def calc_phi_torch(self, device=None, dtype=torch.float64):
        device = "cuda" if device is None else device
        self.device = torch.device(device)
    
        print("Calculating phi on GPU")
        L0 = torch.from_numpy(self.L0).to(self.device, dtype=dtype)
        L1 = torch.from_numpy(self.L1).to(self.device, dtype=dtype)
        L2 = torch.from_numpy(self.L2).to(self.device, dtype=dtype)
        L3 = torch.from_numpy(self.L3).to(self.device, dtype=dtype)
    
        chi = torch.from_numpy(self.chi).to(self.device, dtype=dtype)
        B2  = torch.tensor(self.B2, device=self.device, dtype=dtype)
    
        # phi0
        stt = time.time()
        B4 = B2 * B2
        LHS = (9 * B4 + B2 * (L3 @ L2 + (9 * L1 + 4 * L3) @ L0)
               + L3 @ L2 @ L1 @ L0)
        RHS = (L3 @ L2 @ L1 + B2 * (9 * L1 + 4 * L3)) @ chi
        phi0 = torch.linalg.solve(LHS, RHS.unsqueeze(-1)).squeeze(-1)
        print(f"phi0 Time (torch): {time.time() - stt:.5f} s")
    
        # phi2
        stt = time.time()
        LHS2 = L3 @ L2
        RHS2 = 0.5 * (-9 * B2 * phi0 + (9 * L1 + 4 * L3) @ (L0 @ phi0 - chi))
        phi2 = torch.linalg.solve(LHS2, RHS2.unsqueeze(-1)).squeeze(-1)
        print(f"phi2 Time (torch): {time.time() - stt:.5f} s")
    
        # back to numpy
        self.phi0 = phi0.cpu().numpy()
        self.phi2 = phi2.cpu().numpy()

    def calc_phi_B2(self):
        print('Calc phi B2')
        p0 = []
        p2 = []

        # phi0
        plt.figure()
        for i in range(self.B2.size):
            B2 = self.B2[i]
            print(f"phi0, {i}, B2 = {np.round(B2,5)}")
            B4 = B2 * B2
            LHS = (9 * B4 + B2 * (self.L3 @ self.L2 + (9 * self.L1 + 4 * self.L3) @ self.L0)
                    + self.L3 @ self.L2 @ self.L1 @ self.L0)
            RHS = (self.L3 @ self.L2 @ self.L1 + B2 * (9 * self.L1 + 4 * self.L3)) @ self.chi
            phi0 = np.linalg.solve(LHS,RHS)
            phi0 = self.normalize(phi0)
            p0.append(phi0)
            if i == 0 or i == self.B2.size - 1:
                plt.plot(self.Eplot,phi0,label = f"{np.round(B2,5)}")

        plt.title(fr"$\phi_0 (E,B^2)$ Parametric Study, {self.p0.size} Groups")
        plt.xscale("log")
        plt.xlabel("Energy (eV)")
        plt.ylabel(r"$\phi_0 (E,B^2)$")
        plt.legend()
        plt.grid(which = 'Both')
        plt.savefig(f"{self.chart_dir}phi0_leakage_comparison.png")
        plt.clf()

        p0 = np.array(p0)

        #phi2
        plt.figure()
        for i in range(self.B2.size):
            B2 = self.B2[i]
            print(f"phi2, {i}, B2 = {np.round(B2,5)}")
            LHS = self.L3 @ self.L2
            RHS = .5 * (-9 * B2 * p0[i,:] + (9 * self.L1 + 4 * self.L3)
                    @ (self.L0 @ p0[i,:] - self.chi))
            phi2 = np.linalg.solve(LHS,RHS)
            phi2 = self.normalize(phi2)
            p2.append(phi2)
            if i == 0 or i == self.B2.size - 1:
                plt.plot(self.Eplot,np.abs(phi2),label = f"{np.round(B2,5)}")

        plt.title(fr"$\phi_2 (E,B^2)$ Parametric Study, {self.p0.size} Groups")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Energy (eV)")
        plt.ylabel(fr"$\phi_2$ (E,B2)")
        plt.legend()
        plt.grid(which = 'Both')
        plt.savefig(f"{self.chart_dir}phi2_leakage_comparison.png")
        plt.clf()

        B2 = 0
        # self.phi0
        B4 = B2 * B2
        LHS = (9 * B4 + B2 * (self.L3 @ self.L2 + (9 * self.L1 + 4 * self.L3) @ self.L0)
                    + self.L3 @ self.L2 @ self.L1 @ self.L0)
        RHS = (self.L3 @ self.L2 @ self.L1 + B2 * (9 * self.L1 + 4 * self.L3)) @ self.chi
        self.phi0 = np.linalg.solve(LHS,RHS)
        self.phi0 = self.normalize(self.phi0)

        # self.phi2
        B2 = 0
        LHS = self.L3 @ self.L2
        RHS = .5 * (-9 * B2 * self.phi0 + (9 * self.L1 + 4 * self.L3)
               @ (self.L0 @ self.phi0 - self.chi))
        self.phi2 = np.linalg.solve(LHS,RHS)
        self.phi2 = self.normalize(self.phi2)

        p2 = np.array(p2)

        plt.figure()
        for i in range(self.B2.size):
            plt.plot(self.Eplot, 100 * np.abs(p0[i,:] - self.normalize(self.phi0)) / self.normalize(self.phi0), 
                    label = f"B2: {self.B2[i]:.5g}, L2 = {self.L2_norm(p0[i,:], self.normalize(self.phi0)):.5g}")

        plt.title(fr"$\phi_0 (E,B^2)$ Percent Difference, {self.p0.size} Groups")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Energy (eV)")
        plt.ylabel(fr"Relative % Difference from $\phi_0 (E,B^2 = 0)$")
        plt.legend(loc='upper left')
        plt.grid(which = 'Both')
        plt.savefig(f"{self.chart_dir}phi0_leakage_flux_diff.png")
        plt.clf()

        plt.figure()
        plt.plot(self.Eplot, 100 * np.abs(p2[0,:] - p2[-1,:]) / np.abs(p2[0,:]), 
                label = fr"$B^2$ = {self.B2[0]}, {self.B2[-1]}")
        plt.title(fr"$\phi_2$ Leakage Parameter Percent Difference, $L_2$ = {self.L2_norm(p2[i,:], self.phi2):.5g}")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Energy (eV)")
        plt.ylabel(f"% Difference from Positive and Negative Buckling")
        plt.legend(loc='lower left')
        plt.grid(which = 'Both')
        plt.savefig(f"{self.chart_dir}phi2_leakage_flux_diff.png")
        plt.clf()

    def calc_phi(self, properties = False,dtype=torch.float64,key='fuel'):
        print("Calculating phi on CPU")
        stt=time.time()
        if key == 'mod': self.chi = np.ones_like(self.chi) / self.Evec[:self.L0.shape[0]]

        # phi0
        if self.B2 == 0: self.phi0 = np.linalg.solve(self.L0,self.chi)
        else:
            B4 = self.B2 * self.B2
            LHS = (9 * B4 + self.B2 * (self.L3 @ self.L2 + (9 * self.L1 + 4 * self.L3) @ self.L0)
                    + self.L3 @ self.L2 @ self.L1 @ self.L0)
            RHS = (self.L3 @ self.L2 @ self.L1 + self.B2 * (9 * self.L1 + 4 * self.L3)) @ self.chi
            if properties:
                print("phi0 matrix properties")
                self.print_mat_properties(LHS)
            self.phi0 = np.linalg.solve(LHS,RHS)
        print(f"phi0 Time: {np.round(time.time()-stt,5)} s")

        #phi2
        stt = time.time()
        LHS = self.L3 @ self.L2
        if properties:
            print("phi2 matrix properties")
            self.print_mat_properties(LHS)
        RHS = .5 * (-9 * self.B2 * self.phi0 + (9 * self.L1 + 4 * self.L3) @ (self.L0 @ self.phi0 - self.chi))
        self.phi2 = np.linalg.solve(LHS,RHS)
        print(f"phi2 Time: {np.round(time.time()-stt,5)} s")

    def calc_Phi(self): 
        self.Phi0 = self.phi0 + 2 * self.phi2
        self.Phi2 = self.phi2
        print(f"L2 norm on phi0 and Phi0, B2 = {self.B2}: {self.L2_norm(self.Phi0,self.phi0)}")

    def save_Ln(self):
        if self.save_data == True:
            with h5py.File(f"{self.save_dir}Ln_{self.NH}.h5", "w") as f:
                f.create_dataset("L0", data=self.L0.numpy(),compression="gzip", compression_opts=4)
                f.create_dataset("L1", data=self.L1.numpy(),compression="gzip", compression_opts=4)
                f.create_dataset("L2", data=self.L2.numpy(),compression="gzip", compression_opts=4)
                f.create_dataset("L3", data=self.L3.numpy(),compression="gzip", compression_opts=4)

    @staticmethod
    def save_grp_mat_vectors(sig_t_0, sig_t_2, sig_f_0, sig_f_2, chi,save_dir,key):
        print("Saving Group Vectors")
        df = pd.DataFrame({ 'sig_t_0': sig_t_0,
                            'sig_t_2': sig_t_2,
                            'sig_f_0': sig_f_0,
                            'sig_f_2': sig_f_2,
                            'chi': chi,
                            })
        df.to_csv(f"{save_dir}grp_vectors_{key}.csv")

    @staticmethod
    def save_grp_vectors(sig_t,sig_t_0,sig_t_2,sig_f, sig_f_0, sig_f_2, chi,save_dir):
        print("Saving Group Vectors")
        df = pd.DataFrame({'sig_t': sig_t,
                            'sig_t_0': sig_t_0,
                            'sig_t_2': sig_t_2,
                            'sig_f': sig_f,
                            'sig_f_0': sig_f_0,
                            'sig_f_2': sig_f_2,
                            })
        df.to_csv(f"{save_dir}grp_vectors.csv")

    def save_fluxes(self):
        print("Saving Data...")
        df = pd.DataFrame({'phi0': self.phi0, 'phi2': self.phi2, 
            'Phi0': self.Phi0, 'Phi2': self.Phi2,
            'phi_ref': self.p0, 'chi': self.chi})
        df.to_hdf(f"{self.save_dir}fluxes_{self.NH}.h5", key="df", mode="w", format="table")
        #df.to_csv(f"{self.save_dir}fluxes_{self.NH}.csv")

        df = pd.DataFrame({'sigma_t_U': self.sig_t_U, 'sigma_t_H': self.sig_t_H,'sigma_t_O': self.sig_t_O,
                            'nu_sigma_f': self.sigma_f})
        df.to_hdf(f"{self.save_dir}fine_group_xs_vectors_{self.NH}.h5", key="df", mode="w", format="table")

    def plot_sig_sn_gtg(self, sig_sn_gtg, sig_s0, A):
        Evec = self.Evec[:-1]
        rowsum = np.zeros(Evec.size)
        for i in range(sig_sn_gtg.shape[1]): rowsum[i] = np.sum(sig_sn_gtg[:,i,:])
        self.gtg_and_line_plots(sig_sn_gtg,A,sig_s0,rowsum)

        plt.figure(figsize=(8,6))
        plt.plot(self.Evec,sig_s0, label = r"$\Sigma_{s0}$")
        plt.plot(Evec,rowsum, label = r"$\Sigma_{gtg}$")
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("Energy (MeV)")
        plt.ylabel(r"$\Sigma_{s}$")
        plt.legend()
        plt.grid(which="Both")
        plt.savefig(f'{self.chart_dir}xs_comparison_{A}.png')
        plt.clf()

        plt.figure(figsize=(8,6))
        plt.plot(Evec,100 * np.abs(rowsum - sig_s0[:-1]) / sig_s0[:-1], label = "% Difference")
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("Energy (MeV)")
        plt.ylabel(r"$\Sigma_{s}$ % Difference")
        plt.legend()
        plt.grid(which="Both")
        plt.savefig(f'{self.chart_dir}xs_diff_{A}.png')
        plt.clf()

    def plot_each_l(self,A,sigma_s0,sigma_gtg):
        Evec = self.Evec[:-1]
        plt.figure(figsize=(8,6))
        plt.plot(self.Evec,sigma_s0, label = r"$\Sigma_{s0}$")
        for l in range(self.leg_order):
            rowsum = np.zeros(Evec.size)
            for g in range(Evec.size): rowsum[g] = np.sum(sigma_gtg[l,g,:])
            plt.plot(Evec,rowsum, label = f"Sigma_s{l}")
        plt.title(f"Sigma_sl, A = {A}")
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("Energy (MeV)")
        plt.ylabel(r"$\Sigma_{s}$")
        plt.legend()
        plt.grid(which="Both")
        plt.savefig(f'{self.chart_dir}xs_sl_{A}.png')
        plt.clf()

    def plot_fluxes(self):
        # plot phi0 and phi2
        print("Plotting phi0, phi2")
        phi0 = self.normalize(self.phi0)
        plt.figure()
        plt.plot(self.Eplot,phi0,label=r'$\phi_0$')
        plt.title(fr'$\phi_0(E)$, {self.nbins - 1} Groups')
        plt.xlabel('E')
        plt.ylabel(r'$\phi_0$')
        plt.xscale('log')
        plt.grid(True, which='both')
        plt.legend()
        plt.savefig(f'{self.chart_dir}phi0_{self.NH}.png')
        plt.clf()

        plt.figure()
        plt.plot(self.Eplot,np.abs(self.phi2),label=r'$\phi_2$')
        plt.title(fr'$\phi_2(E)$, {self.nbins - 1} Groups')
        plt.xlabel('E')
        plt.ylabel(r'$\phi_2$')
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True, which='both')
        plt.legend()
        plt.savefig(f'{self.chart_dir}phi2_{self.NH}.png')
        plt.clf()

    def plot_flux_diff_single_axis(self):
        # compare the traditional to new method
        plt.figure()
        plt.plot(self.Eplot,p0,label='Scattering Source')
        plt.plot(self.Eplot,phi0,label='Sp3')
        plt.title(f"Hyperfine Slowing-Down Flux Comparison, {self.nbins-1} Groups")
        plt.ylabel(r"$\phi (E) (n/cm^2)$")
        plt.xscale('log')
        plt.xlabel("Energy (eV)")
        plt.legend()
        plt.grid(True,which='both')
        plt.savefig(f"{self.chart_dir}order_comp.png")
        plt.clf()

        plt.figure()
        plt.plot(self.Eplot, 100 * (phi0 - p0) / p0)
        plt.title(f"Hyperfine Slowing-Down Flux Difference, {self.nbins - 1} Groups")
        plt.ylabel("% Difference Between Calculated and Reference Spectra")
        plt.xscale('log')
        plt.xlabel("Energy (eV)")
        plt.grid(True,which='both')
        plt.savefig(f"{self.chart_dir}flux_difference.png")

    def plot_flux_diff(self):
        fig, ax1 = plt.subplots(figsize=(7, 5))
    
        # Left y-axis: p0 and phi0
        p0 = self.normalize(self.p0)
        phi0 = self.normalize(self.phi0)
        ax1.plot(self.Eplot, p0, label='Scattering Source', color='tab:blue', lw=1.5)
        ax1.plot(self.Eplot, phi0, label='SP3', color='tab:purple', lw=1.5)
        ax1.set_xscale('log')
        ax1.set_xlabel("Energy (eV)")
        ax1.set_ylabel(r"$\phi(E)$  $(n/cm^2)$", color='k')
        ax1.grid(True, which='both', linestyle=':')
        ax1.tick_params(axis='y', labelcolor='k')
    
        # Right y-axis: percent difference
        ax2 = ax1.twinx()
        diff = 100 * (phi0 - p0) / p0
        ax2.plot(self.Eplot, diff, color='tab:red', lw=1.2, alpha=0.8, label='% Difference',linestyle='dotted')
        ax2.set_ylabel("% Difference", color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')
    
        # Title and legend
        fig.suptitle(f"Hyperfine Slowing-Down Flux Comparison, {self.nbins - 1} Groups")
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
    
        fig.tight_layout()
        fig.savefig(f"{self.chart_dir}flux_comparison_dual_axes.png", dpi=300)
        plt.close(fig)

        # evaluate fluxes
        L2 = self.L2_norm(self.phi0,self.p0)
        print(f"L2 norm on phi0 and reference, = {np.round(L2,9)}")
        L2 = self.L2_norm(self.Phi0,self.p0)
        print(f"L2 norm on Phi0 and reference, = {np.round(L2,9)}")

    def gtg_and_line_plots(self,sigma_gtg,A,sigma_s0,rowsum):
        print("Plot Sigma_sl")
        for l in range(self.leg_order):
            plt.figure(figsize=(6, 5))
            im = plt.imshow(
                sigma_gtg[l,:,:],
                cmap='viridis',
                origin='upper',
                norm=LogNorm()
            )
            plt.colorbar(im, label=r'$\Sigma^{sl}_{gtg}$')
            plt.title(f"Scattering Xs's, l = {l}")
            plt.xlabel("Outgoing Energy Group")
            plt.ylabel("Incedent Energy Group")
            plt.savefig(f'{self.chart_dir}sigma_s{l}_A{A}.png')
            plt.clf()

    def few_group_fluxes(self, key):
        # collapse the fine group down to few group fluxes
        few_grp_Phi0 = np.zeros((self.few_groups))
        few_grp_Phi2 = np.zeros((self.few_groups))
        fg_idx = np.linspace(0,self.phi0.size,self.few_groups+1, dtype=int)

        for i in range(few_grp_Phi0.size):
            stt, stp = fg_idx[i], fg_idx[i+1]
            few_grp_Phi0[i] = np.trapz(self.Phi0[stt:stp], self.boundaries[stt:stp])
            few_grp_Phi2[i] = np.trapz(self.Phi2[stt:stp], self.boundaries[stt:stp]) 

        # plotting
        index = 1
        Phi_plot = np.zeros_like(self.phi0)
        for i in range(self.phi0.size):
            if i < fg_idx[index]: Phi_plot[i] = few_grp_Phi0[index-1]
            else:
                index += 1
                Phi_plot[i] = few_grp_Phi0[index-1]

        plt.figure()
        plt.plot(self.Eplot,self.normalize(Phi_plot),label=r"Few Group $\Phi_0 (E)$")
        plt.title("Few Group Scalar Flux")
        plt.ylabel(r"$\Phi_0$")
        plt.xlabel("Energy (ev)")
        plt.xscale("log")
        plt.legend()
        plt.grid(which='Both')
        plt.savefig(f"{self.chart_dir}fg_Phi0.png")

        return few_grp_Phi0, few_grp_Phi2

    def phi_weighted_sigma(self, sigma, A, key):
        # start generating few group xs's
        def sigma_vec_few_grp(sigma, phi, group_idx, E):
            few_grp_xs = np.zeros((group_idx.size - 1))
            # ensure sigma and phi are the same size
            assert sigma.size == phi.size
    
            for i in range(few_grp_xs.size):
                stt, stp = group_idx[i], group_idx[i+1]
                few_grp_xs[i] = np.trapz(sigma[stt:stp] * phi[stt:stp], E[stt:stp]) / np.trapz(phi[stt:stp], E[stt:stp])
    
            return few_grp_xs

        stt=time.time()
        fg_idx = np.linspace(0,self.phi0.size,self.few_groups+1, dtype=int)
        E_ave = np.zeros((self.few_groups))
        u_ave = np.zeros_like(E_ave)
        sigma_fg = np.zeros_like(E_ave)
        for i in range(self.few_groups): 
            E_ave[i] = self.Evec[fg_idx[i]]
            u_ave[i] = self.boundaries[fg_idx[i]]

        phi = self.phi0
        if sigma.size > phi.size: sigma = sigma[:-1]
        sigma_fg = sigma_vec_few_grp(sigma, phi, fg_idx, self.Evec[:-1])
        sigma_fg_0 = sigma_vec_few_grp(sigma, self.Phi0, fg_idx, self.Evec[:-1])
        sigma_fg_2 = sigma_vec_few_grp(sigma, self.Phi2, fg_idx, self.Evec[:-1])
#        print(f"Update A={A} {key} xs: {np.round(time.time()-stt,5)} s")
#        print(f"L2 norm on phi0 and Phi0 weighted xs's for A={A}, {key}: {self.L2_norm(self.normalize(sigma_fg),self.normalize(sigma_fg_0))}")

        # save data to .h5
        if self.save_data == True:
            df = pd.DataFrame({
                "Energy": E_ave,
                "Lethargy": u_ave,
                "Sigma Average": sigma_fg,
                "Sigma Phi0": sigma_fg_0,
                "Sigma Phi2": sigma_fg_2,
            })

            df.to_csv(f"{self.save_dir}fg_xs_{key}_A{A}_NH{self.NH}.csv")

        return sigma_fg, sigma_fg_0, sigma_fg_2

    def phi_weighted_sigma_sl(self, M, A, l):
        # Get gtg flux weighte xs's
        stt=time.time()
        N = self.phi0.size
        E = self.Evec[:-1]
        u = self.boundaries
        fg_idx = np.linspace(0, N, self.few_groups + 1, dtype=int)
        M_fg = np.zeros((self.few_groups,self.few_groups),dtype=float)
        M_fg_0 = np.zeros_like(M_fg)
        M_fg_2 = np.zeros_like(M_fg)
        phi = self.phi0

        for i in range(self.few_groups):
            stt_i, stp_i = fg_idx[i], fg_idx[i+1]
            for j in range(self.few_groups):
                stt_j, stp_j = fg_idx[j], fg_idx[j+1]
                M_fg[i,j] = np.sum(np.trapz(M[stt_i:stp_i,stt_j:stp_j] * phi[stt_i:stp_i], E[stt_i:stp_i]) 
                        / np.trapz(phi[stt_i:stp_i],E[stt_i:stp_i]))
                M_fg_0[i,j] = np.sum(np.trapz(M[stt_i:stp_i,stt_j:stp_j] * self.Phi0[stt_i:stp_i], E[stt_i:stp_i]) 
                        / np.trapz(self.Phi0[stt_i:stp_i],E[stt_i:stp_i]))
                M_fg_2[i,j] = np.sum(np.trapz(M[stt_i:stp_i,stt_j:stp_j] * self.Phi2[stt_i:stp_i], E[stt_i:stp_i]) 
                        / np.trapz(self.Phi2[stt_i:stp_i],E[stt_i:stp_i]))

        M_fg_norm = M_fg / np.linalg.norm(M_fg)
        M_fg_0_norm = M_fg_0 / np.linalg.norm(M_fg_0)
        print(f"Update A={A} sigma_s{l} xs: {np.round(time.time()-stt,5)} s")
        print(f"L2 norm on phi0 vs Phi0 weighted gtg for A={A} and l={l}: {self.L2_norm((M_fg_norm), (M_fg_0_norm))}")

        return M_fg, M_fg_0, M_fg_2

    def Dn_coef(self, sig_s1):
        print("Calculating Phi0/Phi2 weighted Diffusion Coefficients")
        stt=time.time()
        L1_inv = np.linalg.inv(self.L1)
        L3_inv = np.linalg.inv(self.L1)
        N = self.phi0.size
        E = self.Evec[:-1]
        u = self.boundaries
        fg_idx = np.linspace(0, N, self.few_groups + 1, dtype=int)
        D0 = np.zeros((self.few_groups,self.few_groups))
        D2 = np.zeros_like(D0)
        D = np.zeros_like(D0)

        sigma_t = self.sig_t_U + self.sig_t_H
        if isinstance(self.B2, float) or ifinstance(self.B2,int): B2 = self.B2
        else: B2 = 0
        D_tr = self.diff_matrix(sigma_t,sig_s1,B2)
        phi_tr = self.phi0

        for i in range(self.few_groups):
            r0, r1 = fg_idx[i], fg_idx[i+1]
            for j in range(self.few_groups):
                c0, c1 = fg_idx[j], fg_idx[j + 1]
                # Integrate across incident-energy slice for every row, then average over the row block
                D[i, j] = (np.sum(np.trapz(D_tr[r0:r1, c0:c1] * phi_tr[c0:c1], E[c0:c1], axis=1)) 
                            / np.trapz(phi_tr[c0:c1], E[c0:c1]))
                D0[i, j] = (np.sum(np.trapz(L1_inv[r0:r1, c0:c1] * self.Phi0[c0:c1], E[c0:c1], axis=1)) 
                            / np.trapz(self.Phi0[c0:c1], E[c0:c1])).T
                D2[i, j] = (np.sum(np.trapz(L3_inv[r0:r1, c0:c1] * self.Phi2[c0:c1], E[c0:c1], axis=1)) 
                            / np.trapz(self.Phi2[c0:c1], E[c0:c1])).T
                
        print(f"D_coef Time: {np.round(time.time()-stt,5)} s")
        if self.save_data == True:
            with h5py.File(f"{self.save_dir}D_coef_{self.NH}.h5", "w") as f:
                f.create_dataset("D", data=D,compression="gzip", compression_opts=4)
                f.create_dataset("D0", data=D0,compression="gzip", compression_opts=4)
                f.create_dataset("D2", data=D2,compression="gzip", compression_opts=4)

        D_norm = D / np.linalg.norm(D)
        D0_norm = D0 / np.linalg.norm(D0)
        print(f"L2 norm on phi0 vs Phi0 weighted Diffusion Matrix: {self.L2_norm((D_norm), (D0_norm))}")

        return D, D0, D2

    def read_data(self):
        print("Reading Data From File...")
        df = pd.read_hdf(f"{self.save_dir}fluxes_{self.NH}.h5", key="df")
        self.phi0 = df['phi0'].to_numpy()
        self.phi2 = df['phi2'].to_numpy()
        self.Phi0 = df['Phi0'].to_numpy()
        self.Phi2 = df['Phi2'].to_numpy()
        self.p0   = df['phi_ref'].to_numpy()
        print("Phi Read!")

        #with h5py.File(f"{self.save_dir}Ln_{self.NH}.h5", "r") as f:
        #    self.L0 = f["L0"][:]
        #    self.L1 = f["L1"][:]
        #    self.L2 = f["L2"][:]
        #    self.L3 = f["L3"][:]
        #print("Ln Read!")

        with h5py.File(f"{self.save_dir}Sigma_sl_gtg_A238_{self.NH}.h5", "r") as f:
            self.sigma_s_gtg_U[0,:,:] = f["Sigma S0"][:]
            self.sigma_s_gtg_U[1,:,:] = f["Sigma S1"][:]
            self.sigma_s_gtg_U[2,:,:] = f["Sigma S2"][:]
            self.sigma_s_gtg_U[3,:,:] = f["Sigma S3"][:]

        with h5py.File(f"{self.save_dir}Sigma_sl_gtg_A1_{self.NH}.h5", "r") as f:
            self.sigma_s_gtg_H[0,:,:] = f["Sigma S0"][:]
            self.sigma_s_gtg_H[1,:,:] = f["Sigma S1"][:]
            self.sigma_s_gtg_H[2,:,:] = f["Sigma S2"][:]
            self.sigma_s_gtg_H[3,:,:] = f["Sigma S3"][:]
        print("Sigma_gtg Read")

        df = pd.read_csv(f"{self.save_dir}grp_vectors.csv").to_numpy()
        self.Sig_t = df[:,0]
        self.Sig_t_0 = df[:,1]
        self.Sig_t_2 = df[:,2]
        self.Sig_f = df[:,3]
        self.Sig_f_0 = df[:,4]
        self.Sig_f_2 = df[:,5]
        self.Chi = df[:,6]
        print("Group Vectors Read")

        with h5py.File(f"{self.save_dir}Sigma_sl_A238_{self.NH}.h5", "r") as f:
            self.Sig_s0 = f["Sigma S0"][:]
            self.Sig_s0_0 = f["Sigma S0 phi0"][:]
            self.Sig_s0_2 = f["Sigma S0 phi2"][:]
            self.Sig_s1 = f["Sigma S1"][:]
            self.Sig_s1_0 = f["Sigma S1 phi0"][:]
            self.Sig_s1_2 = f["Sigma S1 phi2"][:]
            self.Sig_s2 = f["Sigma S2"][:]
            self.Sig_s2_0 = f["Sigma S2 phi0"][:]
            self.Sig_s2_2 = f["Sigma S2 phi2"][:]
            self.Sig_s3 = f["Sigma S3"][:]
            self.Sig_s3_0 = f["Sigma S3 phi0"][:]
            self.Sig_s3_2 = f["Sigma S3 phi2"][:]

        with h5py.File(f"{self.save_dir}Sigma_sl_A1_{self.NH}.h5", "r") as f:
            self.Sig_s0 += f["Sigma S0"][:]
            self.Sig_s0_0 += f["Sigma S0 phi0"][:]
            self.Sig_s0_2 += f["Sigma S0 phi2"][:]
            self.Sig_s1 += f["Sigma S1"][:]
            self.Sig_s1_0 += f["Sigma S1 phi0"][:]
            self.Sig_s1_2 += f["Sigma S1 phi2"][:]
            self.Sig_s2 += f["Sigma S2"][:]
            self.Sig_s2_0 += f["Sigma S2 phi0"][:]
            self.Sig_s2_2 += f["Sigma S2 phi2"][:]
            self.Sig_s3 += f["Sigma S3"][:]
            self.Sig_s3_0 += f["Sigma S3 phi0"][:]
            self.Sig_s3_2 += f["Sigma S3 phi2"][:]
        print("Phi weighted xs's Read")

        with h5py.File(f"{self.save_dir}Dn_coefs.h5", "r") as f:
            self.D = f["D"][:]
            self.D0 = f["D0"][:]
            self.D2 = f["D2"][:]
        print("Diffusion Coefs Read")

    def mat_grp_constants(self,key):
        if key == 'fuel':
            Sig_t, Sig_t_0, Sig_t_2 = self.phi_weighted_sigma(self.sig_t_U + self.sig_t_O, "UO2", "total") 
            Sig_f, Sig_f_0, Sig_f_2 = self.phi_weighted_sigma(self.sigma_f, "UO2", "nu_sigma_f")
            Chi, _, _ = self.phi_weighted_sigma(self.chi, "UO2", "chi")
            sigma_s0, sigma_s0_0,sigma_s0_2 = self.phi_weighted_sigma_sl(self.sigma_s_gtg_U[0,:,:] + self.sigma_s_gtg_O[0,:,:], "UO2",0)
            sigma_s1, sigma_s1_0,sigma_s1_2 = self.phi_weighted_sigma_sl(self.sigma_s_gtg_U[1,:,:] + self.sigma_s_gtg_O[1,:,:], "UO2",1)
            sigma_s2, sigma_s2_0,sigma_s2_2 = self.phi_weighted_sigma_sl(self.sigma_s_gtg_U[2,:,:] + self.sigma_s_gtg_O[2,:,:], "UO2",2)
            sigma_s3, sigma_s3_0,sigma_s3_2 = self.phi_weighted_sigma_sl(self.sigma_s_gtg_U[3,:,:] + self.sigma_s_gtg_O[3,:,:], "UO2",3)
            D, D0, D2 = self.Dn_coef(self.sigma_s_gtg_U[1,:,:] + self.sigma_s_gtg_O[1,:,:])

        elif key == 'mod':
            Sig_t, Sig_t_0, Sig_t_2 = self.phi_weighted_sigma(self.sig_t_H + self.sig_t_O, "H2O", "total") 
            Sig_f, Sig_f_0, Sig_f_2 = self.phi_weighted_sigma(self.sigma_f, "H2O", "nu_sigma_f")
            Chi, _, _ = self.phi_weighted_sigma(self.chi, "H2O", "chi")
            sigma_s0, sigma_s0_0,sigma_s0_2 = self.phi_weighted_sigma_sl(self.sigma_s_gtg_H[0,:,:] + self.sigma_s_gtg_O[0,:,:], "H2O",0)
            sigma_s1, sigma_s1_0,sigma_s1_2 = self.phi_weighted_sigma_sl(self.sigma_s_gtg_H[1,:,:] + self.sigma_s_gtg_O[1,:,:], "H2O",1)
            sigma_s2, sigma_s2_0,sigma_s2_2 = self.phi_weighted_sigma_sl(self.sigma_s_gtg_H[2,:,:] + self.sigma_s_gtg_O[2,:,:], "H2O",2)
            sigma_s3, sigma_s3_0,sigma_s3_2 = self.phi_weighted_sigma_sl(self.sigma_s_gtg_H[3,:,:] + self.sigma_s_gtg_O[3,:,:], "H2O",3)
            D, D0, D2 = self.Dn_coef(self.sigma_s_gtg_U[1,:,:] + self.sigma_s_gtg_O[1,:,:])

        else: raise ValueError("Key needs to be 'fuel' or 'mod'")
        if self.save_data == True: self.save_grp_mat_vectors(Sig_t_0,Sig_t_2,
                                                            Sig_f_0, Sig_f_2, 
                                                            Chi, self.save_dir,key)

        if self.save_data == True: self.save_sigma_sl(sigma_s0, sigma_s0_0, sigma_s0_2,
                                                        sigma_s1, sigma_s1_0, sigma_s1_2,
                                                        sigma_s2, sigma_s2_0, sigma_s2_2,
                                                        sigma_s3, sigma_s3_0, sigma_s3_2,
                                                        key,self.save_dir,self.NH)

        if self.save_data == True: self.save_Dn(D,D0,D2,self.save_dir,key)

        # save the fluxes
        Phi0, Phi2 = self.few_group_fluxes('fuel')
        df = pd.DataFrame({ 'Phi0': Phi0, 'Phi2': Phi2})
        df.to_csv(f"{self.save_dir}few_grp_fluxes_{key}.csv")

    def upd_grp_constants(self,verbose=False):
        print("Few Group Cross-Sections")
        sig_t_U, sig_t_U_0, sig_t_U_2 = self.phi_weighted_sigma(self.sig_t_U,self.AU, "total")
        pdf = self.get_percent_diff(sig_t_U,sig_t_U_0,0)
        if verbose: print(f"{pdf:5g}, sigma_t U")

        sig_t_H, sig_t_H_0, sig_t_H_2 = self.phi_weighted_sigma(self.sig_t_H,self.AH, "total")
        pdf = self.get_percent_diff(sig_t_H,sig_t_H_0,0)
        if verbose: print(f"{pdf:5g}, sigma_t H")

        self.Sig_f, self.Sig_f_0, self.Sig_f_2 = self.phi_weighted_sigma(self.sigma_f,self.AU, "nu_sigma_f")
        pdf = self.get_percent_diff(self.Sig_f,self.Sig_f_0,0)
        if verbose: print(f"{(pdf/self.nu):5g}, nu * sigma_f")

        self.Chi, _, _ = self.phi_weighted_sigma(self.chi, self.AU, "chi")

        self.Sig_t = sig_t_U + sig_t_H
        self.Sig_t_0 = sig_t_U_0 + sig_t_H_0
        self.Sig_t_2 = sig_t_U_2 + sig_t_H_2

        if self.save_data == True: self.save_grp_vectors(self.Sig_t,self.Sig_t_0,self.Sig_t_2,
                                                            self.Sig_f, self.Sig_f_0, self.Sig_f_2, 
                                                            self.Chi, self.save_dir)

        # Uranium
        print("Uranium Group->Group Few Group Cross-Sections")
        sigma_s0_U, sigma_s0_0_U,sigma_s0_2_U = self.phi_weighted_sigma_sl(self.sigma_s_gtg_U[0,:,:], self.AU,0)
        pdf = self.get_percent_diff(sigma_s0_U,sigma_s0_0_U,0)
        if verbose: print(f"{pdf:5g}, sigma_s0 U")

        sigma_s1_U, sigma_s1_0_U,sigma_s1_2_U = self.phi_weighted_sigma_sl(self.sigma_s_gtg_U[1,:,:], self.AU,1)
        pdf = self.get_percent_diff(sigma_s1_U,sigma_s1_0_U,0)
        if verbose: print(f"{pdf:5g}, sigma_s1 U")

        sigma_s2_U, sigma_s2_0_U,sigma_s2_2_U = self.phi_weighted_sigma_sl(self.sigma_s_gtg_U[2,:,:], self.AU,2)
        pdf = self.get_percent_diff(sigma_s2_U,sigma_s2_0_U,0)
        if verbose: print(f"{pdf:5g}, sigma_s2 U")

        sigma_s3_U, sigma_s3_0_U,sigma_s3_2_U = self.phi_weighted_sigma_sl(self.sigma_s_gtg_U[3,:,:], self.AU,3)
        pdf = self.get_percent_diff(sigma_s3_U,sigma_s3_0_U,0)
        if verbose: print(f"{pdf:5g}, sigma_s3 U")

        if self.save_data == True: self.save_sigma_sl(sigma_s0_U, sigma_s0_0_U, sigma_s0_2_U,
                                                        sigma_s1_U, sigma_s1_0_U, sigma_s1_2_U,
                                                        sigma_s2_U, sigma_s2_0_U, sigma_s2_2_U,
                                                        sigma_s3_U, sigma_s3_0_U, sigma_s3_2_U,
                                                        self.AU,self.save_dir,self.NH)

        # Hydrogen
        print("Hydrogen Group->Group Few Group Cross-Sections")
        sigma_s0_H, sigma_s0_0_H, sigma_s0_2_H = self.phi_weighted_sigma_sl(self.sigma_s_gtg_H[0,:,:], self.AH,0)
        pdf = self.get_percent_diff(sigma_s0_H,sigma_s0_0_H,0)
        if verbose: print(f"{pdf:5g}, sigma_s0 H")

        sigma_s1_H, sigma_s1_0_H, sigma_s1_2_H = self.phi_weighted_sigma_sl(self.sigma_s_gtg_H[1,:,:], self.AH,1)
        pdf = self.get_percent_diff(sigma_s1_H,sigma_s1_0_H,0)
        if verbose: print(f"{pdf:5g}, sigma_s1 H")

        sigma_s2_H, sigma_s2_0_H, sigma_s2_2_H = self.phi_weighted_sigma_sl(self.sigma_s_gtg_H[2,:,:], self.AH,2)
        pdf = self.get_percent_diff(sigma_s2_H,sigma_s2_0_H,0)
        if verbose: print(f"{pdf:5g}, sigma_s2 H")

        sigma_s3_H, sigma_s3_0_H, sigma_s3_2_H = self.phi_weighted_sigma_sl(self.sigma_s_gtg_H[3,:,:], self.AH,3)
        pdf = self.get_percent_diff(sigma_s3_H,sigma_s3_0_H,0)
        if verbose: print(f"{pdf:5g}, sigma_s3 H")

        if self.save_data == True: self.save_sigma_sl(sigma_s0_H, sigma_s0_0_H, sigma_s0_2_H,
                                                        sigma_s1_H, sigma_s1_0_H, sigma_s1_2_H,
                                                        sigma_s2_H, sigma_s2_0_H, sigma_s2_2_H,
                                                        sigma_s3_H, sigma_s3_0_H, sigma_s3_2_H,
                                                        self.AH,self.save_dir,self.NH)

        """
        # Oxygen
        print("Oxygen Group->Group Few Group Cross-Sections")
        sigma_s0_O, sigma_s0_0_O, sigma_s0_2_O = self.phi_weighted_sigma_sl(self.sigma_s_gtg_O[0,:,:], self.AO,0)
        pdf = self.get_percent_diff(sigma_s0_O,sigma_s0_0_O,0)
        if verbose: print(f"{pdf:5g}, sigma_s0 O")

        sigma_s1_O, sigma_s1_0_O, sigma_s1_2_O = self.phi_weighted_sigma_sl(self.sigma_s_gtg_O[1,:,:], self.AO,1)
        pdf = self.get_percent_diff(sigma_s1_O,sigma_s1_0_O,0)
        if verbose: print(f"{pdf:5g}, sigma_s1 O")

        sigma_s2_O, sigma_s2_0_O, sigma_s2_2_O = self.phi_weighted_sigma_sl(self.sigma_s_gtg_O[2,:,:], self.AO,2)
        pdf = self.get_percent_diff(sigma_s2_O,sigma_s2_0_O,0)
        if verbose: print(f"{pdf:5g}, sigma_s2 O")

        sigma_s3_O, sigma_s3_0_O, sigma_s3_2_O = self.phi_weighted_sigma_sl(self.sigma_s_gtg_O[3,:,:], self.AO,3)
        pdf = self.get_percent_diff(sigma_s3_O,sigma_s3_0_O,0)
        if verbose: print(f"{pdf:5g}, sigma_s3 O")

        if self.save_data == True: self.save_sigma_sl(sigma_s0_O, sigma_s0_0_O, sigma_s0_2_O,
                                                        sigma_s1_O, sigma_s1_0_O, sigma_s1_2_O,
                                                        sigma_s2_O, sigma_s2_0_O, sigma_s2_2_O,
                                                        sigma_s3_O, sigma_s3_0_O, sigma_s3_2_O,
                                                        self.AO,self.save_dir,self.NH)
        """
        self.Sig_s0 = sigma_s0_U + sigma_s0_H
        self.Sig_s2 = sigma_s2_U + sigma_s2_H
        self.Sig_s0_0 = sigma_s0_0_U + sigma_s0_0_H
        self.Sig_s0_2 = sigma_s0_2_U + sigma_s0_2_H
        self.Sig_s2_0 = sigma_s2_0_U + sigma_s2_0_H
        self.Sig_s2_2 = sigma_s2_2_U + sigma_s2_2_H


        # calulate diffusion coefs
        self.D, self.D0, self.D2 = self.Dn_coef(self.sigma_s_gtg_U[1,:,:] + self.sigma_s_gtg_H[1,:,:])
        pdf = self.get_percent_diff(self.D,self.D0,0)
        if verbose: print(f"{pdf:5g}, Diffusion Coef 0")

        if self.save_data == True: self.save_Dn(self.D,self.D0,self.D2,self.save_dir,key=None)

    @staticmethod
    def solve_sp3_eqns_new(
        B2: float,
        D0: np.ndarray, D2: np.ndarray,
        Sig_t_0: np.ndarray, Sig_t_2: np.ndarray,
        Sig_s0_0: np.ndarray, Sig_s0_2: np.ndarray, Sig_s2_2: np.ndarray,
        chi: np.ndarray,
        nuSigf0: np.ndarray, nuSigf2: np.ndarray,
        normalize: str = "sum_phi0",  # "sum_phi0" or "sum_scalar"
        norm_value: float = 1.0,
    ):
        """
        Infinite-medium buckling solve for Eq. (59a/59b):
          -D0 ∇² Φ0 + (Σt0-Σs0_0) Φ0 = χ(νΣf0·Φ0 - 2 νΣf2·Φ2) + 2(Σt2-Σs0_2) Φ2
          -D2 ∇² Φ2 + (Σt2-Σs2_2) Φ2 = 2(Σt0 Φ0 - 2Σt2 Φ2) - 2(Σs0_0 Φ0 - 2Σs0_2 Φ2) - 2χ(νΣf0·Φ0 - 2 νΣf2·Φ2)
    
        Buckling: -∇² -> B², so -D∇² -> +B² D.
        Homogeneous system needs a normalization constraint.
        """
        G = chi.size
        chi = chi.astype(np.float64)
        chi = chi / (chi.sum())
    
        # Make diagonal total matrices
        T0 = np.diag(Sig_t_0)
        T2 = np.diag(Sig_t_2)
    
        # Convenience
        R00 = (T0 - Sig_s0_0)              # (Σt0 - Σs0_0)
        C02 = 2.0 * (T2 - Sig_s0_2)        # 2(Σt2 - Σs0_2)
    
        # Fission "outer-product" operator pieces:
        # χ * (nuSigf0·Φ0 - 2 nuSigf2·Φ2) = (χ nuSigf0)Φ0 + (χ * (-2 nuSigf2))Φ2
        F00 = np.outer(chi, nuSigf0)             # GxG
        F02 = np.outer(chi, -2.0 * nuSigf2)      # GxG
    
        # Build 2Gx2G matrix M59 such that M59 @ [Φ0; Φ2] = 0
        # Eq (59a): (B² D0 + R00)Φ0  + ( -C02 )Φ2  + ( -F00 Φ0 - F02 Φ2 ) = 0
        A00 = (B2 * D0) + R00 - F00
        A02 = (-C02)    - F02
    
        # Eq (59b): (B² D2 + (T2 - S2_2) + 4T2 -4S0_2)Φ2 + (-2T0 +2S0_0)Φ0 + 2χ(...) = 0
        # The "+2χ(...)" contributes +2*F00 Φ0 +2*F02 Φ2
        B20 = (-2.0 * T0) + (2.0 * Sig_s0_0) + (2.0 * F00)
        B22 = (B2 * D2) + (T2 - Sig_s2_2) + (4.0 * T2) - (4.0 * Sig_s0_2) + (2.0 * F02)
    
        M = np.block([[A00, A02],
                      [B20, B22]]).astype(np.float64)
    
        # Add normalization constraint by replacing one row
        b = np.zeros(2*G, dtype=np.float64)
    
        if normalize == "sum_phi0":
            # enforce sum_g Φ0_g = norm_value
            M[0, :] = 0.0
            M[0, 0:G] = 1.0
            b[0] = norm_value
        elif normalize == "sum_scalar":
            # enforce sum_g (ϕ_g) = sum_g(Φ0_g - 2Φ2_g) = norm_value
            M[0, :] = 0.0
            M[0, 0:G] = 1.0
            M[0, G:2*G] = -2.0
            b[0] = norm_value
        else:
            raise ValueError("normalize must be 'sum_phi0' or 'sum_scalar'")
    
        x = np.linalg.solve(M, b)
        Phi0 = x[:G]
        Phi2 = x[G:]
        return Phi0, Phi2

    @staticmethod
    def solve_sp3_eqns_conventional(
        B2: float,
        D: np.ndarray,
        Sig_t: np.ndarray,
        Sig_s0: np.ndarray,
        Sig_s2: np.ndarray,
        chi: np.ndarray,
        nuSigf: np.ndarray,
        normalize: str = "sum_phi0",
        norm_value: float = 1.0,
    ):
        """
        Infinite-medium buckling solve for conventional Eq. (61a/61b):
    
          -D ∇² Φ0 + (Σt-Σs0)Φ0 = χ νΣf (Φ0-2Φ2) + 2(Σt-Σs0)Φ2
          -D ∇² Φ2 + (Σt-Σs2)Φ2 = 2(Σt-Σs0)(Φ0-2Φ2) - 2 χ νΣf (Φ0-2Φ2)
    
        We implement χ νΣf (Φ0-2Φ2) as outer(χ,nuSigf) @ (Φ0-2Φ2).
        """
        G = chi.size
        chi = chi.astype(np.float64)
        chi = chi / (chi.sum() + 1e-300)
    
        T = np.diag(Sig_t)
        R0 = (T - Sig_s0)
        R2 = (T - Sig_s2)
    
        F = np.outer(chi, nuSigf)   # χ * (nuSigf·something)
    
        # Let s = (Φ0 - 2Φ2). Then fission term is F @ s.
        # Eq61a: (B²D + R0)Φ0  - 2R0 Φ2  - F(Φ0 - 2Φ2) = 0
        A00 = (B2 * D) + R0 - F
        A02 = (-2.0 * R0) + (2.0 * F)
    
        # Eq61b: (B²D + R2)Φ2  - 2R0(Φ0 - 2Φ2) + 2F(Φ0 - 2Φ2) = 0
        # Expand: (-2R0 + 2F)Φ0 + (B²D + R2 + 4R0 - 4F)Φ2 = 0
        B20 = (-2.0 * R0) + (2.0 * F)
        B22 = (B2 * D) + R2 + (4.0 * R0) - (4.0 * F)
    
        M = np.block([[A00, A02],
                      [B20, B22]]).astype(np.float64)
    
        b = np.zeros(2*G, dtype=np.float64)
        if normalize == "sum_phi0":
            M[0, :] = 0.0
            M[0, 0:G] = 1.0
            b[0] = norm_value
        elif normalize == "sum_scalar":
            M[0, :] = 0.0
            M[0, 0:G] = 1.0
            M[0, G:2*G] = -2.0
            b[0] = norm_value
        else:
            raise ValueError("normalize must be 'sum_phi0' or 'sum_scalar'")
    
        x = np.linalg.solve(M, b)
        return x[:G], x[G:]
    
    def finite_difference(self,tol=1e-5,omega=0.25):
        print("Begin Finite Difference Sweep")
        def sp3_block_thomas(
            D0, D2, A00, A02, A20, A22, dx, N,
            Q0=None, Q2=None, shift=0.0, pin=None, pin_value=1.0
        ):
            G = D0.shape[0]; m = 2*G; dx2 = dx*dx
        
            # Blocks common to all rows
            D0_off = (1.0/dx2) * D0
            D2_off = (1.0/dx2) * D2
            OFF = -np.block([[D0_off, np.zeros((G,G))],
                             [np.zeros((G,G)), D2_off]])            # −D_off
            CENTER = np.block([[ (2.0/dx2)*D0,            np.zeros((G,G))],
                               [ np.zeros((G,G)),          (2.0/dx2)*D2]]) \
                     + np.block([[A00, A02],
                                 [A20, A22]])                        # 2*D_off + A
        
            # Allocate variable tri-diagonal arrays
            Atil = np.empty((N, m, m))
            B    = np.empty((N, m, m))   # lower
            C    = np.empty((N, m, m))   # upper
            rhs  = np.zeros((N, m))
        
            # Fill interior rows
            for i in range(N):
                Atil[i] = CENTER
                B[i]    = OFF
                C[i]    = OFF
                if shift > 0.0:
                    Atil[i, np.arange(m), np.arange(m)] += shift
        
            # Correct Neumann BCs: double the edge neighbor coupling, not the diagonal
            B[0][:]      = 0.0                # no lower neighbor for i=0
            C[0][:]      = 2.0 * OFF          # couple twice to i=1
        
            C[N-1][:]    = 0.0                # no upper neighbor for i=N-1
            B[N-1][:]    = 2.0 * OFF          # couple twice to i=N-2
        
            # Sources
            if Q0 is not None: rhs[:, :G] = Q0
            if Q2 is not None: rhs[:, G:] = Q2
        
            # Optional pin (inhomogeneous row)
            if pin is not None:
                i_pin, comp = pin
                Atil[i_pin,:,:] = 0.0
                Atil[i_pin, comp, comp] = 1.0
                rhs[i_pin,:] = 0.0
                rhs[i_pin, comp] = pin_value
        
            # Forward elimination
            for i in range(1, N):
                T = np.linalg.solve(Atil[i-1], B[i])        # A_{i-1}^{-1} B_i
                Atil[i] = Atil[i] - C[i-1] @ T
                rhs[i]  = rhs[i]  - C[i-1] @ np.linalg.solve(Atil[i-1], rhs[i-1])
        
            # Back substitution
            X = np.zeros((N, m))
            X[-1] = np.linalg.solve(Atil[-1], rhs[-1])
            for i in range(N-2, -1, -1):
                X[i] = np.linalg.solve(Atil[i], rhs[i] - C[i] @ X[i+1])
        
            return X[:, :G], X[:, G:]
        
        G, N = self.few_groups, self.num_cells
        Phi0 = np.ones((N, G)); Phi2 = np.ones((N, G))
        dx2 = self.dx * self.dx
    
        # Matrices (GxG), make sure Sig_f_* include ν; if not, multiply by nu
        Sig_t_0 = np.diag(self.Sig_t_0)
        Sig_t_2 = np.diag(self.Sig_t_2)
        S00 = self.Sig_s0_0
        S02 = self.Sig_s0_2
        S22 = self.Sig_s2_2
        chi = self.Chi / np.sum(self.Chi)
        F0 = np.outer(chi, self.Sig_f_0)   # χ νΣf^0
        F2 = np.outer(chi, self.Sig_f_2)   # χ νΣf^2
        
        # A-blocks from the equations above
        A00 = (Sig_t_0 - S00) - F0
        A02 = -2*(Sig_t_2 - S02) + 2*F2
        A20 = -2*Sig_t_0 + 2*S00 + 2*F0
        A22 = (Sig_t_2 - S22) + 4*Sig_t_2 - 4*S02 - 4*F2
        stt_transport = time.time()
        # initial guess (positive)
        Phi0 = np.ones((N,G))
        Phi2 = np.zeros((N,G))
        k = 1.0
    
        # diag fission operators (if F0,F2 are vectors)
        F0m = np.diag(F0) if F0.ndim == 1 else F0
        F2m = np.diag(F2) if F2.ndim == 1 else F2
    
        # helper: compute fission production at each cell (G-vector)
        def fiss_src(P0_i, P2_i):
            return chi * (F0m @ P0_i - 2.0*(F2m @ P2_i))  # (G,)
    
        # Outer power iteration
        max_outer = 200
        tol_k=1e-8
        tol_phi=1e-8
        for it in range(max_outer):
            Phi0_old = Phi0.copy()
            Phi2_old = Phi2.copy()
            k_old = k
    
            # build Q0,Q2 from old fluxes
            Q0 = np.zeros((N,G))
            Q2 = np.zeros((N,G))
            for i in range(N):
                q = fiss_src(Phi0_old[i], Phi2_old[i])
                Q0[i] = q / k
                Q2[i] = (-2.0*q) / k
    
            # solve fixed-source system with Thomas (your corrected Neumann BC version)
            Phi0, Phi2 = sp3_block_thomas(
                D0=self.D0, D2=self.D2,
                A00=A00, A02=A02, A20=A20, A22=A22,
                dx=self.dx, N=N,
                Q0=Q0, Q2=Q2,
                pin=None, shift=0.0
            )
    
            # update k by Rayleigh quotient (ratio of new to old production)
            # (any consistent inner product works; this is a common choice)
            P_new = 0.0
            P_old = 0.0
            for i in range(N):
                P_new += np.sum(fiss_src(Phi0[i],   Phi2[i]))
                P_old += np.sum(fiss_src(Phi0_old[i], Phi2_old[i]))
            k = k_old * (P_new / (P_old + 1e-300))
    
            # normalize flux to prevent overflow/underflow
            norm = np.max(Phi0)
            if norm > 0:
                Phi0 /= norm
                Phi2 /= norm
    
            # convergence checks
            dk = abs(k - k_old) / (abs(k_old) + 1e-300)
            dphi = np.max(np.abs(Phi0 - Phi0_old))
            print(f"it={it+1}  k={k:.10f}  rel_dk={dk:.3e}  dphi_inf={dphi:.3e}")
    
            if dk < tol_k and dphi < tol_phi:
                break
    
        print(k)

        """
        Q0 = np.ones((N, G), dtype=np.float64)   # uniform source in all groups
        Q2 = np.zeros((N, G), dtype=np.float64)
        Phi0, Phi2 = sp3_block_thomas(self.D0, self.D2, A00, A02, A20, A22,
                              self.dx, self.num_cells, Q0=Q0, Q2=Q2,
                              pin=None, shift=0.0)
        
        print(Phi0.shape)
        print(Phi2.shape)
        """
        #print(Phi0,Phi2)
        # Diffusion blocks
        #D0 = self.D0
        #D2 = self.D2
        #D0_diag = (2.0/dx2) * D0
        #D2_diag = (2.0/dx2) * D2
        #D0_off  = (1.0/dx2) * D0
        #D2_off  = (1.0/dx2) * D2
       # 
       # # Per-cell center matrix (constant across i)
       # M = np.block([[D0_diag + A00,  A02],
       #               [A20,            D2_diag + A22]])
       # 
       # Minv = np.linalg.inv(M)      # outside Numba, once
   # 
   #     gs_sor_fd(Phi0, Phi2, D0_off, D2_off, Minv, self.max_iters, tol, omega)

        """
        # being finite difference for new method
        Phi0 = np.ones((self.num_cells,self.few_groups)) 
        Phi2 = np.ones((self.num_cells,self.few_groups))

        # build data
        sig_t_0 = np.diag(self.Sig_t_0)
        sig_t_2 = np.diag(self.Sig_t_2)
        sig_s0_0 = (self.Sig_s0_0)
        sig_s0_2 = (self.Sig_s0_2)
        sig_s2_0 = (self.Sig_s2_0)
        sig_s2_2 = (self.Sig_s2_2)
        chi = self.Chi
        sig_f_0 = self.Sig_f_0
        sig_f_2 = self.Sig_f_2

        # NEGATIVE BUILT IN HERE
        D0_inv = np.linalg.inv(self.D0)
        D2_inv = np.linalg.inv(self.D2)
        dx2 = self.dx * self.dx

        it = 0
        converged = False
        print(f"Length: {self.L}cm, dx: {self.dx}, Num Nodes: {self.num_cells}")
        stt_transport = time.time()
        # begin Jacobi Solve
        while not converged and it < self.max_iters:
            # Reflecting BC
            Phi0[0,:]  = Phi0[1,:]
            Phi2[0,:]  = Phi2[1,:]
            Phi0[-1,:] = Phi0[-2,:]
            Phi2[-1,:] = Phi2[-2,:]
            phi0_prev = Phi0.copy()
            phi2_prev = Phi2.copy()
            for i in range(1,self.num_cells-1):
                # Phi0(position, Energy)
                t1 = (sig_t_0 - sig_s0_0) @ phi0_prev[i,:]
                t2 = np.outer(chi,sig_f_0) @ phi0_prev[i,:] - 2 * np.outer(chi,sig_f_2) @ phi2_prev[i,:]
                t3 = 2 * (sig_t_2 - sig_s0_2) @ phi0_prev[i,:]
                RHS = (dx2 * D0_inv) @ (-t1 + t2 + t3)
                Phi0[i+1,:] = 2 * phi0_prev[i,:] - phi0_prev[i-1,:] + RHS 
    
                # Phi2(position, Energy)
                t1 = (sig_t_2 - sig_s2_2) @ phi2_prev[i,:]
                t2 = sig_t_0 @ phi0_prev[i,:] - 2 * (sig_t_2 @ phi2_prev[i,:])
                t3 = sig_s0_0 @ phi0_prev[i,:] - 2 * (sig_s0_2 @ phi2_prev[i,:])
                t4 = np.outer(chi, sig_f_0) @ phi0_prev[i,:] - 2 * np.outer(chi,sig_f_2) @ phi2_prev[i,:]
                RHS = (2 * dx2 * D2_inv) @ (-t1 + t2 -t3 - t4)
                Phi2[i+1,:] = 2 * phi2_prev[i,:] - phi2_prev[i-1,:] + RHS

            # Reflecting BC
            Phi0[0,:]  = Phi0[1,:]
            Phi2[0,:]  = Phi2[1,:]
            Phi0[-1,:] = Phi0[-2,:]
            Phi2[-1,:] = Phi2[-2,:]

            L2_0 = self.L2_norm(phi0_prev,Phi0)
            L2_2 = self.L2_norm(phi2_prev,Phi2)
            print(f"L2 Phi0={L2_0:.5e}, Phi2={L2_2:.5e}, it={it+1}")
            converged = (max(L2_0, L2_2) < tol)

            if max(L2_0, L2_2) > 1e3: raise ValueError("Sp3 Solve is Diverging")
            it += 1
        while not converged and it < self.max_iters:
            # enforce reflecting BCs on the *previous* iterate
            Phi0[0,:]  = Phi0[1,:]
            Phi2[0,:]  = Phi2[1,:]
            Phi0[-1,:] = Phi0[-2,:]
            Phi2[-1,:] = Phi2[-2,:]
    
            phi0_prev = Phi0.copy()
            phi2_prev = Phi2.copy()
    
            Phi0_new = phi0_prev.copy()
            Phi2_new = phi2_prev.copy()
    
            # build once
            F0 = np.outer(chi, sig_f_0)
            F2 = np.outer(chi, sig_f_2)
    
            # update the *center* i using neighbors from prev (pure Jacobi)
            for i in range(1, self.num_cells - 1):
                t1_0 = (sig_t_0 - sig_s0_0) @ phi0_prev[i,:]
                t2_0 = F0 @ phi0_prev[i,:] - 2.0 * (F2 @ phi2_prev[i,:])
                t3_0 = 2.0 * (sig_t_2 - sig_s0_2) @ phi0_prev[i,:]
    
                # discrete Laplacian φ'' ≈ (φ_{i+1} - 2 φ_i + φ_{i-1}) / dx^2
                # Solve D0 * φ'' = (-t1 + t2 + t3)  =>  φ'' = D0_inv @ (-t1 + t2 + t3)
                rhs0 = dx2 * (D0_inv @ (-t1_0 + t2_0 + t3_0))
                Phi0_cand = (phi0_prev[i+1,:] - 2.0*phi0_prev[i,:] + phi0_prev[i-1,:]) + rhs0
                # Rearrange to update the center:
                Phi0_new[i,:] = phi0_prev[i,:] + 0.5 * (Phi0_cand)  
    
                t1_2 = (sig_t_2 - sig_s2_2) @ phi2_prev[i,:]
                t2_2 = (sig_t_0 @ phi0_prev[i,:]) - 2.0 * (sig_t_2 @ phi2_prev[i,:])
                t3_2 = (sig_s0_0 @ phi0_prev[i,:]) - 2.0 * (sig_s0_2 @ phi2_prev[i,:])
                t4_2 = F0 @ phi0_prev[i,:] - 2.0 * (F2 @ phi2_prev[i,:])
    
                rhs2 = 2.0 * dx2 * (D2_inv @ (-t1_2 + t2_2 - t3_2 - t4_2))
                Phi2_cand = (phi2_prev[i+1,:] - 2.0*phi2_prev[i,:] + phi2_prev[i-1,:]) + rhs2
                Phi2_new[i,:] = phi2_prev[i,:] + 0.5 * (Phi2_cand)
    
            # reflecting BCs on the *new* iterate
            Phi0_new[0,:]  = Phi0_new[1,:]
            Phi2_new[0,:]  = Phi2_new[1,:]
            Phi0_new[-1,:] = Phi0_new[-2,:]
            Phi2_new[-1,:] = Phi2_new[-2,:]
    
            # under-relax
            Phi0 = (1 - omega) * phi0_prev + omega * Phi0_new
            Phi2 = (1 - omega) * phi2_prev + omega * Phi2_new
    
            L2_0 = self.L2_norm(phi0_prev, Phi0)
            L2_2 = self.L2_norm(phi2_prev, Phi2)
            print(f"L2 Phi0={L2_0:.5e}, Phi2={L2_2:.5e}, it={it+1}")
    
            converged = (max(L2_0, L2_2) < tol)
            if max(L2_0, L2_2) > 1e3:
                raise ValueError("Sp3 Solve is Diverging")
            it += 1
        """
        
        #return k, Phi0, Phi2
        print(f"Convergence Time: {np.round(time.time() - stt_transport, 5)} seconds")
        nodes = np.linspace(0,self.L,self.num_cells)
        phi0 = Phi0 - 2*Phi2
        plt.figure()
        for g in range(self.few_groups): plt.plot(nodes,phi0[:,g],label=fr"$\Phi_0$, g = {g+1}")
        plt.xlabel("Position (cm)")
        plt.ylabel(r"$\phi_0 (x,g)$")
        plt.legend()
        plt.grid(which="Both")
        plt.savefig("New_SP3_solve_phi0.png")
        plt.clf()

        plt.figure()
        for g in range(self.few_groups): plt.plot(nodes,Phi2[:,g],label=fr"$\phi_2$, g = {g+1}")
        plt.xlabel("Position (cm)")
        plt.ylabel(r"$\phi_2 (x,g)$")
        plt.legend()
        plt.grid(which="Both")
        plt.savefig("New_SP3_solve_phi2.png")
        plt.clf()

    def run(self, verbose = False, transport = False):
        self.initial_flux()
        if self.fromH5 == True: self.read_data()
        else:
            print(f"Starting calculation. Saving Data = {self.save_data}")
            print(f'Build Sigma_gtg and Ln for Uranium, NU = {self.NU}')
            self.calc_Ln(self.AU, self.sig_t_U, self.sig_s0_U,verbose)
            print(f'Build Sigma_gtg and Ln for Hydrogen, NH = {self.NH}')
            self.calc_Ln(self.AH, self.sig_t_H, self.sig_s0_H,verbose)
            #print(f'Build Sigma_gtg and Ln for Oxygen, NO = {self.NO}')
            #self.calc_Ln(self.AO, self.sig_t_O, self.sig_s0_O,verbose)
            if verbose: self.save_Ln()

            device = "cuda" if torch.cuda.is_available() else "cpu"
            if isinstance(self.B2, float) or isinstance(self.B2, int):  
                if device == "cuda": self.calc_phi_torch()
                else: self.calc_phi()
            else: self.calc_phi_B2()
            self.calc_Phi()
            self.plot_fluxes()
            if verbose:
                self.plot_flux_diff_single_axis()
                self.plot_flux_diff()
            if self.save_data == True: self.save_fluxes()
            #self.mat_grp_constants("fuel")
            self.upd_grp_constants()

        # few group fluxes
        Phi0_fg, Phi2_fg = self.few_group_fluxes(key=None)
        #if transport: self.finite_difference()
        Phi0_sp3_new, Phi2_sp3_new = self.solve_sp3_eqns_new(
            self.B2,
            self.D0, self.D2,
            self.Sig_t_0, self.Sig_t_2,
            self.Sig_s0_0, self.Sig_s0_2, self.Sig_s2_2,
            self.Chi,
            self.Sig_f_0, self.Sig_f_2)

        Phi0_sp3_conv, Phi2_sp3_conv = self.solve_sp3_eqns_conventional(
            self.B2,
            self.D,
            self.Sig_t,
            self.Sig_s0,
            self.Sig_s2,
            self.Chi,
            self.Sig_f,                
        )

        # plot and compare
        Efg = np.exp(np.linspace(np.log(self.Emin),np.log(self.E0),self.few_groups+1))
        Efg = np.flip(Efg)
        plt.figure(figsize=(8,6))

        plt.step(Efg[:-1], Phi0_sp3_new, where='post', label=r'$\Phi_0^{new}$')
        plt.step(Efg[:-1], Phi0_sp3_conv, where='post', label=r'$\Phi_0^{conv}$')

        plt.xlabel('Energy (MeV)')
        plt.ylabel(r'$\Phi_0$')
        plt.legend()
        plt.grid(True, which='both')

        plt.xscale('log')
        plt.savefig("sp3_fg_Phi0_comp.png")


#        print(Phi0_sp3_new,Phi0_sp3_conv)

    @staticmethod
    def alpha_fn(A): return ((A - 1.0)/(A + 1.0)) ** 2

    @staticmethod
    def lga_fn(alpha): return -np.log(alpha) if alpha != 0 else np.inf

    @staticmethod
    def _Ln(l, sigma_t, sigma_gtg): return (2*l+1) * (np.diag(sigma_t) - sigma_gtg)

    @staticmethod
    @njit(parallel=True, fastmath=True)
    def gen_sig_sn_gtg(A, sig_s0, order, boundaries, gmax_vec, alpha, E0, phi, du, n_sub = 8):
        # du is a vector of differences
        assert order <= 4, (f"Order {order} Not Supproted!")
        G = boundaries.size
        #du = boundaries[1] - boundaries[0]
        den = (1 - alpha) * du
        lga = -np.log(alpha) if A != 1 else np.inf
        sigma_gtg = np.zeros((order, G - 1, G - 1))
        Am1 = A-1
        Ap1 = A+1
        K = 5 * A*A - 1

        for l in range(order):
            for gp in prange(G - 1):
                x1 = boundaries[gp]
                x2 = boundaries[gp+1]

                for g in range(gp, min(gmax_vec[gp], G-1)):
                    y1 = boundaries[g]
                    y2 = boundaries[g+1]
                    c = max(x1, y1 - lga)
                    length = x2 - c
                    n_steps_base = int(np.ceil(length / du[gp]))
                    #n_steps_base = int(np.ceil(length / du))
                    if n_steps_base < 1: n_steps_base = 1
                    n_steps = n_steps_base * n_sub
                    dx = length / n_steps

                    # integrate
                    acc = 0
                    for s in range(n_steps):
                        xm = c + (s + .5) * dx
                        a = y1 if xm < y1 else xm
                        bx = xm + lga
                        b = y2 if bx > y2 else bx

                        if l == 0: val = np.exp(xm-a) - np.exp(xm-b)

                        elif l == 1:
                            if A == 1: val = (1/3) * Ap1 * (np.exp(1.5 * (xm - a)) - np.exp(1.5 * (xm - b)))
                            else: val = (Am1 * (np.exp(.5 * (xm - b)) - np.exp(0.5 * (xm - a))) 
                                 + (1/3) * Ap1 * (np.exp(1.5 * (xm - a)) - np.exp(1.5 * (xm - b))))
                                            
                        elif l == 2:
                            if A == 1:
                                val = (0.25 * (1 - 3*A*A) * (np.exp(xm - a) - np.exp(xm - b)) 
                                    + .1875 * Ap1*Ap1 * (np.exp(2*(xm - a)) - np.exp(2*(xm - b))))

                            else:
                                val = ((.375)*(Am1*Am1)*(b - a)  + 0.25 * (1 - 3*A*A) * (np.exp(xm - a) - np.exp(xm - b)) 
                                    + .1875 * Ap1*Ap1 * (np.exp(2*(xm - a)) - np.exp(2*(xm - b))))

                        elif l == 3:
                            if A == 1:
                                val = 0.0625 * (
                                  +  2 * (Ap1**3) * (np.exp( 2.5 * (xm - a)) - np.exp( 2.5 * (xm - b)))
                                  -  2 * Ap1 * K   * (np.exp( 1.5 * (xm - a)) - np.exp( 1.5 * (xm - b))))
                            else:
                                val = 0.0625 * (
                               10 * (Am1*Am1*Am1) * (np.exp(-0.5 * (xm - a)) - np.exp(-0.5 * (xm - b)))
                              +  2 * (Ap1*Ap1*Ap1) * (np.exp( 2.5 * (xm - a)) - np.exp( 2.5 * (xm - b)))
                              +  6 * Am1 * K   * (np.exp( 0.5 * (xm - a)) - np.exp( 0.5 * (xm - b)))
                              -  2 * Ap1 * K   * (np.exp( 1.5 * (xm - a)) - np.exp( 1.5 * (xm - b))))

                        acc += val

                    sig_foo = (sig_s0[gp] * phi[gp] * acc * dx ) / (den[gp] * phi[gp])
                    sigma_gtg[l, gp, g] = (sig_s0[gp] * phi[gp] * acc * dx ) / (den[gp] * phi[gp])

        return sigma_gtg

    @staticmethod
    def print_mat_properties(LHS):
        """prints important matrix properties"""
        def is_diagonally_dominant(A):
            """checks if a matrix is diagonally dominant"""
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
    def save_sig_s_gtg(S,A,save_dir,NH):
        print(f"Saving A={A} Data")
        with h5py.File(f"{save_dir}Sigma_sl_gtg_A{A}_{NH}.h5", "w") as f:
            f.create_dataset("Sigma S0", data=S[0,:,:],compression="gzip", compression_opts=4)
            f.create_dataset("Sigma S1", data=S[1,:,:],compression="gzip", compression_opts=4)
            f.create_dataset("Sigma S2", data=S[2,:,:],compression="gzip", compression_opts=4)
            f.create_dataset("Sigma S3", data=S[3,:,:],compression="gzip", compression_opts=4)

    @staticmethod
    def save_sigma_sl(sigma_s0, sigma_s0_0, sigma_s0_2,
                        sigma_s1, sigma_s1_0, sigma_s1_2,
                        sigma_s2, sigma_s2_0, sigma_s2_2,
                        sigma_s3, sigma_s3_0, sigma_s3_2,
                        A,save_dir,NH):
        # save
        with h5py.File(f"{save_dir}Sigma_sl_A{A}.h5", "w") as f:
            f.create_dataset("Sigma S0", data=sigma_s0.T,compression="gzip", compression_opts=4)
            f.create_dataset("Sigma S0 phi0", data=sigma_s0_0.T,compression="gzip", compression_opts=4)
            f.create_dataset("Sigma S0 phi2", data=sigma_s0_2.T,compression="gzip", compression_opts=4)
            f.create_dataset("Sigma S1", data=sigma_s1.T,compression="gzip", compression_opts=4)
            f.create_dataset("Sigma S1 phi0", data=sigma_s1_0.T,compression="gzip", compression_opts=4)
            f.create_dataset("Sigma S1 phi2", data=sigma_s1_2.T,compression="gzip", compression_opts=4)
            f.create_dataset("Sigma S2", data=sigma_s2.T,compression="gzip", compression_opts=4)
            f.create_dataset("Sigma S2 phi0", data=sigma_s2_0.T,compression="gzip", compression_opts=4)
            f.create_dataset("Sigma S2 phi2", data=sigma_s2_2.T,compression="gzip", compression_opts=4)
            f.create_dataset("Sigma S3", data=sigma_s3.T,compression="gzip", compression_opts=4)
            f.create_dataset("Sigma S3 phi0", data=sigma_s3_0.T,compression="gzip", compression_opts=4)
            f.create_dataset("Sigma S3 phi2", data=sigma_s3_2.T,compression="gzip", compression_opts=4)

    @staticmethod
    def save_Dn(D,D0,D2,save_dir,key):
        # save
        with h5py.File(f"{save_dir}Dn_coefs_{key}.h5", "w") as f:
            f.create_dataset("D",  data=D,compression="gzip", compression_opts=4)
            f.create_dataset("D0", data=D0,compression="gzip", compression_opts=4)
            f.create_dataset("D2", data=D2,compression="gzip", compression_opts=4)

    @staticmethod
    def normalize(vec):
        if vec.ndim != 1: raise ValueError("Vector isn't 1D")
        return vec / np.sum(vec)    

    @staticmethod
    def L2_norm(A,B): 
        A = A / np.sum(A)
        B = B / np.sum(B)
        assert A.shape == B.shape, ("L2 norm shape mismatch!")
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
            else: return b*np.log((1+b)/(1-b)) / (3*(-2+b**(-1)*np.log((1+b)/(1-b))))

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

        return np.linalg.inv(sigma_tr) / 3

    @staticmethod
    def get_percent_diff(M1,M2,index):
        """Get percent difference is index (i,i) in a matrix"""
        assert M1.shape == M2.shape
        if M1.ndim == 1: return 100 * (np.abs(M1[index] - M2[index]) 
                                        / (np.abs(M1[index])))
        else: return 100 * (np.abs(M1[index,index] - M2[index,index]) 
                    / (np.abs(M1[index,index])))

####################### RUN ########################
process = psutil.Process(os.getpid())
stt = time.time()
NH = 1
xs_tol = 5 # percent
B2 = .0
nbins = 15000
#B2 = np.linspace(-.025,.025,6)
few_groups = 8
fromH5 = False
dx = .0005
verbose = False
adaptive = False

# init class
sp3 = Sp3(nbins, B2, NH, few_groups, fromH5, dx, xs_tol, adaptive)
sp3.run(verbose)

stp = time.time()
print(f"Calculation Time = {np.round((stp - stt),6)}")
mem = process.memory_info().rss / 1e9
print(f"Memory used: {mem:.3f} GB")

