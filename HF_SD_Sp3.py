import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import time
import math
import h5py
from numba import njit, prange
from scipy.special import expi
import psutil

class Sp3:
    def __init__(self, nbins, B2, NH, few_groups, fromH5):
        self.data_dir = 'data/'
        self.save_dir = "/scratch/bckiedro_root/bckiedro0/rsshast/Sp3/results/"
        self.chart_dir = "results/charts/"
        self.B2 = B2
        self.tol = 1e-8 # loop break condition
        self.E0 = 1e7
        self.Emin = 1e-2
        self.nbins = nbins + 1
        self.leg_order = 4
        self.AH = 1
        self.NH = NH
        self.AU = 238
        self.NU = 1
        self.fromH5 = fromH5
        assert nbins % few_groups == 0, ("Number of fine bins must be multiple of number of coarse bins")

        chi35 = pd.read_csv(f'{self.data_dir}chi_u235.txt', sep = '\t',header = 0)
        H1 = pd.read_csv(f'{self.data_dir}xs_h1_T293k.txt', sep  = '\t', header = 0)
        U238 = pd.read_csv(f'{self.data_dir}xs_u238_T293k.txt',sep  = '\t', header = 0)
        sigma_f = pd.read_csv(f'{self.data_dir}xs_u238_fission.csv', sep = ',', dtype=float).to_numpy()
        sigma_fr_U = pd.read_csv(f"{self.data_dir}sigma_fr_U.csv",sep = ';', dtype=float).to_numpy()
        sigma_fr_H = pd.read_csv(f"{self.data_dir}sigma_fr_H.csv",sep = ';', dtype=float).to_numpy()

        chi = np.array([chi35['E'],chi35['chi']]).T
        H = np.array([H1['E'],H1['sigma_t'],H1['sigma_s']]).T
        XS38 = np.array([U238['E'],U238['sigma_t'],U238['sigma_s']]).T
        chi = self.get_data(chi,self.nbins)
        H = self.get_data(H,self.nbins)
        H *= self.NH
        XS38 = self.get_data(XS38,self.nbins)
        self.chi = chi[:,1]
        self.boundaries = XS38[:,0]
        print(f"du = {self.boundaries[1] - self.boundaries[0]}")
        self.Evec = self.E0 * np.exp(-self.boundaries)
        self.sigma_fr_U = self.get_data(sigma_fr_U,self.nbins)[:,1]
        self.sigma_fr_H = self.get_data(sigma_fr_H,self.nbins)[:,1] * NH
        self.sig_t_U   = XS38[:,1]
        self.sig_s0_U  = XS38[:,2]
        self.sig_t_H   = H[:,1]
        self.sig_s0_H  = H[:,2]
        self.sigma_f   = np.flip(self.get_data(sigma_f,self.nbins)[:,1])
        self.T         = 293.15    # degrees Kelvin
        self.k         = 8.617e-5  # eV/K (Boltzmann constant)
        self.kT        = self.k * self.T
        self.m         = 1.66054e-27

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
        #self.L_comp = np.zeros_like(self.L0)
        self.save_data = False
        self.few_groups = few_groups
        self.sigma_s_gtg_U = np.zeros_like(self.L0)
        self.sigma_s_gtg_H = np.zeros_like(self.L0)

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
        #u = self.boundaries
        u = np.zeros_like(self.p0)
        # Problem setup
        chi = np.zeros_like(self.p0)
        StH = np.zeros_like(self.p0)
        SsH = np.zeros_like(self.p0)
        StU = np.zeros_like(self.p0)
        SsU = np.zeros_like(self.p0)

        for i in range(self.p0.size):
            u[i] = .5 * (self.boundaries[i+1] + self.boundaries[i])
            chi[i] = np.trapz(self.chi[i:i+2],self.boundaries[i:i+2]) / (self.boundaries[i+1] - self.boundaries[i])
            self.Eplot[i] = .5 * (self.Evec[i+1] + self.Evec[i]) 
            StH[i] = np.trapz(self.sig_t_H[i:i+2], self.boundaries[i:i+2])  / (self.boundaries[i+1] - self.boundaries[i])
            SsH[i] = np.trapz(self.sig_s0_H[i:i+2], self.boundaries[i:i+2]) / (self.boundaries[i+1] - self.boundaries[i])
            StU[i] = np.trapz(self.sig_t_U[i:i+2], self.boundaries[i:i+2])  / (self.boundaries[i+1] - self.boundaries[i])
            SsU[i] = np.trapz(self.sig_s0_U[i:i+2], self.boundaries[i:i+2]) / (self.boundaries[i+1] - self.boundaries[i])

        du = u[1] - u[0]
        alphaU = self.alpha_fn(self.AU)
        inv1ma = 1 / (1 - alphaU)
        lga = -np.log(alphaU)                

        expu = np.exp(u)
        expm = 1 / expu
        phi = np.zeros_like(u, dtype=float)
        scatH = 0
        scatU = 0    
        gmax = int(np.floor(lga / du))
        f = (lga / du) - gmax  

        for i in range(u.size):
            Sigma_R = (StH[i] + StU[i]) - du * (SsU[i] * inv1ma) - du * SsH[i]
            if i == 0: phi[i] = chi[i] / Sigma_R
    
            # Add scattering sources from previous bin
            scatH += SsH[i-1] * phi[i-1] * expu[i-1] * du
            scatU += (SsU[i-1] * inv1ma) * phi[i-1] * expu[i-1] * du
    
            # Subtract contributions outside uranium lethargy window
            if i > gmax:
                h = i - gmax
                scatU -= (SsU[h] * inv1ma) * (1 - f) * phi[h] * expu[h] * du
                scatU -= (SsU[h-1] * inv1ma) * f * phi[h-1] * expu[h-1] * du
    
            phi[i] = (chi[i] + expm[i] * (scatH + scatU)) / Sigma_R
    
        self.p0 = phi
        #self.p0 = self.normalize(phi)
        self.chi = chi
        #df = pd.DataFrame({
        #        "E": self.Eplot,
        #        "phi_ref": self.p0,
        #})
        #df.to_csv(f"grid_flux_comp.csv",index=False)

        # calculate L_comp
        #b = self.chi.reshape(-1,1)
        #x = self.p0.reshape(-1,1)
        #xT = x.T
        #L_comp = b@xT / (xT@x)

        #plt.figure()
        #plt.plot(self.Eplot, self.p0, label=r'$\phi_0$')
        #plt.title(r'Initial Scalar Flux $\phi_0(E)$')
        #plt.xlabel('Energy [eV]')
        #plt.ylabel(r'$\phi_0$')
        #plt.xscale('log')
        #plt.grid(True, which='both')
        #plt.legend()
        #plt.savefig(f"{self.chart_dir}initial_flux.png")   
        print(f"Initial Flux Time: {np.round(time.time()-stt,5)} s")

        #x = self.p0.reshape(-1,1)
        #b = chi.reshape(-1,1)
        #self.L_comp = b @ np.linalg.pinv(x)
        #plt.figure()
        #im = plt.imshow(
        #    L_comp,
        #    cmap='viridis',
        #    origin='upper',
        #    norm=LogNorm()
        #)
        #plt.colorbar(im, label=r'$\Sigma^{sl}_{gtg}$')
        #plt.xlabel("Outgoing Energy Group")
        #plt.ylabel("Incedent Energy Group")
        #plt.savefig(f'{self.chart_dir}initial_L0.png')
        #plt.clf()
        #assert 0 == 1

    def group_bound(self, A, g, lga): return (self.boundaries.size if A == 1
                #else np.searchsorted(self.boundaries, self.boundaries[g] + lga))
                else 1 + np.searchsorted(self.boundaries, self.boundaries[g] + lga))

    def up_group_bound(self,A,g,lga):
        return (0 if A == 1 
                else np.searchsorted(self.boundaries, self.boundaries[g] - lga))

    def gmin_vec_fn(self,A,lga):
        gmin_vec = np.zeros_like(self.boundaries,dtype=int)
        for g in range(self.boundaries.size): gmin_vec[g] = self.up_group_bound(A,g,lga)
        return gmin_vec

    def gmax_vec_fn(self, A, lga):
        gmax_vec = np.zeros_like(self.boundaries, dtype = int)
        for g in range(self.boundaries.size): gmax_vec[g] = self.group_bound(A,g,lga)
        return gmax_vec

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

    """
    def plot_flux_diff(self):
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
    """
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

    def calc_Ln(self,A,sigma_t, sigma_s, sigma_fr):
        # l x g-1 x g-1 matrix
        phi = np.ones_like(self.p0)
        stt = time.time()
        sig_t = np.zeros_like(self.p0)
        sig_s = np.zeros_like(self.p0)
        for i in range(phi.size):
            sig_t[i] = np.trapz(sigma_t[i:i+2],self.boundaries[i:i+2]) / (self.boundaries[i+1] - self.boundaries[i])
            sig_s[i] = np.trapz(sigma_s[i:i+2],self.boundaries[i:i+2]) / (self.boundaries[i+1] - self.boundaries[i])

        sigma_gtg = self.gen_sig_sn_gtg(A,sig_s,self.leg_order, self.boundaries,
                                        self.gmax_vec_fn(A,self.lga_fn(self.alpha_fn(A))),
                                        self.alpha_fn(A), self.E0, phi)

        # Upscatter
#        sigma_gtg[0,:,:] = self.upscatter(A,sigma_fr,self.kT,self.Evec,self.m)
        # transpose with gp on x axis and g on y axis
        S = np.transpose(sigma_gtg, (0, 2, 1))          

        if A == 1: self.sigma_s_gtg_H = S
        else: self.sigma_s_gtg_U = S
        print(f"Sigma_gtg A={A} Time: {np.round(time.time()-stt,5)} s")

#        self.plot_sig_sn_gtg(sigma_gtg, sigma_s, A)
#        self.plot_each_l(A,sigma_s,sigma_gtg)

        stt = time.time()
        self.L0 += self._Ln(0, sig_t, S[0])
        self.L1 += self._Ln(1, sig_t, S[1])
        self.L2 += self._Ln(2, sig_t, S[2])
        self.L3 += self._Ln(3, sig_t, S[3])

        print(f"Ln A = {A} Time: {np.round(time.time()-stt,5)} s")

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
        #RHS = (self.L3 @ self.L2 @ self.L1 + B2 * (9 * self.L1 + 4 * self.L3)) @ self.chi[:-1]
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
            #plt.plot(self.Evec[:-1], 100 * np.abs(p0[i,:] - self.p0) / self.p0, 
            plt.plot(self.Eplot, 100 * np.abs(p0[i,:] - self.normalize(self.phi0)) / self.normalize(self.phi0), 
                    #label = f"B2: {np.round(self.B2[i],5)}, L2 = {self.L2_norm(p0[i,:],self.p0)}")
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

    def calc_phi(self, properties = False):
        print('Calc phi')
        stt=time.time()
        # phi0
        B4 = self.B2 * self.B2
        LHS = (9 * B4 + self.B2 * (self.L3 @ self.L2 + (9 * self.L1 + 4 * self.L3) @ self.L0)
                + self.L3 @ self.L2 @ self.L1 @ self.L0)
        RHS = (self.L3 @ self.L2 @ self.L1 + self.B2 * (9 * self.L1 + 4 * self.L3)) @ self.chi
        #RHS = (self.L3 @ self.L2 @ self.L1 + self.B2 * (9 * self.L1 + 4 * self.L3)) @ self.chi[:-1]
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
        #RHS = .5 * (-9 * self.B2 * self.phi0 + (9 * self.L1 + 4 * self.L3) @ (self.L0 @ self.phi0 - self.chi[:-1]))
        self.phi2 = np.linalg.solve(LHS,RHS)
        print(f"phi2 Time: {np.round(time.time()-stt,5)} s")

        #self.phi0 = self.normalize(self.phi0)
        #self.phi2 = self.normalize(self.phi2)
    
    def calc_Phi(self): 
        self.Phi0 = self.phi0 + 2 * self.phi2
        self.Phi2 = self.phi2
        print(f"L2 norm on phi0 and Phi0, B2 = {self.B2}: {self.L2_norm(self.Phi0,self.phi0)}")

    def save_Ln(self):
        if self.save_data == True:
            with h5py.File(f"{self.save_dir}Ln_{self.NH}.h5", "w") as f:
                f.create_dataset("L0", data=self.L0)
                f.create_dataset("L1", data=self.L1)
                f.create_dataset("L2", data=self.L2)
                f.create_dataset("L3", data=self.L3)

    def save_fluxes(self):
        print("Saving Data...")
        df = pd.DataFrame({'phi0': self.phi0, 'phi2': self.phi2, 
            'Phi0': self.Phi0, 'Phi2': self.Phi2,
            'phi_ref': self.p0})
        df.to_hdf(f"{self.save_dir}fluxes_{self.NH}.h5", key="df", mode="w", format="table")
        df.to_csv(f"{self.save_dir}fluxes_{self.NH}.csv")

    def run(self):
        self.initial_flux()
        if self.fromH5 == True: self.read_data()
        else:
            print(f"Starting calculation. Saving Data = {self.save_data}")
            print('Build Sigma_gtg and Ln for Uranium')
            self.calc_Ln(self.AU, self.sig_t_U, self.sig_s0_U, self.sigma_fr_U)
            print(f'Build Sigma_gtg and Ln for Hydrogen, NH = {self.NH}')
            self.calc_Ln(self.AH, self.sig_t_H, self.sig_s0_H, self.sigma_fr_H)
            self.save_Ln()

            if isinstance(self.B2, float):  self.calc_phi()
            else: self.calc_phi_B2()
            self.calc_Phi()
            self.plot_fluxes()
            self.plot_flux_diff()
            if self.save_data == True: self.save_fluxes()

        #df = pd.DataFrame({
        #        "E": self.Eplot,
        #        "phi_ref": self.p0,
        #        "phi0": self.phi0,
        #})
        #df.to_csv(f"grid_flux_comp.csv",index=False)

        print("Few Group Cross-Sections")
        self.phi_weighted_sigma(self.sig_t_U,self.AU, "total")
        self.phi_weighted_sigma(self.sig_t_H,self.AH, "total")
        self.phi_weighted_sigma(self.sigma_f,self.AU, "fission")

        # Uranium
        print("Uranium Group->Group Few Group Cross-Sections")
        self.sigma_s0, self.sigma_s0_0, self.sigma_s0_2 = self.phi_weighted_sigma_sl(self.sigma_s_gtg_U[0,:,:], self.AU,0)
        self.sigma_s1, self.sigma_s1_0, self.sigma_s1_2 = self.phi_weighted_sigma_sl(self.sigma_s_gtg_U[1,:,:], self.AU,1)
        self.sigma_s2, self.sigma_s2_0, self.sigma_s2_2 = self.phi_weighted_sigma_sl(self.sigma_s_gtg_U[2,:,:], self.AU,2)
        self.sigma_s3, self.sigma_s3_0, self.sigma_s3_2 = self.phi_weighted_sigma_sl(self.sigma_s_gtg_U[3,:,:], self.AU,3)
        if self.save_data == True: self.save_sigma_sl(self.AU)

        # Hydrogen
        print("Hydrogen Group->Group Few Group Cross-Sections")
        self.sigma_s0, self.sigma_s0_0, self.sigma_s0_2 = self.phi_weighted_sigma_sl(self.sigma_s_gtg_H[0,:,:], self.AH,0)
        self.sigma_s1, self.sigma_s1_0, self.sigma_s1_2 = self.phi_weighted_sigma_sl(self.sigma_s_gtg_H[1,:,:], self.AH,1)
        self.sigma_s2, self.sigma_s2_0, self.sigma_s2_2 = self.phi_weighted_sigma_sl(self.sigma_s_gtg_H[2,:,:], self.AH,2)
        self.sigma_s3, self.sigma_s3_0, self.sigma_s3_2 = self.phi_weighted_sigma_sl(self.sigma_s_gtg_H[3,:,:], self.AH,3)
        if self.save_data == True: self.save_sigma_sl(self.AH)

        # calulate diffusion coefs
        self.Dn_coef(self.sigma_s_gtg_U[1,:,:] + self.sigma_s_gtg_H[1,:,:])

    def phi_weighted_sigma(self, sigma, A,key):
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
        sigma_fg = sigma_vec_few_grp(sigma[:-1], phi, fg_idx, self.Evec[:-1])
        sigma_fg_0 = sigma_vec_few_grp(sigma[:-1], self.Phi0, fg_idx, self.Evec[:-1])
        sigma_fg_2 = sigma_vec_few_grp(sigma[:-1], self.Phi2, fg_idx, self.Evec[:-1])
        print(f"Update A={A} {key} xs: {np.round(time.time()-stt,5)} s")
        print(f"L2 norm on phi0 and Phi0 weighted xs's for A={A}, {key}: {self.L2_norm(sigma_fg,sigma_fg_0)}")

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

        print(f"Update A={A} sigma_s{l} xs: {np.round(time.time()-stt,5)} s")
        print(f"L2 norm on phi0 vs Phi0 weighted gtg for A={A} and l={l}: {self.L2_norm(M_fg, M_fg_0)}")

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
        D00 = np.zeros((self.few_groups,self.few_groups))
#        D02 = np.zeros_like(D00)
        D20 = np.zeros_like(D00)
#        D22 = np.zeros_like(D00)
        D = np.zeros_like(D00)

        sigma_t = self.sig_t_U + self.sig_t_H
        B2 = self.B2 if isinstance(self.B2, float) else 0.0
        D_tr = self.diff_matrix(sigma_t,sig_s1,B2)
        phi_tr = self.phi0
        #phi_tr = np.ones_like(self.Phi0)

        for i in range(self.few_groups):
            r0, r1 = fg_idx[i], fg_idx[i+1]
            for j in range(self.few_groups):
                c0, c1 = fg_idx[j], fg_idx[j + 1]
                # Integrate across incident-energy slice for every row, then average over the row block
                D[i, j] = (np.mean(np.trapz(D_tr[r0:r1, c0:c1] * phi_tr[c0:c1], E[c0:c1], axis=1)) 
                            / np.trapz(phi_tr[c0:c1], E[c0:c1]))
                D00[i, j] = (np.mean(np.trapz(L1_inv[r0:r1, c0:c1] * self.Phi0[c0:c1], E[c0:c1], axis=1)) 
                            / np.trapz(self.Phi0[c0:c1], E[c0:c1])).T
#                D02[i, j] = (np.mean(np.trapz(L1_inv[r0:r1, c0:c1] * self.Phi2[c0:c1], E[c0:c1], axis=1)) 
#                            / np.trapz(self.Phi2[c0:c1], E[c0:c1])).T
                D20[i, j] = (np.mean(np.trapz(L3_inv[r0:r1, c0:c1] * self.Phi0[c0:c1], E[c0:c1], axis=1)) 
                            / np.trapz(self.Phi0[c0:c1], E[c0:c1])).T
#                D22[i, j] = (np.mean(np.trapz(L3_inv[r0:r1, c0:c1] * self.Phi2[c0:c1], E[c0:c1], axis=1)) 
#                            / np.trapz(self.Phi2[c0:c1], E[c0:c1])).T
                
        print(f"D_coef Time: {np.round(time.time()-stt,5)} s")
        if self.save_data == True:
            with h5py.File(f"{self.save_dir}D_coef_{self.NH}.h5", "w") as f:
                f.create_dataset("D", data=D)
                f.create_dataset("D00", data=D00)
#                f.create_dataset("D02", data=D02)
                f.create_dataset("D20", data=D20)
#                f.create_dataset("D22", data=D22)

        print(f"L2 norm on phi0 vs Phi0 weighted Diffusion Matrix: {self.L2_norm(D, D00)}")

    def read_data(self):
        print("Reading Data From File...")
        df = pd.read_hdf(f"{self.save_dir}fluxes_{self.NH}.h5", key="df")
        self.phi0 = df['phi0'].to_numpy()
        self.phi2 = df['phi2'].to_numpy()
        self.Phi0 = df['Phi0'].to_numpy()
        self.Phi2 = df['Phi2'].to_numpy()
        self.p0   = df['phi_ref'].to_numpy()
        print("Phi Read!")

        with h5py.File(f"{self.save_dir}Ln_{self.NH}.h5", "r") as f:
            self.L0 = f["L0"][:]
            self.L1 = f["L1"][:]
            self.L2 = f["L2"][:]
            self.L3 = f["L3"][:]
        print("Ln Read!")

    def save_sigma_sl(self,A):
        # save
        with h5py.File(f"{self.save_dir}Sigma_sl_A{A}_{self.NH}.h5", "w") as f:
            f.create_dataset("Sigma S0", data=self.sigma_s0.T)
            f.create_dataset("Sigma S0 phi0", data=self.sigma_s0_0.T)
            f.create_dataset("Sigma S0 phi2", data=self.sigma_s0_2.T)
            f.create_dataset("Sigma S1", data=self.sigma_s1.T)
            f.create_dataset("Sigma S1 phi0", data=self.sigma_s1_0.T)
            f.create_dataset("Sigma S1 phi2", data=self.sigma_s1_2.T)
            f.create_dataset("Sigma S2", data=self.sigma_s2.T)
            f.create_dataset("Sigma S2 phi0", data=self.sigma_s2_0.T)
            f.create_dataset("Sigma S2 phi2", data=self.sigma_s2_2.T)
            f.create_dataset("Sigma S3", data=self.sigma_s3.T)
            f.create_dataset("Sigma S3 phi0", data=self.sigma_s3_0.T)
            f.create_dataset("Sigma S3 phi2", data=self.sigma_s3_2.T)

    @staticmethod
    def alpha_fn(A): return ((A - 1.0)/(A + 1.0)) ** 2

    @staticmethod
    def lga_fn(alpha): return -np.log(alpha) if alpha != 0 else np.inf

    @staticmethod
    #@njit(fastmath=True)
    def _Ln(l, sigma_t, sigma_gtg): return (2*l+1) * (np.diag(sigma_t) - sigma_gtg)
    #def _Ln(l, sigma_t, sigma_gtg): return (2*l+1) * (np.diag(sigma_t[:-1]) - sigma_gtg)

    @staticmethod
    @njit(parallel=True, fastmath=True)
    def gen_sig_sn_gtg(A, sig_s0, order, boundaries, gmax_vec, alpha, E0, phi, n_sub = 8, tol=1e-10):
        G = boundaries.size
        du = boundaries[1] - boundaries[0]
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
                    n_steps_base = int(np.ceil(length / du))
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
                            if A == 1:
                                val = (1/3) * Ap1 * (np.exp(1.5 * (xm - a)) - np.exp(1.5 * (xm - b)))
                            else:
                                val = (Am1 * (np.exp(.5 * (xm - b)) - np.exp(0.5 * (xm - a))) 
                                 + (1/3) * Ap1 * (np.exp(1.5 * (xm - a)) - np.exp(1.5 * (xm - b))))
                                            
                        elif l == 2:
                            if A == 1:
                                val = (0.25 * (1 - 3*A*A) * (np.exp(xm - a) - np.exp(xm - b)) 
                                    + .1875 * Ap1*Ap1 * (np.exp(2*(xm - a)) - np.exp(2*(xm - b))))

                            else:
                                val = ((.375)*(Am1*Am1)*(b - a)  + 0.25 * (1 - 3*A*A) * (np.exp(xm - a) - np.exp(xm - b)) 
                                    + .1875 * Ap1*Ap1 * (np.exp(2*(xm - a)) - np.exp(2*(xm - b))))

                        else:
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

                    sig_foo = (sig_s0[gp] * phi[gp] * acc * dx ) / (den * phi[gp])
#                    if sig_foo > tol: sigma_gtg[l,gp,g] = sig_foo
                    sigma_gtg[l, gp, g] = (sig_s0[gp] * phi[gp] * acc * dx ) / (den * phi[gp])

        return sigma_gtg

    @staticmethod
#    @njit(parallel=True,fastmath=True)
    def upscatter(A,sigma_fr,kT,Evec,m):
        print("Upscatter")
        G = Evec.size - 1
        sigma_gtg = np.zeros((G,G))
        #B2 = m / (2 * kT)
        #h2 = (A+1) / A * B2
        Ap1 = A+1
        J = 1.602e-19
        E = J * Evec
        kT = J*kT

        def mu(Ep,E): return (.5 * (A+1)) * np.sqrt(E / Ep) - (.5 * (A-1)) * np.sqrt(Ep / E) 

        def E_to_v(E): return np.sqrt(2*E/(m))

        def v_rel_fn(Ep,E):
            Mu = mu(Ep,E)
            vp = E_to_v(Ep)
            V = np.sqrt(2*kT/(A*m))
            v_rel_2 = vp*vp + V*V - 2*v*V*Mu
            return np.sqrt(v_rel_2)

        for gp in range(G-1):
            print(gp)
            vp = E_to_v(E[gp])
            sigma = sigma_fr[gp]
            for g in range(gp):
                v  = E_to_v(E[g]) 
                v_rel = v_rel_fn(E[gp],E[g])
                t1 = ((Ap1*Ap1) / (4 * A) * (vp / v) * sigma * np.exp(m/(2*A*kT) * (v*v - vp*vp)) 
                        * math.erf(np.sqrt(m/(2*A*kT)) * (v_rel-vp)))
                t2 = ((Ap1**(2.5) * vp * sigma / (4 * A**(1.5) * v)) 
                        * (np.exp(m / (2*A*Ap1*kT) * (Ap1*v*v - A*v_rel*v_rel)))
                        * (math.erf(np.sqrt(m / (2*A*Ap1*kT)) * (Ap1 * vp - A * v_rel)) - 1))

                sigma_gtg[gp,g] = t1+t2

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

####################### RUN ########################
process = psutil.Process(os.getpid())
stt = time.time()
NH = 5
B2 = .0
#B2 = np.linspace(-.025,.025,6)
nbins = 75000
few_groups = 8
fromH5 = False

# init class
print(f"Initializing, {nbins} Groups")
sp3 = Sp3(nbins, B2, NH, few_groups, fromH5)
sp3.run()

stp = time.time()
print(f"Calculation Time = {np.round((stp - stt),6)}")
mem = process.memory_info().rss / 1e9
print(f"Memory used: {mem:.3f} GB")

