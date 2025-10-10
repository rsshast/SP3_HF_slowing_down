import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from numba import njit, prange
import os
import h5py

class Sp3:
    def __init__(self, nbins, B2, NH, few_groups, fromH5):
        self.data_dir = 'data/'
        self.save_dir = "results/data/"
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

        chi35 = pd.read_csv(f'{self.data_dir}chi_u235.txt', sep = '\t',header = 0)
        H1 = pd.read_csv(f'{self.data_dir}xs_h1_T293k.txt', sep  = '\t', header = 0)
        U238 = pd.read_csv(f'{self.data_dir}xs_u238_T293k.txt',sep  = '\t', header = 0)
        sigma_f = pd.read_csv(f'{self.data_dir}xs_u238_fission.csv', sep = ',', dtype=float).to_numpy()

        chi = np.array([chi35['E'],chi35['chi']]).T
        H = np.array([H1['E'],H1['sigma_t'],H1['sigma_s']]).T
        XS38 = np.array([U238['E'],U238['sigma_t'],U238['sigma_s']]).T
        chi = self.get_data(chi,self.nbins)
        H = self.get_data(H,self.nbins)
        H *= self.NH
        XS38 = self.get_data(XS38,self.nbins)
        self.chi = chi[:,1]
        self.boundaries = XS38[:,0]
        self.Evec = self.E0 * np.exp(-self.boundaries)
        self.sig_t_U = XS38[:,1]
        self.sig_s0_U = XS38[:,2]
        self.sig_t_H = H[:,1]
        self.sig_s0_H = H[:,2]
        self.sigma_f = self.get_data(sigma_f,self.nbins)[:,1]

        G = self.boundaries.size
        self.phi0 = np.zeros((G-1))
        self.phi2 = np.zeros_like(self.phi0)
        self.Phi0 = np.zeros_like(self.phi0)
        self.Phi2 = np.zeros_like(self.phi0)
        self.p0 = np.zeros((G))
        self.L0 = np.zeros((G-1,G-1))
        self.L1 = np.zeros_like(self.L0)
        self.L2 = np.zeros_like(self.L0)
        self.L3 = np.zeros_like(self.L0)
        self.chart_dir = "results/charts/"
        self.save_data = True
        self.few_groups = few_groups

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
        # Problem setup
        u = self.boundaries                   
        du = u[1] - u[0]
        alphaU = self.alpha_fn(self.AU)
        inv1ma = 1.0 / (1.0 - alphaU)
        lga = -np.log(alphaU)                
        StH, SsH = self.sig_t_H, self.sig_s0_H
        StU, SsU = self.sig_t_U, self.sig_s0_U
        chi = self.chi
    
        expu = np.exp(u)
        expm = 1.0 / expu
        phi = np.zeros_like(u, dtype=float)
        scatH = 0.0
        scatU = 0.0    
        gmax = int(np.floor(lga / du))
        f = (lga / du) - gmax  # fractional overlap for partial bin
        
        for i in range(u.size):
        #for i in range(1, u.size):
            # Removal cross section
            Sigma_R = (StH[i] + StU[i]) - du * (SsU[i] * inv1ma) - du * SsH[i]
            if i == 0: phi[i] = chi[i] / Sigma_R
    
            # Add scattering sources from previous bin
            e_im1 = expu[i - 1]
            scatH += SsH[i - 1] * phi[i - 1] * e_im1 * du
            scatU += (SsU[i - 1] * inv1ma) * phi[i - 1] * e_im1 * du
    
            # Subtract contributions outside uranium lethargy window
            if i > gmax:
                h = i - gmax
                scatU -= (SsU[h] * inv1ma) * (1.0 - f) * phi[h] * expu[h] * du
                scatU -= (SsU[h - 1] * inv1ma) * f * phi[h - 1] * expu[h - 1] * du
    
            # Update flux (extra exp(-u_i) factor per handbook formulation)
            phi[i] = (chi[i] + expm[i] * (scatH + scatU)) / Sigma_R
    
        plt.figure()
        plt.plot(self.Evec, phi, label=r'$\phi_0$')
        plt.title(r'Initial Scalar Flux $\phi_0(E)$')
        plt.xlabel('Energy [eV]')
        plt.ylabel(r'$\phi_0$')
        plt.xscale('log')
        plt.grid(True, which='both')
        plt.legend()
        plt.savefig(f"{self.chart_dir}initial_flux.png")   
        self.p0 = phi

    def group_bound(self, A, g, lga): return (self.boundaries.size if A == 1
                else np.searchsorted(self.boundaries, self.boundaries[g] + lga))
                #else 1 + np.searchsorted(self.boundaries, self.boundaries[g] + lga))

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
        plt.legend()
        plt.grid(which="Both")
        plt.savefig(f'{self.chart_dir}xs_comparison_{A}.png')
        plt.clf()

        plt.figure(figsize=(8,6))
        plt.plot(Evec,np.abs(rowsum - sig_s0[:-1]), label = "Abs Difference")
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.grid(which="Both")
        plt.savefig(f'{self.chart_dir}xs_diff_{A}.png')
        plt.clf()

    def plot_each_l(self,A,sigma_s0,sigma_gtg):
        Evec = self.Evec[:-1]
        plt.figure(figsize=(8,6))
        plt.plot(self.Evec,sigma_s0, label = "Sigma_{s0}")
        for l in range(self.leg_order):
            rowsum = np.zeros(Evec.size)
            for g in range(Evec.size): rowsum[g] = np.sum(sigma_gtg[l,g,:])
            plt.plot(Evec,rowsum, label = f"Sigma_s{l}")
        plt.title(f"Sigma_sl, A = {A}")
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.grid(which="Both")
        plt.savefig(f'{self.chart_dir}xs_sl_{A}.png')
        plt.clf()

    def plot_fluxes(self):
        # plot phi0 and phi2
        print("Plotting phi0, phi2")
        plt.figure()
        E = self.Evec[:-1]
        plt.plot(E,self.phi0,label=r'$\phi_0$')
        plt.title(r'$\phi_0(E)$')
        plt.xlabel('E')
        plt.ylabel(r'$\phi_0$')
        plt.xscale('log')
        plt.grid(True, which='both')
        plt.legend()
#        plt.show()
        plt.savefig(f'{self.chart_dir}phi0_{self.NH}.png')
        plt.clf()

        plt.figure()
        plt.plot(E,self.phi2,label=r'$\phi_2$')
        plt.title(r'$\phi_2(E)$')
        plt.xlabel('E')
        plt.ylabel(r'$\phi_2$')
        plt.xscale('log')
        plt.grid(True, which='both')
        plt.legend()
#        plt.show()
        plt.savefig(f'{self.chart_dir}phi2_{self.NH}.png')
        plt.clf()

    def plot_flux_diff(self):
        # compare the traditional to new method
        plt.figure()
        plt.plot(self.Evec,self.p0,label='Scattering Source')
        plt.plot(self.Evec[:-1],self.phi0,label='Sp3')
        plt.title("Hyperfine Slowing-Down Flux Comparison")
        plt.ylabel(r"$\phi (E) (n/cm^2)$")
        plt.xscale('log')
        plt.xlabel("Energy (eV)")
        plt.legend()
        plt.grid(True,which='both')
        plt.savefig(f"{self.chart_dir}order_comp.png")
        plt.clf()

        plt.figure()
        plt.plot(self.Evec[:-1],(self.phi0 - self.p0[:-1]) / self.p0[:-1])
        plt.title("Hyperfine Slowing-Down Flux Difference")
        plt.ylabel("% Difference")
        plt.xscale('log')
        plt.xlabel("Energy (eV)")
        plt.grid(True,which='both')
        plt.savefig(f"{self.chart_dir}flux_difference.png")

        # evaluate fluxes
        L2 = self.L2_norm(self.phi0,self.p0[:-1])
        print(f"L2 norm on phi0 and reference, = {L2}")
        L2 = self.L2_norm(self.Phi0,self.p0[:-1])
        print(f"L2 norm on Phi0 and reference, = {L2}")

    def gtg_and_line_plots(self,sigma_gtg,A,sigma_s0,rowsum):
        print("Plot Sigma_sl")
        for l in range(self.leg_order):
            plt.figure(figsize=(6, 5))
            im = plt.imshow(
                sigma_gtg[l, :, :],
                cmap='viridis',
                origin='upper',
                norm=LogNorm()
            )
            plt.colorbar(im, label=r'$\Sigma^{sl}_{gtg}$')
            plt.title(f"Scattering Xs's, l = {l}")
            plt.savefig(f'{self.chart_dir}sigma_s{l}_A{A}.png')
            plt.close()
            plt.clf()

    def calc_Ln(self,A,sigma_t, sigma_s):
        # l x g-1 x g-1 matrix
        phi = np.ones(self.boundaries.size)
        sigma_gtg = self.gen_sig_sn_gtg(A,sigma_s,self.leg_order, self.boundaries,
                                        self.gmax_vec_fn(A,self.lga_fn(self.alpha_fn(A))),
                                        self.alpha_fn(A), self.E0, self.tol,phi)
        self.plot_sig_sn_gtg(sigma_gtg, sigma_s, A)
        self.plot_each_l(A,sigma_s,sigma_gtg)

        self.L0 += self._Ln(0, sigma_t, sigma_gtg[0,:,:])
        self.L1 += self._Ln(1, sigma_t, sigma_gtg[1,:,:])
        self.L2 += self._Ln(2, sigma_t, sigma_gtg[2,:,:])
        self.L3 += self._Ln(3, sigma_t, sigma_gtg[3,:,:])

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
            RHS = (self.L3 @ self.L2 @ self.L1 + B2 * (9 * self.L1 + 4 * self.L3)) @ self.chi[:-1]
            phi0 = np.linalg.solve(LHS,RHS)
            p0.append(phi0)
            plt.plot(self.Evec[:-1],phi0,label = f"{np.round(B2,5)}")

        plt.title(r"$\phi_0 (E,B^2)$ Leakage Parameter Parametric Study")
        plt.xscale("log")
        plt.xlabel("Energy (eV)")
        plt.ylabel(r"$\phi_0 (E,B^2)$")
        plt.legend()
        plt.grid(which = 'Both')
        plt.savefig(f"{self.chart_dir}phi0_leakage_comparison.png")
        plt.clf()

        #phi2
        plt.figure()
        for i in range(self.B2.size):
            B2 = self.B2[i]
            print(f"phi2, {i}, B2 = {np.round(B2,5)}")
            LHS = self.L3 @ self.L2
            RHS = .5 * (-9 * B2 * self.phi0 + (9 * self.L1 + 4 * self.L3)
                    @ (self.L0 @ self.phi0 - self.chi[:-1]))
            phi2 = np.linalg.solve(LHS,RHS)
            p2.append(phi2)
            plt.plot(self.Evec[:-1],phi2,label = f"{np.round(B2,5)}")

        plt.title(r"$\phi_2 (E,B^2)$ Leakage Parameter Parametric Study")
        plt.xscale("log")
        plt.xlabel("Energy (eV)")
        plt.ylabel(r"$\phi_0$ (E,B2)")
        plt.legend()
        plt.grid(which = 'Both')
        plt.savefig(f"{self.chart_dir}phi2_leakage_comparison.png")
        plt.clf()

        B2 = 0
        # self.phi0
        B4 = B2 * B2
        LHS = (9 * B4 + B2 * (self.L3 @ self.L2 + (9 * self.L1 + 4 * self.L3) @ self.L0)
                    + self.L3 @ self.L2 @ self.L1 @ self.L0)
        RHS = (self.L3 @ self.L2 @ self.L1 + B2 * (9 * self.L1 + 4 * self.L3)) @ self.chi[:-1]
        self.phi0 = np.linalg.solve(LHS,RHS)

        # self.phi2
        LHS = self.L3 @ self.L2
        RHS = .5 * (-9 * B2 * self.phi0 + (9 * self.L1 + 4 * self.L3)
               @ (self.L0 @ self.phi0 - self.chi[:-1]))
        self.phi2 = np.linalg.solve(LHS,RHS)

        p0 = np.array(p0)
        p2 = np.array(p2)

        plt.figure()
        for i in range(self.B2.size):
            plt.plot(self.Evec[:-1], np.abs(p0[i,:] - self.phi0) / self.phi0, 
                    label = f"B2: {np.round(self.B2[i],5)}, L2 = {self.L2_norm(p0[i,:],self.phi0)}")
        plt.title(r"$\phi_0 (E,B^2)$ Leakage Parameter Percent Difference")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Energy (eV)")
        plt.ylabel(f"% Difference from $\phi_0 (E,B^2 = 0)$")
        plt.legend()
        plt.grid(which = 'Both')
        plt.savefig(f"{self.chart_dir}phi0_leakage_flux_diff.png")
        plt.clf()

        plt.figure()
        for i in range(self.B2.size):
            plt.plot(self.Evec[:-1], np.abs(p2[i,:] - self.phi2) / self.phi2, 
                    label = f"B2: {np.round(self.B2[i],5)}, L2 = {self.L2_norm(p2[i,:],self.phi2)}")
        plt.title(r"$\phi_2 (E,B^2)$ Leakage Parameter Percent Difference")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Energy (eV)")
        plt.ylabel(f"% Difference from $\phi_2 (E,B^2 = 0)$")
        plt.legend()
        plt.grid(which = 'Both')
        plt.savefig(f"{self.chart_dir}phi2_leakage_flux_diff.png")
        plt.clf()


    def calc_phi(self, properties = False):
        # phi0
        print('Calc phi')
        B4 = self.B2 * self.B2
        LHS = (9 * B4 + self.B2 * (self.L3 @ self.L2 + (9 * self.L1 + 4 * self.L3) @ self.L0)
                + self.L3 @ self.L2 @ self.L1 @ self.L0)
        RHS = (self.L3 @ self.L2 @ self.L1 + self.B2 * (9 * self.L1 + 4 * self.L3)) @ self.chi[:-1]
        if properties:
            print("phi0 matrix properties")
            self.print_mat_properties(LHS)
        self.phi0 = np.linalg.solve(LHS,RHS)

        #phi2
        LHS = self.L3 @ self.L2
        if properties:
            print("phi2 matrix properties")
            self.print_mat_properties(LHS)
        RHS = .5 * (-9 * self.B2 * self.phi0 + (9 * self.L1 + 4 * self.L3)
                @ (self.L0 @ self.phi0 - self.chi[:-1]))
        self.phi2 = np.linalg.solve(LHS,RHS)

    def calc_Phi(self): 
        self.Phi0 = self.phi0 + 2 * self.phi2
        self.Phi2 = self.phi2

    def transpose_and_save_Ln(self):
        self.L0 = self.L0.T
        self.L1 = self.L1.T
        self.L2 = self.L2.T
        self.L3 = self.L3.T
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
            'phi_ref': self.p0[:-1]})
        df.to_hdf(f"{self.save_dir}fluxes_{self.NH}.h5", key="df", mode="w", format="table")
        df.to_csv(f"{self.save_dir}fluxes_{self.NH}.csv")

    def run(self):
        self.initial_flux()
        if self.fromH5 == True: self.read_data()
        else:
            print(f"Starting calculation. Saving Data = {self.save_data}")
            print('Build Sigma_gtg and Ln for Uranium')
            self.calc_Ln(self.AU, self.sig_t_U, self.sig_s0_U)
            print(f'Build Sigma_gtg and Ln for Hydrogen, NH = {self.NH}')
            self.calc_Ln(self.AH, self.sig_t_H, self.sig_s0_H)
            self.transpose_and_save_Ln()
            if isinstance(self.B2, float): self.calc_phi()
            else: self.calc_phi_B2()
            self.calc_Phi()
            self.plot_fluxes()
            self.plot_flux_diff()
            if self.save_data == True: self.save_fluxes()

        print("Few Group Cross-Sections")
        self.phi_weighted_sigma(self.sig_t_U,self.AU, "total")
        self.phi_weighted_sigma(self.sig_t_H,self.AH, "total")
        self.phi_weighted_sigma(self.sigma_f,self.AU, "fission")

        sig_s1_diffusion = np.zeros_like(self.L1)
        # Uranium
        print("Uranium Group->Group Few Group Cross-Sections")
        sigma_sl = self.gen_sig_sn_gtg(self.AU,self.sig_s0_U,self.leg_order, self.boundaries,
                                        self.gmax_vec_fn(self.AU,self.lga_fn(self.alpha_fn(self.AU))),
                                        self.alpha_fn(self.AU), self.E0, self.tol,phi=np.ones_like(self.phi0))
        self.sigma_s0, self.sigma_s0_0, self.sigma_s0_2 = self.phi_weighted_sigma_sl(sigma_sl[0,:,:], self.AU, 0)
        self.sigma_s1, self.sigma_s1_0, self.sigma_s1_2 = self.phi_weighted_sigma_sl(sigma_sl[1,:,:], self.AU, 1)
        self.sigma_s2, self.sigma_s2_0, self.sigma_s2_2 = self.phi_weighted_sigma_sl(sigma_sl[2,:,:], self.AU, 2)
        self.sigma_s3, self.sigma_s3_0, self.sigma_s3_2 = self.phi_weighted_sigma_sl(sigma_sl[3,:,:], self.AU, 3)
        if self.save_data == True: self.save_sigma_sl(self.AU)

        sig_s1_diffusion += sigma_sl[1,:,:]

        # Hydrogen
        print("Hydrogen Group->Group Few Group Cross-Sections")
        sigma_sl = self.gen_sig_sn_gtg(self.AH,self.sig_s0_H,self.leg_order, self.boundaries,
                                        self.gmax_vec_fn(self.AH,self.lga_fn(self.alpha_fn(self.AH))),
                                        self.alpha_fn(self.AH), self.E0, self.tol,phi=np.ones_like(self.phi0))
        self.sigma_s0, self.sigma_s0_0, self.sigma_s0_2 = self.phi_weighted_sigma_sl(sigma_sl[0,:,:], self.AH, 0)
        self.sigma_s1, self.sigma_s1_0, self.sigma_s1_2 = self.phi_weighted_sigma_sl(sigma_sl[1,:,:], self.AH, 1)
        self.sigma_s2, self.sigma_s2_0, self.sigma_s2_2 = self.phi_weighted_sigma_sl(sigma_sl[2,:,:], self.AH, 2)
        self.sigma_s3, self.sigma_s3_0, self.sigma_s3_2 = self.phi_weighted_sigma_sl(sigma_sl[3,:,:], self.AH, 3)
        if self.save_data == True: self.save_sigma_sl(self.AH)

        sig_s1_diffusion += sigma_sl[1,:,:]

        # calulate diffusion coefs
        self.Dn_coef(sig_s1_diffusion)
        
        # End

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

        fg_idx = np.linspace(0,self.phi0.size,self.few_groups+1, dtype=int)
        E_ave = np.zeros((self.few_groups))
        u_ave = np.zeros_like(E_ave)
        sigma_fg = np.zeros_like(E_ave)
        for i in range(self.few_groups): 
            E_ave[i] = self.Evec[fg_idx[i]]
            u_ave[i] = self.boundaries[fg_idx[i]]

        # Uranium flux weighted xs's
        phi = np.ones_like(self.phi0)
        sigma_fg = sigma_vec_few_grp(sigma[:-1], phi, fg_idx, self.Evec[:-1])
        #sigma_fg = sigma_vec_few_grp(sigma[:-1], self.phi0, fg_idx, self.Evec[:-1])
        sigma_fg_0 = sigma_vec_few_grp(sigma[:-1], self.Phi0, fg_idx, self.Evec[:-1])
        sigma_fg_2 = sigma_vec_few_grp(sigma[:-1], self.Phi2, fg_idx, self.Evec[:-1])
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
        """Get gtg flux weighte xs's"""
        N = self.phi0.size
        E = self.Evec[:-1]
        u = self.boundaries
        fg_idx = np.linspace(0, N, self.few_groups + 1, dtype=int)
        M_fg = np.zeros((self.few_groups,self.few_groups),dtype=float)
        M_fg_0 = np.zeros_like(M_fg)
        M_fg_2 = np.zeros_like(M_fg)
        phi = np.ones_like(self.phi0)

        for i in range(self.few_groups):
            r0, r1 = fg_idx[i], fg_idx[i+1]
            for j in range(self.few_groups):
                c0, c1 = fg_idx[j], fg_idx[j + 1]
                # Integrate across incident-energy slice for every row, then average over the row block
                M_fg[i, j]   = np.mean(np.trapz(M[r0:r1, c0:c1] * phi[c0:c1], E[c0:c1], axis=1)) / np.trapz(phi[c0:c1], E[c0:c1])
                M_fg_0[i, j] = np.mean(np.trapz(M[r0:r1, c0:c1] * self.Phi0[c0:c1], E[c0:c1], axis=1)) / np.trapz(self.Phi0[c0:c1], E[c0:c1])
                M_fg_2[i, j] = np.mean(np.trapz(M[r0:r1, c0:c1] * self.Phi2[c0:c1], E[c0:c1], axis=1)) / np.trapz(self.Phi2[c0:c1], E[c0:c1])
                
    
        print(f"L2 norm on phi0 vs Phi0 weighted gtg for A={A} and l={l}: {self.L2_norm(M_fg, M_fg_0)}")

        return M_fg, M_fg_0, M_fg_2

    def Dn_coef(self, sig_s1):
        print("Calculating Phi0/Phi2 weighted Diffusion Coefficients")
        L1_inv = np.linalg.inv(self.L1)
        L3_inv = np.linalg.inv(self.L1)
        N = self.phi0.size
        E = self.Evec[:-1]
        u = self.boundaries
        fg_idx = np.linspace(0, N, self.few_groups + 1, dtype=int)
        D00 = np.zeros((self.few_groups,self.few_groups))
        D02 = np.zeros_like(D00)
        D20 = np.zeros_like(D00)
        D22 = np.zeros_like(D00)
        D = np.zeros_like(D00)

        sigma_t = self.sig_t_U + self.sig_t_H
        B2 = self.B2 if isinstance(self.B2, float) else 0.0
        D_tr = self.diff_matrix(sigma_t,sig_s1,B2)
        phi_tr = np.ones_like(self.Phi0)

        for i in range(self.few_groups):
            r0, r1 = fg_idx[i], fg_idx[i+1]
            for j in range(self.few_groups):
                c0, c1 = fg_idx[j], fg_idx[j + 1]
                # Integrate across incident-energy slice for every row, then average over the row block
                D[i, j] = np.mean(np.trapz(D_tr[r0:r1, c0:c1] * phi_tr[c0:c1], E[c0:c1], axis=1)) / np.trapz(phi_tr[c0:c1], E[c0:c1])
                D00[i, j] = np.mean(np.trapz(L1_inv[r0:r1, c0:c1] * self.Phi0[c0:c1], E[c0:c1], axis=1)) / np.trapz(self.Phi0[c0:c1], E[c0:c1])
                D02[i, j] = np.mean(np.trapz(L1_inv[r0:r1, c0:c1] * self.Phi2[c0:c1], E[c0:c1], axis=1)) / np.trapz(self.Phi2[c0:c1], E[c0:c1])
                D20[i, j] = np.mean(np.trapz(L3_inv[r0:r1, c0:c1] * self.Phi0[c0:c1], E[c0:c1], axis=1)) / np.trapz(self.Phi0[c0:c1], E[c0:c1])
                D22[i, j] = np.mean(np.trapz(L3_inv[r0:r1, c0:c1] * self.Phi2[c0:c1], E[c0:c1], axis=1)) / np.trapz(self.Phi2[c0:c1], E[c0:c1])
                
        if self.save_data == True:
            with h5py.File(f"{self.save_dir}D_coef_{self.NH}.h5", "w") as f:
                f.create_dataset("D", data=D)
                f.create_dataset("D00", data=D00)
                f.create_dataset("D02", data=D02)
                f.create_dataset("D20", data=D20)
                f.create_dataset("D22", data=D22)

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
            f.create_dataset("Sigma S0", data=self.sigma_s0)
            f.create_dataset("Sigma S0 phi0", data=self.sigma_s0_0)
            f.create_dataset("Sigma S0 phi2", data=self.sigma_s0_2)
            f.create_dataset("Sigma S1", data=self.sigma_s1)
            f.create_dataset("Sigma S1 phi0", data=self.sigma_s1_0)
            f.create_dataset("Sigma S1 phi2", data=self.sigma_s1_2)
            f.create_dataset("Sigma S2", data=self.sigma_s2)
            f.create_dataset("Sigma S2 phi0", data=self.sigma_s2_0)
            f.create_dataset("Sigma S2 phi2", data=self.sigma_s2_2)
            f.create_dataset("Sigma S3", data=self.sigma_s3)
            f.create_dataset("Sigma S3 phi0", data=self.sigma_s3_0)
            f.create_dataset("Sigma S3 phi2", data=self.sigma_s3_2)

    @staticmethod
    def alpha_fn(A): return ((A - 1.0)/(A + 1.0)) ** 2

    @staticmethod
    def lga_fn(alpha): return -np.log(alpha) if alpha != 0 else np.inf

    @staticmethod
    def _Ln(l, sigma_t, sigma_gtg): return (2 * l + 1) * (np.diag(sigma_t[:-1]) - sigma_gtg)

    @staticmethod
    @njit(parallel=True, fastmath=True)
    def gen_sig_sn_gtg(A, sig_s0, order, boundaries, gmax_vec, alpha, E0, tol, phi, n_sub = 8):
        """Parallel midpoint integration (no SciPy quad).
        Integrates over x in [c, x2] using n_sub midpoints per base-bin."""
        G = boundaries.size
        du = boundaries[1] - boundaries[0]
        den = (1 - alpha) * du
        lga = -np.log(alpha) if A != 1 else np.inf
        sigma_gtg = np.zeros((order, G - 1, G - 1))
        Am1 = A-1
        Ap1 = A+1

        for l in range(order):
            for gp in prange(G - 1):
                x1 = boundaries[gp]
                x2 = boundaries[gp+1]

                for g in range(gp, min(gmax_vec[gp], G-1)):
                    y1 = boundaries[g]
                    y2 = boundaries[g + 1]
                    c = max(x1, y1 - lga)

                    if (y1 < x1) or (c >= x2) or den == 0.0: continue

                    # choose number of midpoint samples
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

                        if l == 0: val = np.exp(-(a - xm)) - np.exp(-(b - xm))

                        elif l == 1:
                            if A == 1:
                                val = ((Ap1) / 3) * (np.exp(1.5 * (xm - a)) - np.exp(1.5 * (xm - b)))
                            else:
                                val = (((Ap1) / 3) * (np.exp(1.5 * (xm - a)) - np.exp(1.5 * (xm - b))) 
                                  - (Am1) * (np.exp(.5 * (xm - a)) - np.exp(.5 * (xm - b))))

                                            
                        elif l == 2:
                            if A == 1:
                                val = .0625 * ((np.exp(xm - 2*a - b) - np.exp(xm - a - 2*b)) *
                                (-4*(3*A*A - 1)*np.exp(a+b) + (3*Ap1*Ap1*(np.exp(a+xm)+np.exp(b+xm)))))
                            else: 
                                val = .0625 * (6 * Am1 *  Am1 * (b-a)
                                + (np.exp(xm - 2*a - b) - np.exp(xm - a - 2*b)) *
                                (-4*(3*A*A - 1)*np.exp(a+b) + (3*Ap1*Ap1 * (np.exp(a+xm)+np.exp(b+xm)))))
                         
                        else:
                            if A == 1: 
                                val = 0.0625 * (
                                  + 2*(Ap1*Ap1*Ap1) * (np.exp(2.5*(xm - a)) - np.exp(2.5*(xm - b)))
                                  - 2*(Ap1)*(5*A*A - 1) * (np.exp(1.5*(xm - a)) - np.exp(1.5*(xm - b))))
                            else: 
                                val = 0.0625 * (
                                    10*(Am1*Am1*Am1) * (np.exp(.5*(a - xm)) - np.exp(.5*(b - xm)))
                                  + 2*(Ap1*Ap1*Ap1) * (np.exp(2.5*(xm - a)) - np.exp(2.5*(xm - b)))
                                  + 6*(Am1)*(5*A*A - 1) * (np.exp(.5*(xm - a)) - np.exp(.5*(xm - b)))
                                  - 2*(Ap1)*(5*A*A - 1) * (np.exp(1.5*(xm - a)) - np.exp(1.5*(xm - b))))

                        acc += val

                    sigma_gtg[l, gp, g] = (sig_s0[gp] * phi[gp] * acc * dx ) / (den * phi[gp])

                    if np.abs(sigma_gtg[l,gp,g]) < tol: break

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
    def L2_norm(A,B): return np.linalg.norm(A-B, ord=2) if A.shape == B.shape else np.inf

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


####################### RUN ########################
stt = time.time()
NH = 5
B2 = .01
#B2 = np.linspace(-1,1,10)
nbins = 5000
few_groups = 8
fromH5 = False

# init class
print(f"Initializing, {nbins} Groups")
sp3 = Sp3(nbins, B2, NH, few_groups, fromH5)
sp3.run()

stp = time.time()
print(f"Calculation Time = {np.round((stp - stt),6)}")

