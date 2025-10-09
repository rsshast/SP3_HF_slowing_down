import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from numba import njit, prange
import os
import h5py

class Sp3:
    def __init__(self,data_dir, nbins, B2, NH):
        self.data_dir = data_dir
        self.save_dir = "results/data/"
        self.B2 = B2
        self.tol = 1e-7 # loop break condition
        self.E0 = 1e7
        self.Emin = 1e-2
        self.nbins = nbins + 1
        self.leg_order = 4
        self.AH = 1
        self.NH = NH
        self.AU = 238
        self.NU = 1

        chi35 = pd.read_csv(f'{data_dir}chi_u235.txt', sep = '\t',header = 0)
        H1 = pd.read_csv(f'{data_dir}xs_h1_T293k.txt', sep  = '\t', header = 0)
        U238 = pd.read_csv(f'{data_dir}xs_u238_T293k.txt',sep  = '\t', header = 0)

        chi = np.array([chi35['E'],chi35['chi']]).T
        H = np.array([H1['E'],H1['sigma_t'],H1['sigma_s']]).T
        XS38 = np.array([U238['E'],U238['sigma_t'],U238['sigma_s']]).T
        chi = self.get_data(chi,self.nbins,self.NU)
        H = self.get_data(H,self.nbins,self.NH)
        H *= self.NH
        XS38 = self.get_data(XS38,self.nbins,self.NU)
        self.chi = chi[:,1]
        self.boundaries = XS38[:,0]
        self.Evec = self.E0 * np.exp(-self.boundaries)
        self.sig_t_U = XS38[:,1]
        self.sig_s0_U = XS38[:,2]
        self.sig_t_H = H[:,1]
        self.sig_s0_H = H[:,2]
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
        self.save_data = False

    def get_data(self,data, gridpoints, N):
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
        u = self.boundaries                   
        du = u[1] - u[0]
        alphaU = self.alpha_fn(self.AU)
        inv1ma = 1.0 / (1.0 - alphaU)
        lga = -np.log(alphaU)                
    
        # --- Cross sections and fission source on SAME grid ---
        StH, SsH = self.sig_t_H, self.sig_s0_H
        StU, SsU = self.sig_t_U, self.sig_s0_U
        chi = self.chi
    
        # --- Precompute exponentials ---
        expu = np.exp(u)
        expm = 1.0 / expu
        phi = np.zeros_like(u, dtype=float)
        scatH = 0.0
        scatU = 0.0    
        # --- Convert lethargy window (lga) to bin count for uranium scattering ---
        gmax = int(np.floor(lga / du))
        f = (lga / du) - gmax  # fractional overlap for partial bin
        
        for i in range(1, u.size):
            # Removal cross section
            Sigma_R = (StH[i] + StU[i]) - du * (SsU[i] * inv1ma) - du * SsH[i]
    
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
    
        # --- Plot result ---
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
        plt.plot(self.Evec[:-1],self.phi0 - self.p0[:-1])
        plt.title("Hyperfine Slowing-Down Flux Difference")
        plt.ylabel(r"$\phi (E) (n/cm^2)$")
        plt.xscale('log')
        plt.xlabel("Energy (eV)")
        plt.grid(True,which='both')
        plt.savefig(f"{self.chart_dir}flux_difference.png")

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
        sigma_gtg = self.gen_sig_sn_gtg(A,sigma_s,self.leg_order, self.boundaries,
                                        self.gmax_vec_fn(A,self.lga_fn(self.alpha_fn(A))),
                                        self.alpha_fn(A), self.E0, self.tol)
        self.plot_sig_sn_gtg(sigma_gtg, sigma_s, A)
        self.plot_each_l(A,sigma_s,sigma_gtg)

        self.L0 += self._Ln(0, sigma_t, sigma_gtg[0,:,:])
        self.L1 += self._Ln(1, sigma_t, sigma_gtg[1,:,:])
        self.L2 += self._Ln(2, sigma_t, sigma_gtg[2,:,:])
        self.L3 += self._Ln(3, sigma_t, sigma_gtg[3,:,:])

    def calc_phi(self):
        # phi0
        print('Calc phi')
        B4 = self.B2 * self.B2
        LHS = (9 * B4 + self.B2 * (self.L3 @ self.L2 + (9 * self.L1 + 4 * self.L3) @ self.L0)
                + self.L3 @ self.L2 @ self.L1 @ self.L0)
        RHS = (self.L3 @ self.L2 @ self.L1 + self.B2 * (9 * self.L1 + 4 * self.L3)) @ self.chi[:-1]
        self.phi0 = np.linalg.solve(LHS,RHS)

        #phi2
        LHS = self.L3 @ self.L2
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
        df = pd.DataFrame({'phi0': self.phi0, 'phi2': self.phi2, 'Phi0': self.Phi0, 'Phi2': self.Phi2})
        df.to_hdf(f"{self.save_dir}/fluxes_{self.NH}.h5", key="df", mode="w", format="table")

    def run(self):
        self.initial_flux()
        print('Build Sigma_gtg and Ln for Uranium')
        self.calc_Ln(self.AU, self.sig_t_U, self.sig_s0_U)
        print(f'Build Sigma_gtg and Ln for Hydrogen, NH = {self.NH}')
        self.calc_Ln(self.AH, self.sig_t_H, self.sig_s0_H)
        self.transpose_and_save_Ln()
        self.calc_phi()
        self.plot_fluxes()
        self.plot_flux_diff()
        self.calc_Phi()
        if self.save_data == True: self.save_fluxes()


    @staticmethod
    def alpha_fn(A): return ((A - 1.0)/(A + 1.0)) ** 2

    @staticmethod
    def lga_fn(alpha): return -np.log(alpha) if alpha != 0 else np.inf

    @staticmethod
    def _Ln(l, sigma_t, sigma_gtg): return (2 * l + 1) * (np.diag(sigma_t[:-1]) - sigma_gtg)

    @staticmethod
    @njit(parallel=True, fastmath=True)
    def gen_sig_sn_gtg(A, sig_s0, order, boundaries, gmax_vec, alpha, E0, tol, n_sub = 8):
        """Parallel midpoint integration (no SciPy quad).
        Integrates over x in [c, x2] using n_sub midpoints per base-bin."""
        G = boundaries.size
        du = boundaries[1] - boundaries[0]
        den = (1.0 - alpha) * du
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

                    sigma_gtg[l, gp, g] = (sig_s0[gp] * (acc * dx) )/ den
                    if np.abs(sigma_gtg[l,gp,g]) < tol: break

        return sigma_gtg

#######################RUN########################
stt = time.time()
NH = 5
data_dir = 'data/'
B2 = .01
nbins = 20000

# init class
print(f"Initializing, {nbins} Groups")
sp3 = Sp3(data_dir,nbins,B2,NH)
sp3.run()

stp = time.time()
print(f"Calculation Time = {np.round((stp - stt),6)}")

