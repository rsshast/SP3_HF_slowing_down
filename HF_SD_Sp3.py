import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import time
import math
import argparse
import csv
import json
from pathlib import Path
from scipy.linalg import null_space
import h5py
from scipy.integrate import simpson, quad
from numba import njit, prange
import copy
import psutil
import torch # tensor decomps, gpu
import gc
from ce_buckled_transport_reference import build_scatter_matrix as build_ce_scatter_matrix

# upscattering: easier to put outside of the class
@njit(parallel=True)
def build_p0_thermal_to_all(E_full, sigma_fr_full, gmax_vec, A, kT, m, x, w, dE_full, insert_idx):
    G = len(E_full)
    N_th = G - insert_idx
    sigma_th_all = np.zeros((N_th, G))

    for i in prange(N_th):
        # gp is the absolute index in the full grid
        gp = insert_idx + i
        sig_val = sigma_fr_full[gp]
        E_prime = E_full[gp]

        gmax = min(gmax_vec[gp],G)
        for g in range(0, gmax):
            E_exit = E_full[g]
            val = eval_p0_kernel_numba(E_prime, E_exit, A, sig_val, kT, m, x, w)
            sigma_th_all[i, g] = val * dE_full[g]

    return sigma_th_all

@njit(parallel=True)
def build_p1_thermal_to_all(E_full, sigma_fr_full, gmax_vec, A, kT, m, x, w, dE_full, insert_idx):
    G = len(E_full)
    N_th = G - insert_idx
    sigma_th_all = np.zeros((N_th, G))

    for i in prange(N_th):
        gp = insert_idx + i
        sig_val = sigma_fr_full[gp]
        E_prime = E_full[gp]
        gmax = min(gmax_vec[gp],G)

        for g in range(0, gmax):
            E_exit = E_full[g]
            val = eval_p1_kernel_numba(E_prime, E_exit, A, sig_val, kT, m, x, w)
            sigma_th_all[i, g] = val * dE_full[g]

    return sigma_th_all

@njit(parallel=True)
def build_p2_thermal_to_all(E_full, sigma_fr_full, gmax_vec, A, kT, m, x, w, dE_full, insert_idx):
    G = len(E_full)
    N_th = G - insert_idx
    sigma_th_all = np.zeros((N_th, G))

    for i in prange(N_th):
        gp = insert_idx + i
        sig_val = sigma_fr_full[gp]
        E_prime = E_full[gp]

        gmax = min(gmax_vec[gp],G)
        for g in range(0, gmax):
            E_exit = E_full[g]
            val = eval_p2_kernel_numba(E_prime, E_exit, A, sig_val, kT, m, x, w)
            sigma_th_all[i, g] = val * dE_full[g]

    return sigma_th_all

@njit(parallel=True)
def build_p3_thermal_to_all(E_full, sigma_fr_full, gmax_vec, A, kT, m, x, w, dE_full, insert_idx):
    G = len(E_full)
    N_th = G - insert_idx
    sigma_th_all = np.zeros((N_th, G))

    for i in prange(N_th):
        gp = insert_idx + i
        sig_val = sigma_fr_full[gp]
        E_prime = E_full[gp]

        gmax = min(gmax_vec[gp],G)
        for g in range(0, gmax):
            E_exit = E_full[g]
            val = eval_p3_kernel_numba(E_prime, E_exit, A, sig_val, kT, m, x, w)
            sigma_th_all[i, g] = val * dE_full[g]

    return sigma_th_all

# P0 kernel
@njit
def eval_p0_kernel_numba(E_prime, E, A, Sigma_fr, kT, m, x, w):
    if A == 1.0:
        if E <= E_prime: # Downscatter
            return (Sigma_fr / E_prime) * math.erf(np.sqrt(E / kT))
        else: # Upscatter
            return (Sigma_fr / E_prime) * np.exp((E_prime - E) / kT) * math.erf(np.sqrt(E_prime / kT))

    kappa_min = np.sqrt(2.0 * m) * np.abs(np.sqrt(E_prime) - np.sqrt(E))
    kappa_max = np.sqrt(2.0 * m) * (np.sqrt(E_prime) + np.sqrt(E))

    half_width = 0.5 * (kappa_max - kappa_min)
    midpoint = 0.5 * (kappa_max + kappa_min)
    integral = 0.0

    for i in range(len(x)):
        kappa = half_width * x[i] + midpoint
        kappa2 = kappa * kappa

        exp_inner = E_prime - E - (kappa2 / (2.0 * A * m))
        exp_arg = - (A * m) / (2.0 * kT * kappa2) * (exp_inner**2)

        integrand = np.exp(exp_arg)
        integral += w[i] * integrand

    integral *= half_width

    term1 = (1 + 1 / A)**2
    term2 = np.sqrt(E / E_prime)
    term3 = np.sqrt((A * m) / (2.0 * np.pi * kT))
    coeff = (Sigma_fr / (8.0 * m * E_prime * E)) * term1 * term2 * term3

    return coeff * integral

# P1 kernel
@njit
def eval_p1_kernel_numba(E_prime, E, A, Sigma_fr, kT, m, x, w):
    if A == 1.0:
        mu_bar = np.sqrt(E / E_prime)
        if E <= E_prime:
            return (Sigma_fr / E_prime) * math.erf(np.sqrt(E / kT)) * mu_bar
        else:
            return (Sigma_fr / E_prime) * np.exp((E_prime - E) / kT) * math.erf(np.sqrt(E_prime / kT)) * mu_bar

    kappa_min = np.sqrt(2.0 * m) * np.abs(np.sqrt(E_prime) - np.sqrt(E))
    kappa_max = np.sqrt(2.0 * m) * (np.sqrt(E_prime) + np.sqrt(E))

    half_width = 0.5 * (kappa_max - kappa_min)
    midpoint = 0.5 * (kappa_max + kappa_min)
    integral = 0.0

    for i in range(len(x)):
        kappa = half_width * x[i] + midpoint
        kappa2 = kappa * kappa

        poly_part = E_prime + E - (kappa2 / (2.0 * m))
        exp_inner = E_prime - E - (kappa2 / (2.0 * A * m))
        exp_arg = - (A * m) / (2.0 * kT * kappa2) * (exp_inner**2)

        integrand = poly_part * np.exp(exp_arg)
        integral += w[i] * integrand

    integral *= half_width

    term1 = (1 + 1 / A)**2
    term2 = np.sqrt(E / E_prime)
    term3 = np.sqrt((A * m) / (2.0 * np.pi * kT))
    coeff = (Sigma_fr / (8.0 * m * E_prime * E)) * term1 * term2 * term3

    return coeff * integral

# P2 kernel
@njit
def eval_p2_kernel_numba(E_prime, E, A, Sigma_fr, kT, m, x, w):
    if A == 1.0:
        mu_bar = np.sqrt(E / E_prime)
        p2_val = 0.5 * (3.0 * mu_bar**2 - 1.0)
        if E <= E_prime:
            return (Sigma_fr / E_prime) * math.erf(np.sqrt(E / kT)) * p2_val
        else:
            return (Sigma_fr / E_prime) * np.exp((E_prime - E) / kT) * math.erf(np.sqrt(E_prime / kT)) * p2_val

    kappa_min = np.sqrt(2.0 * m) * np.abs(np.sqrt(E_prime) - np.sqrt(E))
    kappa_max = np.sqrt(2.0 * m) * (np.sqrt(E_prime) + np.sqrt(E))

    half_width = 0.5 * (kappa_max - kappa_min)
    midpoint = 0.5 * (kappa_max + kappa_min)
    integral = 0.0

    for i in range(len(x)):
        kappa = half_width * x[i] + midpoint
        kappa2 = kappa * kappa

        poly_inner = (E_prime + E - kappa2 / (2 * m)) / (2 * np.sqrt(E_prime * E))
        poly_part = 3 * poly_inner ** 2 - 1
        exp_inner = E_prime - E - (kappa2 / (2.0 * A * m))
        exp_arg = - (A * m) / (2.0 * kT * kappa2) * (exp_inner**2)

        integrand = poly_part * np.exp(exp_arg)
        integral += w[i] * integrand

    integral *= half_width

    term1 = (1 + 1 / A)**2
    term2 = np.sqrt(E / E_prime)
    term3 = np.sqrt((A * m) / (2.0 * np.pi * kT))
    coeff = (Sigma_fr / (8.0 * m * np.sqrt(E_prime * E))) * term1 * term2 * term3

    return coeff * integral

@njit
def eval_p3_kernel_numba(E_prime, E, A, Sigma_fr, kT, m, x, w):
    if A == 1.0:
        mu_bar = np.sqrt(E / E_prime)
        p3_val = 0.5 * (5.0 * mu_bar**3 - 3.0 * mu_bar)
        if E <= E_prime:
            return (Sigma_fr / E_prime) * math.erf(np.sqrt(E / kT)) * p3_val
        else:
            return (Sigma_fr / E_prime) * np.exp((E_prime - E) / kT) * math.erf(np.sqrt(E_prime / kT)) * p3_val

    kappa_min = np.sqrt(2.0 * m) * np.abs(np.sqrt(E_prime) - np.sqrt(E))
    kappa_max = np.sqrt(2.0 * m) * (np.sqrt(E_prime) + np.sqrt(E))

    half_width = 0.5 * (kappa_max - kappa_min)
    midpoint = 0.5 * (kappa_max + kappa_min)
    integral = 0.0

    for i in range(len(x)):
        kappa = half_width * x[i] + midpoint
        kappa2 = kappa * kappa

        poly_inner = (E_prime + E - kappa2 / (2 * m)) / (2 * np.sqrt(E_prime * E))
        poly_part = 5 * poly_inner ** 3 - (3 * poly_inner)
        exp_inner = E_prime - E - (kappa2 / (2.0 * A * m))
        exp_arg = - (A * m) / (2.0 * kT * kappa2) * (exp_inner**2)

        integrand = poly_part * np.exp(exp_arg)
        integral += w[i] * integrand

    integral *= half_width

    term1 = (1 + 1 / A)**2
    term2 = np.sqrt(E / E_prime)
    term3 = np.sqrt((A * m) / (2.0 * np.pi * kT))
    coeff = (Sigma_fr / (8.0 * m * np.sqrt(E_prime * E))) * term1 * term2 * term3

    return coeff * integral

class Sp3:
    def __init__(
        self,
        nbins,
        B2,
        NH,
        fromH5,
        adaptive,
        xs_tol,
        wims=True,
        group_structure="wims69.txt",
        p0_only_upscatter=False,
        no_upscatter=False,
        thermal_upscatter_cutoff_ev=4.0,
        kernel_quad=64,
    ):
        # directories
        self.data_dir = 'data/'
        self.save_dir = "/scratch/bckiedro_root/bckiedro0/rsshast/Sp3/results/"
        self.chart_dir = "results/charts/"
        self.save_data = False

        # grp constant parameters
        self.use_wims = wims
        self.group_structure = group_structure
        self.p0_only_upscatter = p0_only_upscatter
        self.no_upscatter = no_upscatter
        self.thermal_upscatter_cutoff_ev = thermal_upscatter_cutoff_ev
        if self.use_wims:
            group_table = np.loadtxt(Path(self.data_dir) / self.group_structure)
            self.custom_bounds = np.asarray(group_table[:, 1], dtype=float)
            self.few_groups = self.custom_bounds.size
            self.E0 = np.max(self.custom_bounds)
            self.Emin = np.min(self.custom_bounds)
        else:
            self.few_groups = 8
            self.E0 = 1e7
            self.Emin = .01
        self.fromH5 = fromH5
        self.nu = 2.43

        # anisotropy
        self.B2 = B2
        self.leg_order = 4

        # number densities
        self.AH = 1
        self.NH = NH
        self.AU = 238
        self.NU = .02
        self.AO = 16
        self.NO = .1
        #self.NO = self.NH / 2
        #self.NO = self.NH * 2

        # mass of a neutron
        self.m = 1 # amu
        self.nquad = kernel_quad

        # point-wise cross-sections and fission spectrum
        chi35 = pd.read_csv(f'{self.data_dir}chi_u235.txt', sep = '\t', header = 0)
        H1 = pd.read_csv(f'{self.data_dir}xs_h1_T293k.txt', sep = '\t', header = 0)
        U238 = pd.read_csv(f'{self.data_dir}xs_u238_T293k.txt',sep = '\t', header = 0)
        o16_path = Path(self.data_dir) / "xs_o16_T293k.csv"
        if not o16_path.exists():
            o16_path = Path(self.data_dir) / "xs_o16_T293k (1).csv"
        O16 = pd.read_csv(o16_path, sep = ';', dtype=float).to_numpy()
        sigma_f = pd.read_csv(f'{self.data_dir}xs_u238_fission.csv', sep = ',', dtype=float).to_numpy()
        sigma_fr_U = pd.read_csv(f"{self.data_dir}sigma_fr_U.csv",sep = ';', dtype=float).to_numpy()
        sigma_fr_H = pd.read_csv(f"{self.data_dir}sigma_fr_H.csv",sep = ';', dtype=float).to_numpy()

        # interpolations
        def get_adaptive_Ugrid(data, E0, Emin, tol):
            data = data[(data[:, 0] <= E0) & (data[:, 0] >= Emin)]
            data[:, 0] = np.log(E0 / data[:, 0])
            sort_idx = np.argsort(data[:, 0])
            data = data[sort_idx]
            new_grid = [data[0]]  # first row

            i = 0
            while i < data.shape[0] - 1:  # iterate until the second last element
                found = False
                for j in range(i + 1, data.shape[0]):
                    diff = 100 * np.abs(data[i, 1] - data[j, 1]) / data[i, 1]
                    if diff >= tol:
                        new_grid.append(data[j])  # append entire row
                        i = j
                        found = True
                        break
                if not found: break

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
            H = self.get_data(H,self.nbins) 
            #H = self.get_data(H,self.nbins) * self.NH
            XS38 = self.get_data(XS38,self.nbins)
            XS16 = self.get_data(O16, self.nbins)

        self.chi = chi[:,1]
        self.boundaries = XS38[:,0]
        self.Evec = self.E0 * np.exp(-self.boundaries)
        print(f"Initializing, {self.nbins - 1} Groups")
        print(f"Collapse to {self.few_groups - 1} Groups")
        #assert (self.nbins - 1) % self.few_groups == 0, ("Number of fine bins must be multiple of number of coarse bins")

        # cross-sections and number densities
        self.sigma_fr_U = self.get_data(sigma_fr_U,self.nbins)[:,1] * self.NU
        self.sigma_fr_H = self.get_data(sigma_fr_H,self.nbins)[:,1] * self.NH
        self.sig_t_U   = self.NU * XS38[:,1]
        self.sig_s0_U  = self.NU * XS38[:,2]
        self.sig_t_H   = self.NH * H[:,1]
        self.sig_s0_H  = self.NH * H[:,2]
        #self.sig_t_O   = self.NO * XS16[:,1]
        #self.sig_s0_O  = self.NO * XS16[:,2]
        self.sigma_f   = self.nu * (self.get_data(sigma_f,self.nbins)[:,1]) * self.NU
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
        #self.sigma_s_gtg_O = np.zeros_like(self.sigma_s_gtg_U)

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
        # plot initial flux and chi
        plt.figure(figsize=(8,6))
        plt.plot(self.Eplot, self.chi/np.sum(self.chi), label=r'$\chi$')
        plt.title(r'$^{235}$U Fission Spectrum $\chi (E)$')
        plt.xlabel('Energy [eV]')
        plt.ylabel(r'$\chi (E)$')
        plt.xscale('log')
        plt.grid(True, which='both')
        plt.legend()
        plt.savefig(f"{self.chart_dir}chi.png")
        plt.clf()

        plt.figure()
        plt.plot(self.Eplot, self.p0/np.sum(self.p0), label=r'$\phi_0$')
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

    def gtg_xs(self,A,sigma_s, sigma_fr, verbose = False, upscat = True):
        stt = time.time()

        def average_to_fine_bins(values):
            out = np.zeros_like(self.p0)
            for i in range(out.size):
                out[i] = np.trapz(values[i:i+2], self.boundaries[i:i+2]) / (self.boundaries[i+1] - self.boundaries[i])
            return out

        sig_s = average_to_fine_bins(sigma_s)
        sig_fr = average_to_fine_bins(sigma_fr)
        S = build_ce_scatter_matrix(
            float(A),
            sig_s,
            sig_fr,
            self.leg_order,
            self.boundaries,
            self.Evec,
            self.kT,
            self.nquad,
            upscatter=(upscat and not self.no_upscatter),
            p0_only_upscatter=self.p0_only_upscatter,
            thermal_cutoff_ev=self.thermal_upscatter_cutoff_ev,
        )

        print(f"Sigma_gtg A={A} Time: {np.round(time.time()-stt,5)} s")

        if self.save_data: self.save_sig_s_gtg(S,A,self.save_dir,self.NH)

        if A == 1: self.sigma_s_gtg_H = S
        elif A == 16: self.sigma_s_gtg_O = S
        else: self.sigma_s_gtg_U = S
        if verbose:
            sigma_gtg = np.transpose(S, (0, 2, 1))
            self.plot_sig_sn_gtg(sigma_gtg, sigma_s, A)
            self.plot_each_l(A,sigma_s,sigma_gtg)

        return S

    def calc_Ln(self,A,sigma_t, sigma_s, sigma_fr, verbose):
        # build loss operators from xs's
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

        S = self.gtg_xs(A,sigma_s,sigma_fr,verbose)
        sig_t = np.zeros_like(self.p0)
        for i in range(sig_t.size):
            sig_t[i] = np.trapz(sigma_t[i:i+2],self.boundaries[i:i+2]) / (self.boundaries[i+1] - self.boundaries[i])
        sig_t = torch.from_numpy(np.diag(sig_t)) # sig_t is now a torch tensor

        # calculate Ln
        stt = time.time()
        Ln = _torch_Ln(self.leg_order,sig_t,S,device='auto')
        # unpack
        self.L0 += Ln[0,:,:]
        self.L1 += Ln[1,:,:]
        self.L2 += Ln[2,:,:]
        self.L3 += Ln[3,:,:]

        print(f"Ln A = {A} Time: {np.round(time.time()-stt,5)} s")

    def calc_phi_torch(self, device=None, dtype=torch.float64):
        # gpu torch.linalg.solve
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
        # parametric study phi calculation
        print(f'Calc phi B2, {self.B2}')
        p0 = []
        p2 = []

        # phi0
        plt.figure()
        for i in range(self.B2.size):
            B2 = self.B2[i]
            print(f"phi0, {i+1}, B2 = {np.round(B2,5)}")
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

        # plot all of the fluxes
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

    def phi_weighted_sigma(self, sigma, A, key):
        # start generating few group xs's
        def sigma_vec_few_grp(sigma, phi, group_idx, E):
            # E is actually the lethargy boundaries
            few_grp_xs = np.zeros((group_idx.size - 1))
            # ensure sigma and phi are the same size
            assert sigma.size == phi.size

            for i in range(few_grp_xs.size):
                stt, stp = group_idx[i], group_idx[i+1]
                num = simpson(sigma[stt:stp] * phi[stt:stp], E[stt:stp])
                den = simpson(phi[stt:stp], E[stt:stp])
                few_grp_xs[i] = num/den

            return few_grp_xs

        stt=time.time()
        if self.use_wims:
            self.few_groups = len(self.custom_bounds) - 1
            fg_idx = np.zeros(self.few_groups + 1, dtype=int)
            for i, target_E in enumerate(self.custom_bounds):
                fg_idx[i] = np.argmin(np.abs(self.Evec - target_E))

        else: fg_idx = np.linspace(0,self.phi0.size,self.few_groups+1, dtype=int)
        E_ave = np.zeros((self.few_groups))
        u_ave = np.zeros_like(E_ave)
        sigma_fg = np.zeros_like(E_ave)
        for i in range(self.few_groups):
            E_ave[i] = self.Evec[fg_idx[i]]
            u_ave[i] = self.boundaries[fg_idx[i]]

        phi = self.phi0
        if sigma.size > phi.size: sigma = sigma[:-1]
        sigma_fg = sigma_vec_few_grp(sigma, phi, fg_idx, self.boundaries[:-1])
        sigma_fg_0 = sigma_vec_few_grp(sigma, self.Phi0, fg_idx, self.boundaries[:-1])
        sigma_fg_2 = sigma_vec_few_grp(sigma, self.Phi2, fg_idx, self.boundaries[:-1])
        print(f"Update A={A} {key} xs: {np.round(time.time()-stt,5)} s")
        print(f"L2 norm on phi0 and Phi0 weighted xs's for A={A}, {key}: {self.L2_norm(self.normalize(sigma_fg),self.normalize(sigma_fg_0))}")

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

    def few_group_indices(self):
        if self.use_wims:
            self.few_groups = len(self.custom_bounds) - 1
            fg_idx = np.zeros(self.few_groups + 1, dtype=int)
            for i, target_E in enumerate(self.custom_bounds):
                fg_idx[i] = np.argmin(np.abs(self.Evec - target_E))
            return fg_idx
        return np.linspace(0, self.phi0.size, self.few_groups + 1, dtype=int)

    def few_group_edges(self):
        if self.use_wims:
            return np.asarray(self.custom_bounds, dtype=float)
        return np.flip(np.exp(np.linspace(np.log(self.Emin), np.log(self.E0), self.few_groups + 1)))

    def collapse_chi(self):
        fg_idx = self.few_group_indices()
        chi = self.chi[:-1] if self.chi.size > self.phi0.size else self.chi
        chi_fg = np.zeros(self.few_groups)

        for i in range(self.few_groups):
            stt, stp = fg_idx[i], fg_idx[i + 1]
            if stp <= stt:
                raise ValueError(f"Invalid few-group bounds for chi collapse: group {i}, indices {stt}:{stp}.")
            chi_fg[i] = simpson(chi[stt:stp], self.boundaries[stt:stp])

        chi_sum = np.sum(chi_fg)
        if abs(chi_sum) < 1e-300:
            raise ValueError("Collapsed chi spectrum is numerically zero.")
        return chi_fg / chi_sum

    def phi_weighted_sigma_sl(self, M, A, l):
        # Get gtg flux weighted xs's
        stt=time.time()
        N = self.phi0.size
        E = self.Evec[:-1]
        u = self.boundaries
        if self.use_wims:
            self.few_groups = len(self.custom_bounds) - 1
            fg_idx = np.zeros(self.few_groups + 1, dtype=int)
            for i, target_E in enumerate(self.custom_bounds):
                fg_idx[i] = np.argmin(np.abs(self.Evec - target_E))

        else: fg_idx = np.linspace(0,self.phi0.size,self.few_groups+1, dtype=int)
        M_fg = np.zeros((self.few_groups,self.few_groups),dtype=float)
        M_fg_0 = np.zeros_like(M_fg)
        M_fg_2 = np.zeros_like(M_fg)
        phi = self.phi0

        for i in range(self.few_groups): # Exit group
            r0, r1 = fg_idx[i], fg_idx[i+1]
            for j in range(self.few_groups): # Incident group
                c0, c1 = fg_idx[j], fg_idx[j+1]
                M_fg[i,j] = np.sum(M[r0:r1, c0:c1] @ self.phi0[c0:c1]) / np.sum(self.phi0[c0:c1])
                M_fg_0[i,j] = np.sum(M[r0:r1, c0:c1] @ self.Phi0[c0:c1]) / np.sum(self.Phi0[c0:c1])
                M_fg_2[i,j] = np.sum(M[r0:r1, c0:c1] @ self.Phi2[c0:c1]) / np.sum(self.Phi2[c0:c1])

        M_fg_norm = M_fg / np.linalg.norm(M_fg)
        M_fg_0_norm = M_fg_0 / np.linalg.norm(M_fg_0)
        print(f"Update A={A} sigma_s{l} xs: {np.round(time.time()-stt,5)} s")
        print(f"L2 norm on phi0 vs Phi0 weighted gtg for A={A} and l={l}: {self.L2_norm((M_fg_norm), (M_fg_0_norm))}")

        return M_fg, M_fg_0, M_fg_2

    def Dn_coef(self, sig_s1):
        print(f"Calculating Batched Diffusion Coefficients for {self.few_groups} groups...")
        stt = time.time()

        # map to group struct
        self.few_groups = len(self.custom_bounds) - 1
        fg_idx = np.zeros(self.few_groups + 1, dtype=int)
        for i, target_E in enumerate(self.custom_bounds):
            fg_idx[i] = np.argmin(np.abs(self.Evec - target_E))

        N_fine = self.phi0.size
        N_coarse = self.few_groups

        S0_matrix = np.zeros((N_fine, N_coarse))
        S2_matrix = np.zeros((N_fine, N_coarse))
        den0_vec = np.zeros(N_coarse) # To store integrated Phi0 group flux
        den2_vec = np.zeros(N_coarse) # To store integrated Phi2 group flux

        for j in range(N_coarse):
            c0, c1 = fg_idx[j], fg_idx[j+1]
            S0_matrix[c0:c1, j] = self.Phi0[c0:c1]
            S2_matrix[c0:c1, j] = self.Phi2[c0:c1]
            den0_vec[j] = np.sum(self.Phi0[c0:c1])
            den2_vec[j] = np.sum(self.Phi2[c0:c1])

        small0 = np.where(np.abs(den0_vec) <= 1e-300)[0]
        small2 = np.where(np.abs(den2_vec) <= 1e-300)[0]
        if small0.size:
            raise FloatingPointError(f"D0 collapse has near-zero Phi0 denominator in groups {small0[:10] + 1}.")
        if small2.size:
            raise FloatingPointError(f"D2 collapse has near-zero Phi2 denominator in groups {small2[:10] + 1}.")
        print(
            "D_coef denominator diagnostics: "
            f"Phi0 min/max=({den0_vec.min():.6e}, {den0_vec.max():.6e}), "
            f"Phi2 min/max=({den2_vec.min():.6e}, {den2_vec.max():.6e})"
        )

        # Response_Matrix[fine_exit, coarse_incident]
        #print("   Performing batched matrix solve...")
        Y0_mat = np.linalg.solve(self.L1, S0_matrix)
        Y2_mat = np.linalg.solve(self.L3, S2_matrix)

        # 4. Collapse rows to exit coarse groups
        D0 = np.zeros((N_coarse, N_coarse))
        D2 = np.zeros((N_coarse, N_coarse))

        for i in range(N_coarse):
            r0, r1 = fg_idx[i], fg_idx[i+1]
            # Sum the fine-group responses within coarse exit group i
            D0[i, :] = np.sum(Y0_mat[r0:r1, :], axis=0) / den0_vec
            D2[i, :] = np.sum(Y2_mat[r0:r1, :], axis=0) / den2_vec

        # conventional coefs
        sigma_t = self.sig_t_U[:-1] + self.sig_t_H[:-1]
        B2 = self.B2 if (isinstance(self.B2, (float, int))) else 0
        D_tr_fine = self.diff_matrix(sigma_t, sig_s1, B2)
        D2_tr_fine = 9 / (15 * sigma_t)

        D_conv_0 = np.zeros((N_coarse, N_coarse))
        D_conv_2 = np.zeros((N_coarse, N_coarse))

        # Weight conventional D by phi0 (diagonal weight for efficiency)
        for j in range(N_coarse):
            c0, c1 = fg_idx[j], fg_idx[j+1]
            for i in range(N_coarse):
                r0, r1 = fg_idx[i], fg_idx[i+1]
                # Weighted sum across incident bins
                D_conv_0[i, j] = np.sum(D_tr_fine[r0:r1, c0:c1] @ self.phi0[c0:c1]) / np.sum(self.phi0[c0:c1])
                if i == j: # D2_conv is typically a diagonal approximation
                    D_conv_2[i, j] = np.sum(D2_tr_fine[c0:c1] * self.phi0[c0:c1]) / np.sum(self.phi0[c0:c1])

        print(f"D_coef Time: {time.time()-stt:.3f} s")
        return D_conv_0, D_conv_2, D0, D2

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
        print(Phi0-2*Phi2)
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

        self.Chi = self.collapse_chi()

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
        #sigma_s2_U = np.where(sigma_s2_U > 0, sigma_s2_U, 0)
        #sigma_s3_U = np.where(sigma_s3_U > 0, sigma_s3_U, 0)
        #sigma_s2_0_U = np.where(sigma_s2_0_U > 0, sigma_s2_0_U, 0)
        #sigma_s2_2_U = np.where(sigma_s2_2_U > 0, sigma_s2_2_U, 0)
        #sigma_s3_0_U = np.where(sigma_s3_0_U > 0, sigma_s3_0_U, 0)
        #sigma_s3_2_U = np.where(sigma_s3_2_U > 0, sigma_s3_2_U, 0)

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
        self.Sig_s1 = sigma_s1_U + sigma_s1_H
        self.Sig_s2 = sigma_s2_U + sigma_s2_H
        self.Sig_s3 = sigma_s3_U + sigma_s3_H
        self.Sig_s0_0 = sigma_s0_0_U + sigma_s0_0_H
        self.Sig_s0_2 = sigma_s0_2_U + sigma_s0_2_H
        self.Sig_s1_0 = sigma_s1_0_U + sigma_s1_0_H
        self.Sig_s1_2 = sigma_s1_2_U + sigma_s1_2_H
        self.Sig_s2_0 = sigma_s2_0_U + sigma_s2_0_H
        self.Sig_s2_2 = sigma_s2_2_U + sigma_s2_2_H
        self.Sig_s3_0 = sigma_s3_0_U + sigma_s3_0_H
        self.Sig_s3_2 = sigma_s3_2_U + sigma_s3_2_H

        # calulate diffusion coefs
        self.D_conv_0, self.D_conv_2, self.D0, self.D2 = self.Dn_coef(self.sigma_s_gtg_U[1,:,:] + self.sigma_s_gtg_H[1,:,:])
        pdf = self.get_percent_diff(self.D_conv_0,self.D0,0)
        pdf = self.get_percent_diff(self.D_conv_2,self.D2,0)
        if verbose: print(f"{pdf:5g}, Diffusion Coef 0")

        if self.save_data == True: self.save_Dn(self.D_conv_0,self.D0,self.D2,self.save_dir,key=None)

    def precompute_sp3_operators(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            t_L0 = torch.from_numpy(self.L0).to("cuda")
            t_L1 = torch.from_numpy(self.L1).to("cuda")
            t_L2 = torch.from_numpy(self.L2).to("cuda")
            t_L3 = torch.from_numpy(self.L3).to("cuda")

            self.L3210 = (t_L3 @ t_L2 @ t_L1 @ t_L0).cpu().numpy()
            self.L_B2_term = (t_L3 @ t_L2 + (9 * t_L1 + 4 * t_L3) @ t_L0).cpu().numpy()
            self.L321_source = (t_L3 @ t_L2 @ t_L1).cpu().numpy()
            self.L_source_B2 = (9 * t_L1 + 4 * t_L3).cpu().numpy()
            self.L32 = (t_L3 @ t_L2).cpu().numpy()
            self.L_phi2_source = (9 * t_L1 + 4 * t_L3).cpu().numpy()

            del t_L0, t_L1, t_L2, t_L3
            torch.cuda.empty_cache()
        else:
            self.L3210 = self.L3 @ self.L2 @ self.L1 @ self.L0
            self.L_B2_term = self.L3 @ self.L2 + (9 * self.L1 + 4 * self.L3) @ self.L0
            self.L321_source = self.L3 @ self.L2 @ self.L1
            self.L_source_B2 = 9 * self.L1 + 4 * self.L3
            self.L32 = self.L3 @ self.L2
            self.L_phi2_source = 9 * self.L1 + 4 * self.L3
        gc.collect()

    def solve_fine_sp3_fixed_b2(self, b2):
        self.B2 = b2
        B4 = b2 * b2
        LHS = self.L3210 + b2 * self.L_B2_term
        np.fill_diagonal(LHS, LHS.diagonal() + 9 * B4)
        RHS = (self.L321_source + b2 * self.L_source_B2) @ self.chi
        self.phi0 = np.linalg.solve(LHS, RHS)

        RHS2 = 0.5 * (-9 * b2 * self.phi0 + self.L_phi2_source @ (self.L0 @ self.phi0 - self.chi))
        self.phi2 = np.linalg.solve(self.L32, RHS2)
        self.calc_Phi()

    @staticmethod
    def fixed_b2_new_flux(D0, D2, Sig_t_0, Sig_t_2, Sig_s0_0, Sig_s0_2, Sig_s2_2,
                          chi, nuSigf0, nuSigf2, b2, max_iters=10000, tol=1e-10):
        G = chi.size
        chi = chi / np.sum(chi)
        R0 = np.diag(Sig_t_0) - Sig_s0_0
        R2_new = np.diag(Sig_t_2) - Sig_s0_2
        R2_bot = np.diag(Sig_t_2) - Sig_s2_2

        A00 = R0 + b2 * D0
        A02 = -2 * R2_new
        A20 = -2 * R0
        A22 = R2_bot + 4 * R2_new + b2 * D2
        lhs = np.block([[A00, A02], [A20, A22]])
        rhs = np.concatenate([chi, -2 * chi])
        sol = np.linalg.solve(lhs, rhs)
        Phi0 = sol[:G]
        Phi2 = sol[G:]
        scalar = Phi0 - 2 * Phi2
        k_eff = float(np.sum(nuSigf0 * Phi0 - 2 * nuSigf2 * Phi2))
        residual = np.linalg.norm(lhs @ sol - rhs) / max(np.linalg.norm(rhs), 1e-300)
        return scalar, Phi2, k_eff, 1, residual

    @staticmethod
    def fixed_b2_conv_flux(Sig_t, Sig_s0, Sig_s2, chi, nuSigf, D0, D2,
                           b2, max_iters=10000, tol=1e-10):
        G = chi.size
        chi = chi / np.sum(chi)
        R0 = np.diag(Sig_t) - Sig_s0
        R2 = np.diag(Sig_t) - Sig_s2

        A00 = R0 + b2 * D0
        A02 = -2 * R0
        A20 = -2 * R0
        A22 = R2 + 4 * R0 + b2 * D2
        lhs = np.block([[A00, A02], [A20, A22]])
        rhs = np.concatenate([chi, -2 * chi])
        sol = np.linalg.solve(lhs, rhs)
        Phi0 = sol[:G]
        Phi2 = sol[G:]
        scalar = Phi0 - 2 * Phi2
        k_eff = float(np.sum(nuSigf * scalar))
        residual = np.linalg.norm(lhs @ sol - rhs) / max(np.linalg.norm(rhs), 1e-300)
        return scalar, Phi2, k_eff, 1, residual

    @staticmethod
    def relative_l2(candidate, reference):
        candidate = np.real_if_close(candidate).real
        reference = np.real_if_close(reference).real
        denom = np.linalg.norm(reference.ravel())
        if denom <= 1e-300:
            return np.nan
        return float(np.linalg.norm((candidate - reference).ravel()) / denom)

    @staticmethod
    def normalized_l2(candidate, reference):
        candidate = np.real_if_close(candidate).real
        reference = np.real_if_close(reference).real
        csum = np.sum(candidate)
        rsum = np.sum(reference)
        if abs(csum) <= 1e-300 or abs(rsum) <= 1e-300:
            return np.nan
        return float(np.linalg.norm((candidate / csum - reference / rsum).ravel()))

    def write_fixed_b2_outputs(self, outdir, b2, phi0_fg, phi2_fg, ce_npz=None):
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        self.chart_dir = str(outdir) + os.sep
        edges = self.few_group_edges()
        du = np.log(edges[:-1] / edges[1:])
        scatter_trad = np.stack([self.Sig_s0, self.Sig_s1, self.Sig_s2, self.Sig_s3])
        scatter_new_phi0 = np.stack([self.Sig_s0_0, self.Sig_s1_0, self.Sig_s2_0, self.Sig_s3_0])
        scatter_new_phi2 = np.stack([self.Sig_s0_2, self.Sig_s1_2, self.Sig_s2_2, self.Sig_s3_2])

        sp3_npz = outdir / "sp3_results.npz"
        scalar_diagnostics = {
            "new_phi0_min": float(np.min(np.real_if_close(self.phi0_sp3_new).real)),
            "traditional_phi0_min": float(np.min(np.real_if_close(self.phi0_sp3_conv).real)),
            "new_phi0_negative_count": int(np.count_nonzero(np.real_if_close(self.phi0_sp3_new).real < 0.0)),
            "traditional_phi0_negative_count": int(np.count_nonzero(np.real_if_close(self.phi0_sp3_conv).real < 0.0)),
            "D_new_2_min": float(np.min(np.real_if_close(self.D2).real)),
            "D_new_2_max": float(np.max(np.real_if_close(self.D2).real)),
        }
        np.savez_compressed(
            sp3_npz,
            B2=b2,
            group_edges_e=edges,
            coarse_du=du,
            fine_energy_e=self.Eplot,
            fine_phi0=self.phi0,
            fine_phi2=self.phi2,
            fine_Phi0=self.Phi0,
            fine_Phi2=self.Phi2,
            collapsed_fine_phi0=phi0_fg,
            collapsed_fine_phi2=phi2_fg,
            phi0_new=self.phi0_sp3_new,
            phi2_new=self.phi2_sp3_new,
            phi0_trad=self.phi0_sp3_conv,
            phi2_trad=self.phi2_sp3_conv,
            k_new=self.k_sp3_new,
            k_trad=self.k_sp3_conv,
            iters_new=self.iters_sp3_new,
            iters_trad=self.iters_sp3_conv,
            err_new=self.err_sp3_new,
            err_trad=self.err_sp3_conv,
            chi=self.Chi,
            trad_Sigma_t=self.Sig_t,
            trad_nuSigma_f=self.Sig_f,
            trad_Sigma_s=scatter_trad,
            new_Sigma_t_phi0=self.Sig_t_0,
            new_Sigma_t_phi2=self.Sig_t_2,
            new_nuSigma_f_phi0=self.Sig_f_0,
            new_nuSigma_f_phi2=self.Sig_f_2,
            new_Sigma_s_phi0=scatter_new_phi0,
            new_Sigma_s_phi2=scatter_new_phi2,
            D_trad_0=self.D_conv_0,
            D_trad_2=self.D_conv_2,
            D_new_0=self.D0,
            D_new_2=self.D2,
            scalar_diagnostics=json.dumps(scalar_diagnostics),
        )
        self.write_spectrum_diagnostics(outdir, edges, du)

        if ce_npz is not None:
            ref_path = self.resolve_reference_npz_path(ce_npz, outdir)
            if ref_path is None:
                print(
                    f"Warning: reference npz '{ce_npz}' was not found. "
                    f"Skipping CE/SP3 comparison; SP3 data were still written to {sp3_npz}."
                )
            else:
                self.write_reference_comparison(outdir, b2, ref_path, sp3_npz)

    def write_spectrum_diagnostics(self, outdir, edges, du):
        def safe_frac(phi):
            phi = np.real_if_close(phi).real
            total = np.sum(phi)
            if abs(total) <= 1e-300:
                return np.full_like(phi, np.nan, dtype=float)
            return phi / total

        new_frac = safe_frac(self.phi0_sp3_new)
        trad_frac = safe_frac(self.phi0_sp3_conv)
        new_density = new_frac / du
        trad_density = trad_frac / du
        rows = []
        for i in range(self.few_groups):
            rows.append({
                "group": i + 1,
                "E_hi_eV": edges[i],
                "E_lo_eV": edges[i + 1],
                "du": du[i],
                "new_phi0": self.phi0_sp3_new[i],
                "traditional_phi0": self.phi0_sp3_conv[i],
                "new_fraction": new_frac[i],
                "traditional_fraction": trad_frac[i],
                "new_density": new_density[i],
                "traditional_density": trad_density[i],
            })

        with (Path(outdir) / "sp3_spectrum_diagnostics.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        fast = 0
        print(
            "SP3 fastest-group diagnostic: "
            f"E=[{edges[fast + 1]:.6e}, {edges[fast]:.6e}] eV, "
            f"new_fraction={new_frac[fast]:.6e}, traditional_fraction={trad_frac[fast]:.6e}, "
            f"new_density={new_density[fast]:.6e}, traditional_density={trad_density[fast]:.6e}"
        )

    @staticmethod
    def resolve_reference_npz_path(reference_npz, outdir):
        ref_path = Path(reference_npz)
        default_sn = Path(outdir) / "ce_buckled_transport_reference.npz"
        default_pn = Path(outdir) / "ce_buckled_pn_reference.npz"
        candidates = []
        if ref_path.name == "sp3_results.npz":
            candidates.extend([default_sn, default_pn])
        candidates.append(ref_path)
        if not ref_path.is_absolute():
            candidates.append(Path(outdir) / ref_path)
            candidates.extend([default_sn, default_pn])
            candidates.append(Path.cwd() / ref_path)

        seen = set()
        for candidate in candidates:
            candidate = candidate.resolve()
            if candidate in seen:
                continue
            seen.add(candidate)
            if candidate.exists():
                return candidate
        return None

    @staticmethod
    def validate_reference_npz(reference_npz):
        required = {
            "coarse_phi0",
            "coarse_phi0_density",
            "Sigma_t",
            "nuSigma_f",
            "chi_few",
            "Sigma_s_moment_weighted",
            "k_at_B2",
            "group_edges_e",
        }
        with np.load(reference_npz) as data:
            missing = sorted(required.difference(data.files))
            available = sorted(data.files)
        return missing, available

    @staticmethod
    def reference_label(reference):
        ref_type = "SN"
        if "reference_type" in reference.files:
            raw = reference["reference_type"]
            ref_type = str(raw.item() if getattr(raw, "shape", ()) == () else raw)
        if ref_type.upper() == "PN":
            order = int(reference["pn_order"]) if "pn_order" in reference.files else int(reference["moments"].shape[0] - 1)
            return f"CE P{order}", f"ce_p{order}_k_at_B2"
        if "sn_order" in reference.files:
            return f"CE S{int(reference['sn_order'])}", "ce_sn_k_at_B2"
        return "CE SN", "ce_sn_k_at_B2"

    def write_reference_comparison(self, outdir, b2, reference_npz, sp3_npz):
        outdir = Path(outdir)
        missing, available = self.validate_reference_npz(reference_npz)
        if missing:
            print(
                f"Warning: '{reference_npz}' is not a compatible CE buckled reference npz. "
                f"Missing keys: {', '.join(missing)}. "
                f"Available keys include: {', '.join(available[:12])}. "
                "Skipping CE/SP3 comparison."
            )
            return

        ref = np.load(reference_npz)
        sp3 = np.load(sp3_npz)
        ref_label, ref_k_key = self.reference_label(ref)
        metrics = []

        def add_metric(quantity, method, candidate, reference, metric="relative_l2"):
            if candidate.shape != reference.shape:
                value = np.nan
                note = f"shape mismatch candidate={candidate.shape}, reference={reference.shape}"
            elif not np.all(np.isfinite(np.real_if_close(candidate).real)):
                value = np.nan
                note = "candidate has non-finite values"
            elif not np.all(np.isfinite(np.real_if_close(reference).real)):
                value = np.nan
                note = "reference has non-finite values"
            else:
                value = self.normalized_l2(candidate, reference) if metric == "normalized_l2" else self.relative_l2(candidate, reference)
                note = ""
            metrics.append({"quantity": quantity, "method": method, "metric": metric, "value": value, "note": note})

        add_metric("phi0", "new", sp3["phi0_new"], ref["coarse_phi0"].real, "normalized_l2")
        add_metric("phi0", "traditional", sp3["phi0_trad"], ref["coarse_phi0"].real, "normalized_l2")
        add_metric("Sigma_t", "new_phi0_weighted", sp3["new_Sigma_t_phi0"], ref["Sigma_t"])
        add_metric("Sigma_t", "traditional", sp3["trad_Sigma_t"], ref["Sigma_t"])
        add_metric("nuSigma_f", "new_phi0_weighted", sp3["new_nuSigma_f_phi0"], ref["nuSigma_f"])
        add_metric("nuSigma_f", "traditional", sp3["trad_nuSigma_f"], ref["nuSigma_f"])
        add_metric("chi", "new_and_traditional", sp3["chi"], ref["chi_few"], "normalized_l2")

        ref_scatter = ref["Sigma_s_moment_weighted"]
        for ell in range(min(ref_scatter.shape[0], sp3["trad_Sigma_s"].shape[0])):
            add_metric(f"Sigma_s_l{ell}", "traditional", sp3["trad_Sigma_s"][ell], ref_scatter[ell])
            add_metric(f"Sigma_s_l{ell}", "new_phi0_weighted", sp3["new_Sigma_s_phi0"][ell], ref_scatter[ell])
            add_metric(f"Sigma_s_l{ell}", "new_phi2_weighted", sp3["new_Sigma_s_phi2"][ell], ref_scatter[ell])

        with (outdir / "error_metrics.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["quantity", "method", "metric", "value", "note"])
            writer.writeheader()
            for row in metrics:
                writer.writerow(row)

        summary = {
            "B2": float(b2),
            "reference_file": str(reference_npz),
            "reference_label": ref_label,
            "reference_k_at_B2": float(np.real_if_close(ref["k_at_B2"]).real),
            ref_k_key: float(np.real_if_close(ref["k_at_B2"]).real),
            "sp3_k_new": float(np.real_if_close(sp3["k_new"]).real),
            "sp3_k_traditional": float(np.real_if_close(sp3["k_trad"]).real),
            "metrics": metrics,
        }
        (outdir / "comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        self.plot_reference_sp3_spectra(outdir, b2, ref, sp3, ref_label)

    def plot_reference_sp3_spectra(self, outdir, b2, reference, sp3, reference_label):
        edges = reference["group_edges_e"][::-1]
        ref_phi = np.real_if_close(reference["coarse_phi0"]).real
        ref_du = reference["coarse_du"]
        sp3_edges_desc = sp3["group_edges_e"]
        du = sp3["coarse_du"]

        def norm_density_from_integral(phi, widths):
            phi = np.real_if_close(phi).real
            total = np.sum(phi)
            if abs(total) > 1e-300:
                phi = phi / total
            return (phi / widths)[::-1]

        ref_density = norm_density_from_integral(ref_phi, ref_du)
        new_density = norm_density_from_integral(sp3["phi0_new"], du)
        trad_density = norm_density_from_integral(sp3["phi0_trad"], du)

        plt.figure(figsize=(9, 5.5))
        plt.step(edges, np.r_[ref_density, ref_density[-1]], where="post", label=fr"{reference_label}, $k={float(reference['k_at_B2']):.6g}$")
        plt.step(sp3_edges_desc[::-1], np.r_[new_density, new_density[-1]], where="post", label=fr"new SP3, $k={float(sp3['k_new']):.6g}$")
        plt.step(sp3_edges_desc[::-1], np.r_[trad_density, trad_density[-1]], where="post", label=fr"traditional SP3, $k={float(sp3['k_trad']):.6g}$")
        plt.xscale("log")
        plt.xlabel("Energy (eV)")
        plt.ylabel("Normalized few-group scalar flux per lethargy")
        plt.title(fr"Fixed $B^2={b2:.6e}$")
        plt.grid(True, which="both", alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(Path(outdir) / "ce_sp3_few_group_spectra.png", dpi=200)
        plt.close()

    def run_fixed_b2(self, b2, outdir, ce_npz=None, verbose=False, fixed_iters=10000,
                     fixed_tol=1e-10, plot_b2_zero=True):
        self.save_dir = str(Path(outdir)) + os.sep
        self.chart_dir = str(Path(outdir)) + os.sep
        Path(outdir).mkdir(parents=True, exist_ok=True)
        self.initial_flux()
        if self.fromH5 == True:
            self.read_data()
        else:
            print(f"Starting fixed-B2 SP3 calculation. B2 = {b2:.8e}")
            self.calc_Ln(self.AU, self.sig_t_U, self.sig_s0_U, self.sigma_fr_U, verbose)
            self.calc_Ln(self.AH, self.sig_t_H, self.sig_s0_H, self.sigma_fr_H, verbose)
            self.precompute_sp3_operators()

        self.solve_fine_sp3_fixed_b2(b2)
        self.upd_grp_constants(verbose=False)

        self.phi0_sp3_new, self.phi2_sp3_new, self.k_sp3_new, self.iters_sp3_new, self.err_sp3_new = self.fixed_b2_new_flux(
            self.D0, self.D2,
            self.Sig_t_0, self.Sig_t_2,
            self.Sig_s0_0, self.Sig_s0_2, self.Sig_s2_2,
            self.Chi, self.Sig_f_0, self.Sig_f_2,
            b2, max_iters=fixed_iters, tol=fixed_tol,
        )
        self.phi0_sp3_conv, self.phi2_sp3_conv, self.k_sp3_conv, self.iters_sp3_conv, self.err_sp3_conv = self.fixed_b2_conv_flux(
            self.Sig_t, self.Sig_s0, self.Sig_s2,
            self.Chi, self.Sig_f,
            self.D_conv_0, self.D_conv_2,
            b2, max_iters=fixed_iters, tol=fixed_tol,
        )
        self.B2_new = b2
        self.B2_conv = b2
        phi0_fg, phi2_fg = self.few_group_fluxes(key=None)
        self.plot_few_grp_sp3_eqns(phi0_fg if plot_b2_zero else None)
        self.write_fixed_b2_outputs(outdir, b2, phi0_fg, phi2_fg, ce_npz=ce_npz)

    def run(self, verbose=False, transport=False, b2_tol=1e-5, max_outer_iters=20, plot_b2_zero=True):
        self.initial_flux()
        if self.fromH5 == True:
            self.read_data()
        else:
            print(f"Starting calculation. Saving Data = {self.save_data}")
            self.calc_Ln(self.AU, self.sig_t_U, self.sig_s0_U, self.sigma_fr_U, verbose)
            self.calc_Ln(self.AH, self.sig_t_H, self.sig_s0_H, self.sigma_fr_H, verbose)

            # precompute operators in first step, use downstream
            print("\nPrecomputing constant SP3 operators...")

            # Use GPU if available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cuda":
                t_L0 = torch.from_numpy(self.L0).to("cuda")
                t_L1 = torch.from_numpy(self.L1).to("cuda")
                t_L2 = torch.from_numpy(self.L2).to("cuda")
                t_L3 = torch.from_numpy(self.L3).to("cuda")

                self.L3210 = (t_L3 @ t_L2 @ t_L1 @ t_L0).cpu().numpy()
                self.L_B2_term = (t_L3 @ t_L2 + (9 * t_L1 + 4 * t_L3) @ t_L0).cpu().numpy()
                self.L321_source = (t_L3 @ t_L2 @ t_L1).cpu().numpy()
                self.L_source_B2 = (9 * t_L1 + 4 * t_L3).cpu().numpy()
                self.L32 = (t_L3 @ t_L2).cpu().numpy()
                self.L_phi2_source = (9 * t_L1 + 4 * t_L3).cpu().numpy()

                del t_L0, t_L1, t_L2, t_L3
                torch.cuda.empty_cache()
            else:
                self.L3210 = self.L3 @ self.L2 @ self.L1 @ self.L0
                self.L_B2_term = self.L3 @ self.L2 + (9 * self.L1 + 4 * self.L3) @ self.L0
                self.L321_source = self.L3 @ self.L2 @ self.L1
                self.L_source_B2 = 9 * self.L1 + 4 * self.L3
                self.L32 = self.L3 @ self.L2
                self.L_phi2_source = 9 * self.L1 + 4 * self.L3

            gc.collect()

            # FUNDAMENTAL MODE SEARCH LOOP
            print("\n--- Starting Fundamental Mode Buckling Search ---")
            current_B2 = 0.0001 # Strict infinite medium start

            #omegas = np.linspace(1.55,1.65,21)
            for outer_iter in range(max_outer_iters):
                print(f"\nNew Outer Iteration {outer_iter + 1} | Input B2 = {current_B2:5e}")
                self.B2 = current_B2

                # solve slowing down spectrum on fine grid
                stt = time.time()
                B4 = self.B2 * self.B2
                # LHS
                LHS = self.L3210 + self.B2 * self.L_B2_term
                np.fill_diagonal(LHS, LHS.diagonal() + 9 * B4)
                # RHS
                RHS = (self.L321_source + self.B2 * self.L_source_B2) @ self.chi
                self.phi0 = np.linalg.solve(LHS, RHS)

                # Solve phi2
                RHS2 = 0.5 * (-9 * self.B2 * self.phi0 + self.L_phi2_source @ (self.L0 @ self.phi0 - self.chi))
                self.phi2 = np.linalg.solve(self.L32, RHS2)

                print(f"Hyperfine Solve Time: {time.time()-stt:.5f} s")
                self.calc_Phi()

                # few group constants
                self.upd_grp_constants(verbose=False)

                # eigenvalue solver for B2
                #for omega in (omegas):
                new_B2, fg_phi0, fg_phi2 = self.B2_eigenvalue_new(
                    self.D0, self.D2,
                    self.Sig_t_0, self.Sig_t_2,
                    self.Sig_s0_0, self.Sig_s0_2, self.Sig_s2_2,
                    self.Chi, self.Sig_f_0, self.Sig_f_2,
                    current_B2,
                )
                #new_B2, fg_phi0, fg_phi2 = self.B2_eigenvalue_new(
                #        self.D0, self.D2,
                #        self.Sig_t_0, self.Sig_t_0,
                #        self.Sig_s0_0, self.Sig_s0_0, self.Sig_s2_0,
                #        self.Chi, self.Sig_f_0, self.Sig_f_0,
                #        current_B2,
                #        #omega=omega,
                #    )
                # check convergence
                if abs(current_B2) > 0: err = abs(new_B2 - current_B2) / abs(current_B2)
                else: err = abs(new_B2)

                print(f"Resulting B2 = {new_B2:5e} | Relative Error = {err:5e}")

                if err < b2_tol and outer_iter > 0:
                    print(f"\n*** Fundamental Mode Search Converged in {outer_iter+1} iterations! ***")
                    self.B2_new = new_B2
                    self.phi0_sp3_new = fg_phi0
                    self.phi2_sp3_new = fg_phi2
                    break

                current_B2 = new_B2
                gc.collect()

            else:
                print("\n*** Warning: Reached max iterations without converging! ***")

            self.B2_new = new_B2
            self.phi0_sp3_new = fg_phi0
            self.phi2_sp3_new = fg_phi2

            # =====================================================================
            # CONVENTIONAL FUNDAMENTAL MODE SEARCH LOOP
            # =====================================================================
            print("\n--- Starting Conventional Fundamental Mode Buckling Search ---")
            current_B2_conv = 0.0001 # Start fresh for the conventional solver

            for outer_iter in range(max_outer_iters):
                print(f"\nConv Outer Iteration {outer_iter + 1} | Input B2 = {current_B2_conv:5e}")
                self.B2 = current_B2_conv

                stt = time.time()
                B4 = self.B2 * self.B2
                LHS = self.L3210 + self.B2 * self.L_B2_term
                np.fill_diagonal(LHS, LHS.diagonal() + 9 * B4)
                RHS = (self.L321_source + self.B2 * self.L_source_B2) @ self.chi
                self.phi0 = np.linalg.solve(LHS, RHS)

                RHS2 = 0.5 * (-9 * self.B2 * self.phi0 + self.L_phi2_source @ (self.L0 @ self.phi0 - self.chi))
                self.phi2 = np.linalg.solve(self.L32, RHS2)
                self.calc_Phi()

                self.upd_grp_constants(verbose=False)
                new_B2_conv, fg_phi0_conv, fg_phi2_conv = self.B2_eigenvalue_conv(
                    self.Sig_t,
                    self.Sig_s0, self.Sig_s2,
                    self.Chi, self.Sig_f,
                    self.D_conv_0, self.D_conv_2,
                    current_B2_conv
                )

                if abs(current_B2_conv) > 0:
                    err = abs(new_B2_conv - current_B2_conv) / abs(current_B2_conv)
                else:
                    err = abs(new_B2_conv)

                print(f"Resulting Conv B2 = {new_B2_conv:5e} | Relative Error = {err:5e}")

                if err < b2_tol and outer_iter > 0:
                    print(f"\n*** Conventional Search Converged in {outer_iter+1} iterations! ***")
                    self.B2_conv = new_B2_conv
                    self.phi0_sp3_conv = fg_phi0_conv
                    self.phi2_sp3_conv = fg_phi2_conv
                    break

                current_B2_conv = new_B2_conv
                gc.collect()
            else:
                print("\n*** Warning: Conventional solver reached max iterations without converging! ***")
            self.B2_conv = new_B2_conv
            self.phi0_sp3_conv = fg_phi0_conv
            self.phi2_sp3_conv = fg_phi2_conv
            if self.save_data: self.save_fluxes()

        # plot and save
        phi0_fg, phi2_fg = self.few_group_fluxes(key=None)
        self.plot_few_grp_sp3_eqns(phi0_fg if plot_b2_zero else None)
        # --- End of Run --- #

    # ----- Reading Data, Plotting and Saving----- #
    def save_Ln(self):
        if self.save_data == True:
            with h5py.File(f"{self.save_dir}Ln_{self.NH}.h5", "w") as f:
                f.create_dataset("L0", data=self.L0.numpy(),compression="gzip", compression_opts=4)
                f.create_dataset("L1", data=self.L1.numpy(),compression="gzip", compression_opts=4)
                f.create_dataset("L2", data=self.L2.numpy(),compression="gzip", compression_opts=4)
                f.create_dataset("L3", data=self.L3.numpy(),compression="gzip", compression_opts=4)

    def save_fluxes(self):
        print("Saving Data...")
        df = pd.DataFrame({'phi0': self.phi0, 'phi2': self.phi2,
            'Phi0': self.Phi0, 'Phi2': self.Phi2,
            'phi_ref': self.p0, 'chi': self.chi})
        df.to_hdf(f"{self.save_dir}fluxes_{self.NH}.h5", key="df", mode="w", format="table")
        #df.to_csv(f"{self.save_dir}fluxes_{self.NH}.csv")

        df = pd.DataFrame({'sigma_t_U': self.sig_t_U, 'sigma_t_H': self.sig_t_H,#'sigma_t_O': self.sig_t_O,
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
        plt.xlim([self.Emin,self.E0])
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
        plt.xlim([self.Emin,self.E0])
        plt.grid(True, which='both')
        plt.legend()
        plt.savefig(f'{self.chart_dir}phi2_{self.NH}.png')
        plt.clf()

    def plot_flux_diff_single_axis(self):
        # compare the traditional to new method
        plt.figure()
        plt.plot(self.Eplot,self.p0,label='Scattering Source')
        plt.plot(self.Eplot,self.phi0,label='Sp3')
        plt.title(f"Hyperfine Slowing-Down Flux Comparison, {self.nbins-1} Groups")
        plt.ylabel(r"$\phi (E) (n/cm^2)$")
        plt.xscale('log')
        plt.xlabel("Energy (eV)")
        plt.legend()
        plt.xlim([self.Emin,self.E0])
        plt.grid(True,which='both')
        plt.savefig(f"{self.chart_dir}order_comp_{self.nbins -1}.png")
        plt.clf()

        plt.figure()
        plt.plot(self.Eplot, 100 * (self.phi0 - self.p0) / self.p0)
        plt.title(f"Hyperfine Slowing-Down Flux Difference, {self.nbins - 1} Groups")
        plt.ylabel("% Difference Between Calculated and Reference Spectra")
        plt.xscale('log')
        plt.xlim([self.Emin,self.E0])
        plt.xlabel("Energy (eV)")
        plt.grid(True,which='both')
        plt.savefig(f"{self.chart_dir}flux_difference_{self.nbins -1}.png")

    def plot_flux_diff(self):
        fig, ax1 = plt.subplots(figsize=(7, 5))

        # Left y-axis: p0 and phi0
        p0 = self.normalize(self.p0)
        phi0 = self.normalize(self.phi0)
        ax1.plot(self.Eplot, self.p0, label='Scattering Source', color='tab:blue', lw=1.5)
        ax1.plot(self.Eplot, self.phi0, label='SP3', color='tab:purple', lw=1.5)
        ax1.set_xscale('log')
        ax1.set_xlabel("Energy (eV)")
        ax1.set_ylabel(r"$\phi(E)$  $(n/cm^2)$", color='k')
        ax1.grid(True, which='both', linestyle=':')
        ax1.tick_params(axis='y', labelcolor='k')

        # Right y-axis: percent difference
        ax2 = ax1.twinx()
        diff = 100 * (self.phi0 - self.p0) / self.phi0
        ax2.plot(self.Eplot, diff, color='tab:red', lw=1.2, alpha=0.8, label='% Difference',linestyle='dotted')
        ax2.set_ylabel("% Difference", color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')

        # Title and legend
        fig.suptitle(f"Hyperfine Slowing-Down Flux Comparison, {self.nbins - 1} Groups")
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        fig.tight_layout()
        fig.savefig(f"{self.chart_dir}flux_comparison_dual_axes_{self.nbins -1}.png", dpi=300)
        plt.close(fig)

        # evaluate fluxes
        L2 = self.L2_norm(self.phi0,self.p0)
        print(f"L2 norm on phi0 and reference, = {np.round(L2,9)}")
        L2 = self.L2_norm(self.Phi0,self.p0)
        print(f"L2 norm on Phi0 and reference, = {np.round(L2,9)}")

    def gtg_and_line_plots(self,sigma_gtg,A,sigma_s0,rowsum):
        sigma_gtg[sigma_gtg < 1e-15] = 0.0
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
        self.few_groups = len(self.custom_bounds) - 1
        fg_idx = np.zeros(self.few_groups + 1, dtype=int)
        for i, target_E in enumerate(self.custom_bounds):
            fg_idx[i] = np.argmin(np.abs(self.Evec - target_E))

        plot_density_Phi0 = np.zeros(self.few_groups)
        for i in range(few_grp_Phi0.size):
            stt, stp = fg_idx[i], fg_idx[i+1]
            few_grp_Phi0[i] = simpson(self.Phi0[stt:stp], self.boundaries[stt:stp])
            few_grp_Phi2[i] = simpson(self.Phi2[stt:stp], self.boundaries[stt:stp])
            delta_u = abs(self.boundaries[stp-1] - self.boundaries[stt])
            plot_density_Phi0[i] = few_grp_Phi0[i] / delta_u

        # plotting
        index = 1
        Phi_plot = np.zeros_like(self.phi0)
        for i in range(self.phi0.size):
            # If we cross the index threshold into the next custom group, increment index
            if i >= fg_idx[index+1] and index < self.few_groups - 1:
                index += 1
            Phi_plot[i] = plot_density_Phi0[index]

        plt.figure()
        plt.plot(self.Eplot,self.normalize(Phi_plot),label=r"Few Group $\Phi_0 (E)$")
        for arg in self.custom_bounds: plt.axvline(arg, color='gold',linestyle='dashed',alpha=.5)
        plt.title("Few Group Scalar Flux")
        plt.ylabel(r"$\Phi_0$")
        plt.xlabel("Energy (ev)")
        plt.xscale("log")
        plt.legend()
        plt.grid(which='Both')
        plt.savefig(f"{self.chart_dir}fg_Phi0.png")

        return few_grp_Phi0, few_grp_Phi2

    def plot_few_grp_sp3_eqns(self,phi0_fg):
        # plot and compare
        Efg_desc = self.few_group_edges()
        Efg = Efg_desc[::-1]
        du_coarse = np.log(Efg_desc[:-1] / Efg_desc[1:])

        def normalized_density(phi):
            phi = np.asarray(phi, dtype=float)
            total = np.sum(phi)
            if abs(total) > 1e-300:
                phi = phi / total
            return phi / du_coarse

        def density(phi):
            return np.asarray(phi, dtype=float) / du_coarse

        phi0_new_dens = normalized_density(self.phi0_sp3_new)
        phi0_conv_dens = normalized_density(self.phi0_sp3_conv)
        phi2_new_dens = density(self.phi2_sp3_new)
        phi2_conv_dens = density(self.phi2_sp3_conv)

        phi0_new_plot = np.append(phi0_new_dens[::-1], phi0_new_dens[::-1][-1])
        phi0_conv_plot = np.append(phi0_conv_dens[::-1], phi0_conv_dens[::-1][-1])
        phi2_new_plot = np.append(phi2_new_dens[::-1], phi2_new_dens[::-1][-1])
        phi2_conv_plot = np.append(phi2_conv_dens[::-1], phi2_conv_dens[::-1][-1])
        phi0_fg_plot = None
        if phi0_fg is not None and len(phi0_fg) == self.few_groups:
            phi0_fg_dens = normalized_density(phi0_fg)
            phi0_fg_plot = np.append(phi0_fg_dens[::-1], phi0_fg_dens[::-1][-1])

        plt.figure(figsize=(8,6))
        #plt.step(Efg[:-1], self.phi0_sp3_new, where='post', label=fr'$\phi_0^{{new}}$, $B^2 =$ {self.B2_new:5g}')
        #plt.step(Efg[:-1], self.phi0_sp3_conv, where='post', label=fr'$\phi_0^{{conv}}$, $B^2 =$ {self.B2_conv:5g}')
        plt.step(Efg, phi0_new_plot, where='post', label=fr'$\phi_0^{{new}}$, $B^2 =$ {self.B2_new:5g}')
        plt.step(Efg, phi0_conv_plot, where='post', label=fr'$\phi_0^{{conv}}$, $B^2 =$ {self.B2_conv:5g}')
        if phi0_fg_plot is not None:
            plt.step(Efg, phi0_fg_plot, where='post', label=r'$\phi_0^{FG}$', alpha=.5, linestyle='dashed')
        #plt.step(Efg, phi0_fg/np.sum(phi0_fg), where='post', label=r'$\phi_0^{{FG}}$, $B^2 = 0.0$', alpha = .5, linestyle='dashed')
        plt.title(f"Scalar Flux 0th Moment")
        plt.xlabel('Energy (eV)')
        plt.ylabel(r'$\phi_0 (E)$')
        plt.legend()
        plt.grid(True, which='both')
        plt.xscale('log')
        plt.savefig(f"{self.chart_dir}B2_eigen_phi0_comp_{self.boundaries.size - 1}.png")
        plt.clf()

        plt.figure(figsize=(8,6))
        plt.title(f"Scalar Flux 2nd Moment")
        plt.step(Efg, phi2_new_plot, where='post', label=r'$\phi_2^{new}$')
        plt.step(Efg, phi2_conv_plot, where='post', label=r'$\phi_2^{conv}$')
        #plt.step(Efg[:-1], self.phi2_sp3_new, where='post', label=r'$\phi_2^{new}$')
        #plt.step(Efg[:-1], self.phi2_sp3_conv, where='post', label=r'$\phi_2^{conv}$')
        plt.xlabel('Energy (eV)')
        plt.ylabel(r'$\phi_2 (E)$')
        plt.legend()
        plt.grid(True, which='both')
        plt.xscale('log')
        plt.savefig(f"{self.chart_dir}B2_eigen_phi2_comp_{self.boundaries.size - 1}.png")
        plt.clf()

        print(f"L2 Norm on Conventional and New SP3 Equations, phi0: {self.L2_norm(self.phi0_sp3_new, self.phi0_sp3_conv)}")
        print(f"L2 Norm on Conventional and New SP3 Equations, phi2: {self.L2_norm(self.phi2_sp3_new, self.phi2_sp3_conv)}")
        #print(f"Percent Error on Conventional and New SP3 Equations, B2: {100 * (np.abs(self.B2_new - self.B2_conv) / np.abs(self.B2_conv)):5g}%")

    def read_data(self):
        print("Reading Data From File...")
        df = pd.read_hdf(f"{self.save_dir}fluxes_{self.NH}.h5", key="df")
        self.phi0 = df['phi0'].to_numpy()
        self.phi2 = df['phi2'].to_numpy()
        self.Phi0 = df['Phi0'].to_numpy()
        self.Phi2 = df['Phi2'].to_numpy()
        self.p0   = df['phi_ref'].to_numpy()
        print("Phi Read!")

        # unnecessary to get few-group constants
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

    # ----- Physics Static Methods ----- #
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
    def diff_matrix(sigma_t,sigma_s1,B2): # sigma_s1 is a matrix
        """Traditional diffusion matrix calculation"""
        # gamma buckling correction factor
        def gamma_fn(B2,sigma_t):
            # takes in the individual xs!
            if B2 == 0.0: return 1

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
    def B2_eigenvalue_conv(Sig_t, Sig_s0, Sig_s2, chi, nuSigf, D0, D2, B2_guess = 1e-4,
                            max_iters = 1000, tol = 1e-8, omega = 1.2, boost = 1):
        # eigenvalue solve for critical buckling B2
        print("B2 Eigenvalue Conv")
        G = chi.size
        chi = chi/chi.sum()
        Phi0 = np.ones(G)
        Phi2 = np.zeros(G)
        B2 = B2_guess
        D0 *= boost
        D2 *= boost

        # Diagonal matrices for simpler math
        R0 = np.diag(Sig_t) - Sig_s0
        R2 = np.diag(Sig_t) - Sig_s2
        Fiss = np.outer(chi, nuSigf)
        #Sig_T = np.diag(Sig_t)
        #Sig_a = (Sig_T - Sig_s0)
        #Sig_r2 = (Sig_T - Sig_s2)
        #Fiss = np.outer(chi, nuSigf)

        stt = time.time()
        for k in range(max_iters):
            B2_old = B2
            Phi0_old = Phi0.copy()
            Phi2_old = Phi2.copy()
            Q_fiss = Fiss @ (Phi0_old - 2 * Phi2_old)

            if B2 >= 0:
                lhs0 = D0 * B2 + R0
                rhs0 = Q_fiss + 2 * R0 @ Phi2_old
                Phi0 = np.linalg.solve(lhs0, rhs0)

                lhs2 = D2 * B2 + R2 + 4 * R0
                rhs2 = 2 * R0 @ Phi0 - 2 * Q_fiss
                Phi2 = np.linalg.solve(lhs2, rhs2)
            else:
                lhs0 = R0
                rhs0 = Q_fiss + 2 * R0 @ Phi2_old - (D0 * B2) @ Phi0_old
                Phi0 = np.linalg.solve(lhs0, rhs0)

                lhs2 = R2 + 4 * R0
                rhs2 = 2 * R0 @ Phi0 - 2 * Q_fiss - (D2 * B2) @ Phi2_old
                Phi2 = np.linalg.solve(lhs2, rhs2)

            norm = np.sum(Phi0)
            #norm = np.sum(Phi0 - 2 * Phi2)
            #norm = np.sum(Phi0)
            Phi0 /= norm
            Phi2 /= norm

            # Top Eqn is: D0 * B2 * Phi0 = Q_fiss + 2 * R0 * Phi2 - R0 * Phi0
            RHS_sum = np.sum(Q_fiss + 2 * R0 @ Phi2 - R0 @ Phi0)
            LHS_sum = np.sum(D0 @ Phi0)

            #print("WARNING ONLY RETURNING THE FIRST ITERATION")
            #return B2, Phi0 - 2 * Phi2, Phi2
            B2_target = RHS_sum / LHS_sum
            B2 = B2_old + omega * (B2_target - B2_old)

            err = abs(B2 - B2_old) / abs(B2_old) if abs(B2_old) > 0 else abs(B2)
            if err < tol and k > 5:
                print(f"Converged B2_conv: {B2:5e} in {k+1} iters")
                return B2, Phi0 - 2 * Phi2, Phi2

            if (k % 25 == 0): print(f"Iter {k+1}: B2 = {B2:5g}, Res = {err:5g}")

        print("Warning: B2_conv did not converge!")
        return B2, Phi0 - 2 * Phi2, Phi2

    @staticmethod
    def B2_eigenvalue_new(D0, D2, Sig_t_0, Sig_t_2, Sig_s0_0, Sig_s0_2, Sig_s2_2, chi, nuSigf0, nuSigf2, B2_guess=-1e-4, max_iters=5000, tol=1e-8, omega=1.2, boost=1):
        print("--- B2 Eigenvalue New ---")
        G = chi.size
        chi = chi / np.sum(chi)
        D0 = D0 * boost
        D2 = D2 * boost

        Phi0 = np.ones(G) / G
        Phi2 = np.zeros(G)
        B2 = B2_guess

        # Pre-calculate Removal matrices
        R0     = np.diag(Sig_t_0) - Sig_s0_0
        R2_new = np.diag(Sig_t_2) - Sig_s0_2  # Coupling transport term
        R2_bot = np.diag(Sig_t_2) - Sig_s2_2  # Bottom transport term

        Fiss0 = np.outer(chi, nuSigf0)
        Fiss2 = np.outer(chi, nuSigf2)

        for k in range(max_iters):
            B2_old = B2
            Phi0_old = Phi0.copy()
            Phi2_old = Phi2.copy()

            Q_fiss = Fiss0 @ Phi0_old - 2 * Fiss2 @ Phi2_old
            if B2 >= 0:
                lhs0 = D0 * B2 + R0
                rhs0 = Q_fiss + 2 * R2_new @ Phi2_old
                Phi0 = np.linalg.solve(lhs0, rhs0)

                lhs2 = D2 * B2 + R2_bot + 4 * R2_new
                rhs2 = 2 * R0 @ Phi0 - 2 * Q_fiss
                Phi2 = np.linalg.solve(lhs2, rhs2)
            else:
                lhs0 = R0
                rhs0 = Q_fiss + 2 * R2_new @ Phi2_old - (D0 * B2) @ Phi0_old
                Phi0 = np.linalg.solve(lhs0, rhs0)

                lhs2 = R2_bot + 4 * R2_new
                rhs2 = 2 * R0 @ Phi0 - 2 * Q_fiss - (D2 * B2) @ Phi2_old
                Phi2 = np.linalg.solve(lhs2, rhs2)

            norm = np.sum(Phi0 - 2 * Phi2)
            Phi0 /= norm
            Phi2 /= norm

            RHS_sum = np.sum(Q_fiss + 2 * R2_new @ Phi2 - R0 @ Phi0)
            LHS_sum = np.sum(D0 @ Phi0)

            #print("WARNING ONLY RETURNING THE FIRST ITERATION")
            #return B2, Phi0 - 2 * Phi2, Phi2
            B2_target = RHS_sum / LHS_sum
            B2 = B2_old + omega * (B2_target - B2_old)

            err = abs(B2 - B2_old) / abs(B2_old) if abs(B2_old) > 0 else abs(B2)
            if err < tol and k > 5:
                print(f"Converged B2_new: {B2:5e} in {k+1} iters")
                return B2, Phi0 - 2 * Phi2, Phi2

            if (k % 25 == 0): print(f"Iter {k+1}: B2 = {B2:5g}, Res = {err:5g}")

        print("Warning: B2_new did not converge!")
        return B2, Phi0 - 2 * Phi2, Phi2

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
    def get_percent_diff(M1,M2,index):
        """Get percent difference is index (i,i) in a matrix"""
        assert M1.shape == M2.shape
        if M1.ndim == 1: return 100 * (np.abs(M1[index] - M2[index])
                                        / (np.abs(M1[index])))
        else: return 100 * (np.abs(M1[index,index] - M2[index,index])
                    / (np.abs(M1[index,index])))

def parse_args():
    parser = argparse.ArgumentParser(description="Run SP3 few-group calculations for the CE buckled transport comparison.")
    parser.add_argument("--fixed-b2", type=float, default=None, help="Run one fixed-B2 comparison case instead of the legacy critical search.")
    parser.add_argument("--outdir", default="results/sp3_b2=0.0", help="Directory for fixed-B2 SP3 outputs.")
    parser.add_argument(
        "--ce-npz",
        "--reference-npz",
        dest="ce_npz",
        default=None,
        help=(
            "Optional buckled reference npz for spectra plots and error metrics. "
            "Accepts SN ce_buckled_transport_reference.npz or PN ce_buckled_pn_reference.npz."
        ),
    )
    parser.add_argument("--nbins", type=int, default=int(os.getenv("SP3_FINE_GROUPS", "5000")))
    parser.add_argument("--NH", type=float, default=float(os.getenv("SP3_NH", "0.10")))
    parser.add_argument("--xs-tol", type=float, default=float(os.getenv("SP3_XS_TOL", "5")))
    parser.add_argument("--group-structure", default=os.getenv("SP3_GROUP_STRUCTURE", os.getenv("CE_GROUP_STRUCTURE", "wims69.txt")))
    parser.add_argument("--kernel-quad", type=int, default=int(os.getenv("SP3_KERNEL_QUAD", os.getenv("CE_KERNEL_QUAD", "64"))))
    parser.add_argument("--no-upscatter", action="store_true")
    parser.add_argument(
        "--thermal-upscatter-cutoff-ev",
        type=float,
        default=float(os.getenv("SP3_THERMAL_UPSCATTER_CUTOFF_EV", os.getenv("CE_THERMAL_UPSCATTER_CUTOFF_EV", "4.0"))),
    )
    parser.add_argument(
        "--p0-only-upscatter",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("SP3_P0_ONLY_UPSCATTER", os.getenv("CE_P0_ONLY_UPSCATTER", "false")).lower() == "true",
    )
    parser.add_argument("--from-h5", action="store_true")
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument("--no-wims", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--fixed-iters", type=int, default=int(os.getenv("SP3_FIXED_ITERS", "10000")))
    parser.add_argument("--fixed-tol", type=float, default=float(os.getenv("SP3_FIXED_TOL", "1e-10")))
    parser.add_argument(
        "--plot-b2-zero",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("SP3_PLOT_B2_ZERO", "true").lower() == "true",
        help="Include the collapsed fine-grid B2=0 reference curve on the eigen_plot. Default: true.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    process = psutil.Process(os.getpid())
    stt = time.time()
    b2 = 0.0 if args.fixed_b2 is None else args.fixed_b2
    sp3 = Sp3(
        args.nbins,
        b2,
        args.NH,
        args.from_h5,
        args.adaptive,
        args.xs_tol,
        wims=not args.no_wims,
        group_structure=args.group_structure,
        p0_only_upscatter=args.p0_only_upscatter,
        no_upscatter=args.no_upscatter,
        thermal_upscatter_cutoff_ev=args.thermal_upscatter_cutoff_ev,
        kernel_quad=args.kernel_quad,
    )

    if args.fixed_b2 is None:
        sp3.run(args.verbose, plot_b2_zero=args.plot_b2_zero)
    else:
        sp3.run_fixed_b2(
            args.fixed_b2,
            args.outdir,
            ce_npz=args.ce_npz,
            verbose=args.verbose,
            fixed_iters=args.fixed_iters,
            fixed_tol=args.fixed_tol,
            plot_b2_zero=args.plot_b2_zero,
        )

    stp = time.time()
    print(f"Calculation Time = {np.round((stp - stt),6)}")
    mem = process.memory_info().rss / 1e9
    print(f"Memory used: {mem:.3f} GB")


if __name__ == "__main__":
    main()
