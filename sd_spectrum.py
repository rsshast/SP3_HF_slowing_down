import numpy as np
import pandas as pd
from scipy.integrate import quad
import time
import matplotlib.pyplot as plt

stt = time.time()

class Sp3:
    def __init__(self,data_dir, nbins, B2, NH):
        self.data_dir = data_dir
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
        self.phi0 = np.zeros((G))
        self.phi2 = np.zeros_like(self.phi0)
        self.L0 = np.zeros((G,G))
        self.L1 = np.zeros_like(self.L0)
        self.L2 = np.zeros_like(self.L0)
        self.L3 = np.zeros_like(self.L0)
    
    def get_data(self,data, gridpoints, N):
        data = data[(data[:, 0] <= self.E0) & (data[:, 0] >= self.Emin)]
        data[:, 0] = np.log(self.E0 / data[:, 0])
    
        # Sort rows by lethargy ascending (required by np.interp)
        sort_idx = np.argsort(data[:, 0])
        data = data[sort_idx]
    
        # Define new grid
        new_grid = np.linspace(np.log(self.E0/self.E0), np.log(self.E0 / self.Emin), gridpoints)
        out = np.empty((new_grid.size, data.shape[1]), dtype=float)
        out[:, 0] = new_grid
    
        # Interpolate remaining columns onto the lethargy grid
        xp = data[:, 0]
        for i in range(1, data.shape[1]):
            out[:, i] = np.interp(new_grid, xp, data[:, i])
        return out
    
    def group_bound(self, A, g, lga): return (self.boundaries.size if A == 1
                else np.searchsorted(self.boundaries, self.boundaries[g] + lga))
    
    @staticmethod
    def alpha_fn(A): return ((A - 1.0)/(A + 1.0)) ** 2
    
    @staticmethod
    def lga_fn(alpha): return -np.log(alpha) if alpha != 0 else np.inf
    
    def gmax_vec_fn(self, A, lga):
        gmax_vec = np.zeros_like(self.boundaries, dtype = int)
        for g in range(self.boundaries.size): gmax_vec[g] = self.group_bound(A,g,lga)
        return gmax_vec
    
    def plot_sig_sn_gtg(self, sig_sn_gtg, sig_s0):
        plt.figure(figsize=(8,6))
        plt.plot(self.Evec,sig_s0, label = r"$\Sigma_{s0}$")
        Evec = self.Evec[:-1]
        rowsum = np.zeros(Evec.size)
        for i in range(sig_sn_gtg.shape[1]): rowsum[i] = np.sum(sig_sn_gtg[:,i,:])
                
        plt.plot(Evec,rowsum, label = r"$\Sigma_{gtg}$")
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.grid(which="Both")
        plt.show()
    
    def plot_fluxes(self):
        # plot phi0 and phi2
        plt.figure()
        E = self.Evec[:-1]
        plt.plot(E,self.phi0,label=r'$\phi_0$')
        plt.title(r'$\phi_0(E)$')
        plt.xlabel('E')
        plt.ylabel(r'$\phi_0$')
        plt.xscale('log')
        plt.grid(True, which='both')
        plt.legend()
        plt.show()
        plt.savefig(f'phi0_{self.NH}.png')
        #plt.savefig(f'results/charts/phi0_{self.NH}.png')
        plt.clf()

        plt.figure()
        plt.plot(E,self.phi2,label=r'$\phi_2$')
        plt.title(r'$\phi_2(E)$')
        plt.xlabel('E')
        plt.ylabel(r'$\phi_2$')
        plt.xscale('log')
        plt.grid(True, which='both')
        plt.legend()
        plt.show()
        plt.savefig(f'phi2_{self.NH}.png')
        #plt.savefig(f'results/charts/phi2_{self.NH}_lin.png')
        plt.clf()
        
    @staticmethod
    def sgg_val(lethargy_grid, A, x1, x2, y1, y2, l, 
                       alpha, lga, du):
        # set bounds
        def a(x): return max(y1, x)
        def b(x): return min(y2, x + lga)
        c = max(x1, y1 - lga)
    
        if (y1 < x1) or (c >= x2): return 0.0
    
        # integrate orders 0 -> 3    
        def integrand(x, A=A, l=l):
            if l == 0: return np.exp(-(a(x) - x)) - np.exp(-(b(x) - x))
            
            elif l == 1:
                return ((A+1)/3 * (np.exp(1.5*(x - a(x))) - np.exp(1.5*(x-b(x))))
                     - (A-1) * (np.exp(.5*(x-a(x))) - np.exp(.5*(x-b(x)))))
            
            elif l == 2:
                # 1/16
                return (.0625) * (
                    3*(A+1)**2 * (np.exp(2*(x - a(x))) - np.exp(2*(x - b(x))))
                    - 4*(3*A*A - 1) * (np.exp(x - a(x)) - np.exp(x - b(x)))
                    + 6*(A-1)**2 * (b(x) - a(x)))
            elif l == 3:
                return (.125) * (
                    (A + 1)**3 * (np.exp(2.5*(x - a(x))) - np.exp(2.5*(x - b(x))))                 
                    - (A + 1) * (5*A*A - 1) * (np.exp(1.5*(x - a(x))) - np.exp(1.5*(x - b(x))))     
                    + 3*(A - 1) * (5*A*A - 1) * (np.exp(0.5*(x - a(x))) - np.exp(0.5*(x - b(x)))) 
                    + 5*(A - 1)**3 * (np.exp(-0.5*(x - a(x))) - np.exp(-0.5*(x - b(x)))))
    
            else: raise ValueError("Order Not Supported")
            #return I 
            
        numerator, _ = quad(integrand, c, x2, limit=100, epsabs=1e-8, epsrel=1e-6)
        sig_sgg = numerator / ((1.0 - alpha) * du)
        return sig_sgg
    
    def gen_sig_sn_gtg(self,sig_s0,A,alpha,lga):
        # Output matrix
        sig_sn_gtg = np.zeros((self.leg_order, self.nbins-1, self.nbins-1))
        G = self.boundaries.size
        du = self.boundaries[1] - self.boundaries[0]
        gmax_vec = self.gmax_vec_fn(A, lga)
        
        for l in range(self.leg_order):
            for gp in range(self.nbins-1):  
                x1, x2 = self.boundaries[gp], self.boundaries[gp+1]
                gmax = gmax_vec[gp]
                for g in range(gp, min(gmax,G-1)):  
                    y1, y2 = self.boundaries[g], self.boundaries[g+1]
                    sig_sn_gtg[l, gp, g] = sig_s0[gp] * self.sgg_val(self.boundaries, A, 
                                                                 x1, x2, y1, y2, 
                                                                 l, alpha, lga, du)
        
                    if np.abs(sig_sn_gtg[l, gp, g]) <= self.tol: break  
        
        return sig_sn_gtg
    
    @staticmethod
    def _Ln(l, sigma_t, sigma_gtg): return (2 * l + 1) * (np.diag(sigma_t[:-1]) - sigma_gtg)
    
    def calc_Ln(self,A,sigma_t, sigma_s):
        alpha = self.alpha_fn(A)
        # l x g-1 x g-1 matrix
        sigma_gtg = self.gen_sig_sn_gtg(sigma_s, A, alpha, self.lga_fn(alpha))
        self.plot_sig_sn_gtg(sigma_gtg, sigma_s)
        self.L0 = self._Ln(0, sigma_t, sigma_gtg[0,:,:])
        self.L1 = self._Ln(1, sigma_t, sigma_gtg[1,:,:])
        self.L2 = self._Ln(2, sigma_t, sigma_gtg[2,:,:])
        self.L3 = self._Ln(3, sigma_t, sigma_gtg[3,:,:])

    def calc_phi_np(self):
        B4 = self.B2 * self.B2
        LHS = (9 * B4 + self.B2 * (self.L3 @ self.L2 + (9 * self.L1 + 4 * self.L3) @ self.L0)
                + self.L3 @ self.L2 @ self.L1 @ self.L0)
        RHS = (self.L3 @ self.L2 @ self.L1 + self.B2 * (9 * self.L1 + 4 * self.L3)) @ self.chi[:-1]
        self.phi0 = np.linalg.solve(LHS,RHS)

        LHS = self.L3 @ self.L2
        RHS = .5 * (-9 * self.B2 * self.phi0 + (9 * self.L1 + 4 * self.L3)
                @ (self.L0 @ self.phi0 - self.chi[:-1]))
        self.phi2 = np.linalg.solve(LHS,RHS)
    
    def run(self):
        print(r'Build $\Sigma_{gtg}$ and $\mathcal{L_n}$')
        self.calc_Ln(self.AU, self.sig_t_U, self.sig_s0_U)
        print(r'Calc $\phi (E)$')
        self.calc_phi_np()
        self.plot_fluxes()

#######################Constants #########################################
NH = 5
data_dir = './'
B2 = .01
nbins = 7500

# init class
sp3 = Sp3(data_dir,nbins,B2,NH)
sp3.run()
stp = time.time()
print(f"T = {np.round((stp - stt),6)}")
