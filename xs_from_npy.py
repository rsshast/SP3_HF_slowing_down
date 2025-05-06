#from setup_sp3 import *
import numpy as np
import matplotlib.pyplot as plt
import time
# import data
data_dir = "data/openmc"
tol = 1e-10
B2 = .001
gridpoints = 10000
leg_order = 4

# SP3 xs's from openmc
t = time.time()
sigma_s = np.zeros((gridpoints,gridpoints,leg_order))
for i in range(leg_order): sigma_s[:,:,i] = np.load(f"{data_dir}/sigma_s{i}.npy")

#sigma_s0 = np.load(f"{data_dir}/sigma_s0.npy"),
#sigma_s1 = np.load(f"{data_dir}/sigma_s1.npy"),
#sigma_s2 = np.load(f"{data_dir}/sigma_s2.npy"),
#sigma_s3 = np.load(f"{data_dir}/sigma_s3.npy"),
sigma_t  = np.load(f"{data_dir}/total_xs.npy")
chi      = np.load(f"{data_dir}/chi.npy")
groups   = chi[:,0]
E        = np.flip(np.exp(groups))
print(f"Data Load Time = {np.round(time.time() - t, 5)}")
#chi      = chi[:,1]
#chi     /= np.trapz(chi,E)

def _Ln(n, xs_t, xs_s): return (2 * n + 1) * (np.diag(xs_t) - xs_s)

def plot_flux(phi,n):
    plt.clf()
    plt.figure(figsize=(8,6))
    plt.plot(E,np.abs(phi),label=f'$\\phi_{n}$')
    plt.title(f"$\\phi_{n}(E)$")
    plt.xlabel('Energy (eV)')
    plt.ylabel(f'$\\phi_{n}$')
    plt.xscale('log')
    if n == 2: plt.yscale('log')
    plt.grid(True, which='both')
    plt.legend()
    plt.savefig(f"phi{n}.png")
#    plt.show()

t = time.time()
L0 = _Ln(0,sigma_t,sigma_s0)
L1 = _Ln(1,sigma_t,sigma_s1)
L2 = _Ln(2,sigma_t,sigma_s2)
L3 = _Ln(3,sigma_t,sigma_s3)
print(f"Operator Calculation Time = {np.round(time.time() - t, 5)}")

# linalg for phi0 and phi2
phi0 = np.zeros_like(groups)
phi2 = np.zeros_like(phi0)

print("Solving phi0")
t = time.time()
I = np.eye(groups.size)
B4 = B2 ** 2 * I
LHS = (9 * B4 + B2 * (L3 @ L2 + (9 * L1 + 4 * L3) * L0) + L3 @ L2 @ L1 @ L0)
RHS = ((L3 @ L2 @ L1 + B2 * (9 * L1 + 4 * L3)) @ groups)
phi0 = np.linalg.solve(LHS,RHS) 
plot_flux(phi0, 0)
print(f"Phi0 Calc Time = {np.round(time.time() - t, 5)}")

print("Solving phi2")
t = time.time()
LHS = L3 @ L2 
RHS = (-9 * B2 * phi0 + (9 * L1 + 4 * L3) @ (L0 @ phi0 - groups)) * .5 
phi2 = np.linalg.solve(LHS,RHS) 
plot_flux(phi2, 2)
print(f"Phi2 Calc Time = {np.round(time.time() - t, 5)}")



"""
def plotting():
    # sigma_t
    plt.figure(figsize=(8,6))
    plt.plot(E,sigma_t, label = "$\\Sigma_t$")
    plt.xscale('log')
    plt.yscale('log')
    plt.title("$\\Sigma_t$ vs. E for Homogenous Mixture from OpenMC")
    plt.xlabel("Energy (eV)")
    plt.ylabel("\\Sigma_t(E): NH = 5, NU = 1")
    plt.legend()
    plt.grid(which = 'both')
    plt.show()
    
    plt.figure(figsize=(8, 6))
    plt.imshow(sigma_s0, aspect='auto', cmap='viridis', origin='upper')
    plt.colorbar(label="$\\Sigma_{s0}$")
    plt.xlabel("g'")
    plt.ylabel("g")
    plt.title("$\\Sigma_{s0}$")
    plt.show()
    
    plt.figure(figsize=(8, 6))
    plt.imshow(sigma_s1, aspect='auto', cmap='viridis', origin='upper')
    plt.colorbar(label="$\\Sigma_{s1}$")
    plt.xlabel("g'")
    plt.ylabel("g")
    plt.title("$\\Sigma_{s1}$")
    plt.show()
    
    plt.figure(figsize=(8, 6))
    plt.imshow(sigma_s2, aspect='auto', cmap='viridis', origin='upper')
    plt.colorbar(label="$\\Sigma_{s2}$")
    plt.xlabel("g")
    plt.ylabel("g'")
    plt.title("$\\Sigma_{s2}$")
    plt.show()
    
    plt.figure(figsize=(8, 6))
    plt.imshow(sigma_s3, aspect='auto', cmap='viridis', origin='upper')
    plt.colorbar(label="$\\Sigma_{s3}$")
    plt.xlabel("g'")
    plt.ylabel("g")
    plt.title("$\\Sigma_{s3}$")
    plt.show()

    L0[L0 != 0] = 1
    plt.figure(figsize=(8, 6))
    plt.imshow(L0, aspect='auto', cmap='plasma', origin='upper')
    plt.title("L0 Structure")
    plt.xlabel("g'")
    plt.ylabel("g")
    plt.colorbar(label="Memory")
    plt.savefig("L0.png")

"""
