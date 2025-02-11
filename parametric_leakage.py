import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from setup_sp3 import B2
from run_sp3 import groups

# init and define comparison flux
df = pd.read_hdf(f"results/h5s/fluxes_{np.round(B2[11],5)}.h5", key="df")
df = df.to_numpy()

phi0_comp, phi2_comp = df[:,0], df[:,1]
phi0 = np.zeros((phi0_comp.size,B2.size))
phi2 = np.zeros_like(phi0)

for i in range(B2.size):
    # read in the data
    df = pd.read_hdf(f"results/h5s/fluxes_{np.round(B2[i],5)}.h5", key="df")
    df = df.to_numpy()
    phi0[:,i], phi2[:,i] = df[:, 0], df[:, 1]

# compare the different leakages against B2 = 0
for i in range(B2.size):
    L2_norm_0 = np.linalg.norm(phi0_comp - phi0[:,i], ord = 2)
    L2_norm_2 = np.linalg.norm(phi2_comp - phi2[:,i], ord = 2)
    print(f"L2 norm 0 for B2 = {np.round(B2[i],5)}: {L2_norm_0}")
    print(f"L2 norm 2 for B2 = {np.round(B2[i],5)}: {L2_norm_2}")

plt.figure()
plt.plot(groups,np.abs(phi2_comp),label = "abs phi2")
plt.xscale('log')
plt.yscale('log')
plt.title(r"ln($\phi_2$) vs ln(E)")
plt.grid()
plt.show()
