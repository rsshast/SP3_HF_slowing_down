import numpy as np
import time
from Sp3_method import Sp3
from setup_sp3 import *

#get data
print("Reading Data...")
t_data=time.time()
chi = get_data(chi,gridpoints,NU)
H = get_data(H,gridpoints,NH)
XS38 = get_data(XS38,gridpoints,NU)
sigma_f = get_data(sigma_f,gridpoints,NU)
print(f"Data Read, time = {np.round(time.time() - t_data,5)}s")
print(f"Groups = {gridpoints}")

# init class
sp3 = Sp3(H,XS38,sigma_f,chi,B2)
print("plotting sigma_t")
#sp3.plot_xs_t()

# run
st = time.time()
sp3.run()
et = time.time()

print(f'Computation time = {np.round(et - st,5)} s')

