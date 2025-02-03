import numpy as np
import time
from Sp3 import Sp3
from setup_sp3 import *

#get data
print("Reading Data...")
t_data=time.time()
chi = get_data(chi,gridpoints,NU)
H = get_data(H,gridpoints,NH)
XS38 = get_data(XS38,gridpoints,NU)
sigma_f = get_data(sigma_f,gridpoints,NU) #FIXME xs's are wrong here.
print(f"Data Read, time = {np.round(time.time() - t_data,5)}s")

# init class
sp3 = Sp3(H,XS38,sigma_f[:,1],chi,B2)
print(f"Class initialized. Groups = {gridpoints}")
print("plotting sigma_t")
sp3.plot_xs_t()
plt.show()

# run
st = time.time()
sp3.run()
et = time.time()

print(f'Computation time = {np.round(et - st,5)} s')

