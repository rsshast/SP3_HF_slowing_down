from Sp3 import Sp3
from setup_sp3 import *

#get data
print("Reading Data...")
t_data = time.time()
chi = get_data(chi,gridpoints,NU)
H = get_data(H,gridpoints,NH)
H *= NH
XS38 = get_data(XS38,gridpoints,NU)
sigma_f = get_fission_data(XS38[:,0],sigma_f)

# init class
sp3 = Sp3(H, XS38, sigma_f, chi, B2, NH)
print(f"Class initialized. Groups = {gridpoints}")
#print("Plotting Sigma_t")
#sp3.plot_xs_t()

# run
st = time.time()
sp3.run(properties,from_h5)
et = time.time()

print(f'Computation time = {np.round(et - st,5)}s')
