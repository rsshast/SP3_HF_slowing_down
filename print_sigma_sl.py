import numpy as np
import h5py

"""
fname = "/scratch/bckiedro_root/bckiedro0/rsshast/Sp3/results/D_coef_5.h5"

with h5py.File(fname, "r") as f:
    # List all datasets in the file
    print("Datasets in file:", list(f.keys()))
    print()

    # Read and print each dataset
    D = f["D"][:]
    D00 = f["D00"][:]
    D02 = f["D02"][:]
    D20 = f["D20"][:]
    D22 = f["D22"][:]

    print(D,"\n")
    print(D00,"\n")
    print(D02,"\n")
    print(D20,"\n")
    print(D22,"\n")
"""
save_dir = "/scratch/bckiedro_root/bckiedro0/rsshast/Sp3/results/"
#A = 238  # or whatever isotope you're using
#fname = f"{save_dir}Sigma_sl_A{A}_5.h5"

#with h5py.File(fname, "r") as f:
#    print("Datasets in file:", list(f.keys()))
#    print()
#
#    for key in f.keys():
#        data = f[key][:]
#        print(f"{key}: shape = {data.shape}")
#        print(np.round(data, 5))
#        print()

A = 1  # or whatever isotope you're using
fname = f"{save_dir}Sigma_sl_A{A}_5.h5"

with h5py.File(fname, "r") as f:
    print("Datasets in file:", list(f.keys()))
    print()

    for key in f.keys():
        data = f[key][:]
        print(f"{key}: shape = {data.shape}")
        print(np.round(data, 8))
        print()
