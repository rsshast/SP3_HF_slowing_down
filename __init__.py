import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import time
import os
import concurrent.futures
import h5py
from numba import njit, prange
from scipy.sparse import csr_matrix
from joblib import Parallel, delayed
from scipy.integrate import quad
from scipy.special import legendre, erf
import tensorly as tl
#import scikit_tt as sktt

from setup_sp3 import scratch_dir
