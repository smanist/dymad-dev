"""
The original script comes with the CYLINDER_ALL.mat data from DMDBook.

Modified to reduce data.
"""

import numpy as np
from scipy.io import loadmat
raw = loadmat('CYLINDER_ALL.mat')
dat = np.array([np.moveaxis(raw[_n].reshape(449,199,151),(0,1,2),(2,1,0))
                for _n in ['UALL','VALL','VORTALL']])
np.savez_compressed('raw.npz', vor=dat[2])
