import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

from dymad.io import DataInterface

config_path = "noise_wf.yaml"
dat = np.load("data/lti_denoise.npz")
ttrn = dat["t"]
Xtrn = dat["x"]

trn_sgf = {
    "type": "denoise",
    "method": 'savgol',
    "window_length": 15,
    "polyorder": 5
    }

di = DataInterface(config_path=config_path, config_mod={"transform_x": [trn_sgf]})
Ztrn = di.encode(Xtrn)
Zref = savgol_filter(Xtrn, window_length=trn_sgf["window_length"], polyorder=trn_sgf["polyorder"], axis=1)

f = plt.figure()
plt.plot(ttrn[0], Xtrn[0,:,0], 'r:')
plt.plot(ttrn[0], Ztrn[0,:,0], 'b-')
plt.plot(ttrn[0], Zref[0,:,0], 'g--')

plt.show()