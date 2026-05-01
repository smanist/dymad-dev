import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

from dymad.io import DataInterface
from dymad.numerics import denoise

config_path = "noise_wf.yaml"
dat = np.load("data/lti_denoise.npz")
ttrn = dat["t"]
Xtrn = dat["x"]

trn_sgf = {"type": "denoise", "method": "savgol", "window_length": 15, "polyorder": 5}
trn_gau = {
    "type": "denoise",
    "method": "kernel_smoothing",
    "kernel": "gaussian",
    "anchor_count": 64,
    "bandwidth_multiplier": 2.0,
}
trn_cpp = {
    "type": "denoise",
    "method": "kernel_smoothing",
    "kernel": "compact_polynomial",
    "anchor_count": 64,
    "bandwidth_multiplier": 2.0,
    "degree": 4.0,
}


def _kernel_reference(config):
    kwargs = {key: value for key, value in config.items() if key != "type"}
    return np.stack([denoise(trajectory, **kwargs) for trajectory in Xtrn], axis=0)


cases = [
    (
        "Savitzky-Golay",
        trn_sgf,
        savgol_filter(
            Xtrn, window_length=trn_sgf["window_length"], polyorder=trn_sgf["polyorder"], axis=1
        ),
    ),
    ("Gaussian", trn_gau, _kernel_reference(trn_gau)),
    ("Compact polynomial", trn_cpp, _kernel_reference(trn_cpp)),
]

f, ax = plt.subplots(nrows=2, ncols=len(cases), sharex=True, sharey="row")
for _j, (title, config, Zref) in enumerate(cases):
    di = DataInterface(config_path=config_path, config_mod={"transform_x": [config]})
    Ztrn = di.encode(Xtrn)
    for _i in range(2):
        ax[_i, _j].plot(ttrn[0], Xtrn[0, :, _i], "r:")
        ax[_i, _j].plot(ttrn[0], Ztrn[0, :, _i], "b-")
        ax[_i, _j].plot(ttrn[0], Zref[0, :, _i], "g--")
    ax[0, _j].set_title(title)

plt.show()
