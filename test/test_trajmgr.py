import copy
import numpy as np
import torch

def check_data(out, ref, label=''):
    for _s, _t in zip(out, ref):
        assert np.allclose(_s, _t), f"{label} failed: {_s} != {_t}"

from dymad.src.data import make_transform, TrajectoryManager
from dymad.src.utils import TrajectorySampler

# --------------------
# Data generation
B = 128
N = 501
t_grid = np.linspace(0, 5, N)

A = np.array([
            [0., 1.],
            [-1., -0.1]])
def f(t, x, u):
    return (x @ A.T) + u
g = lambda t, x, u: x

config_chr = {
    "control" : {
        "kind": "chirp",
        "params": {
            "t1": 4.0,
            "freq_range": (0.5, 2.0),
            "amp_range": (0.5, 1.0),
            "phase_range": (0.0, 360.0)}}}

sampler = TrajectorySampler(f, g, config='lti_data.yaml', config_mod=config_chr)
ts, xs, us, ys = sampler.sample(t_grid, batch=B)
np.savez_compressed('lti.npz', t=ts, x=ys, u=us)

# --------------------
# Configure metadata
metadata = {
    "config" : {
        "data": {
            "path": "./lti.npz",
            "n_samples": 128,
            "n_steps": 501,
            "double_precision": False},
        "transform_x": [
            {
            "type": "Scaler",
            "mode" : "01"},
            {
            "type": "delay",
            "delay": 2}],
        "transform_u": {
            "type": "Scaler",
            "mode" : "-11"},
        "model": {
            "type": "NN"},
}}

# --------------------
# First pass
tm = TrajectoryManager(metadata, device="cpu")
tm.process_all()

data = np.load('./lti.npz', allow_pickle=True)
xs = data['x']
ts = data['t']
us = data['u']

trnx = make_transform(metadata['config']['transform_x'])
trnu = make_transform(metadata['config']['transform_u'])

xtrn = [xs[_i] for _i in tm.train_set_index]
xtst = [xs[_i] for _i in tm.test_set_index]
trnx.fit(xtrn)
Xtst = trnx.transform(xtst)

utrn = [us[_i] for _i in tm.train_set_index]
utst = [us[_i] for _i in tm.test_set_index]
trnu.fit(utrn)
Utst = trnu.transform(utst)

Dtst = [np.hstack([xt, ut[2:]]) for xt, ut in zip(Xtst, Utst)]
check_data(Dtst, tm.test_set, label='Transform X and U')

Xrec = [_d[:,:6] for _d in Dtst]
tmp = trnx.inverse_transform(Xrec)
check_data(tmp, xtst, label='Inverse Transform X')

Urec = [_d[:,6:].reshape(-1,1) for _d in Dtst]
tmp = trnu.inverse_transform(Urec)
Uref = [us[_i][2:] for _i in tm.test_set_index]
check_data(tmp, Uref, label='Inverse Transform U')

# --------------------
# Second pass - reinitialize and reload
old_metadata = copy.deepcopy(tm.metadata)

new_tm = TrajectoryManager(old_metadata, device="cpu")
new_tm.process_all()

check_data(new_tm.train_set, tm.train_set, label="New Train")
check_data(new_tm.valid_set, tm.valid_set, label="New Valid")
check_data(new_tm.test_set,  tm.test_set,  label="New Test")

# --------------------
# Clean up
import os
if os.path.exists('./lti.npz'):
    os.remove('./lti.npz')
