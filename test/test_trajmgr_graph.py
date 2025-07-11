import copy
import numpy as np
import torch
from torch_geometric.utils import dense_to_sparse

from dymad.data import DynGeoData, make_transform, TrajectoryManagerGraph
from dymad.utils import TrajectorySampler

def check_data(out, ref, label=''):
    for _s, _t in zip(out, ref):
        assert np.allclose(_s.x, _t.x), f"{label} failed: {_s.x} != {_t.x}"
        assert np.allclose(_s.u, _t.u), f"{label} failed: {_s.u} != {_t.u}"
        assert np.allclose(_s.edge_index, _t.edge_index), f"{label} failed: {_s.edge_index} != {_t.edge_index}"
    print(f"{label} passed.")

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

# Pretending a 3-node graph
np.savez_compressed('lti.npz', t=ts, x=np.concatenate([ys, ys, ys], axis=-1), u=np.concatenate([us, us, us], axis=-1))

# --------------------
# Configure metadata
metadata = {
    "config" : {
        "data": {
            "path": "./lti.npz",
            "n_samples": 128,
            "n_steps": 501,
            "double_precision": False,
            "n_nodes": 3,},
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

adj = np.array([
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0]
])
edge_index, _ = dense_to_sparse(torch.Tensor(adj))

# --------------------
# First pass
tm = TrajectoryManagerGraph(metadata, adj=adj, device="cpu")
tm.process_all()

data = np.load('./lti.npz', allow_pickle=True)
xs = data['x']
ts = data['t']
us = data['u']

# ---
# Forward transform test
# ---
# Manually transform data
# Transform only one node, and then replicate it for all nodes
trnx = make_transform(metadata['config']['transform_x'])
trnu = make_transform(metadata['config']['transform_u'])

xtrn = [xs[_i][:,:2] for _i in tm.train_set_index]
xtst = [xs[_i][:,:2] for _i in tm.test_set_index]
trnx.fit(xtrn)
Xtst = trnx.transform(xtst)
Xtst = [np.concatenate([xt, xt, xt], axis=-1) for xt in Xtst]

utrn = [us[_i][:,0].reshape(-1,1) for _i in tm.train_set_index]
utst = [us[_i][:,0].reshape(-1,1) for _i in tm.test_set_index]
trnu.fit(utrn)
Utst = trnu.transform(utst)
Utst = [np.concatenate([ut, ut, ut], axis=-1) for ut in Utst]

Dtst = [DynGeoData(xt, ut[:,2:], edge_index) for xt, ut in zip(Xtst, Utst)]
check_data(Dtst, tm.test_set, label='Transform X and U')

# ---
# Inverse transform test
# ---
# Again manually inverse transform data
# Inverse transform only one node, and then replicate it for all nodes
Xrec = trnx.inverse_transform([_d.x[:,:6] for _d in Dtst])
Xrec = [np.concatenate([xr, xr, xr], axis=-1) for xr in Xrec]
Urec = trnu.inverse_transform([_d.u[:,0].reshape(-1,1) for _d in Dtst])
Urec = [np.concatenate([ur, ur, ur], axis=-1) for ur in Urec]

Drec = [DynGeoData(xr, ur, edge_index) for xr, ur in zip(Xrec, Urec)]
Xref = [xs[_i] for _i in tm.test_set_index]
Uref = [us[_i][:,2:] for _i in tm.test_set_index]
Dref = [DynGeoData(xr, ur, edge_index) for xr, ur in zip(Xref, Uref)]
check_data(Drec, Dref, label='Inverse Transform X and U')

# --------------------
# Second pass - reinitialize and reload
old_metadata = copy.deepcopy(tm.metadata)

new_tm = TrajectoryManagerGraph(old_metadata, adj=adj, device="cpu")
new_tm.process_all()

check_data(new_tm.train_set, tm.train_set, label="New Train")
check_data(new_tm.valid_set, tm.valid_set, label="New Valid")
check_data(new_tm.test_set,  tm.test_set,  label="New Test")

# --------------------
# Clean up
import os
if os.path.exists('./lti.npz'):
    os.remove('./lti.npz')
