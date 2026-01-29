import matplotlib.pyplot as plt
import numpy as np
import torch

from dymad.io import load_model
from dymad.models import LDM, KBF, LTI
from dymad.training import WeakFormTrainer, NODETrainer, LinearTrainer
from dymad.utils import plot_summary, plot_trajectory, TrajectorySampler

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

config_gau = {
    "control" : {
        "kind": "gaussian",
        "params": {
            "mean": 0.5,
            "std":  1.0,
            "t1":   4.0,
            "dt":   0.2,
            "mode": "zoh"}}}

cases = [
    {"name": "ldm_wf",   "model" : LDM, "trainer": WeakFormTrainer, "config": 'lti_ldm_wf.yaml'},
    {"name": "ldm_node", "model" : LDM, "trainer": NODETrainer,     "config": 'lti_ldm_node.yaml'},
    {"name": "kbf_wf",   "model" : KBF, "trainer": WeakFormTrainer, "config": 'lti_kbf_wf.yaml'},
    {"name": "kbf_node", "model" : KBF, "trainer": NODETrainer,     "config": 'lti_kbf_node.yaml'},
    {"name": "kbf_ln",   "model" : KBF, "trainer": LinearTrainer,   "config": 'lti_kbf_ln.yaml'},
    {"name": "lti_wf",   "model" : LTI, "trainer": WeakFormTrainer, "config": 'lti_lti_wf.yaml'},
    {"name": "lti_ln",   "model" : LTI, "trainer": LinearTrainer,   "config": 'lti_lti_ln.yaml'}
]
IDX = [0, 1, 2, 3, 4, 5, 6]
labels = [cases[i]['name'] for i in IDX]

ifdat = 0
iftrn = 1
ifplt = 1
ifprd = 1

if ifdat:
    sampler = TrajectorySampler(f, g, config='lti_data.yaml', config_mod=config_chr)
    ts, xs, us, ys = sampler.sample(t_grid, batch=B)
    np.savez_compressed('./data/lti.npz', t=ts, x=ys, u=us)

if iftrn:
    for _i in IDX:
        Model = cases[_i]['model']
        Trainer = cases[_i]['trainer']
        config_path = cases[_i]['config']

        trainer = Trainer(config_path, Model)
        trainer.train()

if ifplt:
    npz_files = [f'lti_{mdl}' for mdl in labels]
    npzs = plot_summary(npz_files, labels = labels, ifclose=False)
    for lbl, npz in zip(labels, npzs):
        print(f"Epoch time: {lbl} - {npz['avg_epoch_time']}")

if ifprd:
    sampler = TrajectorySampler(f, g, config='lti_data.yaml', config_mod=config_gau)

    ts, xs, us, ys = sampler.sample(t_grid, batch=1)
    x_data = xs[0]
    t_data = ts[0]
    u_data = us[0]

    res = [x_data]
    for _i in IDX:
        mdl, MDL = cases[_i]['name'], cases[_i]['model']
        _, prd_func = load_model(MDL, f'lti_{mdl}.pt')

        with torch.no_grad():
            _pred = prd_func(x_data, t_data, u=u_data)
        res.append(_pred)

    plot_trajectory(
        np.array(res), t_data, "LTI",
        us=u_data, labels=['Truth']+labels, ifclose=False)

plt.show()
