import logging
import matplotlib.pyplot as plt
import numpy as np
import torch

from dymad.data import DynData
from dymad.models import LDM, KBF
from dymad.training import WeakFormTrainer, NODETrainer
from dymad.utils import load_model, plot_summary, plot_trajectory, setup_logging, TrajectorySampler

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

MDL, mdl = LDM, 'ldm'
# MDL, mdl = KBF, 'kbf'

ifdat = 0
iftrn = 0
ifplt = 1
ifprd = 1

if ifdat:
    sampler = TrajectorySampler(f, g, config='ltd_data.yaml', config_mod=config_chr)
    ts, xs, us, ys = sampler.sample(t_grid, batch=B)
    np.savez_compressed('./data/ltd.npz', t=ts, x=ys, u=us)

if iftrn:
    cases = [
        {"model" : LDM, "trainer": WeakFormTrainer, "config": 'ltd_ldm_wf.yaml'},
        {"model" : LDM, "trainer": NODETrainer,     "config": 'ltd_ldm_node.yaml'},
        {"model" : KBF, "trainer": WeakFormTrainer, "config": 'ltd_kbf_wf.yaml'},
        {"model" : KBF, "trainer": NODETrainer,     "config": 'ltd_kbf_node.yaml'}
    ]

    for _i in [0, 1, 2, 3]:
        Model = cases[_i]['model']
        Trainer = cases[_i]['trainer']
        config_path = cases[_i]['config']

        setup_logging(config_path, mode='info', prefix='results')
        logging.info(f"Config: {config_path}")
        trainer = Trainer(config_path, Model)
        trainer.train()

if ifplt:
    npz_files = [f'results/ltd_{mdl}_node_summary.npz', f'results/ltd_{mdl}_wf_summary.npz']
    npzs = plot_summary(npz_files, labels = [f'{mdl}/NODE', f'{mdl}/WF'], ifclose=False)

    print("Epoch time NODE/WF:", npzs[0]['avg_epoch_time']/npzs[1]['avg_epoch_time'])

if ifprd:
    mdl_wf, prd_wf = load_model(MDL, f'ltd_{mdl}_wf.pt', f'ltd_{mdl}_wf.yaml')
    mdl_nd, prd_nd = load_model(MDL, f'ltd_{mdl}_node.pt', f'ltd_{mdl}_node.yaml')

    sampler = TrajectorySampler(f, g, config='ltd_data.yaml', config_mod=config_gau)

    ts, xs, us, ys = sampler.sample(t_grid, batch=1)
    x_data = xs[0]
    t_data = ts[0]
    u_data = us[0]

    with torch.no_grad():
        weak_pred = prd_wf(x_data, u_data, t_data[:-1])
        node_pred = prd_nd(x_data, u_data, t_data[:-1])

    plot_trajectory(
        np.array([x_data, weak_pred, node_pred]), t_data, "LTI", metadata={'n_state_features': 2},
        us=u_data, labels=['Truth', 'Weak Form', 'NODE'], ifclose=False)

plt.show()
