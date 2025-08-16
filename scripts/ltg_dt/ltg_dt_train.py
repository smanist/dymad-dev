import logging
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.utils import dense_to_sparse

from dymad.models import DGLDM, DGKBF
from dymad.training import NODETrainer, LinearTrainer
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

adj = np.array([
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0]
])

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
    {"name" : "dldm", "model" : DGLDM, "trainer": NODETrainer,   "config": 'ltg_dldm.yaml'},
    {"name" : "dkbf", "model" : DGKBF, "trainer": NODETrainer,   "config": 'ltg_dkbf.yaml'},
    {"name" : "dkbl", "model" : DGKBF, "trainer": LinearTrainer, "config": 'ltg_dkbl.yaml'}
]

IDX = [1, 2]
labels = [cases[i]['name'] for i in IDX]

ifdat = 0
iftrn = 1
ifplt = 1
ifprd = 1

if ifdat:
    sampler = TrajectorySampler(f, g, config='ltg_data.yaml', config_mod=config_chr)
    ts, xs, us, ys = sampler.sample(t_grid, batch=B)
    # Pretending a 3-node graph
    np.savez_compressed(
        './data/ltg.npz',
        t=ts, x=np.concatenate([ys, ys, ys], axis=-1), u=np.concatenate([us, us, us], axis=-1),
        adj_mat=adj)

if iftrn:
    for _i in IDX:
        Model = cases[_i]['model']
        Trainer = cases[_i]['trainer']
        config_path = cases[_i]['config']

        setup_logging(config_path, mode='info', prefix='results')
        logging.info(f"Config: {config_path}")
        trainer = Trainer(config_path, Model)
        trainer.train()

if ifplt:
    npz_files = [f'results/ltg_{mdl}_summary.npz' for mdl in labels]
    npzs = plot_summary(npz_files, labels=labels, ifclose=False)

if ifprd:
    sampler = TrajectorySampler(f, g, config='ltg_data.yaml', config_mod=config_gau)
    edge_index = dense_to_sparse(torch.Tensor(adj))[0]

    ts, xs, us, ys = sampler.sample(t_grid, batch=1)
    x_data = np.concatenate([ys[0], ys[0], ys[0]], axis=-1)
    t_data = ts[0]
    u_data = np.concatenate([us[0], us[0], us[0]], axis=-1)

    res = [x_data]
    for i in IDX:
        MDL, mdl = cases[i]['model'], cases[i]['name']
        _, prd_func = load_model(MDL, f'ltg_{mdl}.pt', f'ltg_{mdl}.yaml')

        with torch.no_grad():
            pred = prd_func(x_data, u_data, t_data, ei=edge_index)
            res.append(pred)

    plot_trajectory(
        np.array(res), t_data, "LTG",
        us=u_data, labels=['Truth']+labels, ifclose=False)

plt.show()
