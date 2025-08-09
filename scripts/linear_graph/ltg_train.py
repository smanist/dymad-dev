import logging
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.utils import dense_to_sparse

from dymad.models import GLDM, GKBF
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

cfgs = [(GLDM, 'ldm'), (GKBF, 'kbf')]

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
    cases = [
        {"model" : GLDM, "trainer": WeakFormTrainer, "config": 'ltg_ldm_wf.yaml'},
        {"model" : GLDM, "trainer": NODETrainer,     "config": 'ltg_ldm_node.yaml'},
        {"model" : GKBF, "trainer": WeakFormTrainer, "config": 'ltg_kbf_wf.yaml'},
        {"model" : GKBF, "trainer": NODETrainer,     "config": 'ltg_kbf_node.yaml'}
    ]

    for _i in [0, 1]:
    # for _i in [2, 3]:
        Model = cases[_i]['model']
        Trainer = cases[_i]['trainer']
        config_path = cases[_i]['config']

        setup_logging(config_path, mode='info', prefix='results')
        logging.info(f"Config: {config_path}")
        trainer = Trainer(config_path, Model)
        trainer.train()

if ifplt:
    for cfg in [cfgs[0]]:
        MDL, mdl = cfg
        npz_files = [f'results/ltg_{mdl}_node_summary.npz', f'results/ltg_{mdl}_wf_summary.npz']
        npzs = plot_summary(npz_files, labels = [f'{mdl}/NODE', f'{mdl}/WF'], ifclose=False)

        print("Epoch time NODE/WF:", npzs[0]['avg_epoch_time']/npzs[1]['avg_epoch_time'])

if ifprd:
    sampler = TrajectorySampler(f, g, config='ltg_data.yaml', config_mod=config_gau)
    edge_index = dense_to_sparse(torch.Tensor(adj))[0]

    ts, xs, us, ys = sampler.sample(t_grid, batch=1)
    x_data = np.concatenate([ys[0], ys[0], ys[0]], axis=-1)
    t_data = ts[0]
    u_data = np.concatenate([us[0], us[0], us[0]], axis=-1)

    for cfg in [cfgs[0]]:
        MDL, mdl = cfg

        mdl_wf, prd_wf = load_model(MDL, f'ltg_{mdl}_wf.pt', f'ltg_{mdl}_wf.yaml')
        mdl_nd, prd_nd = load_model(MDL, f'ltg_{mdl}_node.pt', f'ltg_{mdl}_node.yaml')

        with torch.no_grad():
            weak_pred = prd_wf(x_data, u_data, t_data, ei=edge_index)
            node_pred = prd_nd(x_data, u_data, t_data, ei=edge_index)

        plot_trajectory(
            np.array([x_data, weak_pred, node_pred]), t_data, "LTG",
            us=u_data, labels=['Truth', 'Weak Form', 'NODE'], ifclose=False)

plt.show()
