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
def f(t, x):
    return (x @ A.T)
g = lambda t, x: x

adj = np.array([
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0]
])

mdl_kb = {
    "name" : 'ltga_model',
    "encoder_layers" : 1,
    "decoder_layers" : 1,
    "latent_dimension" : 32,
    "koopman_dimension" : 2,
    "activation" : "none",
    "gcl" : "sage",
    "weight_init" : "xavier_uniform"}
mdl_ld = {
    "name": "ltga_model",
    "encoder_layers": 1,
    "processor_layers": 1,
    "decoder_layers": 1,
    "latent_dimension": 32,
    "activation": "none",
    "gcl" : "sage",
    "weight_init": "xavier_uniform"}

trn_wf = {
    "n_epochs": 500,
    "save_interval": 10,
    "load_checkpoint": False,
    "learning_rate": 1e-2,
    "decay_rate": 0.999,
    "reconstruction_weight": 1.0,
    "dynamics_weight": 1.0,
    "weak_form_params": {
        "N": 13,
        "dN": 2,
        "ordpol": 2,
        "ordint": 2}}
trn_nd = {
    "n_epochs": 500,
    "save_interval": 10,
    "load_checkpoint": False,
    "learning_rate": 1e-2,
    "decay_rate": 0.999,
    "reconstruction_weight": 1.0,
    "dynamics_weight": 1.0,
    "sweep_lengths": [50, 100, 200, 300, 501],
    "sweep_epoch_step": 100,
    "ode_method": "dopri5",
    "rtol": 1e-7,
    "atol": 1e-9
}
config_path = 'ltga_model.yaml'

cfgs = [
    ('ldm_wf',   GLDM, WeakFormTrainer, {"model": mdl_ld, "training" : trn_wf}),
    ('ldm_node', GLDM, NODETrainer,     {"model": mdl_ld, "training" : trn_nd}),
    ('kbf_wf',   GKBF, WeakFormTrainer, {"model": mdl_kb, "training" : trn_wf}),
    ('kbf_node', GKBF, NODETrainer,     {"model": mdl_kb, "training" : trn_nd}),
    ]

IDX = [0, 1]

ifdat = 0
iftrn = 1
ifplt = 1
ifprd = 1

if ifdat:
    sampler = TrajectorySampler(f, g, config='ltga_data.yaml')
    ts, xs, ys = sampler.sample(t_grid, batch=B)
    # Pretending a 3-node graph
    np.savez_compressed(
        './data/ltga.npz',
        t=ts, x=np.concatenate([ys, ys, ys], axis=-1),
        adj_mat=adj)

if iftrn:
    for i in IDX:
        mdl, MDL, Trainer, opt = cfgs[i]
        opt["model"]["name"] = f"ltga_{mdl}"
        setup_logging(config_path, mode='info', prefix='results')
        logging.info(f"Config: {config_path}")
        trainer = Trainer(config_path, MDL, config_mod=opt)
        trainer.train()

if ifplt:
    labels = [cfgs[i][0] for i in IDX]
    npz_files = [f'results/ltga_{l}_summary.npz' for l in labels]
    npzs = plot_summary(npz_files, labels=labels, ifclose=False)

    print(f"Epoch time {labels[0]}/{labels[1]}: {npzs[0]['avg_epoch_time']/npzs[1]['avg_epoch_time']}")

if ifprd:
    sampler = TrajectorySampler(f, config='ltga_data.yaml')
    ts, xs, ys = sampler.sample(t_grid, batch=1)
    x_data = np.concatenate([xs[0], xs[0], xs[0]], axis=-1)
    t_data = ts[0]
    edge_index = dense_to_sparse(torch.Tensor(adj))[0]

    res = [x_data]
    for i in IDX:
        mdl, MDL, Trainer, opt = cfgs[i]
        opt["model"]["name"] = f"ltga_{mdl}"
        _, prd_func = load_model(MDL, f'ltga_{mdl}.pt', f'ltga_model.yaml', config_mod=opt)
        with torch.no_grad():
            pred = prd_func(x_data, t_data, ei=edge_index)
        res.append(pred)

    labels = ['Truth'] + [cfgs[i][0] for i in IDX]
    plot_trajectory(
        np.array(res), t_data, "LTGA",
        labels=labels, ifclose=False)

plt.show()
