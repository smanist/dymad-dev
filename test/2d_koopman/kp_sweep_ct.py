import logging
import matplotlib.pyplot as plt
import numpy as np
import torch

from dymad.models import KBF
from dymad.training import NODETrainer
from dymad.utils import load_model, plot_summary, plot_trajectory, setup_logging, TrajectorySampler

B = 256
N = 301
t_grid = np.linspace(0, 6, N)

mu = -0.5
lm = -3
def f(t, x):
    _d = np.array([mu*x[0], lm*(x[1]-x[0]**2)])
    return _d

mdl_kb = {
    "name" : 'kp_model',
    "encoder_layers" : 2,
    "decoder_layers" : 2,
    "latent_dimension" : 32,
    "koopman_dimension" : 4,
    "autoencoder_type": "cat",
    "activation" : "prelu",
    "weight_init" : "xavier_uniform"}

trn_ref = {
    "n_epochs": 400,
    "save_interval": 20,
    "load_checkpoint": False,
    "learning_rate": 5e-3,
    "decay_rate": 0.999,
    "reconstruction_weight": 1.0,
    "dynamics_weight": 1.0,
    "ode_method": "dopri5",
    "rtol": 1e-7,
    "atol": 1e-9,
    "sweep_epoch_step": 100,
    "sweep_lengths": [10, 20, 30, 50],
}
# Simple sweep
trn_nd1 = {}
trn_nd1.update(trn_ref)
# Sweep with possible early stop by tolerance
trn_nd2 = {
    "sweep_tols": [0.4, 0.1],
}
trn_nd2.update(trn_ref)
# Sweep with cycling through all lengths for all tolerances
trn_nd3 = {
    "sweep_tols": [0.4, 0.1],
    "sweep_mode": "full",
}
trn_nd3.update(trn_ref)
# Simple sweep but use all trajectories
trn_nd4 = {
    "chop_mode": "unfold",
    "chop_step": 0.5,
}
trn_nd4.update(trn_ref)

trn_opts = [trn_nd1, trn_nd2, trn_nd3, trn_nd4]
config_path = 'kp_model.yaml'

IDX = [0, 1, 2, 3]

iftrn = 1
ifplt = 1
ifprd = 1

if iftrn:
    for i in IDX:
        opt = {"model": mdl_kb, "training": trn_opts[i]}
        opt["model"]["name"] = f"kp_nd{i+1}"
        setup_logging(config_path, mode='info', prefix='results')
        logging.info(f"Config: {config_path}")
        trainer = NODETrainer(config_path, KBF, config_mod=opt)
        trainer.train()

if ifplt:
    labels = [f"nd{i+1}" for i in IDX]
    npz_files = [f'results/kp_{l}_summary.npz' for l in labels]
    npzs = plot_summary(npz_files, labels=labels, ifscl=False, ifclose=False)

    for i in IDX:
        print(f"nd{i+1} Epoch time:", npzs[i]['avg_epoch_time'])

if ifprd:
    sampler = TrajectorySampler(f, config='kp_data.yaml')
    ts, xs, ys = sampler.sample(t_grid, batch=1)
    x_data = xs[0]
    t_data = ts[0]

    res = [x_data]
    for i in IDX:
        opt = {"model": mdl_kb, "training": trn_opts[i]}
        opt["model"]["name"] = f"kp_nd{i+1}"
        _, prd_func = load_model(KBF, f'kp_nd{i+1}.pt', f'kp_model.yaml', config_mod=opt)
        with torch.no_grad():
            pred = prd_func(x_data, t_data)
        res.append(pred)

    labels = ['Truth'] + [f"nd{i+1}" for i in IDX]
    plot_trajectory(
        np.array(res), t_data, "KP", metadata={'n_state_features': 2},
        labels=labels, ifclose=False)

plt.show()
