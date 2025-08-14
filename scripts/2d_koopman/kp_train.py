import logging
import matplotlib.pyplot as plt
import numpy as np
import torch

from dymad.models import LDM, KBF
from dymad.training import WeakFormTrainer, NODETrainer, LinearTrainer
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
    "activation" : "prelu",
    "weight_init" : "xavier_uniform"}
mdl_ld = {
    "name": "kp_model",
    "encoder_layers": 0,
    "processor_layers": 2,
    "decoder_layers": 0,
    "latent_dimension": 32,
    "activation": "prelu",
    "weight_init": "xavier_uniform"}
mdl_kl = {
    "name" : 'kp_model',
    "encoder_layers" : 1,
    "decoder_layers" : 1,
    "latent_dimension" : 32,
    "koopman_dimension" : 8,
    "activation" : "tanh",
    "autoencoder_type" : "cat",
    "weight_init" : "xavier_uniform"}

trn_wf = {
    "n_epochs": 2000,
    "save_interval": 50,
    "load_checkpoint": False,
    "learning_rate": 5e-3,
    "decay_rate": 0.999,
    "reconstruction_weight": 1.0,
    "dynamics_weight": 1.0,
    "weak_form_params": {
        "N": 13,
        "dN": 2,
        "ordpol": 2,
        "ordint": 2}}
trn_nd = {
    "n_epochs": 2000,
    "save_interval": 20,
    "load_checkpoint": False,
    "learning_rate": 5e-3,
    "decay_rate": 0.999,
    "reconstruction_weight": 1.0,
    "dynamics_weight": 1.0,
    "sweep_lengths": [30, 50, 100, 200, 301],
    "sweep_epoch_step": 400,
    "ode_method": "dopri5",
    "rtol": 1e-7,
    "atol": 1e-9}
trn_ln = {
    "n_epochs": 1,
    "save_interval": 1,
    "load_checkpoint": False,
    "learning_rate": 5e-3,
    "decay_rate": 0.999,
    "reconstruction_weight": 1.0,
    "dynamics_weight": 1.0,
    "method": "truncated",
    "params": 8}
config_path = 'kp_model.yaml'

cfgs = [
    ('ldm_wf',   LDM, WeakFormTrainer, {"model": mdl_ld, "training" : trn_wf}),
    ('ldm_node', LDM, NODETrainer,     {"model": mdl_ld, "training" : trn_nd}),
    ('kbf_wf',   KBF, WeakFormTrainer, {"model": mdl_kb, "training" : trn_wf}),
    ('kbf_node', KBF, NODETrainer,     {"model": mdl_kb, "training" : trn_nd}),
    ('kbf_ln',   KBF, LinearTrainer,   {"model": mdl_kl, "training" : trn_ln}),
    ]

# IDX = [0, 1]
# IDX = [2, 3]
IDX = [4]
labels = [cfgs[i][0] for i in IDX]

ifdat = 0
iftrn = 1
ifplt = 0
ifprd = 1

if ifdat:
    sampler = TrajectorySampler(f, config='kp_data.yaml')
    ts, xs, ys = sampler.sample(t_grid, batch=B, save='./data/kp.npz')

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    for i in range(B):
        ax.plot(xs[i, :, 0], xs[i, :, 1], alpha=0.5)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title('2D Trajectories')
    plt.tight_layout()

    fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    for i in range(B):
        axs[0].plot(ts[i], xs[i, :, 0], alpha=0.5)
        axs[1].plot(ts[i], xs[i, :, 1], alpha=0.5)
    axs[0].set_ylabel('x1')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('x2')
    plt.tight_layout()

if iftrn:
    for i in IDX:
        mdl, MDL, Trainer, opt = cfgs[i]
        opt["model"]["name"] = f"kp_{mdl}"
        setup_logging(config_path, mode='info', prefix='results')
        logging.info(f"Config: {config_path}")
        trainer = Trainer(config_path, MDL, config_mod=opt)
        trainer.train()

if ifplt:
    npz_files = [f'results/kp_{l}_summary.npz' for l in labels]
    npzs = plot_summary(npz_files, labels=labels, ifclose=False)

    for lbl, npz in zip(labels, npzs):
        print(f"Epoch time {lbl}: {npz['avg_epoch_time']}")

if ifprd:
    sampler = TrajectorySampler(f, config='kp_data.yaml')
    ts, xs, ys = sampler.sample(t_grid, batch=1)
    x_data = xs[0]
    t_data = ts[0]

    res = [x_data]
    for i in IDX:
        mdl, MDL, Trainer, opt = cfgs[i]
        opt["model"]["name"] = f"kp_{mdl}"
        _, prd_func = load_model(MDL, f'kp_{mdl}.pt', f'kp_model.yaml', config_mod=opt)
        with torch.no_grad():
            pred = prd_func(x_data, t_data)
        res.append(pred)

    plot_trajectory(
        np.array(res), t_data, "KP",
        labels=['Truth'] + labels, ifclose=False)

plt.show()