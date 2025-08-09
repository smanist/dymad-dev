import logging
import matplotlib.pyplot as plt
import numpy as np
import torch

from dymad.models import KBF, DKBF
from dymad.training import WeakFormTrainer, NODETrainer
from dymad.utils import load_model, plot_summary, plot_trajectory, setup_logging, TrajectorySampler

mdl_kb = {
    "name" : 'vor_model',
    "encoder_layers" : 1,
    "decoder_layers" : 1,
    "latent_dimension" : 32,
    "koopman_dimension" : 8,
    "activation" : "none",
    "weight_init" : "xavier_uniform",
    "gain" : 0.01}

trn_wf = {
    "n_epochs": 400,
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
    "n_epochs": 500,
    "save_interval": 20,
    "load_checkpoint": False,
    "learning_rate": 5e-3,
    "decay_rate": 0.999,
    "reconstruction_weight": 1.0,
    "dynamics_weight": 1.0,
    # "sweep_lengths": [5, 10, 30, 50, 140],
    "sweep_lengths": [2, 3, 4, 6, 8],
    "sweep_epoch_step": 100,
    "chop_mode": "unfold",
    "chop_step": 1,
    "ode_method": "dopri5",
    "rtol": 1e-7,
    "atol": 1e-9}
trn_dt = {
    "n_epochs": 500,
    "save_interval": 20,
    "load_checkpoint": False,
    "learning_rate": 5e-3,
    "decay_rate": 0.999,
    "reconstruction_weight": 1.0,
    "dynamics_weight": 1.0,
    "sweep_lengths": [2, 3, 4, 6, 8],
    "sweep_epoch_step": 100,
    "chop_mode": "unfold",
    "chop_step": 1}
config_path = 'vor_model.yaml'

cfgs = [
    ('kbf_wf',   KBF,  WeakFormTrainer, {"model": mdl_kb, "training" : trn_wf}),
    ('kbf_node', KBF,  NODETrainer,     {"model": mdl_kb, "training" : trn_nd}),
    ('dkbf_nd',  DKBF, NODETrainer,     {"model": mdl_kb, "training" : trn_dt}),
    ]

IDX = [2]

ifdat = 0
iftrn = 1
ifplt = 0
ifprd = 1

if ifdat:
    dat = np.load('./data/raw.npz')['vor']
    Nt, Nx, Ny = dat.shape
    ts = np.arange(Nt)
    dt = 1.
    X = dat.reshape(Nt, -1)

    Nspl = 140
    ttrn = ts[:Nspl]
    Xtrn = X[:Nspl]
    ttst = ts[Nspl:]
    Xtst = X[Nspl:]

    np.savez_compressed('data/cylinder.npz', x=Xtrn, t=ttrn)
    np.savez_compressed('data/test.npz', x=Xtst, t=ttst)

if iftrn:
    for i in IDX:
        mdl, MDL, Trainer, opt = cfgs[i]
        opt["model"]["name"] = f"kp_{mdl}"
        setup_logging(config_path, mode='info', prefix='results')
        logging.info(f"Config: {config_path}")
        trainer = Trainer(config_path, MDL, config_mod=opt)
        trainer.train()

if ifplt:
    labels = [cfgs[i][0] for i in IDX]
    npz_files = [f'results/kp_{l}_summary.npz' for l in labels]
    npzs = plot_summary(npz_files, labels=labels, ifclose=False)

if ifprd:
    dat = np.load('./data/test.npz')
    x_data, t_data = dat['x'], dat['t']

    res = [x_data]
    for i in IDX:
        mdl, MDL, Trainer, opt = cfgs[i]
        opt["model"]["name"] = f"kp_{mdl}"
        _, prd_func = load_model(MDL, f'kp_{mdl}.pt', f'vor_model.yaml', config_mod=opt)
        with torch.no_grad():
            pred = prd_func(x_data, t_data)
        res.append(pred)

    labels = ['Truth'] + [cfgs[i][0] for i in IDX]
    plot_trajectory(
        np.array(res), t_data, "VOR",
        labels=labels, ifclose=False,
        xidx=[10000, 15000, 20000, 30000, 40000, 45000], grid=(3, 2))

plt.show()
