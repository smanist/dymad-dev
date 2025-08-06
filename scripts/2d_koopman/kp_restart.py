import logging
import matplotlib.pyplot as plt
import numpy as np
import torch

from dymad.models import LDM, KBF
from dymad.training import WeakFormTrainer, NODETrainer
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
mdl_ld = {
    "name": "kp_model",
    "encoder_layers": 0,
    "processor_layers": 2,
    "decoder_layers": 0,
    "latent_dimension": 32,
    "autoencoder_type": "smp",
    "activation": "prelu",
    "weight_init": "xavier_uniform"}

trn_wf = {
    "n_epochs": 1000,
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
    "n_epochs": 1000,
    "save_interval": 20,
    "load_checkpoint": False,
    "learning_rate": 5e-3,
    "decay_rate": 0.999,
    "reconstruction_weight": 1.0,
    "dynamics_weight": 1.0,
    "sweep_lengths": [30, 50, 100, 200, 301],
    "sweep_epoch_step": 200,
    "ode_method": "dopri5",
    "rtol": 1e-7,
    "atol": 1e-9
}
config_path = 'kp_model.yaml'

cfgs = [
    ('ldm_wf',   LDM, WeakFormTrainer, {"model": mdl_ld, "training" : trn_wf}),
    ('ldm_node', LDM, NODETrainer,     {"model": mdl_ld, "training" : trn_nd}),
    ('kbf_wf',   KBF, WeakFormTrainer, {"model": mdl_kb, "training" : trn_wf}),
    ('kbf_node', KBF, NODETrainer,     {"model": mdl_kb, "training" : trn_nd}),
    ]

IDX = [0, 1]
# IDX = [2, 3]

iftrn = 1
ifrst = 1
ifplt = 1
ifprd = 1

if iftrn:
    for i in IDX:
        mdl, MDL, Trainer, opt = cfgs[i]
        opt["model"]["name"] = f"kp_{mdl}"
        setup_logging(config_path, mode='info', prefix='results')
        logging.info(f"Config: {config_path}")
        trainer = Trainer(config_path, MDL, config_mod=opt)
        trainer.train()

if ifrst:
    mdl, MDL, Trainer, opt = cfgs[IDX[0]]
    opt["model"]["name"] = f"kp_{mdl}_rst"
    opt["training"]["n_epochs"] = 500
    setup_logging(config_path, mode='info', prefix='results')
    logging.info(f"Config: {config_path}")
    trainer = Trainer(config_path, MDL, config_mod=opt)
    trainer.train()

    old_mld = cfgs[IDX[0]][0]
    mdl, MDL, Trainer, opt = cfgs[IDX[1]]
    opt["model"]["name"] = f"kp_{mdl}_rst"
    opt["training"]["load_checkpoint"] = f"./checkpoints/kp_{old_mld}_rst_checkpoint.pt"
    opt["training"]["n_epochs"] = 500
    # opt["training"]["sweep_lengths"] = None
    opt["training"]["sweep_epoch_step"] = 100
    setup_logging(config_path, mode='info', prefix='results')
    logging.info(f"Config: {config_path}")
    trainer = Trainer(config_path, MDL, config_mod=opt)
    trainer.train()

if ifplt:
    labels = [cfgs[i][0] for i in IDX] + [cfgs[IDX[1]][0]+'_rst']
    npz_files = [f'results/kp_{l}_summary.npz' for l in labels]
    npzs = plot_summary(npz_files, labels=labels, ifscl=False, ifclose=False)

    print(f"Epoch time {labels[0]}/{labels[1]}: {npzs[0]['avg_epoch_time']/npzs[1]['avg_epoch_time']}")
    print(f"Epoch time {labels[2]}/{labels[1]}: {npzs[2]['avg_epoch_time']/npzs[1]['avg_epoch_time']}")

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
    mdl, MDL, Trainer, opt = cfgs[IDX[1]]
    opt["model"]["name"] = f"kp_{mdl}_rst"
    _, prd_func = load_model(MDL, f'kp_{mdl}_rst.pt', f'kp_model.yaml', config_mod=opt)
    with torch.no_grad():
        pred = prd_func(x_data, t_data)
    res.append(pred)

    labels = ['Truth'] + [cfgs[i][0] for i in IDX] + [cfgs[IDX[1]][0]+'_rst']
    plot_trajectory(
        np.array(res), t_data, "KP", metadata={'n_state_features': 2},
        labels=labels, ifclose=False)

plt.show()
