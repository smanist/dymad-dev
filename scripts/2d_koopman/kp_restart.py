import matplotlib.pyplot as plt
import numpy as np
import torch

from dymad.io import load_model
from dymad.models import LDM, KBF
from dymad.training import StackedTrainer
from dymad.utils import plot_summary, plot_trajectory, TrajectorySampler

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
    "hidden_dimension" : 32,
    "koopman_dimension" : 4,
    "autoencoder_type": "cat",
    "activation" : "prelu",
    "weight_init" : "xavier_uniform"}
mdl_ld = {
    "name": "kp_model",
    "encoder_layers": 0,
    "processor_layers": 2,
    "decoder_layers": 0,
    "hidden_dimension": 32,
    "autoencoder_type": "smp",
    "activation": "prelu",
    "weight_init": "xavier_uniform"}

config_path = 'kp_stack.yaml'

cfgs = [
    ('ldm_st', LDM, StackedTrainer, {"model": mdl_ld}),
    ('kbf_st', KBF, StackedTrainer, {"model": mdl_kb}),
    ]

IDX = [0, 1]
# prf, MDL = 'ldm', LDM
prf, MDL = 'kbf', KBF
models = [prf+'_wf', prf+'_node', prf+'_st']

iftrn = 0
ifplt = 1
ifprd = 1

if iftrn:
    for i in IDX:
        mdl, MDL, Trainer, opt = cfgs[i]
        opt["model"]["name"] = f"kp_{mdl}"
        trainer = Trainer(config_path, MDL, config_mod=opt)
        trainer.train()

if ifplt:
    npz_files = [f'kp_{l}' for l in models]
    npzs = plot_summary(npz_files, labels=models, ifscl=False, ifclose=False)

    print(f"Epoch time {models[0]}/{models[1]}: {npzs[0]['avg_epoch_time']/npzs[1]['avg_epoch_time']}")
    print(f"Epoch time {models[2]}/{models[1]}: {npzs[2]['avg_epoch_time']/npzs[1]['avg_epoch_time']}")

if ifprd:
    sampler = TrajectorySampler(f, config='kp_data.yaml')
    ts, xs, ys = sampler.sample(t_grid, batch=1)
    x_data = xs[0]
    t_data = ts[0]

    res = [x_data]
    for mdl in models:
        _, prd_func = load_model(MDL, f'kp_{mdl}.pt')
        with torch.no_grad():
            pred = prd_func(x_data, t_data)
        res.append(pred)

    labels = ['Truth'] + models
    plot_trajectory(
        np.array(res), t_data, "KP",
        labels=labels, ifclose=False)

plt.show()
