import copy
import matplotlib.pyplot as plt
import numpy as np
import torch

from dymad.io import load_model
from dymad.models import KBF, DKBF, DKMSK
from dymad.training import NODETrainer, LinearTrainer
from dymad.utils import animate, compare_contour, plot_summary, setup_logging

def gen_mdl_kb(e, l, k):
    return {
        "name" : 'vor_model',
        "encoder_layers" : e,
        "decoder_layers" : e,
        "hidden_dimension" : l,
        "koopman_dimension" : k,
        "activation" : "prelu",
        "weight_init" : "xavier_uniform",
        "predictor_type" : "exp"}

mdl_kl = {
    "name" : 'ker_model',
    "encoder_layers" : 0,
    "decoder_layers" : 0,
    "kernel_dimension" : 12,
    "type": "share",
    "kernel": {
        "type": "sc_dm",
        "input_dim": 12,
        "eps_init": None
    },
    "dtype": torch.float64,
    "ridge_init": 1e-10
    }

crit = {
    "dynamics": {"weight": 1.0},
    "recon":    {"weight": 1.0},
}
trn_nd = {
    "n_epochs": 200,
    "save_interval": 50,
    "load_checkpoint": False,
    "learning_rate": 5e-3,
    "decay_rate": 0.999,
    "sweep_lengths": [2],
    "sweep_epoch_step": 100,
    "chop_mode": "unfold",
    "chop_step": 1,
    "ode_method": "dopri5",
    "ode_args": {
        "rtol": 1.e-7,
        "atol": 1.e-9},
    "ls_update": {
        "method": "full",
        "interval": 50,
        "times": 1}
    }
trn_ae = copy.deepcopy(trn_nd)
trn_ae["n_epochs"] = 2000
trn_ae["sweep_lengths"] = [2, 10, 50]
trn_ae["sweep_epoch_step"] = 500
trn_ae["ls_update"]["interval"] = 500
trn_ae["ls_update"]["times"] = 2
trn_ae["ls_update"]["reset"] = False

trn_ln = {
    "n_epochs": 500,
    "save_interval": 50,
    "load_checkpoint": False,
    "ls_update": {
        "method": "full"}
    }
trn_rw = copy.deepcopy(trn_ln)
trn_rw["ls_update"]["method"] = "raw"

trn_svd = {
    "type" : "svd",
    "ifcen": True,
    "order": 12
}
trn_scl = {
    "type" : "scaler",
    "mode" : "std",
}
trn_add = {
    "type" : "add_one"
}
trn_dmf = {
    "type" : "dm",
    "edim": 3,
    "Knn" : 15,
    "Kphi": 3,
    "inverse": "gmls",
    "order": 1,
    "mode": "full"
}

config_path = 'vor_model.yaml'

cfgs = [
    ('kbf_node', KBF,  NODETrainer,     {"model": gen_mdl_kb(0,0,13), "criterion": crit, "training" : trn_nd, "transform_x" : [trn_svd, trn_add]}),
    ('dkbf_ln',  DKBF, LinearTrainer,   {"model": gen_mdl_kb(0,0,13), "training" : trn_ln, "transform_x" : [trn_svd, trn_add]}),
    ('dkbf_ae',  DKBF, NODETrainer,     {"model": gen_mdl_kb(3,64,3), "criterion": crit, "training" : trn_ae, "transform_x" : [trn_svd]}),
    ('dkbf_dm',  DKBF, LinearTrainer,   {"model": gen_mdl_kb(0,0,3),  "training" : trn_ln, "transform_x" : [trn_svd, trn_dmf]}),
    ('dks_ln',  DKMSK, LinearTrainer,   {"model": mdl_kl, "training" : trn_rw, "transform_x" : [trn_svd, trn_scl]}),
    ]

IDX = [0, 1, 2, 3, 4]

iftrn = 1
ifplt = 1
ifprd = 1

if iftrn:
    for i in IDX:
        mdl, MDL, Trainer, opt = cfgs[i]
        opt["model"]["name"] = f"kp_{mdl}"
        trainer = Trainer(config_path, MDL, config_mod=opt)
        trainer.train()

if ifplt:
    labels = [cfgs[i][0] for i in IDX]
    npz_files = [f'kp_{l}' for l in labels]
    npzs = plot_summary(npz_files, labels=labels, ifclose=False)

if ifprd:
    # dat = np.load('./data/test.npz')
    dat = np.load('./data/cylinder.npz')
    x_data, t_data = dat['x'], dat['t']
    Nx, Ny = 199, 449

    res = [x_data]
    for i in IDX:
        mdl, MDL, Trainer, _ = cfgs[i]
        model, prd_func = load_model(MDL, f'kp_{mdl}.pt')
        with torch.no_grad():
            pred = prd_func(x_data, t_data)
        res.append(pred)

    setup_logging()

    N = len(IDX)
    def contour_fig(j):
        fig, ax = plt.subplots(N, 3, sharex=True, sharey=True, figsize=(12, 1.5*N))
        colorbar = j == 0
        for i in range(N):
            compare_contour(
                res[0][j].reshape(Nx, Ny), res[i+1][j].reshape(Nx, Ny), vmin=-12, vmax=12,
                axes=(fig, ax[i]), colorbar=colorbar)
            ax[i,1].set_title(cfgs[IDX[i]][0])
        for _ax in ax.flatten():
            _ax.set_axis_off()
        return fig, ax

    animate(contour_fig, filename="vis.mp4", fps=10, n_frames=len(t_data))

plt.show()
