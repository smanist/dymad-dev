import copy

import matplotlib.pyplot as plt
import numpy as np
import torch

from dymad.io import load_model
from dymad.losses import vpt_loss
from dymad.models import DKMSK
from dymad.training import LinearTrainer
from dymad.utils import TrajectorySampler, plot_cv_results, plot_multi_trajs, plot_trajectory

M = 2048
V = 2000
t_grid = np.linspace(0, 120, 12000)
t_pred = np.linspace(0, 60, 6000)

sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0


def f(t, x):
    dxdt = np.zeros_like(x)
    dxdt[0] = sigma * (x[1] - x[0])
    dxdt[1] = x[0] * (rho - x[2]) - x[1]
    dxdt[2] = x[0] * x[1] - beta * x[2]
    return dxdt


# Training options
RIDGE = 1e-6
DIM = 3

## Multi-output shared scalar kernels
opt_rbf = {"type": "sc_rbf", "input_dim": DIM, "lengthscale_init": None}
opt_dm = {
    "type": "sc_dm",
    "input_dim": DIM,
    "eps_init": None,
}
mdl_rbf = {
    "name": "ker_model",
    "encoder_layers": 0,
    "decoder_layers": 0,
    "kernel_dimension": DIM,
    "type": "share",
    "kernel": opt_rbf,
    "dtype": torch.float64,
    "ridge_init": RIDGE,
    "jitter": 0.0,
}
mdl_dm = copy.deepcopy(mdl_rbf)
mdl_dm["kernel"] = opt_dm

trn_ln = {
    "n_epochs": 1,
    "method": "raw",
}

cv_rbf = {
    "param_grid": {
        # "model.kernel.lengthscale_init": ('logspace', (1.0, 7.0, 25, True, 2)),
        "model.kernel.lengthscale_init": ("linspace", (4.0, 4.35, 25)),
        "model.ridge_init": [1e-13],
    },
    "metric": "vpt",
}
cv_dm = {
    "param_grid": {
        # "model.kernel.eps_init": ('logspace', (1.0, 7.0, 25, True, 2)),
        "model.kernel.eps_init": ("linspace", (0.7, 1.6, 10)),
        "model.ridge_init": [1e-13],
    },
    "metric": "vpt",
}

config_path = "lor_model.yaml"

cfgs = [
    ("dks_rbf", DKMSK, LinearTrainer, {"model": mdl_rbf, "cv": cv_rbf, "training": trn_ln}),
    ("ddm_dm", DKMSK, LinearTrainer, {"model": mdl_dm, "cv": cv_dm, "training": trn_ln}),
]

IDX = [0, 1]
labels = [cfgs[i][0] for i in IDX]

ifdat = 0
iftrn = 1
ifplt = 1
ifprd = 1
resume_training = False

if ifdat:
    sampler = TrajectorySampler(f, config="lor_data.yaml")
    ts, xs, ys = sampler.sample(t_grid, batch=1)

    np.savez_compressed("./data/l63_train.npz", t=ts[0][:M], x=xs[0][:M])
    np.savez_compressed(
        "./data/l63_valid.npz",
        t=ts[0][:V],
        x=np.array([xs[0][M : M + V], xs[0][M + V : M + 2 * V]]),
    )

    tt, xt, yt = sampler.sample(t_pred, batch=50)
    np.savez_compressed("./data/l63_test.npz", t=tt, x=xt)

    plot_trajectory(np.array(xs[0][:M]), ts[0][:M], None, labels=["Truth"], ifclose=False)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    B, T, D = xs.shape
    for i in range(B):
        ax.plot(xs[i, :M, 0], xs[i, :M, 1], xs[i, :M, 2])
    ax.set_title("Lorenz63")

if iftrn:
    for i in IDX:
        mdl, MDL, Trainer, opt = cfgs[i]
        opt["model"]["name"] = f"lor_{mdl}"
        trainer = Trainer(config_path, MDL, config_mod=opt)
        trainer.train(continue_training=resume_training)

if ifplt:
    for _i in IDX:
        mdl = cfgs[_i][0]
        if _i == 0:
            _key = "model.kernel.lengthscale_init"
        else:
            _key = "model.kernel.eps_init"
        # keys = [_key, 'model.ridge_init']
        # keys = [_key]
        keys = [_key]
        _, ax = plot_cv_results(f"lor_{mdl}", keys, ifclose=False)
        # ax.set_xscale('log')
        ax.set_yscale("log")

if ifprd:
    data = np.load("./data/l63_test.npz")
    ts = torch.tensor(data["t"], dtype=torch.float64)
    xs = torch.tensor(data["x"], dtype=torch.float64)
    JDX = 50

    res = [xs[:JDX]]
    for _i in IDX:
        mdl, MDL, _, _ = cfgs[_i]
        _, prd_func = load_model(MDL, f"lor_{mdl}.pt")
        with torch.no_grad():
            pred = prd_func(xs[:JDX], ts[:JDX])
        res.append(pred)

    vpt_rb, avr_rb = vpt_loss(res[1], res[0], gamma=0.3)
    vpt_dm, avr_dm = vpt_loss(res[2], res[0], gamma=0.3)

    fig = plt.figure()
    plt.violinplot([vpt_rb, vpt_dm], showmeans=True)
    plt.xticks([1, 2], labels)
    plt.ylabel("VPT (steps)")

    plot_multi_trajs(
        np.array([r[:2] for r in res]), ts[0], "L63", labels=["Truth"] + labels, ifclose=False
    )

plt.show()
