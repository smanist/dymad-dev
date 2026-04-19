import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from dymad.io import load_model
from dymad.models import LTI
from dymad.training import StackedTrainer
from dymad.utils import TrajectorySampler, plot_summary, plot_trajectory

B = 128
N = 501
t_grid = np.linspace(0, 5, N)

A = np.array([[0.0, 1.0], [-1.0, -0.1]])


def f(t, x, u):
    return (x @ A.T) + u


def g(t, x, u):
    return x


config_chr = {
    "control": {
        "kind": "chirp",
        "params": {
            "t1": 4.0,
            "freq_range": (0.5, 2.0),
            "amp_range": (0.5, 1.0),
            "phase_range": (0.0, 360.0),
        },
    }
}

config_gau = {
    "control": {
        "kind": "gaussian",
        "params": {"mean": 0.5, "std": 1.0, "t1": 4.0, "dt": 0.2, "mode": "zoh"},
    }
}

config_noise = {
    "noise": {
        "kind": "gaussian",
        "params": {"mean": 0.0, "std": 0.1},
    }
}

cases = [
    {
        "name": "lti_denoise_wf",
        "model": LTI,
        "trainer": StackedTrainer,
        "config": "noise_wf.yaml",
    }
]
IDX = [0]
labels = [cases[i]["name"] for i in IDX]

ifdat = 1
iftrn = 1
ifplt = 1
ifprd = 1

if ifdat:
    os.makedirs("./data", exist_ok=True)
    sampler = TrajectorySampler(
        f,
        g,
        config="noise_data.yaml",
        config_mod={**config_chr, **config_noise},
    )
    ts, xs, us, ys = sampler.sample(t_grid, batch=B)
    np.savez_compressed("./data/lti_denoise.npz", t=ts, x=ys, u=us)

if iftrn:
    for _i in IDX:
        Model = cases[_i]["model"]
        Trainer = cases[_i]["trainer"]
        config_path = cases[_i]["config"]

        trainer = Trainer(config_path, Model)
        trainer.train()

if ifplt:
    npz_files = [labels[i] for i in range(len(labels))]
    npzs = plot_summary(npz_files, labels=labels, ifclose=False)
    for lbl, npz in zip(labels, npzs, strict=False):
        print(f"Epoch time: {lbl} - {npz['avg_epoch_time']}")

if ifprd:
    sampler = TrajectorySampler(
        f,
        g,
        config="noise_data.yaml",
        config_mod={**config_gau, **config_noise},
    )

    ts, xs, us, ys = sampler.sample(t_grid, batch=1)
    x_truth = xs[0]
    y_noisy = ys[0]
    t_data = ts[0]
    u_data = us[0]

    res = [x_truth, y_noisy]
    for _i in IDX:
        mdl, MDL = cases[_i]["name"], cases[_i]["model"]
        _, prd_func = load_model(MDL, f"{mdl}.pt")

        with torch.no_grad():
            _pred = prd_func(y_noisy, t_data, u=u_data)
        res.append(_pred)

    plot_trajectory(
        np.array(res),
        t_data,
        "LTI_denoise",
        us=u_data,
        labels=["Truth", "Noisy"] + labels,
        ifclose=False,
    )

plt.show()
