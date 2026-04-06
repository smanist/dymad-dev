import matplotlib.pyplot as plt
import numpy as np
import torch

from dymad.io import load_model
from dymad.models import KBF
from dymad.training import StackedTrainer
from dymad.utils import TrajectorySampler, plot_multi_trajs, plot_summary

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

cases = [
    {"name": "kbf_two", "model": KBF, "trainer": StackedTrainer, "config": "lti_kbf_two.yaml"},
    {"name": "kbf_mcri", "model": KBF, "trainer": StackedTrainer, "config": "lti_kbf_mcri.yaml"},
]
IDX = [1]
labels = [cases[i]["name"] for i in IDX]

iftrn = 1
ifplt = 1
ifprd = 1

if iftrn:
    for _i in IDX:
        Model = cases[_i]["model"]
        Trainer = cases[_i]["trainer"]
        config_path = cases[_i]["config"]

        trainer = Trainer(config_path, Model)
        trainer.train()

if ifplt:
    npz_files = [f"lti_{mdl}" for mdl in labels]
    npzs = plot_summary(npz_files, labels=labels, ifclose=False)
    for lbl, npz in zip(labels, npzs, strict=False):
        print(f"Epoch time: {lbl} - {npz['avg_epoch_time']}")

if ifprd:
    sampler = TrajectorySampler(f, g, config="lti_data.yaml", config_mod=config_gau)
    ts, xs, us, ys = sampler.sample(t_grid, batch=3)

    res = [xs]
    for _i in IDX:
        mdl, MDL = cases[_i]["name"], cases[_i]["model"]
        _, prd_func = load_model(MDL, f"lti_{mdl}.pt")

        with torch.no_grad():
            _pred = prd_func(xs, ts, u=us)
        res.append(_pred)

    plot_multi_trajs(np.array(res), ts[0], "LTI", us=us, labels=["Truth"] + labels, ifclose=False)

plt.show()
