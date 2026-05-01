import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sps
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

noise_kind = 3

noise_options = [
    {
        "kind": "gaussian",
        "params": {"mean": 0.0, "std": 0.1},
    },
    {
        "kind": "laplace",
        "params": {"loc": [0.0, 0.0], "scale": [0.1, 0.1]},
    },
    {
        "kind": "student_t",
        "params": {"df": [5.0, 5.0], "loc": [0.0, 0.0], "scale": [0.1, 0.1]},
    },
    {
        "kind": "uniform",
        "params": {"bounds": [[-0.1, 0.1], [-0.1, 0.1]]},
    },
]
config_noise = noise_options[noise_kind]


def expected_noise_rvs(config_noise, dim):
    kind = config_noise["kind"]
    params = config_noise["params"]

    if kind == "gaussian":
        mean = np.broadcast_to(params["mean"], (dim,))
        std = np.broadcast_to(params["std"], (dim,))
        return [sps.norm(loc=mean[i], scale=std[i]) for i in range(dim)]

    if kind == "laplace":
        loc = np.broadcast_to(params["loc"], (dim,))
        scale = np.broadcast_to(params["scale"], (dim,))
        return [sps.laplace(loc=loc[i], scale=scale[i]) for i in range(dim)]

    if kind == "student_t":
        df = np.broadcast_to(params["df"], (dim,))
        loc = np.broadcast_to(params.get("loc", 0.0), (dim,))
        scale = np.broadcast_to(params.get("scale", 1.0), (dim,))
        return [sps.t(df=df[i], loc=loc[i], scale=scale[i]) for i in range(dim)]

    if kind == "uniform":
        bounds = np.broadcast_to(params["bounds"], (dim, 2))
        return [
            sps.uniform(loc=bounds[i, 0], scale=bounds[i, 1] - bounds[i, 0]) for i in range(dim)
        ]

    raise KeyError(f"Unknown noise kind '{kind}'. Available: {list(noise_options)}")


def plot_noise_distribution_check(xs, ys, config_noise):
    noise = (ys - xs).reshape(-1, ys.shape[-1]).T
    rvs = expected_noise_rvs(config_noise, noise.shape[0])

    fig, ax = plt.subplots(nrows=noise.shape[0], sharex=True)
    ax = np.atleast_1d(ax)
    for i, (noise_i, rv) in enumerate(zip(noise, rvs, strict=False)):
        q_lo, q_hi = rv.ppf([1e-3, 1 - 1e-3])
        xx = np.linspace(min(np.min(noise_i), q_lo), max(np.max(noise_i), q_hi), 200)
        ax[i].hist(noise_i, bins=20, density=True, alpha=0.6, label="Sampled")
        ax[i].plot(xx, rv.pdf(xx), linewidth=2, label="Expected")
        ax[i].set_ylabel(f"y{i + 1}")
        ax[i].legend()
    ax[-1].set_xlabel("Noise")
    fig.suptitle(f"Noise check: {config_noise['kind']}")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    return fig, ax


cases = [
    {
        "name": "lti_denoise_wf",
        "model": LTI,
        "trainer": StackedTrainer,
        "config": "noise_wf.yaml",
    },
    {
        "name": "lti_denoise_cpp",
        "model": LTI,
        "trainer": StackedTrainer,
        "config": "noise_cpp.yaml",
    },
]
IDX = [0, 1]
labels = [cases[i]["name"] for i in IDX]

ifdat = 0
ifdst = 0
iftrn = 0
ifplt = 1
ifprd = 1

if ifdat:
    os.makedirs("./data", exist_ok=True)
    sampler = TrajectorySampler(
        f,
        g,
        config="noise_data.yaml",
        config_mod={**config_chr, **{"noise": config_noise}},
    )
    ts, xs, us, ys = sampler.sample(t_grid, batch=B)
    np.savez_compressed("./data/lti_denoise.npz", t=ts, x=ys, u=us)

    if ifdst:
        plot_noise_distribution_check(xs, ys, config_noise)

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
        config_mod={**config_gau, **{"noise": config_noise}},
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
