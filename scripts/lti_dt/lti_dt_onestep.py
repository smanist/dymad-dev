"""
Continuous-time one-step nonlinear solver example for the controlled LTI case.

This expands the original one-step example in place. The `lti_dt/` location is
historical; the comparison below now targets the continuous-time `LDM` setup so
the missing CT workflow lives next to the original development script. The
system and hyperparameter baseline match the existing continuous-time LTI
examples, while keeping the same `if`-block control flow for easy regression
checks.
"""

import copy
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from dymad.io import load_model
from dymad.models import LDM
from dymad.training import OneStepTrainer, StackedTrainer
from dymad.utils import TrajectorySampler, plot_summary, plot_trajectory

BASE_DIR = Path(__file__).resolve().parent
DATA_CONFIG_PATH = BASE_DIR.parent / "linear_time_invariant" / "lti_data.yaml"
TRAINING_CONFIG_PATH = BASE_DIR.parent / "linear_time_invariant" / "lti_ldm_node.yaml"
os.chdir(BASE_DIR)

B = 128
N = 501
t_grid = np.linspace(0, 5, N)
DATA_PATH = BASE_DIR / "data" / "lti.npz"

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
        "params": {
            "mean": 0.5,
            "std": 1.0,
            "t1": 4.0,
            "dt": 0.2,
            "mode": "zoh",
        },
    }
}

mdl_ld = {
    "name": "lti_ldm",
    "encoder_layers": 0,
    "processor_layers": 1,
    "decoder_layers": 0,
    "hidden_dimension": 32,
    "activation": "none",
    "weight_init": "xavier_uniform",
    "gain": 0.01,
}

# The one-step-only run is intentionally longer because it does not get a NODE refinement phase.
trn_step = {
    "n_epochs": 1200,
    "save_interval": 20,
    "load_checkpoint": False,
    "learning_rate": 1e-3,
    "decay_rate": 0.999,
}
trn_step_warm = copy.deepcopy(trn_step)
trn_step_warm["n_epochs"] = 300

trn_node = {
    "n_epochs": 900,
    "save_interval": 20,
    "load_checkpoint": False,
    "learning_rate": 1e-3,
    "decay_rate": 0.999,
    "sweep_lengths": [50, 100, 200, 300, 501],
    "sweep_epoch_step": 100,
    "ode_method": "dopri5",
    "ode_args": {"rtol": 1.0e-7, "atol": 1.0e-9},
}


def _optimizer_phase(trainer, cfg):
    phase = copy.deepcopy(cfg)
    phase.update({"type": "optimizer", "trainer": trainer})
    return phase


def generate_data():
    sampler = TrajectorySampler(f, g, config=DATA_CONFIG_PATH, config_mod=config_chr)
    ts, _xs, us, ys = sampler.sample(t_grid, batch=B)
    DATA_PATH.parent.mkdir(exist_ok=True)
    np.savez_compressed(DATA_PATH, t=ts, x=ys, u=us)
    print(f"Generated data: {DATA_PATH}")


cases = [
    {
        "name": "ldm_step",
        "model": LDM,
        "trainer": OneStepTrainer,
        "config_mod": {"model": mdl_ld, "training": trn_step},
    },
    {
        "name": "ldm_step_node",
        "model": LDM,
        "trainer": StackedTrainer,
        "config_mod": {
            "model": mdl_ld,
            # Clear the legacy single-stage block from the base YAML so it does not
            # overwrite the explicit warm-start phase during config normalization.
            "training": None,
            "phases": [
                _optimizer_phase("OneStep", trn_step_warm),
                _optimizer_phase("NODE", trn_node),
            ],
        },
    },
]

IDX = [0, 1]
labels = [cases[i]["name"] for i in IDX]

ifdat = 0
iftrn = 1
ifplt = 1
ifprd = 1

if ifdat:
    generate_data()

if iftrn:
    if not DATA_PATH.exists():
        generate_data()
    for i in IDX:
        case = cases[i]
        opt = copy.deepcopy(case["config_mod"])
        opt["model"]["name"] = f"lti_{case['name']}"
        trainer = case["trainer"](TRAINING_CONFIG_PATH, case["model"], config_mod=opt)
        trainer.train()

if ifplt:
    npz_files = [f"lti_{label}" for label in labels]
    npzs = plot_summary(npz_files, labels=labels, ifclose=False)

    for label, npz in zip(labels, npzs, strict=False):
        print(f"Epoch time {label}: {npz['avg_epoch_time']}")

if ifprd:
    sampler = TrajectorySampler(f, g, config=DATA_CONFIG_PATH, config_mod=config_gau)

    ts, xs, us, ys = sampler.sample(t_grid, batch=1)
    x_data = xs[0]
    t_data = ts[0]
    u_data = us[0]

    res = [x_data]
    for i in IDX:
        case = cases[i]
        _, prd_func = load_model(case["model"], f"lti_{case['name']}.pt")
        with torch.no_grad():
            pred = prd_func(x_data, t_data, u=u_data)
        res.append(pred)

    plot_trajectory(
        np.array(res),
        t_data,
        "LTI CT",
        us=u_data,
        labels=["Truth"] + labels,
        ifclose=False,
    )

plt.show()
