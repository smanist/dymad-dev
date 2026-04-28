"""
Standalone one-step optimizer examples for controlled LTI systems.

The folder owns its local data generation and training configs. It compares:

- continuous-time: one-step only, one-step followed by weak-form refinement
- discrete-time: one-step only, one-step followed by NODE refinement
"""

import copy
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from dymad.io import load_model
from dymad.models import DLDM, LDM
from dymad.training import OneStepTrainer, StackedTrainer
from dymad.utils import TrajectorySampler, plot_summary, plot_trajectory

BASE_DIR = Path(__file__).resolve().parent
DATA_CONFIG_PATH = BASE_DIR / "lti_1s_data.yaml"
CT_CONFIG_PATH = BASE_DIR / "lti_1s_ct.yaml"
DT_CONFIG_PATH = BASE_DIR / "lti_1s_dt.yaml"
DATA_PATH = BASE_DIR / "data" / "lti_1s.npz"

os.chdir(BASE_DIR)

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
        "params": {
            "mean": 0.5,
            "std": 1.0,
            "t1": 4.0,
            "dt": 0.2,
            "mode": "zoh",
        },
    }
}

mdl_ct = {
    "encoder_layers": 0,
    "processor_layers": 1,
    "decoder_layers": 0,
    "hidden_dimension": 32,
    "activation": "none",
    "weight_init": "xavier_uniform",
    "gain": 0.01,
    "input_order": "cubic",
}

mdl_dt = {
    "encoder_layers": 0,
    "processor_layers": 1,
    "decoder_layers": 0,
    "hidden_dimension": 32,
    "activation": "none",
    "weight_init": "xavier_uniform",
    "gain": 0.01,
}

trn_step = {
    "n_epochs": 2000,
    "save_interval": 50,
    "load_checkpoint": False,
    "learning_rate": 1e-3,
    "decay_rate": 0.999,
}
trn_step_warm = copy.deepcopy(trn_step)
trn_step_warm["n_epochs"] = 1000

trn_weak = {
    "n_epochs": 500,
    "save_interval": 50,
    "load_checkpoint": False,
    "learning_rate": 5e-3,
    "decay_rate": 0.999,
    "weak_form_params": {
        "N": 13,
        "dN": 2,
        "ordpol": 2,
        "ordint": 2,
    },
}

trn_dt_node = {
    "n_epochs": 500,
    "save_interval": 50,
    "load_checkpoint": False,
    "learning_rate": 1e-3,
    "decay_rate": 0.999,
    "chop_mode": "unfold",
    "chop_step": 0.5,
    "sweep_lengths": [3, 5, 7],
    "sweep_epoch_step": 200,
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
        "label": "ct_step",
        "artifact_name": "lti_1s_ct_step",
        "config_path": CT_CONFIG_PATH,
        "model": LDM,
        "trainer": OneStepTrainer,
        "config_mod": {
            "model": {"name": "lti_1s_ct_step", **mdl_ct},
            "training": trn_step,
        },
    },
    {
        "label": "ct_step_wf",
        "artifact_name": "lti_1s_ct_step_wf",
        "config_path": CT_CONFIG_PATH,
        "model": LDM,
        "trainer": StackedTrainer,
        "config_mod": {
            "model": {"name": "lti_1s_ct_step_wf", **mdl_ct},
            "training": None,
            "phases": [
                _optimizer_phase("OneStep", trn_step_warm),
                _optimizer_phase("Weak", {**trn_weak, "reset_optimizer": True}),
            ],
        },
    },
    {
        "label": "dt_step",
        "artifact_name": "lti_1s_dt_step",
        "config_path": DT_CONFIG_PATH,
        "model": DLDM,
        "trainer": OneStepTrainer,
        "config_mod": {
            "model": {"name": "lti_1s_dt_step", **mdl_dt},
            "training": trn_step,
        },
    },
    {
        "label": "dt_step_node",
        "artifact_name": "lti_1s_dt_step_node",
        "config_path": DT_CONFIG_PATH,
        "model": DLDM,
        "trainer": StackedTrainer,
        "config_mod": {
            "model": {"name": "lti_1s_dt_step_node", **mdl_dt},
            "training": None,
            "phases": [
                _optimizer_phase("OneStep", trn_step_warm),
                _optimizer_phase("NODE", {**trn_dt_node, "reset_optimizer": True}),
            ],
        },
    },
]

IDX = [0, 1, 2, 3]
labels = [cases[i]["label"] for i in IDX]

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
        trainer = case["trainer"](
            case["config_path"],
            case["model"],
            config_mod=copy.deepcopy(case["config_mod"]),
        )
        trainer.train()

if ifplt:
    npz_files = [cases[i]["artifact_name"] for i in IDX]
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
        _, prd_func = load_model(case["model"], f"{case['artifact_name']}.pt")
        with torch.no_grad():
            pred = prd_func(x_data, t_data, u=u_data)
        res.append(pred)

    plot_trajectory(
        np.array(res),
        t_data,
        "LTI",
        us=u_data,
        labels=["Truth"] + labels,
        ifclose=False,
    )

plt.show()
