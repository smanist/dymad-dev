import matplotlib.pyplot as plt
import numpy as np
import torch

from dymad.io import load_model
from dymad.models import LDM
from dymad.training import NODETrainer, WeakFormTrainer
from dymad.utils import TrajectorySampler, plot_summary, plot_trajectory

# Keep a shared sampling interval across trajectories; the weak-form path currently
# builds one set of integration weights from dataset metadata.
DT = 0.05
LENGTHS = [41, 61, 81, 101]
TRAJ_PER_LENGTH = 8

A = np.array([[0.0, 1.0], [-1.0, -0.1]])


def f(t, x, u):
    return (x @ A.T) + u


def g(t, x, u):
    return x


def _to_object_array(items):
    data = np.empty(len(items), dtype=object)
    for index, item in enumerate(items):
        data[index] = np.asarray(item)
    return data


cases = [
    {
        "name": "ldm_node",
        "label": "NODE",
        "model": LDM,
        "trainer": NODETrainer,
        "config": "lti_vlen_ldm_node.yaml",
    },
    {
        "name": "ldm_wf",
        "label": "Weak form",
        "model": LDM,
        "trainer": WeakFormTrainer,
        "config": "lti_vlen_ldm_wf.yaml",
    },
]
# IDX = [0, 1]
IDX = [1]
labels = [cases[i]["label"] for i in IDX]
npz_files = [f"lti_vlen_{cases[i]['name']}" for i in IDX]


ifdat = 0
iftrn = 1
ifplt = 1
ifprd = 1

if ifdat:
    sampler = TrajectorySampler(f, g, config="lti_vlen_data.yaml", rng=7)

    ts_all = []
    ys_all = []
    us_all = []
    for n_steps in LENGTHS:
        t_grid = DT * np.arange(n_steps)
        ts, _, us, ys = sampler.sample(t_grid, batch=TRAJ_PER_LENGTH)
        ts_all.extend(np.array(item) for item in ts)
        ys_all.extend(np.array(item) for item in ys)
        us_all.extend(np.array(item) for item in us)

    # Store ragged object arrays on purpose; padding here would defeat the example.
    np.savez_compressed(
        "./data/lti_vlen.npz",
        t=_to_object_array(ts_all),
        x=_to_object_array(ys_all),
        u=_to_object_array(us_all),
    )

if iftrn:
    for i in IDX:
        Model = cases[i]["model"]
        Trainer = cases[i]["trainer"]
        config_path = cases[i]["config"]
        trainer = Trainer(config_path, Model)
        trainer.train()

if ifplt:
    plot_summary(npz_files, labels=labels, ifclose=False)

if ifprd:
    sampler = TrajectorySampler(f, g, config="lti_vlen_data.yaml", rng=11)

    t_grid = DT * np.arange(LENGTHS[-1])
    ts, xs, us, ys = sampler.sample(t_grid, batch=1)
    x_data = xs[0]
    t_data = ts[0]
    u_data = us[0]

    res = [x_data]
    for i in IDX:
        mdl = cases[i]["name"]
        MDL = cases[i]["model"]
        _, prd_func = load_model(MDL, f"lti_vlen_{mdl}.pt")

        with torch.no_grad():
            pred = prd_func(x_data, t_data, u=u_data)
        res.append(pred)

    plot_trajectory(
        np.array(res), t_data, "lti_vlen", us=u_data, labels=["Truth"] + labels, ifclose=False
    )

plt.show()
