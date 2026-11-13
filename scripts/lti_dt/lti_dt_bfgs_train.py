"""Compare Adam, Adam → BFGS, and Adam → L-BFGS with identical Adam warm-ups.

All phases use full-batch, full-horizon losses and equal update caps, not equal
wall time. Curve times include updates and validation, excluding setup and I/O.
"""

# ruff: noqa: E402
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.runtime_env import configure_script_runtime

configure_script_runtime(__file__, matplotlib=True)

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy.signal import cont2discrete

from dymad.io import load_model
from dymad.models import DKBF, DLDM
from dymad.training import StackedTrainer
from dymad.utils import TrajectorySampler, plot_trajectory

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = BASE_DIR / "runs" / "lti_dt_bfgs"
B = 128
N = 501
SEED = 12345
WARMUP = 200
REFINE = 800
HISTORY_SIZE = 100
t_grid = np.linspace(0, 5, N)
cases = [("dldm", DLDM), ("dkbf", DKBF)]
IDX = [0, 1]
optimizers = ["adam", "bfgs", "lbfgs"]
labels = ["Adam", "Adam → full BFGS", "Adam → L-BFGS"]
ifdat, iftrn, ifplt, ifprd = 1, 1, 1, 1
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

A = np.array([[0.0, 1.0], [-1.0, -0.1]])
Ad, Bd, *_ = cont2discrete((A, np.ones((2, 1)), np.eye(2), np.zeros((2, 1))), t_grid[1] - t_grid[0])


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


def sample(control, seed, batch):
    sampler = TrajectorySampler(
        f, g, config=BASE_DIR / "lti_data.yaml", rng=seed, config_mod=control
    )
    ts, xs, us, _ = sampler.sample(t_grid, batch=batch)
    # Exact sample-held teacher: the stored u[k] drives the entire k -> k+1 interval.
    for k in range(N - 1):
        xs[:, k + 1] = xs[:, k] @ Ad.T + us[:, k] @ Bd.T
    return ts, xs, us


if ifdat:
    ts, xs, us = sample(config_chr, SEED, B)
    np.savez_compressed(OUTPUT_ROOT / "lti.npz", t=ts, x=xs, u=us)

for i in IDX:
    mdl, Model = cases[i]
    names = [f"lti_{mdl}_{optimizer}" for optimizer in optimizers]

    if iftrn:
        for optimizer, name in zip(optimizers, names, strict=True):
            config = yaml.safe_load((BASE_DIR / f"lti_{mdl}.yaml").read_text())
            config.pop("training")
            config["seed"] = SEED
            config["data"].update(
                path=str(OUTPUT_ROOT / "lti.npz"),
                n_samples=B,
                n_steps=N,
                double_precision=True,
                split_seed=SEED,
            )
            config["model"]["name"] = name
            # Small-time-step maps start near identity rather than erasing the state.
            if mdl == "dldm":
                config["model"].update(processor_type="res", gain=0.001)
            else:
                config["model"].update(autoencoder_type="cat", linear_dynamics_init_scale=1.0)
            config["dataloader"]["batch_size"] = B  # One optimizer update per epoch.
            warmup = {
                "trainer": "NODE",
                "optimizer": "adam",
                "n_epochs": WARMUP,
                "learning_rate": 1e-3,
                "decay_rate": 1.0,
                "save_interval": WARMUP,
                "chop_mode": "initial",
                "sweep_lengths": [None],
                "sweep_epoch_step": WARMUP + REFINE + 1,
            }
            refinement = {
                **warmup,
                "optimizer": optimizer,
                "n_epochs": REFINE,
                "save_interval": REFINE,
            }
            if optimizer != "adam":
                refinement.update(learning_rate=1.0, max_eval=40)
            if optimizer == "lbfgs":
                refinement["history_size"] = HISTORY_SIZE
            config["phases"] = [warmup, refinement]
            config_path = OUTPUT_ROOT / f"{name}.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False))
            StackedTrainer(str(config_path), Model, device=torch.device("cpu")).train()

    if ifplt:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        for name, label in zip(names, labels, strict=True):
            checkpoint = torch.load(OUTPUT_ROOT / name / f"{name}.pt", weights_only=False)
            # Validation is post-update for every optimizer; training-loss timing differs.
            loss = [value for phase in checkpoint["hist"] for value in phase["valid_total"]]
            axes[0].semilogy(np.arange(1, len(loss) + 1), loss, label=label)
            axes[1].semilogy(np.cumsum(checkpoint["epoch_times"]), loss, label=label)
        axes[0].axvline(WARMUP, color="gray", linestyle=":")
        axes[0].set_xlabel("Optimizer updates")
        axes[1].set_xlabel("Update + validation time (s)")
        for axis in axes:
            axis.set_ylabel("Validation loss")
            axis.grid(alpha=0.3)
            axis.legend()
        fig.suptitle(mdl.upper())
        fig.savefig(OUTPUT_ROOT / f"{mdl}_loss.png", dpi=150)

    if ifprd:
        ts, xs, us = sample(config_gau, SEED + 1, 1)
        res = [xs[0]]
        for name in names:
            _, predict = load_model(Model, OUTPUT_ROOT / name / f"{name}.pt")
            with torch.no_grad():
                res.append(predict(xs[0], ts[0], u=us[0]))
        plot_trajectory(
            np.array(res),
            ts[0],
            mdl,
            us=us[0],
            labels=["Truth"] + labels,
            prefix=str(OUTPUT_ROOT),
            cmp_err=False,
            ifclose=False,
        )
        np.savez_compressed(OUTPUT_ROOT / f"{mdl}_predictions.npz", t=ts[0], x=res, u=us[0])

plt.show()
