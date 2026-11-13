"""CLI counterpart of lti_dt_bfgs_train.py, with adjustable experiment sizes."""

# ruff: noqa: E402
import argparse
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
from scripts.cli_helpers import (
    add_common_cli_args,
    print_case_table,
    resolve_case_indices,
    set_seed,
)

from dymad.io import load_model
from dymad.models import DKBF, DLDM
from dymad.training import StackedTrainer
from dymad.utils import TrajectorySampler, plot_trajectory

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = BASE_DIR / "runs" / "lti_dt_bfgs"
CASES = [{"name": "dldm", "model": DLDM}, {"name": "dkbf", "model": DKBF}]
OPTIMIZERS = ["adam", "bfgs", "lbfgs"]
LABELS = ["Adam", "Adam → full BFGS", "Adam → L-BFGS"]
A = np.array([[0.0, 1.0], [-1.0, -0.1]])
CONFIG_CHR = {
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
CONFIG_GAU = {
    "control": {
        "kind": "gaussian",
        "params": {"mean": 0.5, "std": 1.0, "t1": 4.0, "dt": 0.2, "mode": "zoh"},
    }
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_cli_args(
        parser,
        plot_help="Skip loss and prediction comparison plots.",
        predict_help="Skip held-out predictions and their plots.",
    )
    parser.set_defaults(seed=12345)
    parser.add_argument("--trajectories", type=int, default=128)
    parser.add_argument("--steps", type=int, default=501)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--refine", type=int, default=800)
    parser.add_argument("--history-size", type=int, default=100)
    args = parser.parse_args()
    if args.trajectories < 2 or args.steps < 2:
        parser.error("--trajectories and --steps must each be at least 2")
    if not np.isfinite(args.duration) or args.duration <= 0:
        parser.error("--duration must be finite and positive")
    if min(args.warmup, args.refine, args.history_size) <= 0:
        parser.error("--warmup, --refine, and --history-size must be positive")
    return args


def f(t, x, u):
    return (x @ A.T) + u


def g(t, x, u):
    return x


def sample(control, seed, batch, t_grid):
    sampler = TrajectorySampler(
        f, g, config=BASE_DIR / "lti_data.yaml", rng=seed, config_mod=control
    )
    ts, xs, us, _ = sampler.sample(t_grid, batch=batch)
    ad, bd, *_ = cont2discrete(
        (A, np.ones((2, 1)), np.eye(2), np.zeros((2, 1))), t_grid[1] - t_grid[0]
    )
    # Exact sample-held teacher: stored u[k] drives the complete k -> k+1 interval.
    for k in range(len(t_grid) - 1):
        xs[:, k + 1] = xs[:, k] @ ad.T + us[:, k] @ bd.T
    return ts, xs, us


def generate_data(root, args, t_grid):
    ts, xs, us = sample(CONFIG_CHR, args.seed, args.trajectories, t_grid)
    np.savez_compressed(root / "lti.npz", t=ts, x=xs, u=us)


def validate_data(root, args, t_grid):
    expected = {
        "t": (args.trajectories, args.steps),
        "x": (args.trajectories, args.steps, 2),
        "u": (args.trajectories, args.steps, 1),
    }
    with np.load(root / "lti.npz") as data:
        if any(
            key not in data or data[key].shape != shape or not np.isfinite(data[key]).all()
            for key, shape in expected.items()
        ) or not np.allclose(data["t"], t_grid, rtol=1e-12, atol=1e-14):
            raise ValueError(
                "Existing lti.npz does not match this run; use --data to regenerate it."
            )


def train(case, root, args):
    mdl, model = case["name"], case["model"]
    for optimizer in OPTIMIZERS:
        name = f"lti_{mdl}_{optimizer}"
        config = yaml.safe_load((BASE_DIR / f"lti_{mdl}.yaml").read_text())
        config.pop("training")
        config["seed"] = args.seed
        config["data"].update(
            path=str(root / "lti.npz"),
            n_samples=args.trajectories,
            n_steps=args.steps,
            double_precision=True,
            split_seed=args.seed,
        )
        config["model"]["name"] = name
        if mdl == "dldm":
            config["model"].update(processor_type="res", gain=0.001)
        else:
            config["model"].update(autoencoder_type="cat", linear_dynamics_init_scale=1.0)
        config["dataloader"]["batch_size"] = args.trajectories
        warmup = {
            "trainer": "NODE",
            "optimizer": "adam",
            "n_epochs": args.warmup,
            "learning_rate": 1e-3,
            "decay_rate": 1.0,
            "save_interval": args.warmup,
            "chop_mode": "initial",
            "sweep_lengths": [None],
            "sweep_epoch_step": args.warmup + args.refine + 1,
        }
        refinement = {
            **warmup,
            "optimizer": optimizer,
            "n_epochs": args.refine,
            "save_interval": args.refine,
        }
        if optimizer != "adam":
            refinement.update(learning_rate=1.0, max_eval=40)
        if optimizer == "lbfgs":
            refinement["history_size"] = args.history_size
        config["phases"] = [warmup, refinement]
        config_path = root / f"{name}.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        StackedTrainer(str(config_path), model, device=torch.device("cpu")).train()


def plot(case, root, args):
    mdl = case["name"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for optimizer, label in zip(OPTIMIZERS, LABELS, strict=True):
        name = f"lti_{mdl}_{optimizer}"
        checkpoint = torch.load(root / name / f"{name}.pt", weights_only=False)
        # Validation losses are post-update for every optimizer.
        loss = [value for phase in checkpoint["hist"] for value in phase["valid_total"]]
        axes[0].semilogy(np.arange(1, len(loss) + 1), loss, label=label)
        axes[1].semilogy(np.cumsum(checkpoint["epoch_times"]), loss, label=label)
    axes[0].axvline(args.warmup, color="gray", linestyle=":")
    axes[0].set_xlabel("Optimizer updates")
    axes[1].set_xlabel("Update + validation time (s)")
    for axis in axes:
        axis.set_ylabel("Validation loss")
        axis.grid(alpha=0.3)
        axis.legend()
    fig.suptitle(mdl.upper())
    fig.savefig(root / f"{mdl}_loss.png", dpi=150)


def predict(case, root, args, t_grid):
    mdl = case["name"]
    ts, xs, us = sample(CONFIG_GAU, args.seed + 1, 1, t_grid)
    initial = np.zeros_like(xs[0])
    initial[0] = xs[0, 0]
    res = [xs[0]]
    for optimizer in OPTIMIZERS:
        name = f"lti_{mdl}_{optimizer}"
        _, predict_fn = load_model(case["model"], root / name / f"{name}.pt")
        with torch.no_grad():
            res.append(predict_fn(initial, ts[0], u=us[0]))
    np.savez_compressed(root / f"{mdl}_predictions.npz", t=ts[0], x=res, u=us[0])
    if not args.no_plot:
        plot_trajectory(
            np.array(res),
            ts[0],
            mdl,
            us=us[0],
            labels=["Truth"] + LABELS,
            prefix=str(root),
            cmp_err=False,
            ifclose=False,
        )


def main():
    args = parse_args()
    if args.list_cases:
        print_case_table(CASES)
        return 0
    selected = resolve_case_indices(args.case, len(CASES), range(len(CASES)))
    set_seed(args.seed)
    root = (OUTPUT_ROOT if args.workdir is None else args.workdir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    t_grid = np.linspace(0, args.duration, args.steps)
    if args.data or not (root / "lti.npz").exists():
        generate_data(root, args, t_grid)
    validate_data(root, args, t_grid)
    for index in selected:
        case = CASES[index]
        if not args.no_train:
            train(case, root, args)
        if not args.no_plot:
            plot(case, root, args)
        if not args.no_predict:
            predict(case, root, args, t_grid)
    if not args.no_show:
        plt.show()
    plt.close("all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
