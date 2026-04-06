import argparse
import copy
import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.cli_helpers import add_common_cli_args, print_case_table, resolve_case_indices, set_seed, stage_workdir

from dymad.io import load_model
from dymad.losses import vpt_loss
from dymad.models import DKMSK
from dymad.training import LinearTrainer
from dymad.utils import TrajectorySampler, plot_cv_results, plot_multi_trajs, plot_trajectory


BASE_DIR = Path(__file__).resolve().parent

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


RIDGE = 1e-6
DIM = 3
opt_rbf = {"type": "sc_rbf", "input_dim": DIM, "lengthscale_init": None}
opt_dm = {"type": "sc_dm", "input_dim": DIM, "eps_init": None}
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

trn_ln = {"n_epochs": 1, "method": "raw"}
cv_rbf = {
    "param_grid": {
        "model.kernel.lengthscale_init": ("linspace", (54.0, 75.0, 25)),
        "model.ridge_init": [1e-13],
    },
    "metric": "vpt",
}
cv_rbf_resume = {
    "param_grid": {
        "model.kernel.lengthscale_init": ("linspace", (76.0, 85.0, 10)),
        "model.ridge_init": [1e-13],
    },
    "metric": "vpt",
}
cv_dm = {
    "param_grid": {
        "model.kernel.eps_init": ("linspace", (1.0, 4.0, 10)),
        "model.ridge_init": [1e-13],
    },
    "metric": "vpt",
}
cv_dm_resume = {
    "param_grid": {
        "model.kernel.eps_init": ("linspace", (4.25, 7.0, 10)),
        "model.ridge_init": [1e-13],
    },
    "metric": "vpt",
}

cases = [
    {
        "name": "dks_rbf",
        "model": DKMSK,
        "trainer": LinearTrainer,
        "config": "lor_model.yaml",
        "setups": [
            {"name": "base_25", "resume": False, "options": {"model": mdl_rbf, "cv": cv_rbf, "training": trn_ln}},
            {"name": "resume_10", "resume": True, "options": {"model": mdl_rbf, "cv": cv_rbf_resume, "training": trn_ln}},
        ],
    },
    {
        "name": "ddm_dm",
        "model": DKMSK,
        "trainer": LinearTrainer,
        "config": "lor_model.yaml",
        "setups": [
            {"name": "base_10", "resume": False, "options": {"model": mdl_dm, "cv": cv_dm, "training": trn_ln}},
            {"name": "resume_10", "resume": True, "options": {"model": mdl_dm, "cv": cv_dm_resume, "training": trn_ln}},
        ],
    },
]
DEFAULT_CASES = [0, 1]
DEFAULT_SETUP = 0


def parse_args():
    parser = argparse.ArgumentParser(description="Run Lorenz63 kernel cases.")
    add_common_cli_args(parser, include_data=True, plot_help="Skip CV plotting.")
    parser.add_argument("--resume", action="store_true", help="Resume training from an existing checkpoint.")
    parser.add_argument("--setup", type=int, default=DEFAULT_SETUP, help="Setup index to run for the selected cases.")
    parser.add_argument("--list-setups", action="store_true", help="List setup indices for each case.")
    return parser.parse_args()


def print_setup_table():
    for case_idx, case in enumerate(cases):
        print(f"{case_idx}: {case['name']}")
        for setup_idx, setup in enumerate(case["setups"]):
            suffix = " (resume)" if setup["resume"] else ""
            print(f"  {setup_idx}: {setup['name']}{suffix}")


def resolve_setup_index(setup: int) -> int:
    if setup < 0:
        raise ValueError(f"Setup index must be non-negative, got {setup}.")
    max_setups = min(len(case["setups"]) for case in cases)
    if setup >= max_setups:
        raise ValueError(f"Setup index {setup} is out of range; expected < {max_setups}.")
    return setup


def prepare_workdir(root: Path):
    stage_workdir(root, BASE_DIR, ["lor_model.yaml", "lor_data.yaml"])


def generate_data(root: Path):
    sampler = TrajectorySampler(f, config=BASE_DIR / "lor_data.yaml")
    ts, xs, _ = sampler.sample(t_grid, batch=1)

    np.savez_compressed(root / "data" / "l63_train.npz", t=ts[0][:M], x=xs[0][:M])
    np.savez_compressed(
        root / "data" / "l63_valid.npz",
        t=ts[0][:V],
        x=np.array([xs[0][M : M + V], xs[0][M + V : M + 2 * V]]),
    )

    tt, xt, _ = sampler.sample(t_pred, batch=50)
    np.savez_compressed(root / "data" / "l63_test.npz", t=tt, x=xt)
    print(f"Generated data under: {root / 'data'}")


def train(selected: list[int], *, setup_idx: int, resume_override: bool):
    for idx in selected:
        case = cases[idx]
        setup = case["setups"][setup_idx]
        options = copy.deepcopy(setup["options"])
        options["model"]["name"] = f"lor_{case['name']}"
        trainer = case["trainer"](case["config"], case["model"], config_mod=options)
        trainer.train(continue_training=(setup["resume"] or resume_override))


def plot(selected: list[int]):
    for idx in selected:
        name = cases[idx]["name"]
        key = "model.kernel.lengthscale_init" if idx == 0 else "model.kernel.eps_init"
        _, ax = plot_cv_results(f"lor_{name}", [key], ifclose=False)
        ax.set_yscale("log")


def predict(selected: list[int]):
    data = np.load("./data/l63_test.npz")
    ts = torch.tensor(data["t"], dtype=torch.float64)
    xs = torch.tensor(data["x"], dtype=torch.float64)
    n_plot = 20

    res = [xs[:n_plot]]
    labels = ["Truth"]
    for idx in selected:
        name = cases[idx]["name"]
        _, prd_func = load_model(cases[idx]["model"], f"lor_{name}.pt")
        with torch.no_grad():
            pred = prd_func(xs[:n_plot], ts[:n_plot])
        res.append(pred)
        labels.append(name)

    if len(res) >= 3:
        vpts = []
        for pred in res[1:]:
            vpt, _ = vpt_loss(pred, res[0], gamma=0.3)
            vpts.append(vpt)
        plt.figure()
        plt.violinplot(vpts, showmeans=True)
        plt.xticks(np.arange(1, len(vpts) + 1), labels[1:])
        plt.ylabel("VPT (steps)")

    plot_multi_trajs(np.array([r[:2] for r in res]), ts[0], "L63", labels=labels, ifclose=False)


def main():
    args = parse_args()
    if args.seed is not None:
        set_seed(args.seed)
    root = BASE_DIR if args.workdir is None else args.workdir.resolve()
    if args.workdir is not None:
        prepare_workdir(root)
    os.chdir(root)

    if args.list_cases:
        print_case_table(cases)
        return 0
    if args.list_setups:
        print_setup_table()
        return 0

    selected = resolve_case_indices(args.case, len(cases), DEFAULT_CASES)
    setup_idx = resolve_setup_index(args.setup)
    required = [root / "data" / name for name in ("l63_train.npz", "l63_valid.npz", "l63_test.npz")]
    if args.data or (args.workdir is not None and not all(path.exists() for path in required)):
        generate_data(root)
    if not args.no_train:
        train(selected, setup_idx=setup_idx, resume_override=args.resume)
    if not args.no_plot:
        plot(selected)
    if not args.no_predict:
        predict(selected)
    if not args.no_show and (not args.no_plot or not args.no_predict):
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
