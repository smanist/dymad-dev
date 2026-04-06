import argparse
import copy
import os
import random
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from dymad.io import load_model
from dymad.models import DKM, DKMSK, KM
from dymad.training import LinearTrainer, StackedTrainer
from dymad.utils import TrajectorySampler, plot_trajectory

BASE_DIR = Path(__file__).resolve().parent

B = 30
N = 41
t_grid = np.linspace(0, 2, N)

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

RIDGE = 1e-10
opt_rbf1 = {"type": "sc_rbf", "input_dim": 2, "lengthscale_init": 1.0}
opt_opk1 = {
    "type": "op_sep",
    "input_dim": 2,
    "output_dim": 2,
    "kopts": [opt_rbf1],
    "Ls": np.array([[[1, 0], [0, 1]]]),
}
opt_opval = {"type": "opval", "kernel": opt_opk1, "dtype": torch.float64, "ridge_init": RIDGE}
mdl_kl = {
    "name": "ker_model",
    "encoder_layers": 0,
    "decoder_layers": 0,
    "kernel_dimension": 2,
    **opt_opval,
}

trn_ln = {
    "n_epochs": 1,
    "save_interval": 100,
    "load_checkpoint": False,
    "learning_rate": 1e-2,
    "decay_rate": 0.999,
    "method": "raw",
}
trn_ct = {
    "n_epochs": 200,
    "save_interval": 50,
    "load_checkpoint": False,
    "learning_rate": 1e-2,
    "decay_rate": 0.999,
    "sweep_lengths": [4],
    "chop_mode": "initial",
    "chop_step": 0.5,
}
trn_dt = {
    "n_epochs": 400,
    "save_interval": 50,
    "load_checkpoint": False,
    "learning_rate": 1e-2,
    "decay_rate": 0.999,
}


def _optimizer_phase(trainer, cfg):
    phase = dict(cfg)
    phase["type"] = "optimizer"
    phase["trainer"] = trainer
    return phase


def _linear_solve_phase(method, params=None, *, kwargs=None, reset_optimizer=True):
    phase = {"type": "linear_solve", "method": method, "reset_optimizer": reset_optimizer}
    if params is not None:
        phase["params"] = params
    if kwargs is not None:
        phase["kwargs"] = kwargs
    return phase


def _alternating_schedule(
    trainer, base_cfg, chunk_epochs, *, method, params=None, kwargs=None, reset_optimizer=True
):
    phases = []
    for n_epochs in chunk_epochs:
        phases.append(
            _linear_solve_phase(
                method, params=params, kwargs=kwargs, reset_optimizer=reset_optimizer
            )
        )
        chunk_cfg = dict(base_cfg)
        chunk_cfg["n_epochs"] = n_epochs
        phases.append(_optimizer_phase(trainer, chunk_cfg))
    phases.append(
        _linear_solve_phase(method, params=params, kwargs=kwargs, reset_optimizer=reset_optimizer)
    )
    return phases


config_path = "ker_model.yaml"
cfgs = [
    ("km_ln", KM, LinearTrainer, {"model": mdl_kl, "training": trn_ln}),
    (
        "km_nd",
        KM,
        StackedTrainer,
        {
            "model": mdl_kl,
            "phases": _alternating_schedule(
                "NODE", trn_ct, [50, 50, 50, 50], method="raw", reset_optimizer=False
            ),
        },
    ),
    ("dkm_ln", DKM, LinearTrainer, {"model": mdl_kl, "training": trn_ln}),
    (
        "dkm_nd",
        DKM,
        StackedTrainer,
        {
            "model": mdl_kl,
            "phases": _alternating_schedule(
                "NODE", trn_dt, [100, 100, 100, 100], method="raw", reset_optimizer=False
            ),
        },
    ),
    ("dks_ln", DKMSK, LinearTrainer, {"model": mdl_kl, "training": trn_ln}),
    (
        "dks_nd",
        DKMSK,
        StackedTrainer,
        {
            "model": mdl_kl,
            "phases": _alternating_schedule(
                "NODE", trn_dt, [100, 100, 100, 100], method="raw", reset_optimizer=False
            ),
        },
    ),
]
DEFAULT_CASES = [0, 1, 2, 3, 4, 5]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Run kernel LTI cases.")
    parser.add_argument("--case", nargs="+", type=int, help="Case indices to run.")
    parser.add_argument(
        "--list-cases", action="store_true", help="Print available case indices and exit."
    )
    parser.add_argument(
        "--data", action="store_true", help="Generate ./data/ker.npz before other actions."
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        help="Run in a separate working directory and stage the needed files there.",
    )
    parser.add_argument("--seed", type=int, help="Set random seeds for reproducible runs.")
    parser.add_argument("--no-train", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--no-predict", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def resolve_indices(values):
    indices = DEFAULT_CASES if values is None else values
    invalid = [idx for idx in indices if idx < 0 or idx >= len(cfgs)]
    if invalid:
        raise ValueError(f"Invalid case indices: {invalid}")
    return indices


def print_cases():
    for idx, (name, _, _, _) in enumerate(cfgs):
        print(f"{idx}: {name}")


def prepare_workdir(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    (root / "data").mkdir(exist_ok=True)
    shutil.copy2(BASE_DIR / "ker_data.yaml", root / "ker_data.yaml")
    shutil.copy2(BASE_DIR / "ker_model.yaml", root / "ker_model.yaml")


def generate_data(root: Path):
    sampler = TrajectorySampler(f, g, config=BASE_DIR / "ker_data.yaml", config_mod=config_chr)
    sampler.sample(t_grid, batch=B, save=str(root / "data" / "ker.npz"))
    print(f"Generated data: {root / 'data' / 'ker.npz'}")


def train(selected):
    for i in selected:
        mdl, MDL, Trainer, opt = cfgs[i]
        opt_local = copy.deepcopy(opt)
        opt_local["model"]["name"] = f"ker_{mdl}"
        trainer = Trainer(config_path, MDL, config_mod=opt_local)
        trainer.train()


def predict(selected):
    sampler = TrajectorySampler(f, g, config="ker_data.yaml", config_mod=config_chr)
    ts, xs, us, ys = sampler.sample(t_grid, batch=1)
    x_data = xs[0]
    t_data = ts[0]
    u_data = us[0]
    res = [x_data]
    for i in selected:
        mdl, MDL, _, _ = cfgs[i]
        _, prd_func = load_model(MDL, f"ker_{mdl}.pt")
        with torch.no_grad():
            pred = prd_func(x_data, t_data, u=u_data)
        res.append(pred)
    plot_trajectory(
        np.array(res),
        t_data,
        "LTI",
        us=u_data,
        labels=["Truth"] + [cfgs[i][0] for i in selected],
        ifclose=False,
    )


def main():
    args = parse_args()
    if args.seed is not None:
        set_seed(args.seed)
    root = BASE_DIR if args.workdir is None else args.workdir.resolve()
    if args.workdir is not None:
        prepare_workdir(root)
    os.chdir(root)
    if args.list_cases:
        print_cases()
        return 0
    selected = resolve_indices(args.case)
    data_path = root / "data" / "ker.npz"
    if args.data or (args.workdir is not None and not data_path.exists()):
        generate_data(root)
    if not args.no_train:
        train(selected)
    if not args.no_predict:
        predict(selected)
    if not args.no_show and not args.no_predict:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
