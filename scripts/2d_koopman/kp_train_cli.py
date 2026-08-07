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
from dymad.models import KBF, LDM
from dymad.training import LinearTrainer, NODETrainer, WeakFormTrainer
from dymad.utils import TrajectorySampler, plot_multi_trajs, plot_summary

BASE_DIR = Path(__file__).resolve().parent

B = 256
N = 301
t_grid = np.linspace(0, 6, N)

mu = -0.5
lm = -3


def f(t, x):
    return np.array([mu * x[0], lm * (x[1] - x[0] ** 2)])


mdl_kb = {
    "name": "kp_model",
    "encoder_layers": 2,
    "decoder_layers": 2,
    "hidden_dimension": 32,
    "koopman_dimension": 4,
    "autoencoder_type": "cat",
    "activation": "prelu",
    "weight_init": "xavier_uniform",
}
mdl_ld = {
    "name": "kp_model",
    "encoder_layers": 0,
    "processor_layers": 2,
    "decoder_layers": 0,
    "hidden_dimension": 32,
    "autoencoder_type": "smp",
    "activation": "prelu",
    "weight_init": "xavier_uniform",
}
mdl_kl = {
    "name": "kp_model",
    "encoder_layers": 1,
    "decoder_layers": 1,
    "hidden_dimension": 32,
    "koopman_dimension": 8,
    "autoencoder_type": "cat",
    "activation": "tanh",
    "weight_init": "xavier_uniform",
}

trn_wf = {
    "n_epochs": 2000,
    "save_interval": 50,
    "load_checkpoint": False,
    "learning_rate": 5e-3,
    "decay_rate": 0.999,
    "weak_form_params": {"N": 13, "dN": 2, "ordpol": 2, "ordint": 2},
}
trn_wf_kbf = {**trn_wf, "n_epochs": 3000, "decay_rate": 1.0}
trn_nd = {
    "n_epochs": 2000,
    "save_interval": 20,
    "load_checkpoint": False,
    "learning_rate": 5e-3,
    "decay_rate": 0.999,
    "sweep_lengths": [30, 50, 100, 200, 301],
    "sweep_epoch_step": 400,
    "ode_method": "dopri5",
    "ode_args": {"rtol": 1.0e-7, "atol": 1.0e-9},
}
trn_ln = {
    "n_epochs": 1,
    "save_interval": 1,
    "load_checkpoint": False,
    "learning_rate": 5e-3,
    "decay_rate": 0.999,
    "method": "truncated",
    "params": 8,
}
config_path = "kp_model.yaml"

cfgs = [
    ("ldm_wf", LDM, WeakFormTrainer, {"model": mdl_ld, "training": trn_wf}),
    ("ldm_node", LDM, NODETrainer, {"model": mdl_ld, "training": trn_nd}),
    ("kbf_wf", KBF, WeakFormTrainer, {"model": mdl_kb, "training": trn_wf_kbf}),
    ("kbf_node", KBF, NODETrainer, {"model": mdl_kb, "training": trn_nd}),
    ("kbf_ln", KBF, LinearTrainer, {"model": mdl_kl, "training": trn_ln}),
]
DEFAULT_CASES = [0, 1, 2, 3, 4]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Run 2D Koopman training cases.")
    parser.add_argument("--case", nargs="+", type=int)
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--data", action="store_true")
    parser.add_argument("--workdir", type=Path)
    parser.add_argument("--seed", type=int)
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
    shutil.copy2(BASE_DIR / "kp_data.yaml", root / "kp_data.yaml")
    shutil.copy2(BASE_DIR / "kp_model.yaml", root / "kp_model.yaml")


def generate_data(root: Path, seed: int | None = None):
    sampler = TrajectorySampler(f, config=BASE_DIR / "kp_data.yaml", rng=seed)
    sampler.sample(t_grid, batch=B, save=str(root / "data" / "kp.npz"))
    print(f"Generated data: {root / 'data' / 'kp.npz'}")


def train(selected, seed: int | None = None):
    for i in selected:
        mdl, MDL, Trainer, opt = cfgs[i]
        opt_local = copy.deepcopy(opt)
        opt_local["model"]["name"] = f"kp_{mdl}"
        if seed is not None:
            opt_local.setdefault("data", {})["split_seed"] = seed
        trainer = Trainer(config_path, MDL, config_mod=opt_local)
        trainer.train()


def plot(selected):
    labels = [cfgs[i][0] for i in selected]
    npz_files = [f"kp_{label}" for label in labels]
    plot_summary(npz_files, labels=labels, ifclose=False)


def predict(selected, seed: int | None = None):
    sampler = TrajectorySampler(f, config="kp_data.yaml", rng=seed)
    ts, xs, ys = sampler.sample(t_grid, batch=3)
    res = [xs]
    for i in selected:
        mdl, MDL, _, _ = cfgs[i]
        _, prd_func = load_model(MDL, f"kp_{mdl}.pt")
        with torch.no_grad():
            pred = prd_func(xs, ts)
        res.append(pred)
    plot_multi_trajs(
        np.array(res), ts[0], "KP", labels=["Truth"] + [cfgs[i][0] for i in selected], ifclose=False
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
    data_path = root / "data" / "kp.npz"
    if args.data or (args.workdir is not None and not data_path.exists()):
        generate_data(root, args.seed)
    if not args.no_train:
        train(selected, args.seed)
    if not args.no_plot:
        plot(selected)
    if not args.no_predict:
        predict(selected, args.seed)
    if not args.no_show and (not args.no_plot or not args.no_predict):
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
