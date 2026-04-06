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
from dymad.models import DKBF
from dymad.training import NODETrainer
from dymad.utils import TrajectorySampler, plot_summary, plot_trajectory

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
    "activation": "tanh",
    "weight_init": "xavier_uniform",
}
trn_ref = {
    "n_epochs": 2000,
    "save_interval": 20,
    "load_checkpoint": False,
    "learning_rate": 5e-3,
    "decay_rate": 0.999,
    "sweep_epoch_step": 100,
    "sweep_lengths": [2, 4, 6, 8],
    "chop_mode": "unfold",
    "chop_step": 0.5,
}
trn_nd1 = dict(trn_ref)
trn_nd2 = {"sweep_tols": [1e-1, 1e-2, 1e-3], **trn_ref}
trn_nd3 = {"sweep_tols": [1e-1, 1e-2, 1e-3], "sweep_mode": "full", **trn_ref}
trn_opts = [trn_nd1, trn_nd2, trn_nd3]
DEFAULT_CASES = [0, 1, 2]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Run 2D Koopman discrete sweep cases.")
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
    invalid = [idx for idx in indices if idx < 0 or idx >= len(trn_opts)]
    if invalid:
        raise ValueError(f"Invalid case indices: {invalid}")
    return indices


def print_cases():
    for idx in DEFAULT_CASES:
        print(f"{idx}: dt{idx + 1}")


def prepare_workdir(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    (root / "data").mkdir(exist_ok=True)
    shutil.copy2(BASE_DIR / "kp_data.yaml", root / "kp_data.yaml")
    shutil.copy2(BASE_DIR / "kp_model.yaml", root / "kp_model.yaml")


def generate_data(root: Path):
    sampler = TrajectorySampler(f, config=BASE_DIR / "kp_data.yaml")
    sampler.sample(t_grid, batch=B, save=str(root / "data" / "kp.npz"))
    print(f"Generated data: {root / 'data' / 'kp.npz'}")


def train(selected):
    for i in selected:
        opt = {"model": copy.deepcopy(mdl_kb), "training": copy.deepcopy(trn_opts[i])}
        opt["model"]["name"] = f"kp_dt{i + 1}"
        trainer = NODETrainer("kp_model.yaml", DKBF, config_mod=opt)
        trainer.train()


def plot(selected):
    labels = [f"dt{i + 1}" for i in selected]
    npz_files = [f"kp_{label}" for label in labels]
    plot_summary(npz_files, labels=labels, ifscl=False, ifclose=False)


def predict(selected):
    sampler = TrajectorySampler(f, config="kp_data.yaml")
    ts, xs, ys = sampler.sample(t_grid, batch=1)
    x_data = xs[0]
    t_data = ts[0]
    res = [x_data]
    for i in selected:
        _, prd_func = load_model(DKBF, f"kp_dt{i + 1}.pt")
        with torch.no_grad():
            pred = prd_func(x_data, t_data)
        res.append(pred)
    plot_trajectory(
        np.array(res),
        t_data,
        "KP",
        labels=["Truth"] + [f"dt{i + 1}" for i in selected],
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
    data_path = root / "data" / "kp.npz"
    if args.data or (args.workdir is not None and not data_path.exists()):
        generate_data(root)
    if not args.no_train:
        train(selected)
    if not args.no_plot:
        plot(selected)
    if not args.no_predict:
        predict(selected)
    if not args.no_show and (not args.no_plot or not args.no_predict):
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
