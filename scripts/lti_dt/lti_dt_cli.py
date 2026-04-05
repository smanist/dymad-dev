import argparse
import os
from pathlib import Path
import random
import shutil

import matplotlib.pyplot as plt
import numpy as np
import torch

from dymad.io import load_model
from dymad.models import DKBF, DLDM, DLTI
from dymad.training import LinearTrainer, NODETrainer
from dymad.utils import TrajectorySampler, plot_summary, plot_trajectory


BASE_DIR = Path(__file__).resolve().parent

B = 128
N = 501
t_grid = np.linspace(0, 5, N)

A = np.array([[0., 1.], [-1., -0.1]])


def f(t, x, u):
    return (x @ A.T) + u


g = lambda t, x, u: x

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

cases = [
    {"name": "dldm", "model": DLDM, "trainer": NODETrainer, "config": "lti_dldm.yaml"},
    {"name": "dkbf", "model": DKBF, "trainer": NODETrainer, "config": "lti_dkbf.yaml"},
    {"name": "dkbl", "model": DKBF, "trainer": LinearTrainer, "config": "lti_dkbl.yaml"},
    {"name": "ltil", "model": DLTI, "trainer": LinearTrainer, "config": "lti_ltil.yaml"},
]
DEFAULT_CASES = [0, 1, 2, 3]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Run LTI discrete-time cases.")
    parser.add_argument("--case", nargs="+", type=int, help="Case indices to run.")
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
    invalid = [idx for idx in indices if idx < 0 or idx >= len(cases)]
    if invalid:
        raise ValueError(f"Invalid case indices: {invalid}")
    return indices


def print_cases():
    for idx, case in enumerate(cases):
        print(f"{idx}: {case['name']} [{case['config']}]")


def prepare_workdir(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    (root / "data").mkdir(exist_ok=True)
    shutil.copy2(BASE_DIR / "lti_data.yaml", root / "lti_data.yaml")
    for case in cases:
        src = BASE_DIR / case["config"]
        dst = root / case["config"]
        if not dst.exists():
            shutil.copy2(src, dst)


def generate_data(root: Path):
    sampler = TrajectorySampler(f, g, config=BASE_DIR / "lti_data.yaml", config_mod=config_chr)
    ts, xs, us, ys = sampler.sample(t_grid, batch=B)
    out_path = root / "data" / "lti.npz"
    np.savez_compressed(out_path, t=ts, x=ys, u=us)
    print(f"Generated data: {out_path}")


def train(selected, root: Path):
    for idx in selected:
        trainer = cases[idx]["trainer"](root / cases[idx]["config"], cases[idx]["model"])
        trainer.train()


def plot(selected):
    labels = [cases[idx]["name"] for idx in selected]
    npz_files = [f"lti_{label}" for label in labels]
    plot_summary(npz_files, labels=labels, ifclose=False)


def predict(selected):
    sampler = TrajectorySampler(f, g, config="lti_data.yaml", config_mod=config_gau)
    ts, xs, us, ys = sampler.sample(t_grid, batch=1)
    x_data = xs[0]
    t_data = ts[0]
    u_data = us[0]
    res = [x_data]
    for idx in selected:
        mdl = cases[idx]["name"]
        MDL = cases[idx]["model"]
        _, prd_func = load_model(MDL, f"lti_{mdl}.pt")
        with torch.no_grad():
            pred = prd_func(x_data, t_data, u=u_data)
        res.append(pred)
    plot_trajectory(np.array(res), t_data, "LTI", us=u_data, labels=["Truth"] + [cases[idx]["name"] for idx in selected], ifclose=False)


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
    data_path = root / "data" / "lti.npz"
    if args.data or (args.workdir is not None and not data_path.exists()):
        generate_data(root)
    if not args.no_train:
        train(selected, root)
    if not args.no_plot:
        plot(selected)
    if not args.no_predict:
        predict(selected)
    if not args.no_show and (not args.no_plot or not args.no_predict):
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
