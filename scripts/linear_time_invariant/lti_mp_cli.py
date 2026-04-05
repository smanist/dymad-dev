import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing as mp

from dymad.io import load_model
from dymad.models import KBF
from dymad.training import WeakFormTrainer
from dymad.utils import TrajectorySampler, plot_cv_results, plot_multi_trajs


BASE_DIR = Path(__file__).resolve().parent

B = 128
N = 501
t_grid = np.linspace(0, 5, N)

A = np.array([
            [0., 1.],
            [-1., -0.1]])


def f(t, x, u):
    return (x @ A.T) + u


g = lambda t, x, u: x

config_gau = {
    "control" : {
        "kind": "gaussian",
        "params": {
            "mean": 0.5,
            "std":  1.0,
            "t1":   4.0,
            "dt":   0.2,
            "mode": "zoh"}}}

cases = [
    {"name": "kbf_cv", "model": KBF, "trainer": WeakFormTrainer, "config": 'lti_kbf_cv.yaml', "max_workers": 4},
]
DEFAULT_CASES = [0]


def parse_args():
    parser = argparse.ArgumentParser(description="Run LTI multiprocessing/CV cases.")
    parser.add_argument(
        "--case",
        nargs="+",
        type=int,
        help="Case indices to run, for example '--case 0'.",
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="Print available case indices and exit.",
    )
    parser.add_argument("--no-train", action="store_true", help="Skip training.")
    parser.add_argument("--no-plot", action="store_true", help="Skip CV plotting.")
    parser.add_argument("--no-predict", action="store_true", help="Skip prediction plotting.")
    parser.add_argument("--no-show", action="store_true", help="Skip plt.show().")
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


def train(selected):
    mp.set_start_method("spawn", force=True)
    for idx in selected:
        Model = cases[idx]['model']
        Trainer = cases[idx]['trainer']
        config_path = cases[idx]['config']
        max_workers = cases[idx]['max_workers']

        trainer = Trainer(config_path, Model, max_workers=max_workers)
        trainer.train()


def plot(selected):
    if len(selected) != 1:
        raise ValueError("CV plotting expects a single selected case.")
    mdl = cases[selected[0]]['name']
    keys = ['model.koopman_dimension', 'training.weak_form_params.N']
    plot_cv_results(f'lti_{mdl}', keys, ifclose=False)


def predict(selected):
    sampler = TrajectorySampler(f, g, config='lti_data.yaml', config_mod=config_gau)
    ts, xs, us, ys = sampler.sample(t_grid, batch=3)

    res = [xs]
    for idx in selected:
        mdl = cases[idx]['name']
        MDL = cases[idx]['model']
        _, prd_func = load_model(MDL, f'lti_{mdl}.pt')

        with torch.no_grad():
            _pred = prd_func(xs, ts, u=us)
        res.append(_pred)

    plot_multi_trajs(
        np.array(res), ts[0], "LTI",
        us=us, labels=['Truth'] + [cases[idx]['name'] for idx in selected], ifclose=False)


def main():
    args = parse_args()
    os.chdir(BASE_DIR)

    if args.list_cases:
        print_cases()
        return 0

    selected = resolve_indices(args.case)

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
