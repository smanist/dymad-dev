import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.cli_helpers import (
    add_common_cli_args,
    print_case_table,
    resolve_case_indices,
    set_seed,
    stage_workdir,
)

from dymad.io import load_model
from dymad.models import DLTI, DSDM, KBF, LDM
from dymad.training import NODETrainer, WeakFormTrainer
from dymad.utils import TrajectorySampler, plot_summary, plot_trajectory

BASE_DIR = Path(__file__).resolve().parent

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

cases = [
    {"name": "ldm_wf", "model": LDM, "trainer": WeakFormTrainer, "config": "ltd_ldm_wf.yaml"},
    {"name": "ldm_node", "model": LDM, "trainer": NODETrainer, "config": "ltd_ldm_node.yaml"},
    {"name": "kbf_wf", "model": KBF, "trainer": WeakFormTrainer, "config": "ltd_kbf_wf.yaml"},
    {"name": "kbf_node", "model": KBF, "trainer": NODETrainer, "config": "ltd_kbf_node.yaml"},
    {"name": "lti_node", "model": DLTI, "trainer": NODETrainer, "config": "ltd_lti_node.yaml"},
    {"name": "sdm_smp", "model": DSDM, "trainer": NODETrainer, "config": "ltd_sdm_smp.yaml"},
    {"name": "sdm_std", "model": DSDM, "trainer": NODETrainer, "config": "ltd_sdm_std.yaml"},
]
DEFAULT_CASES = list(range(len(cases)))


def parse_args():
    parser = argparse.ArgumentParser(description="Run delayed LTI cases.")
    add_common_cli_args(parser, include_data=True)
    return parser.parse_args()


def prepare_workdir(root: Path):
    stage_workdir(root, BASE_DIR, ["ltd_data.yaml"] + [case["config"] for case in cases])


def generate_data(root: Path, seed: int | None = None):
    sampler = TrajectorySampler(
        f,
        g,
        config=BASE_DIR / "ltd_data.yaml",
        rng=seed,
        config_mod=config_chr,
    )
    ts, xs, us, ys = sampler.sample(t_grid, batch=B)
    out_path = root / "data" / "ltd.npz"
    np.savez_compressed(out_path, t=ts, x=ys, u=us)
    print(f"Generated data: {out_path}")


def train(selected: list[int], root: Path, seed: int | None = None):
    for idx in selected:
        case = cases[idx]
        config_mod = {"prediction_diagnostic": {"sample_seed": seed}} if seed is not None else None
        trainer = case["trainer"](root / case["config"], case["model"], config_mod=config_mod)
        trainer.train()


def plot(selected: list[int]):
    labels = [cases[idx]["name"] for idx in selected]
    npz_files = [f"ltd_{label}" for label in labels]
    npzs = plot_summary(npz_files, labels=labels, ifclose=False)
    for label, npz in zip(labels, npzs, strict=False):
        print(f"Epoch time: {label} - {npz['avg_epoch_time']}")


def predict(selected: list[int], seed: int | None = None):
    sampler = TrajectorySampler(f, g, config="ltd_data.yaml", rng=seed, config_mod=config_gau)
    ts, xs, us, _ = sampler.sample(t_grid, batch=1)
    x_data = xs[0]
    t_data = ts[0]
    u_data = us[0]

    preds = []
    for idx in selected:
        case = cases[idx]
        _, prd_func = load_model(case["model"], f"ltd_{case['name']}.pt")
        with torch.no_grad():
            pred = prd_func(x_data, t_data, u=u_data)
        preds.append(pred)

    res = [x_data] + preds
    plot_trajectory(
        np.array(res),
        t_data,
        "LTI",
        us=u_data,
        labels=["Truth"] + [cases[idx]["name"] for idx in selected],
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
        print_case_table(cases)
        return 0

    selected = resolve_case_indices(args.case, len(cases), DEFAULT_CASES)
    data_path = root / "data" / "ltd.npz"
    if args.data or (args.workdir is not None and not data_path.exists()):
        generate_data(root, args.seed)
    if not args.no_train:
        train(selected, root, args.seed)
    if not args.no_plot:
        plot(selected)
    if not args.no_predict:
        predict(selected, args.seed)
    if not args.no_show and (not args.no_plot or not args.no_predict):
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
