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
from dymad.training import NODETrainer
from dymad.utils import TrajectorySampler, plot_multi_trajs, plot_summary

from scripts.pirom_dyn.dyn_train import DPT, DPJ, f, g, mdl_kl, t_grid, trn_nd


BASE_DIR = Path(__file__).resolve().parent

cases = [
    {"name": "dp_nd", "model": DPT, "trainer": NODETrainer, "config": "dyn_model.yaml"},
    {"name": "dj_nd", "model": DPJ, "trainer": NODETrainer, "config": "dyn_model.yaml"},
]
DEFAULT_CASES = [0, 1]


def parse_args():
    parser = argparse.ArgumentParser(description="Run PI-ROM dynamics correction cases.")
    add_common_cli_args(parser, include_data=False)
    return parser.parse_args()


def prepare_workdir(root: Path):
    stage_workdir(root, BASE_DIR, ["dyn_model.yaml", "dyn_test.yaml", "data/dyn.npz"], data_dir=True)


def train(selected: list[int], root: Path):
    for idx in selected:
        case = cases[idx]
        opt = {"data": {"path": str(root / "data" / "dyn.npz")}, "model": copy.deepcopy(mdl_kl), "training": copy.deepcopy(trn_nd)}
        opt["model"]["name"] = f"dyn_{case['name']}"
        trainer = case["trainer"](root / case["config"], case["model"], config_mod=opt)
        trainer.train()


def plot(selected: list[int]):
    labels = [cases[idx]["name"] for idx in selected]
    npz_files = [f"dyn_{label}" for label in labels]
    npzs = plot_summary(npz_files, labels=labels, ifclose=False)
    for label, npz in zip(labels, npzs):
        print(f"Epoch time {label}: {npz['avg_epoch_time']}")


def predict(selected: list[int]):
    sampler = TrajectorySampler(f, g, config="dyn_test.yaml")
    ts, xs, us, ys, ps = sampler.sample(t_grid, batch=5)
    x_data = ys
    u_data = us
    t_data = ts[0]

    res = [x_data]
    for idx in selected:
        case = cases[idx]
        _, prd_func = load_model(case["model"], f"dyn_{case['name']}.pt")
        with torch.no_grad():
            pred = np.stack(
                [prd_func(x_data[j], t_data, u=u_data[j], p=ps[j]) for j in range(len(x_data))],
                axis=0,
            )
        res.append(pred)

    plot_multi_trajs(np.array(res), t_data, "DP", us=u_data, labels=["Truth"] + [cases[idx]["name"] for idx in selected], ifclose=False)


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
