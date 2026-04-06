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
from dymad.utils import adj_to_edge, plot_multi_trajs

from scripts.kuramoto.train import DSDMSKG, mdl_sdm, trn_nd


BASE_DIR = Path(__file__).resolve().parent
DATA_STEM = "data/data_n4_s5_k4_s5"

cases = [
    {"name": "sdm_skip", "model": DSDMSKG, "trainer": NODETrainer, "config": "kur_seq.yaml"},
]
DEFAULT_CASES = [0]


def parse_args():
    parser = argparse.ArgumentParser(description="Run Kuramoto sequence model cases.")
    add_common_cli_args(parser, include_data=True)
    return parser.parse_args()


def prepare_workdir(root: Path):
    stage_workdir(root, BASE_DIR, ["kur_seq.yaml", f"{DATA_STEM}_train.npz", f"{DATA_STEM}_test.npz"], data_dir=True)


def stage_data(root: Path):
    stage_workdir(root, BASE_DIR, [f"{DATA_STEM}_train.npz", f"{DATA_STEM}_test.npz"], data_dir=True)
    print(f"Staged data under: {root / 'data'}")


def train(selected: list[int], root: Path):
    for idx in selected:
        case = cases[idx]
        opt = {
            "data": {"path": str(root / f"{DATA_STEM}_train.npz")},
            "model": copy.deepcopy(mdl_sdm),
            "training": copy.deepcopy(trn_nd),
        }
        opt["model"]["name"] = case["name"]
        trainer = case["trainer"](root / case["config"], case["model"], config_mod=opt)
        trainer.train()


def predict(selected: list[int], root: Path):
    dat = np.load(root / f"{DATA_STEM}_test.npz", allow_pickle=True)
    x_data = dat["x"]
    t_data = np.arange(x_data.shape[1]) * 0.01
    u_data = dat["u"]
    ei_data, ew_data = adj_to_edge(dat["adj"])

    res = []
    labels = ["Truth"]
    time = None
    control = None
    for idx in selected:
        case = cases[idx]
        _, prd_func = load_model(case["model"], f"{case['name']}.pt")
        with torch.no_grad():
            pred = np.stack(
                [
                    prd_func(
                        x_data[j],
                        t_data,
                        u=u_data[j],
                        ei=ei_data[j],
                        ew=ew_data[j],
                    )
                    for j in range(len(x_data))
                ],
                axis=0,
            )
        if time is None:
            plot_len = min(x_data.shape[1], pred.shape[1])
            time = t_data[-plot_len:]
            control = u_data[:, -plot_len:]
            res.append(x_data[:, -plot_len:])
        pred = pred[:, -len(time):]
        res.append(pred)
        labels.append(case["name"])

    plot_multi_trajs(np.array(res), time, "KURA", us=control, labels=labels, ifclose=False, xidx=[0, 1, 2, 3, 4], uidx=[0])


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
    required = [root / f"{DATA_STEM}_train.npz", root / f"{DATA_STEM}_test.npz"]
    if args.data or (args.workdir is not None and not all(path.exists() for path in required)):
        stage_data(root)
    if not args.no_train:
        train(selected, root)
    if not args.no_predict:
        predict(selected, root)
    if not args.no_show and not args.no_predict:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
