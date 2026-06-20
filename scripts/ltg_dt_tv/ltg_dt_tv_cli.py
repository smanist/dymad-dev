import argparse
import copy
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
from dymad.models import DLDMG
from dymad.training import NODETrainer
from dymad.utils import plot_summary, plot_trajectory

BASE_DIR = Path(__file__).resolve().parent

mdl_kb = {
    "name": "kura_model",
    "encoder_layers": 0,
    "decoder_layers": 0,
    "processor_layers": 1,
    "hidden_dimension": 2,
    "gcl": "gcnv",
    "gcl_opts": {"bias": False},
    "activation": "none",
    "weight_init": "xavier_uniform",
}

trn_nd = {
    "n_epochs": 1000,
    "save_interval": 50,
    "load_checkpoint": False,
    "learning_rate": 1e-2,
    "decay_rate": 0.999,
    "reconstruction_weight": 1.0,
    "dynamics_weight": 1.0,
    "sweep_lengths": [2, 4, 8],
    "sweep_epoch_step": 100,
    "chop_mode": "unfold",
    "chop_step": 0.5,
}

cases = [
    {
        "name": "dldmg",
        "model": DLDMG,
        "trainer": NODETrainer,
        "config": "config.yaml",
        "options": {"model": mdl_kb, "training": trn_nd},
        "train_data": "data/data_n2_s3_k4_s10.pkl",
        "predict_data": "data/data_n2_s3_k4_s20.pkl",
    }
]
DEFAULT_CASES = [0]


def parse_args():
    parser = argparse.ArgumentParser(description="Run LTG discrete-time time-varying graph cases.")
    add_common_cli_args(parser, include_data=True)
    return parser.parse_args()


def prepare_workdir(root: Path):
    stage_workdir(
        root,
        BASE_DIR,
        [
            "config.yaml",
            "data/data_n2_s3_k4_s10.pkl",
            "data/data_n2_s3_k4_s20.pkl",
        ],
    )


def stage_data(root: Path):
    stage_workdir(
        root,
        BASE_DIR,
        [
            "data/data_n2_s3_k4_s10.pkl",
            "data/data_n2_s3_k4_s20.pkl",
        ],
        data_dir=True,
    )
    print(f"Staged data under: {root / 'data'}")


def train(selected: list[int], root: Path, seed: int | None = None):
    for idx in selected:
        case = cases[idx]
        options = copy.deepcopy(case["options"])
        options["data"] = {"path": str(root / case["train_data"])}
        if seed is not None:
            options["data"]["split_seed"] = seed
        trainer = case["trainer"](root / case["config"], case["model"], config_mod=options)
        trainer.train()


def plot(selected: list[int]):
    labels = [cases[idx]["name"] for idx in selected]
    plot_summary(["kura_model"], labels=labels, ifclose=False)


def _load_case_data(path: Path):
    data = np.load(path, allow_pickle=True)
    return data.item() if isinstance(data, np.ndarray) and data.shape == () else data


def predict(selected: list[int], root: Path):
    data = _load_case_data(root / cases[0]["predict_data"])
    tdx = 10
    x_data = np.asarray(data["x"][tdx])
    t_data = np.arange(x_data.shape[0])
    ei_data = data["ei"][tdx]
    ew_data = data["ew"][tdx]

    res = [x_data]
    for idx in selected:
        case = cases[idx]
        _, prd_func = load_model(case["model"], "kura_model.pt")
        with torch.no_grad():
            pred = prd_func(x_data, t_data, ei=ei_data, ew=ew_data)
        res.append(pred)

    plot_trajectory(
        np.array(res),
        t_data,
        "LTGV",
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
    required = [root / cases[0]["train_data"], root / cases[0]["predict_data"]]
    if args.data or (args.workdir is not None and not all(path.exists() for path in required)):
        stage_data(root)
    if not args.no_train:
        train(selected, root, args.seed)
    if not args.no_plot:
        plot(selected)
    if not args.no_predict:
        predict(selected, root)
    if not args.no_show and (not args.no_plot or not args.no_predict):
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
