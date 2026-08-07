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
from dymad.models import GKBF, GLDM
from dymad.training import LinearTrainer, NODETrainer, WeakFormTrainer
from dymad.utils import TrajectorySampler, adj_to_edge, plot_summary, plot_trajectory

BASE_DIR = Path(__file__).resolve().parent

B = 128
N = 501
t_grid = np.linspace(0, 5, N)

A = np.array([[0.0, 1.0], [-1.0, -0.1]])


def f(t, x):
    return x @ A.T


def g(t, x):
    return x


adj = np.array(
    [
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
    ]
)

mdl_kb = {
    "encoder_layers": 1,
    "decoder_layers": 1,
    "hidden_dimension": 32,
    "koopman_dimension": 4,
    "activation": "none",
    "gcl": "sage",
    "weight_init": "xavier_uniform",
}
mdl_ld = {
    "encoder_layers": 1,
    "processor_layers": 1,
    "decoder_layers": 1,
    "hidden_dimension": 32,
    "activation": "none",
    "gcl": "sage",
    "weight_init": "xavier_uniform",
}
mdl_kl = {
    "encoder_layers": 1,
    "decoder_layers": 1,
    "hidden_dimension": 32,
    "koopman_dimension": 4,
    "activation": "none",
    "autoencoder_type": "cat",
    "gcl": "sage",
    "weight_init": "xavier_uniform",
}

trn_wf = {
    "n_epochs": 500,
    "save_interval": 10,
    "load_checkpoint": False,
    "learning_rate": 1e-2,
    "decay_rate": 0.999,
    "weak_form_params": {"N": 13, "dN": 2, "ordpol": 2, "ordint": 2},
}
trn_wf_kbf = {**trn_wf, "n_epochs": 1400, "decay_rate": 1.0}
trn_nd = {
    "n_epochs": 500,
    "save_interval": 10,
    "load_checkpoint": False,
    "learning_rate": 1e-2,
    "decay_rate": 0.999,
    "sweep_lengths": [2, 5, 10, 50, 100],
    "sweep_epoch_step": 100,
    "ode_method": "dopri5",
    "ode_args": {"rtol": 1e-7, "atol": 1e-9},
}
trn_ln = {
    "n_epochs": 1,
    "save_interval": 1,
    "load_checkpoint": False,
    "learning_rate": 1e-2,
    "decay_rate": 0.999,
    "method": "truncated",
    "params": 2,
}

cases = [
    {
        "name": "ldm_wf",
        "model": GLDM,
        "trainer": WeakFormTrainer,
        "config_mod": {"model": mdl_ld, "training": trn_wf},
    },
    {
        "name": "ldm_node",
        "model": GLDM,
        "trainer": NODETrainer,
        "config_mod": {"model": mdl_ld, "training": trn_nd},
    },
    {
        "name": "kbf_wf",
        "model": GKBF,
        "trainer": WeakFormTrainer,
        "config_mod": {"model": mdl_kb, "training": trn_wf_kbf},
    },
    {
        "name": "kbf_node",
        "model": GKBF,
        "trainer": NODETrainer,
        "config_mod": {"model": mdl_kb, "training": trn_nd},
    },
    {
        "name": "kbf_ln",
        "model": GKBF,
        "trainer": LinearTrainer,
        "config_mod": {"model": mdl_kl, "training": trn_ln},
    },
]
DEFAULT_CASES = list(range(len(cases)))


def parse_args():
    parser = argparse.ArgumentParser(description="Run linear graph auto training cases.")
    add_common_cli_args(parser, include_data=True)
    return parser.parse_args()


def prepare_workdir(root: Path):
    stage_workdir(root, BASE_DIR, ["ltga_data.yaml", "ltga_model.yaml"])


def generate_data(root: Path, seed: int | None = None):
    sampler = TrajectorySampler(f, g, config=BASE_DIR / "ltga_data.yaml", rng=seed)
    ts, xs, ys = sampler.sample(t_grid, batch=B)
    out_path = root / "data" / "ltga.npz"
    np.savez_compressed(
        out_path,
        t=ts,
        x=np.concatenate([ys, ys, ys], axis=-1),
        adj=adj,
    )
    print(f"Generated data: {out_path}")


def train(selected: list[int], root: Path, seed: int | None = None):
    config_path = root / "ltga_model.yaml"
    for idx in selected:
        case = cases[idx]
        config_mod = copy.deepcopy(case["config_mod"])
        config_mod["model"]["name"] = f"ltga_{case['name']}"
        if seed is not None:
            config_mod.setdefault("data", {})["split_seed"] = seed
        trainer = case["trainer"](config_path, case["model"], config_mod=config_mod)
        trainer.train()


def plot(selected: list[int]):
    labels = [cases[idx]["name"] for idx in selected]
    npz_files = [f"ltga_{label}" for label in labels]
    npzs = plot_summary(npz_files, labels=labels, ifclose=False)
    for label, npz in zip(labels, npzs, strict=False):
        print(f"Epoch time: {label} - {npz['avg_epoch_time']}")


def predict(selected: list[int], seed: int | None = None):
    sampler = TrajectorySampler(f, config="ltga_data.yaml", rng=seed)
    ts, xs, ys = sampler.sample(t_grid, batch=1)
    x_data = np.concatenate([xs[0], xs[0], xs[0]], axis=-1)
    t_data = ts[0]
    edge_index = adj_to_edge(adj)[0]

    res = [x_data]
    for idx in selected:
        case = cases[idx]
        _, predict_fn = load_model(case["model"], f"ltga_{case['name']}.pt")
        with torch.no_grad():
            pred = predict_fn(x_data, t_data, ei=edge_index)
        res.append(pred)

    plot_trajectory(
        np.array(res),
        t_data,
        "LTGA",
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
    data_path = root / "data" / "ltga.npz"
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
