import argparse
import os
import random
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from dymad.io import load_model
from dymad.models import DGKBF, DGKM, DGKMSK, DGLDM
from dymad.training import LinearTrainer, NODETrainer
from dymad.utils import TrajectorySampler, adj_to_edge, plot_summary, plot_trajectory

BASE_DIR = Path(__file__).resolve().parent

B = 128
N = 501
t_grid = np.linspace(0, 5, N)

A = np.array([[0.0, 1.0], [-1.0, -0.1]])


def f(t, x, u):
    return (x @ A.T) + u


def g(t, x, u):
    return x


adj = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])

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
        "params": {"mean": 0.5, "std": 1.0, "t1": 4.0, "dt": 0.2, "mode": "zoh"},
    }
}

cases = [
    {"name": "dldm", "model": DGLDM, "trainer": NODETrainer, "config": "ltg_dldm.yaml"},
    {"name": "dkbf", "model": DGKBF, "trainer": NODETrainer, "config": "ltg_dkbf.yaml"},
    {"name": "dkbl", "model": DGKBF, "trainer": LinearTrainer, "config": "ltg_dkbl.yaml"},
    {"name": "ltil", "model": DGKBF, "trainer": LinearTrainer, "config": "ltg_ltil.yaml"},
    {"name": "dkm", "model": DGKM, "trainer": LinearTrainer, "config": "ltg_dkm.yaml"},
    {"name": "dkmsk", "model": DGKMSK, "trainer": LinearTrainer, "config": "ltg_dkmsk.yaml"},
]
DEFAULT_CASES = [0, 1, 2, 3, 4, 5]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Run LTG discrete-time training cases.")
    parser.add_argument(
        "--case",
        nargs="+",
        type=int,
        help="Case indices to run, for example '--case 0 1 2'.",
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="Print available case indices and exit.",
    )
    parser.add_argument(
        "--data",
        action="store_true",
        help="Generate ./data/ltg.npz before other actions.",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        help="Run in a separate working directory and stage the needed YAML files there.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Set random seeds for reproducible runs.",
    )
    parser.add_argument("--no-train", action="store_true", help="Skip training.")
    parser.add_argument("--no-plot", action="store_true", help="Skip summary plotting.")
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


def prepare_workdir(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    (root / "data").mkdir(exist_ok=True)
    shutil.copy2(BASE_DIR / "ltg_data.yaml", root / "ltg_data.yaml")
    for case in cases:
        src = BASE_DIR / case["config"]
        dst = root / case["config"]
        if not dst.exists():
            shutil.copy2(src, dst)


def generate_data(root: Path, seed: int | None = None):
    (root / "data").mkdir(exist_ok=True)
    sampler = TrajectorySampler(
        f,
        g,
        config=BASE_DIR / "ltg_data.yaml",
        rng=seed,
        config_mod=config_chr,
    )
    ts, xs, us, ys = sampler.sample(t_grid, batch=B)
    edge_index = adj_to_edge(adj)[0]
    out_path = root / "data" / "ltg.npz"
    np.savez_compressed(
        out_path,
        t=ts,
        x=np.concatenate([ys, ys, ys], axis=-1),
        u=np.concatenate([us, us, us], axis=-1),
        ei=edge_index,
    )
    print(f"Generated data: {out_path}")
    return out_path


def normalize_legacy_data(data_path: Path) -> bool:
    if not data_path.exists():
        return False

    with np.load(data_path, allow_pickle=True) as data:
        if "ei" in data.files or "adj" not in data.files:
            return False
        payload = {key: data[key] for key in data.files if key != "adj"}
        edge_index = adj_to_edge(data["adj"])[0]

    # Use an explicit edge index to avoid expanding static adjacency weights across time.
    payload["ei"] = edge_index
    np.savez_compressed(data_path, **payload)
    print(f"Normalized legacy graph data: {data_path}")
    return True


def train(selected, root: Path, seed: int | None = None):
    for idx in selected:
        Model = cases[idx]["model"]
        Trainer = cases[idx]["trainer"]
        config_path = root / cases[idx]["config"]
        config_mod = {"prediction_diagnostic": {"sample_seed": seed}} if seed is not None else None
        trainer = Trainer(config_path, Model, config_mod=config_mod)
        trainer.train()


def plot(selected):
    labels = [cases[idx]["name"] for idx in selected]
    npz_files = [f"ltg_{label}" for label in labels]
    npzs = plot_summary(npz_files, labels=labels, ifclose=False)
    for label, npz in zip(labels, npzs, strict=False):
        print(f"Epoch time: {label} - {npz['avg_epoch_time']}")


def predict(selected, seed: int | None = None):
    sampler = TrajectorySampler(f, g, config="ltg_data.yaml", rng=seed, config_mod=config_gau)
    edge_index = adj_to_edge(adj)[0]

    ts, xs, us, ys = sampler.sample(t_grid, batch=1)
    x_data = np.concatenate([ys[0], ys[0], ys[0]], axis=-1)
    t_data = ts[0]
    u_data = np.concatenate([us[0], us[0], us[0]], axis=-1)

    res = [x_data]
    for idx in selected:
        mdl = cases[idx]["name"]
        MDL = cases[idx]["model"]
        _, prd_func = load_model(MDL, f"ltg_{mdl}.pt")

        with torch.no_grad():
            pred = prd_func(x_data, t_data, u=u_data, ei=edge_index)
        res.append(pred)

    plot_trajectory(
        np.array(res),
        t_data,
        "LTG",
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
        print_cases()
        return 0

    selected = resolve_indices(args.case)

    data_path = root / "data" / "ltg.npz"
    if args.data or (args.workdir is not None and not data_path.exists()):
        generate_data(root, args.seed)
    normalize_legacy_data(data_path)
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
