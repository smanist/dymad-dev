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
from dymad.models import LDM
from dymad.training import NODETrainer, WeakFormTrainer
from dymad.utils import TrajectorySampler, plot_summary, plot_trajectory

BASE_DIR = Path(__file__).resolve().parent

# Keep a shared sampling interval across trajectories; the weak-form path currently
# builds one set of integration weights from dataset metadata.
DT = 0.05
LENGTHS = [41, 61, 81, 101]
TRAJ_PER_LENGTH = 8

A = np.array([[0.0, 1.0], [-1.0, -0.1]])


def f(t, x, u):
    return (x @ A.T) + u


def g(t, x, u):
    return x


def _to_object_array(items):
    data = np.empty(len(items), dtype=object)
    for index, item in enumerate(items):
        data[index] = np.asarray(item)
    return data


cases = [
    {
        "name": "ldm_node",
        "label": "NODE",
        "model": LDM,
        "trainer": NODETrainer,
        "config": "lti_vlen_ldm_node.yaml",
    },
    {
        "name": "ldm_wf",
        "label": "Weak form",
        "model": LDM,
        "trainer": WeakFormTrainer,
        "config": "lti_vlen_ldm_wf.yaml",
    },
]
DEFAULT_CASES = list(range(len(cases)))


def parse_args():
    parser = argparse.ArgumentParser(description="Run variable-length LTI cases.")
    add_common_cli_args(parser, include_data=True)
    return parser.parse_args()


def prepare_workdir(root: Path):
    stage_workdir(
        root,
        BASE_DIR,
        ["lti_vlen_data.yaml"] + [case["config"] for case in cases],
    )


def generate_data(root: Path, seed: int | None = None):
    sampler = TrajectorySampler(
        f,
        g,
        config=root / "lti_vlen_data.yaml",
        rng=seed,
    )

    ts_all = []
    ys_all = []
    us_all = []
    for n_steps in LENGTHS:
        t_grid = DT * np.arange(n_steps)
        ts, _, us, ys = sampler.sample(t_grid, batch=TRAJ_PER_LENGTH)
        ts_all.extend(np.array(item) for item in ts)
        ys_all.extend(np.array(item) for item in ys)
        us_all.extend(np.array(item) for item in us)

    out_path = root / "data" / "lti_vlen.npz"
    np.savez_compressed(
        out_path,
        t=_to_object_array(ts_all),
        x=_to_object_array(ys_all),
        u=_to_object_array(us_all),
    )
    print(f"Generated data: {out_path}")


def train(selected: list[int], root: Path, seed: int | None = None):
    for idx in selected:
        case = cases[idx]
        config_mod = {"prediction_diagnostic": {"sample_seed": seed}} if seed is not None else None
        trainer = case["trainer"](root / case["config"], case["model"], config_mod=config_mod)
        trainer.train()


def plot(selected: list[int]):
    labels = [cases[idx]["label"] for idx in selected]
    npz_files = [f"lti_vlen_{cases[idx]['name']}" for idx in selected]
    plot_summary(npz_files, labels=labels, ifclose=False)


def predict(selected: list[int], seed: int | None = None):
    sampler = TrajectorySampler(f, g, config="lti_vlen_data.yaml", rng=seed)

    t_grid = DT * np.arange(LENGTHS[-1])
    ts, xs, us, _ = sampler.sample(t_grid, batch=1)
    x_data = xs[0]
    t_data = ts[0]
    u_data = us[0]

    res = [x_data]
    for idx in selected:
        case = cases[idx]
        _, predict_fn = load_model(case["model"], f"lti_vlen_{case['name']}.pt")
        with torch.no_grad():
            pred = predict_fn(x_data, t_data, u=u_data)
        res.append(pred)

    plot_trajectory(
        np.array(res),
        t_data,
        "lti_vlen",
        us=u_data,
        labels=["Truth"] + [cases[idx]["label"] for idx in selected],
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
    data_path = root / "data" / "lti_vlen.npz"
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
