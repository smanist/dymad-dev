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
from dymad.models import DLDM, LDM
from dymad.training import OneStepTrainer, StackedTrainer
from dymad.utils import TrajectorySampler, plot_summary, plot_trajectory

BASE_DIR = Path(__file__).resolve().parent
DATA_CONFIG_NAME = "lti_1s_data.yaml"
CT_CONFIG_NAME = "lti_1s_ct.yaml"
DT_CONFIG_NAME = "lti_1s_dt.yaml"
DEFAULT_CASES = [0, 1, 2, 3]

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

mdl_ct = {
    "encoder_layers": 0,
    "processor_layers": 1,
    "decoder_layers": 0,
    "hidden_dimension": 32,
    "activation": "none",
    "weight_init": "xavier_uniform",
    "gain": 0.01,
    "input_order": "cubic",
}

mdl_dt = {
    "encoder_layers": 0,
    "processor_layers": 1,
    "decoder_layers": 0,
    "hidden_dimension": 32,
    "activation": "none",
    "weight_init": "xavier_uniform",
    "gain": 0.01,
}

trn_step = {
    "n_epochs": 2000,
    "save_interval": 50,
    "load_checkpoint": False,
    "learning_rate": 1e-3,
    "decay_rate": 0.999,
}
trn_step_warm = copy.deepcopy(trn_step)
trn_step_warm["n_epochs"] = 1000

trn_weak = {
    "n_epochs": 500,
    "save_interval": 50,
    "load_checkpoint": False,
    "learning_rate": 5e-3,
    "decay_rate": 0.999,
    "weak_form_params": {
        "N": 13,
        "dN": 2,
        "ordpol": 2,
        "ordint": 2,
    },
}

trn_dt_node = {
    "n_epochs": 500,
    "save_interval": 50,
    "load_checkpoint": False,
    "learning_rate": 1e-3,
    "decay_rate": 0.999,
    "chop_mode": "unfold",
    "chop_step": 0.5,
    "sweep_lengths": [3],
    "sweep_epoch_step": 500,
}


def _optimizer_phase(trainer, cfg):
    phase = copy.deepcopy(cfg)
    phase.update({"type": "optimizer", "trainer": trainer})
    return phase


cases = [
    {
        "label": "ct_step",
        "artifact_name": "lti_1s_ct_step",
        "config": CT_CONFIG_NAME,
        "model": LDM,
        "trainer": OneStepTrainer,
        "config_mod": {
            "model": {"name": "lti_1s_ct_step", **mdl_ct},
            "training": trn_step,
        },
    },
    {
        "label": "ct_step_wf",
        "artifact_name": "lti_1s_ct_step_wf",
        "config": CT_CONFIG_NAME,
        "model": LDM,
        "trainer": StackedTrainer,
        "config_mod": {
            "model": {"name": "lti_1s_ct_step_wf", **mdl_ct},
            "training": None,
            "phases": [
                _optimizer_phase("OneStep", trn_step_warm),
                _optimizer_phase("Weak", {**trn_weak, "reset_optimizer": True}),
            ],
        },
    },
    {
        "label": "dt_step",
        "artifact_name": "lti_1s_dt_step",
        "config": DT_CONFIG_NAME,
        "model": DLDM,
        "trainer": OneStepTrainer,
        "config_mod": {
            "model": {"name": "lti_1s_dt_step", **mdl_dt},
            "training": trn_step,
        },
    },
    {
        "label": "dt_step_node",
        "artifact_name": "lti_1s_dt_step_node",
        "config": DT_CONFIG_NAME,
        "model": DLDM,
        "trainer": StackedTrainer,
        "config_mod": {
            "model": {"name": "lti_1s_dt_step_node", **mdl_dt},
            "training": None,
            "phases": [
                _optimizer_phase("OneStep", trn_step_warm),
                _optimizer_phase("NODE", {**trn_dt_node, "reset_optimizer": True}),
            ],
        },
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run LTI one-step comparison cases.")
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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_indices(values):
    indices = DEFAULT_CASES if values is None else values
    invalid = [idx for idx in indices if idx < 0 or idx >= len(cases)]
    if invalid:
        raise ValueError(f"Invalid case indices: {invalid}")
    return indices


def print_cases():
    for idx, case in enumerate(cases):
        print(f"{idx}: {case['label']} [{case['config']}]")


def prepare_workdir(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    (root / "data").mkdir(exist_ok=True)
    for name in (DATA_CONFIG_NAME, CT_CONFIG_NAME, DT_CONFIG_NAME):
        src = BASE_DIR / name
        dst = root / name
        if not dst.exists():
            shutil.copy2(src, dst)


def generate_data(root: Path = BASE_DIR, seed: int | None = None):
    sampler = TrajectorySampler(
        f, g, config=root / DATA_CONFIG_NAME, rng=seed, config_mod=config_chr
    )
    ts, _xs, us, ys = sampler.sample(t_grid, batch=B)
    data_path = root / "data" / "lti_1s.npz"
    data_path.parent.mkdir(exist_ok=True)
    np.savez_compressed(data_path, t=ts, x=ys, u=us)
    print(f"Generated data: {data_path}")


def train(selected, root: Path = BASE_DIR):
    for idx in selected:
        case = cases[idx]
        trainer = case["trainer"](
            root / case["config"],
            case["model"],
            config_mod=copy.deepcopy(case["config_mod"]),
        )
        trainer.train()


def plot(selected):
    labels = [cases[idx]["label"] for idx in selected]
    npz_files = [cases[idx]["artifact_name"] for idx in selected]
    npzs = plot_summary(npz_files, labels=labels, ifclose=False)
    for label, npz in zip(labels, npzs, strict=False):
        print(f"Epoch time {label}: {npz['avg_epoch_time']}")


def predict(selected, root: Path = BASE_DIR, seed: int | None = None):
    sampler = TrajectorySampler(
        f, g, config=root / DATA_CONFIG_NAME, rng=seed, config_mod=config_gau
    )
    ts, xs, us, ys = sampler.sample(t_grid, batch=1)
    x_data = xs[0]
    t_data = ts[0]
    u_data = us[0]

    res = [x_data]
    labels = [cases[idx]["label"] for idx in selected]
    for idx in selected:
        case = cases[idx]
        _, prd_func = load_model(case["model"], f"{case['artifact_name']}.pt")
        with torch.no_grad():
            pred = prd_func(x_data, t_data, u=u_data)
        res.append(pred)

    plot_trajectory(
        np.array(res),
        t_data,
        "LTI",
        us=u_data,
        labels=["Truth"] + labels,
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
    data_path = root / "data" / "lti_1s.npz"
    if args.data or (args.workdir is not None and not data_path.exists()):
        generate_data(root, args.seed)
    if not args.no_train:
        train(selected, root)
    if not args.no_plot:
        plot(selected)
    if not args.no_predict:
        predict(selected, root, args.seed)
    if not args.no_show and (not args.no_plot or not args.no_predict):
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
