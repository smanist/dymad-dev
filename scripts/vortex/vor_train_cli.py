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
from dymad.models import DKBF, DKMSK, KBF
from dymad.training import LinearTrainer, StackedTrainer
from dymad.utils import animate, compare_contour, plot_summary, setup_logging

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "vor_model.yaml"
DEFAULT_DATA_PATH = BASE_DIR / "data" / "cylinder.npz"
DEFAULT_CASES = [0, 1, 2, 3, 4]
NX = 199
NY = 449


def gen_mdl_kb(e, l, k):
    return {
        "name": "vor_model",
        "encoder_layers": e,
        "decoder_layers": e,
        "hidden_dimension": l,
        "koopman_dimension": k,
        "activation": "prelu",
        "weight_init": "xavier_uniform",
        "predictor_type": "exp",
    }


mdl_kl = {
    "name": "ker_model",
    "encoder_layers": 0,
    "decoder_layers": 0,
    "kernel_dimension": 12,
    "type": "share",
    "kernel": {"type": "sc_dm", "input_dim": 12, "eps_init": None},
    "dtype": torch.float64,
    "ridge_init": 1e-10,
}

crit = {
    "dynamics": {"weight": 1.0},
    "recon": {"weight": 1.0},
}
trn_nd = {
    "n_epochs": 200,
    "save_interval": 50,
    "load_checkpoint": False,
    "learning_rate": 5e-3,
    "decay_rate": 0.999,
    "sweep_lengths": [2],
    "sweep_epoch_step": 100,
    "chop_mode": "unfold",
    "chop_step": 1,
    "ode_method": "dopri5",
    "ode_args": {"rtol": 1.0e-7, "atol": 1.0e-9},
}
trn_ae = copy.deepcopy(trn_nd)
trn_ae["n_epochs"] = 2000
trn_ae["sweep_lengths"] = [2, 10, 50]
trn_ae["sweep_epoch_step"] = 500

trn_ln = {
    "n_epochs": 1,
    "save_interval": 50,
    "load_checkpoint": False,
    "method": "full",
}
trn_rw = copy.deepcopy(trn_ln)
trn_rw["method"] = "raw"

trn_svd = {"type": "svd", "ifcen": True, "order": 12}
trn_scl = {
    "type": "scaler",
    "mode": "std",
}
trn_add = {"type": "add_one"}
trn_dmf = {
    "type": "dm",
    "edim": 3,
    "Knn": 15,
    "Kphi": 3,
    "inverse": "gmls",
    "order": 1,
    "mode": "full",
}


def _optimizer_phase(trainer, cfg):
    phase = dict(cfg)
    phase["type"] = "optimizer"
    phase["trainer"] = trainer
    return phase


def _linear_solve_phase(method, params=None, *, kwargs=None, reset_optimizer=True):
    phase = {"type": "linear_solve", "method": method, "reset_optimizer": reset_optimizer}
    if params is not None:
        phase["params"] = params
    if kwargs is not None:
        phase["kwargs"] = kwargs
    return phase


def _alternating_schedule(
    trainer, base_cfg, chunk_epochs, *, method, params=None, kwargs=None, reset_optimizer=True
):
    phases = []
    for n_epochs in chunk_epochs:
        phases.append(
            _linear_solve_phase(
                method, params=params, kwargs=kwargs, reset_optimizer=reset_optimizer
            )
        )
        chunk_cfg = dict(base_cfg)
        chunk_cfg["n_epochs"] = n_epochs
        phases.append(_optimizer_phase(trainer, chunk_cfg))
    phases.append(
        _linear_solve_phase(method, params=params, kwargs=kwargs, reset_optimizer=reset_optimizer)
    )
    return phases


cfgs = [
    (
        "kbf_node",
        KBF,
        StackedTrainer,
        {
            "model": gen_mdl_kb(0, 0, 13),
            "criterion": crit,
            "phases": _alternating_schedule("NODE", trn_nd, [50, 150], method="full"),
            "transform_x": [trn_svd, trn_add],
        },
    ),
    (
        "dkbf_ln",
        DKBF,
        LinearTrainer,
        {"model": gen_mdl_kb(0, 0, 13), "training": trn_ln, "transform_x": [trn_svd, trn_add]},
    ),
    (
        "dkbf_ae",
        DKBF,
        StackedTrainer,
        {
            "model": gen_mdl_kb(3, 64, 3),
            "criterion": crit,
            "phases": _alternating_schedule(
                "NODE", trn_ae, [500, 500, 2000], method="full", reset_optimizer=False
            ),
            "transform_x": [trn_svd],
        },
    ),
    (
        "dkbf_dm",
        DKBF,
        LinearTrainer,
        {"model": gen_mdl_kb(0, 0, 3), "training": trn_ln, "transform_x": [trn_svd, trn_dmf]},
    ),
    (
        "dks_ln",
        DKMSK,
        LinearTrainer,
        {"model": mdl_kl, "training": trn_rw, "transform_x": [trn_svd, trn_scl]},
    ),
]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Run vortex training cases.")
    parser.add_argument("--case", nargs="+", type=int)
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--workdir", type=Path)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--no-train", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--no-predict", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def resolve_indices(values):
    indices = DEFAULT_CASES if values is None else values
    invalid = [idx for idx in indices if idx < 0 or idx >= len(cfgs)]
    if invalid:
        raise ValueError(f"Invalid case indices: {invalid}")
    return indices


def print_cases():
    for idx, (name, _, _, _) in enumerate(cfgs):
        print(f"{idx}: {name}")


def prepare_workdir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    config_path = root / "vor_model.yaml"
    if not config_path.exists():
        shutil.copy2(CONFIG_PATH, config_path)
    return config_path


def resolve_data_path(path: Path) -> Path:
    data_path = path if path.is_absolute() else (Path.cwd() / path).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Vortex data not found: {data_path}")
    return data_path


def train(selected, data_path: Path, config_path: Path):
    for idx in selected:
        mdl, model_class, trainer_class, opt = cfgs[idx]
        opt_local = copy.deepcopy(opt)
        opt_local["model"]["name"] = f"kp_{mdl}"
        opt_local["data"] = {"path": str(data_path)}
        trainer = trainer_class(str(config_path), model_class, config_mod=opt_local)
        trainer.train()


def plot(selected):
    labels = [cfgs[idx][0] for idx in selected]
    npz_files = [f"kp_{label}" for label in labels]
    npzs = plot_summary(npz_files, labels=labels, ifclose=False)
    for label, npz in zip(labels, npzs, strict=False):
        print(f"Epoch time: {label} - {npz['avg_epoch_time']}")


def predict(selected, data_path: Path):
    dat = np.load(data_path)
    x_data, t_data = dat["x"], dat["t"]

    res = [x_data]
    for idx in selected:
        mdl, model_class, _, _ = cfgs[idx]
        _, predict_fn = load_model(model_class, f"kp_{mdl}.pt")
        with torch.no_grad():
            pred = predict_fn(x_data, t_data)
        res.append(pred)

    setup_logging()
    n_cases = len(selected)

    def contour_fig(step_idx):
        fig, ax = plt.subplots(n_cases, 3, sharex=True, sharey=True, figsize=(12, 1.5 * n_cases))
        ax = np.atleast_2d(ax)
        colorbar = step_idx == 0
        for row, cfg_idx in enumerate(selected):
            compare_contour(
                res[0][step_idx].reshape(NX, NY),
                res[row + 1][step_idx].reshape(NX, NY),
                vmin=-12,
                vmax=12,
                axes=(fig, ax[row]),
                colorbar=colorbar,
            )
            ax[row, 1].set_title(cfgs[cfg_idx][0])
        for axis in ax.flatten():
            axis.set_axis_off()
        return fig, ax

    animate(contour_fig, filename="vis.mp4", fps=10, n_frames=len(t_data))


def main():
    args = parse_args()
    if args.seed is not None:
        set_seed(args.seed)

    root = BASE_DIR if args.workdir is None else args.workdir.resolve()
    config_path = CONFIG_PATH if args.workdir is None else prepare_workdir(root)
    root.mkdir(parents=True, exist_ok=True)
    os.chdir(root)

    if args.list_cases:
        print_cases()
        return 0

    selected = resolve_indices(args.case)
    data_path = resolve_data_path(args.data_path.resolve())

    if not args.no_train:
        train(selected, data_path, config_path)
    if not args.no_plot:
        plot(selected)
    if not args.no_predict:
        predict(selected, data_path)
    if not args.no_show and (not args.no_plot or not args.no_predict):
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
