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
from dymad.models import DKMSK, KM, KMM
from dymad.training import LinearTrainer
from dymad.utils import TrajectorySampler, plot_trajectory

BASE_DIR = Path(__file__).resolve().parent

B = 1
N = 201
t_grid = np.linspace(0, 8, N)
t_pred = np.linspace(0, 8 * 16, N * 16)

s5 = np.sqrt(5)
K0, D0 = 3, 0.5


def dyn(tt, K=K0, D=D0):
    vv = 2 * np.arctan(np.tan(s5 * tt / 4) / s5)
    rr = 1 + D * np.cos(K * vv)
    uu = np.array([rr * np.cos(vv), rr * np.sin(vv)]).T
    return vv, uu


t_ref = np.linspace(0, 6, 51)
_ref = dyn(t_ref)[1]


def f(t, x):
    _x = np.atleast_2d(x)
    _t = np.arctan2(_x[:, 1], _x[:, 0])
    _v = 1.5 - np.cos(_t)
    _r = 1 + D0 * np.cos(K0 * _t)
    _d = -K0 * D0 * np.sin(K0 * _t)
    _c, _s = np.cos(_t) * _v, np.sin(_t) * _v
    return np.vstack([-_r * _s + _d * _c, _r * _c + _d * _s]).T.squeeze()


RIDGE = 1e-6
opt_rbf = {"type": "sc_rbf", "input_dim": 2, "lengthscale_init": None}
opt_exp = {"type": "sc_exp", "input_dim": 2, "lengthscale_init": 1.0}
opt_dm = {"type": "sc_dm", "input_dim": 2, "eps_init": None}
mdl_rbf = {
    "name": "ker_model",
    "encoder_layers": 0,
    "decoder_layers": 0,
    "kernel_dimension": 2,
    "type": "share",
    "kernel": opt_rbf,
    "dtype": torch.float64,
    "ridge_init": RIDGE,
}
mdl_exp = copy.deepcopy(mdl_rbf)
mdl_exp["kernel"] = opt_exp
mdl_dm = copy.deepcopy(mdl_rbf)
mdl_dm["kernel"] = opt_dm

opt_opk = {"type": "op_tan", "input_dim": 2, "output_dim": 2, "kopts": copy.deepcopy(opt_rbf)}
opt_opk["kopts"]["lengthscale_init"] = 1.0
opt_tange = {"type": "tangent", "kernel": opt_opk, "dtype": torch.float64, "ridge_init": RIDGE}
GT = {0.0: (6, 3), 0.1: (7, 5), 0.3: (6, 5), 0.5: (5, 4)}
mdl_mn = {
    "name": "ker_model",
    "encoder_layers": 0,
    "decoder_layers": 0,
    "kernel_dimension": 2,
    "manifold": {"d": 1, "T": GT[D0][1], "g": GT[D0][0]},
    **opt_tange,
}

trn_ln = {"n_epochs": 1, "save_interval": 100, "load_checkpoint": False, "method": "raw"}
trn_l1 = copy.deepcopy(trn_ln)
trn_l1["kwargs"] = {"order": 1}

smpl = {"x0": {"kind": "perturb", "params": {"bounds": [0, 0], "ref": _ref}}}
config_path = "ker_model.yaml"
cfgs = [
    ("km_exp", KM, LinearTrainer, {"model": mdl_exp, "training": trn_ln}),
    ("kmm_tn", KMM, LinearTrainer, {"model": mdl_mn, "training": trn_l1}),
    ("dks_rbf", DKMSK, LinearTrainer, {"model": mdl_rbf, "training": trn_ln}),
    ("dks_exp", DKMSK, LinearTrainer, {"model": mdl_exp, "training": trn_ln}),
    ("ddm_dm", DKMSK, LinearTrainer, {"model": mdl_dm, "training": trn_ln}),
]
DEFAULT_CASES = [0, 1, 2, 3, 4]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Run kernel S1 cases.")
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


def resolve_indices(values):
    indices = DEFAULT_CASES if values is None else values
    invalid = [idx for idx in indices if idx < 0 or idx >= len(cfgs)]
    if invalid:
        raise ValueError(f"Invalid case indices: {invalid}")
    return indices


def print_cases():
    for idx, (name, _, _, _) in enumerate(cfgs):
        print(f"{idx}: {name}")


def prepare_workdir(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    (root / "data").mkdir(exist_ok=True)
    shutil.copy2(BASE_DIR / "ker_data.yaml", root / "ker_data.yaml")
    shutil.copy2(BASE_DIR / "ker_model.yaml", root / "ker_model.yaml")


def generate_data(root: Path, seed: int | None = None):
    sampler = TrajectorySampler(f, config=BASE_DIR / "ker_data.yaml", rng=seed, config_mod=smpl)
    sampler.sample(t_grid, batch=B, save=str(root / "data" / "ker.npz"))
    print(f"Generated data: {root / 'data' / 'ker.npz'}")


def train(selected):
    for i in selected:
        mdl, MDL, Trainer, opt = cfgs[i]
        opt_local = copy.deepcopy(opt)
        opt_local["model"]["name"] = f"ker_{mdl}"
        trainer = Trainer(config_path, MDL, config_mod=opt_local)
        trainer.train()


def predict(selected, seed: int | None = None):
    sampler = TrajectorySampler(f, config="ker_data.yaml", rng=seed, config_mod=smpl)
    ts, xs, ys = sampler.sample(t_pred, batch=1)
    x_data = xs[0]
    t_data = ts[0]
    res = [x_data]
    for i in selected:
        mdl, MDL, _, _ = cfgs[i]
        _, prd_func = load_model(MDL, f"ker_{mdl}.pt")
        with torch.no_grad():
            pred = prd_func(x_data, t_data)
        res.append(pred)
    plot_trajectory(
        np.array(res),
        t_data,
        "LTI",
        labels=["Truth"] + [cfgs[i][0] for i in selected],
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
    data_path = root / "data" / "ker.npz"
    if args.data or (args.workdir is not None and not data_path.exists()):
        generate_data(root, args.seed)
    if not args.no_train:
        train(selected)
    if not args.no_predict:
        predict(selected, args.seed)
    if not args.no_show and not args.no_predict:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
