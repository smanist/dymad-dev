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

from scripts.cli_helpers import print_case_table, resolve_case_indices, set_seed, stage_workdir

from dymad.io import load_model
from dymad.models import DKBF
from dymad.numerics import complex_plot, scaled_eig
from dymad.sako import SpectralAnalysis
from dymad.training import LinearTrainer
from dymad.utils import TrajectorySampler, plot_trajectory

BASE_DIR = Path(__file__).resolve().parent

B = 64
N = 21
t_grid = np.linspace(0, 10, N)
dt = t_grid[1] - t_grid[0]

_t = 0.2
_T = np.array([[1, _t], [0, 1]])
_S = np.array([[1, -_t], [0, 1]])

mu = -0.5
lm = -3


def f(t, x):
    _y = _T.dot(x)
    _d = np.array([mu * _y[0], lm * (_y[1] - _y[0] ** 2)])
    return _S.dot(_d)


def func_obs(x):
    _x1, _x2 = x.T
    return _x1 + _x2


Jac = _S.dot(np.diag([mu, lm])).dot(_T)

w0 = np.array([mu, lm]) + 1j * 0
w0 = np.hstack([w0, 2 * w0[0], 2 * w0[1], w0[0] + w0[1]])
wa = np.exp(w0 * dt)

mdl_kl = {
    "name": "kp_model",
    "encoder_layers": 0,
    "decoder_layers": 0,
    "koopman_dimension": 9,
    "activation": "none",
    "weight_init": "xavier_uniform",
    "predictor_type": "exp",
}
trn_kl = [{"type": "lift", "fobs": "poly", "Ks": [3, 3]}]

ref = {
    "n_epochs": 1,
    "save_interval": 1,
    "load_checkpoint": False,
}
trn_ln = {
    "method": "full",
}
trn_ln.update(ref)
trn_tr = {
    "method": "truncated",
    "params": 0.999,
}
trn_tr.update(ref)
trn_sa = {
    "method": "sako",
    "params": 4,
    "kwargs": {"remove_one": True},
}
trn_sa.update(ref)

CASES = [
    {
        "name": "dkbf_ln",
        "model": DKBF,
        "trainer": LinearTrainer,
        "opt": {"model": mdl_kl, "transform_x": trn_kl, "training": trn_ln},
    },
    {
        "name": "dkbf_tr",
        "model": DKBF,
        "trainer": LinearTrainer,
        "opt": {"model": mdl_kl, "transform_x": trn_kl, "training": trn_tr},
    },
    {
        "name": "dkbf_sa",
        "model": DKBF,
        "trainer": LinearTrainer,
        "opt": {"model": mdl_kl, "transform_x": trn_kl, "training": trn_sa},
    },
]
DEFAULT_CASES = [0, 1, 2]
CONFIG_PATH = "kp_model.yaml"


def parse_args():
    parser = argparse.ArgumentParser(description="Run 2D Koopman spectral-analysis cases.")
    parser.add_argument(
        "--case", nargs="+", type=int, help="Case indices to run, for example '--case 0 2'."
    )
    parser.add_argument(
        "--list-cases", action="store_true", help="Print available case indices and exit."
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        help="Run in a separate working directory and stage the needed files there.",
    )
    parser.add_argument("--seed", type=int, help="Set random seeds for reproducible runs.")
    parser.add_argument(
        "--data", action="store_true", help="Generate or refresh the training data file."
    )
    parser.add_argument(
        "--train", action="store_true", help="Train the selected checkpoints before analysis."
    )
    parser.add_argument(
        "--no-predict", action="store_true", help="Skip prediction trajectory plots."
    )
    parser.add_argument(
        "--no-analysis", action="store_true", help="Skip spectral-analysis plots and diagnostics."
    )
    parser.add_argument("--no-show", action="store_true", help="Skip plt.show().")
    return parser.parse_args()


def prepare_workdir(root: Path):
    stage_workdir(root, BASE_DIR, ["kp_data.yaml", "kp_model.yaml"], data_dir=True)


def generate_data(root: Path):
    sampler = TrajectorySampler(f, config=root / "kp_data.yaml")
    ts, xs, _ = sampler.sample(t_grid, batch=B, save=str(root / "data" / "kp.npz"))

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    for i in range(B):
        ax.plot(xs[i, :, 0], xs[i, :, 1], alpha=0.5)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("2D Trajectories")
    plt.tight_layout()

    fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    for i in range(B):
        axs[0].plot(ts[i], xs[i, :, 0], alpha=0.5)
        axs[1].plot(ts[i], xs[i, :, 1], alpha=0.5)
    axs[0].set_ylabel("x1")
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("x2")
    plt.tight_layout()


def _checkpoint_name(case_idx: int) -> str:
    return f"kp_{CASES[case_idx]['name']}"


def _checkpoint_path(case_idx: int, root: Path) -> Path:
    name = _checkpoint_name(case_idx)
    return root / name / f"{name}.pt"


def _require_checkpoints(selected: list[int], root: Path):
    missing = [
        str(_checkpoint_path(idx, root))
        for idx in selected
        if not _checkpoint_path(idx, root).exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing checkpoint(s): "
            + ", ".join(missing)
            + ". Train them first with '--train' or stage existing checkpoints."
        )


def train(selected: list[int], root: Path):
    for idx in selected:
        case = CASES[idx]
        opt_local = copy.deepcopy(case["opt"])
        opt_local["data"] = {"path": str(root / "data" / "kp.npz")}
        opt_local["model"]["name"] = _checkpoint_name(idx)
        trainer = case["trainer"](root / CONFIG_PATH, case["model"], config_mod=opt_local)
        trainer.train()


def predict(selected: list[int], root: Path):
    _require_checkpoints(selected, root)
    sampler = TrajectorySampler(f, config=root / "kp_data.yaml")
    ts, xs, _ = sampler.sample(t_grid, batch=1)
    x_data = xs[0]
    t_data = ts[0]

    res = [x_data]
    for idx in selected:
        case = CASES[idx]
        _, predict_fn = load_model(case["model"], _checkpoint_path(idx, root))
        with torch.no_grad():
            pred = predict_fn(x_data, t_data)
        res.append(pred)

    plot_trajectory(
        np.array(res),
        t_data,
        "KP",
        labels=["Truth"] + [CASES[idx]["name"] for idx in selected],
        ifclose=False,
    )


def _load_analyses(selected: list[int], root: Path) -> tuple[list[SpectralAnalysis], list[str]]:
    _require_checkpoints(selected, root)
    analyses = []
    labels = []
    for idx in selected:
        case = CASES[idx]
        analyses.append(
            SpectralAnalysis(
                case["model"], _checkpoint_path(idx, root), dt=dt, reps=1e-10, etol=None
            )
        )
        labels.append(case["name"].replace("dkbf_", "DT-").upper())
    return analyses, labels


def compute_analysis_diagnostics(
    selected: list[int],
    root: Path,
    *,
    pred_batch: int = 16,
    map_batch: int = 64,
    ps_points: int = 51,
    measure_thetas: int = 501,
):
    sas, lbs = _load_analyses(selected, root)
    sampler = TrajectorySampler(f, config=root / "kp_data.yaml")
    ts, xs, _ = sampler.sample(t_grid, batch=pred_batch)
    x0s = xs[:, 0, :].squeeze()

    xs_disc = np.linspace(-1.3, 1.3, ps_points)
    grid_disc = np.vstack([xs_disc, xs_disc])
    zs = np.linspace(-8.0, 0.5, ps_points)
    ws = np.linspace(-4.0, 4.0, ps_points)
    grid_cont = np.vstack([zs, ws])

    yy = np.linspace(-1, 1, 41)
    s0 = _S.dot(np.array([yy, yy**2])).T
    _, sol, _ = sampler.sample(t_grid, batch=map_batch)
    traj = sol[0]

    cases = []
    for sa in sas:
        grid_disc_eval, ps_standard_disc = sa.estimate_ps(
            grid_disc, mode="disc", method="standard", return_vec=False
        )
        _, ps_sako_disc = sa.estimate_ps(grid_disc, mode="disc", method="sako", return_vec=False)
        grid_cont_eval, ps_standard_cont = sa.estimate_ps(
            grid_cont, mode="cont", method="standard", return_vec=False
        )
        _, ps_sako_cont = sa.estimate_ps(grid_cont, mode="cont", method="sako", return_vec=False)
        theta, measure = sa.estimate_measure(func_obs, 6, 0.1, thetas=measure_thetas)

        sa.set_conj_map(Jac)
        traj_cnj = sa.mapto_cnj(traj)
        traj_nrm = sa.mapto_nrm(traj)
        s0_cnj = sa.mapto_cnj(s0)
        s0_nrm = sa.mapto_nrm(s0)

        cases.append(
            {
                "label": lbs[len(cases)],
                "analysis": sa,
                "discrete": {
                    "grid": grid_disc_eval,
                    "standard": ps_standard_disc,
                    "sako": ps_sako_disc,
                },
                "continuous": {
                    "grid": grid_cont_eval,
                    "standard": ps_standard_cont,
                    "sako": ps_sako_cont,
                },
                "measure": {"theta": theta, "values": measure},
                "conjugacy": {
                    "trajectory": traj,
                    "trajectory_cnj": traj_cnj,
                    "trajectory_nrm": traj_nrm,
                    "slow_manifold": s0,
                    "slow_cnj": s0_cnj,
                    "slow_nrm": s0_nrm,
                },
            }
        )
    return {
        "cases": cases,
        "pred": {"ts": ts[0], "ref": xs, "x0s": x0s},
    }


def analyze(
    selected: list[int],
    root: Path,
    *,
    pred_batch: int = 16,
    map_batch: int = 64,
    ps_points: int = 51,
    measure_thetas: int = 501,
):
    diagnostics = compute_analysis_diagnostics(
        selected,
        root,
        pred_batch=pred_batch,
        map_batch=map_batch,
        ps_points=ps_points,
        measure_thetas=measure_thetas,
    )
    cases = diagnostics["cases"]
    n_cases = len(cases)
    if n_cases == 0:
        return diagnostics

    ts = diagnostics["pred"]["ts"]
    xs = diagnostics["pred"]["ref"]
    x0s = diagnostics["pred"]["x0s"]
    lbs = [case["label"] for case in cases]
    sas = [case["analysis"] for case in cases]
    for sa, label in zip(sas, lbs, strict=False):
        sa.plot_pred(x0s, ts, ref=xs, idx="all", figsize=(6, 8), title=label)

    fig, ax = plt.subplots(ncols=n_cases, sharey=True, figsize=(5 * n_cases, 5))
    ax = np.atleast_1d(ax)
    for i, (sa, label) in enumerate(zip(sas, lbs, strict=False)):
        fig, ax[i], lines = sa.plot_eigs(fig=(fig, ax[i]))
        (truth_line,) = ax[i].plot(wa.real, wa.imag, "kx", markersize=15)
        ax[i].set_title(f"{label}\nMax res: {sa._res[-1]:4.3e}")
        ax[i].legend(lines + [truth_line], [label, "Filtered", "Truth"], loc=1)

    disc_rng = np.array([0.1, 0.25])
    for i, case in enumerate(cases):
        grid = case["discrete"]["grid"]
        ps_standard = case["discrete"]["standard"]
        ps_sako = case["discrete"]["sako"]
        fig, ax[i] = complex_plot(
            grid, 1 / ps_standard, disc_rng, fig=(fig, ax[i]), mode="line", lwid=2, lsty="dotted"
        )
        fig, ax[i] = complex_plot(
            grid, 1 / ps_sako, disc_rng, fig=(fig, ax[i]), mode="line", lwid=1
        )
        ax[i].set_xlim([-0.4, 1.3])
        ax[i].set_ylim([-0.7, 0.7])

    fig, ax = plt.subplots(ncols=n_cases, sharey=True, figsize=(5 * n_cases, 5))
    ax = np.atleast_1d(ax)
    for i, (sa, label) in enumerate(zip(sas, lbs, strict=False)):
        fig, ax[i], lines = sa.plot_eigs(fig=(fig, ax[i]), mode="cont")
        (truth_line,) = ax[i].plot(w0.real, w0.imag, "kx", markersize=15)
        ax[i].set_title(f"{label}\nMax res: {sa._res[-1]:4.3e}")
        ax[i].legend(lines + [truth_line], [label, "Filtered", "Truth"], loc=1)

    cont_rng = np.array([0.25, 0.5])
    for i, case in enumerate(cases):
        grid = case["continuous"]["grid"]
        ps_standard = case["continuous"]["standard"]
        ps_sako = case["continuous"]["sako"]
        fig, ax[i] = complex_plot(
            grid, 1 / ps_standard, cont_rng, fig=(fig, ax[i]), mode="line", lwid=2, lsty="dotted"
        )
        fig, ax[i] = complex_plot(
            grid, 1 / ps_sako, cont_rng, fig=(fig, ax[i]), mode="line", lwid=1
        )
        ax[i].set_xlim([-8.0, 0.5])
        ax[i].set_ylim([-4.0, 4.0])

    arg = np.angle(wa)
    amp = np.max(cases[0]["measure"]["values"])
    fig = plt.figure()
    styles = ["b-", "r-", "g--", "m--", "c-"]
    for i, case in enumerate(cases):
        theta = case["measure"]["theta"]
        measure = case["measure"]["values"]
        plt.plot(theta, measure, styles[i], label=lbs[i], markerfacecolor="none")
    plt.plot([arg[0], arg[0]], [0, amp], "k:", label="System frequency")
    for angle in arg[1:]:
        plt.plot([angle, angle], [0, amp], "k:")
    plt.legend()
    plt.xlabel("Angle, rad")
    plt.ylabel("Spectral measure")

    rngs = [[-1.5, 1.5], [-1.5, 1.5]]
    Ns = [101, 101]
    fig, ax = plt.subplots(
        nrows=n_cases, ncols=4, sharex=True, sharey=True, figsize=(10, 3 * n_cases)
    )
    ax = np.atleast_2d(ax)
    for i, (sa, label) in enumerate(zip(sas, lbs, strict=False)):
        n_plot = min(sa._Nrank, 4)
        sa.plot_eigfun_2d(rngs, Ns, n_plot, mode="real", fig=(fig, ax[i]))
        ax[i][0].set_ylabel(label)

    fig, ax = plt.subplots(nrows=n_cases, ncols=3, figsize=(10, 4 * n_cases), squeeze=False)
    styles = ["b-", "r-", "b--", "r--"]
    _, _, jac_vecs = scaled_eig(Jac)
    for row, case in enumerate(cases):
        conj = case["conjugacy"]
        traj = conj["trajectory"]
        ax[row][0].plot(traj[:, 0], traj[:, 1], styles[row % 4])
        r0, r1 = conj["trajectory_cnj"].real.T
        ax[row][1].plot(r0, r1, styles[row % 4])
        r0, r1 = conj["trajectory_nrm"].real.T
        ax[row][2].plot(r0, r1, styles[row % 4])
        s0 = conj["slow_manifold"]
        ax[row][0].plot(s0[:, 0], s0[:, 1], "k-")
        r0, r1 = conj["slow_cnj"].real.T
        ax[row][1].plot(r0, r1, "k-")
        r0, r1 = conj["slow_nrm"].real.T
        ax[row][2].plot(r0, r1, "k-")
        for i in range(2):
            for j in range(2):
                ax[row][i].plot([0, jac_vecs[0, j]], [0, jac_vecs[1, j]], "go-")
        ax[row][2].plot([0, 0, 1], [1, 0, 0], "go-")
        ax[row][0].set_ylabel(lbs[row])
        ax[row][0].set_xlabel(r"$x_1$")
        ax[row][0].set_title("Physical space")
        ax[row][0].set_aspect("equal")
        ax[row][1].set_xlabel(r"$y_1$")
        ax[row][1].set_title('"Flatten" space')
        ax[row][1].set_aspect("equal")
        ax[row][2].set_xlabel(r"$y_1^*$")
        ax[row][2].set_title("Orthogonalized space")
        ax[row][2].set_aspect("equal")

    return diagnostics


def main():
    args = parse_args()
    if args.seed is not None:
        set_seed(args.seed)
    root = BASE_DIR if args.workdir is None else args.workdir.resolve()
    if args.workdir is not None:
        prepare_workdir(root)
    os.chdir(root)
    if args.list_cases:
        print_case_table(CASES)
        return 0

    selected = resolve_case_indices(args.case, len(CASES), DEFAULT_CASES)
    data_path = root / "data" / "kp.npz"
    if args.data or (args.train and not data_path.exists()):
        generate_data(root)
    if args.train:
        train(selected, root)
    if not args.no_predict:
        predict(selected, root)
    if not args.no_analysis:
        analyze(selected, root)
    if not args.no_show and (not args.no_predict or not args.no_analysis or args.data):
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
