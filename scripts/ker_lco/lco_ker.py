import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as spi
import torch

from dymad.io import load_model
from dymad.models import DKM, DKMSK, KM
from dymad.training import LinearTrainer, StackedTrainer
from dymad.utils import TrajectorySampler, plot_multi_trajs

B = 50
N = 81
t_grid = np.linspace(0, 8, N)
dt = t_grid[1] - t_grid[0]

mu = 1.0


def f(t, x):
    _x, _y = x
    dx = np.array([_y, mu * (1 - _x**2) * _y - _x])
    return dx


def g(t, x):
    return x


# Reference trajectory
_Nt = 161
_ts = np.linspace(0, 40.0, 8 * _Nt)
_res = spi.solve_ivp(f, [0, _ts[-1]], [2, 2], t_eval=_ts)
_ref = _res.y[:, -220:].T

# Transition to LCO
db = 0.4

# Training options
RIDGE = 1e-10
opt_rbf1 = {"type": "sc_rbf", "input_dim": 2, "lengthscale_init": 1.0}
opt_opk1 = {
    "type": "op_sep",
    "input_dim": 2,
    "output_dim": 2,
    "kopts": [opt_rbf1],
    "Ls": np.array([[[1, 0], [0, 1]]]),
}
opt_share = {"type": "share", "kernel": opt_rbf1, "dtype": torch.float64, "ridge_init": RIDGE}
opt_indep = {
    "type": "indep",
    "kernel": [opt_rbf1, opt_rbf1],
    "dtype": torch.float64,
    "ridge_init": RIDGE,
}
opt_opval = {
    "type": "opval",
    "kernel": opt_opk1,
    "dtype": torch.float64,
    "ridge_init": RIDGE,
}

# opt_krr = opt_share
# opt_krr = opt_indep
opt_krr = opt_opval
mdl_kl = {"name": "ker_model", "encoder_layers": 0, "decoder_layers": 0, "kernel_dimension": 2}
mdl_kl.update(**opt_krr)

trn_ln = {
    "n_epochs": 1,
    "save_interval": 100,
    "load_checkpoint": False,
    "learning_rate": 1e-2,
    "decay_rate": 0.999,
    "method": "raw",
}
trn_ct = {
    "n_epochs": 200,
    "save_interval": 50,
    "load_checkpoint": False,
    "learning_rate": 1e-2,
    "decay_rate": 0.999,
    "sweep_lengths": [4],
    "chop_mode": "initial",
    "chop_step": 0.5,
}
trn_dt = {
    "n_epochs": 400,
    "save_interval": 50,
    "load_checkpoint": False,
    "learning_rate": 1e-2,
    "decay_rate": 0.999,
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


smpl = {"x0": {"kind": "perturb", "params": {"bounds": [-db, db], "ref": _ref}}}
config_path = "ker_model.yaml"

cfgs = [
    ("km_ln", KM, LinearTrainer, {"model": mdl_kl, "training": trn_ln}),
    (
        "km_nd",
        KM,
        StackedTrainer,
        {
            "model": mdl_kl,
            "phases": _alternating_schedule(
                "NODE", trn_ct, [50, 50, 50, 50], method="raw", reset_optimizer=False
            ),
        },
    ),
    ("dkm_ln", DKM, LinearTrainer, {"model": mdl_kl, "training": trn_ln}),
    (
        "dkm_nd",
        DKM,
        StackedTrainer,
        {
            "model": mdl_kl,
            "phases": _alternating_schedule(
                "NODE", trn_dt, [100, 100, 100, 100], method="raw", reset_optimizer=False
            ),
        },
    ),
    ("dks_ln", DKMSK, LinearTrainer, {"model": mdl_kl, "training": trn_ln}),
    (
        "dks_nd",
        DKMSK,
        StackedTrainer,
        {
            "model": mdl_kl,
            "phases": _alternating_schedule(
                "NODE", trn_dt, [100, 100, 100, 100], method="raw", reset_optimizer=False
            ),
        },
    ),
]

# IDX = [0, 1, 2, 3, 4, 5]
# IDX = [0, 1]
# IDX = [2, 3]
IDX = [0, 2, 4]
labels = [cfgs[i][0] for i in IDX]

ifdat = 0
iftrn = 1
ifprd = 1

if ifdat:
    sampler = TrajectorySampler(f, g, config="ker_data.yaml", config_mod=smpl)
    ts, xs, ys = sampler.sample(t_grid, batch=B, save="./data/ker.npz")

    for i in range(B):
        plt.plot(ys[i, :, 0], ys[i, :, 1])
    plt.plot(_ref[:, 0], _ref[:, 1], "k--", linewidth=2)

if iftrn:
    for i in IDX:
        mdl, MDL, Trainer, opt = cfgs[i]
        opt["model"]["name"] = f"ker_{mdl}"
        trainer = Trainer(config_path, MDL, config_mod=opt)
        trainer.train()

if ifprd:
    J = 32
    sampler = TrajectorySampler(f, g, config="ker_data.yaml", config_mod=smpl)
    ts, xs, ys = sampler.sample(t_grid, batch=J)
    x_data = xs
    t_data = ts[0]

    res = [x_data]
    for i in IDX:
        mdl, MDL, _, _ = cfgs[i]
        _, prd_func = load_model(MDL, f"ker_{mdl}.pt")
        with torch.no_grad():
            pred = prd_func(x_data, t_data)
        res.append(pred)

    plot_multi_trajs(np.array(res), t_data, "KER", labels=["Truth"] + labels, ifclose=False)

plt.show()
