import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Tuple, Union

from dymad.io import load_model
from dymad.models import TemplateCorrAlg
from dymad.numerics import rk4_step
from dymad.training import NODETrainer
from dymad.utils import JaxWrapper, plot_multi_trajs, plot_summary, TrajectorySampler

B = 16
N = 301
t_grid = np.linspace(0, 6, N)
dt = t_grid[1] - t_grid[0]

g = 9.81
def f(t, x, u=None, p=[1.0]):
    dtheta = x[1]
    domega = - (g / p[0]) * (np.sin(x[0]) + 0.1 * x[1] + u[0])
    return np.array([dtheta, domega])

class DPT(TemplateCorrAlg):
    CONT = False
    def _step(self, x: torch.Tensor, u: torch.Tensor, f: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        _f = torch.zeros_like(x)
        _f[..., 0] = x[..., 1]
        _f[..., 1] = - (g / p[..., 0]) * (torch.sin(x[..., 0]) + f[..., 0] + u[..., 0])
        return _f

    def base_dynamics(self, x: torch.Tensor, u: torch.Tensor, f: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        func = lambda t, x: self._step(x, u, f, p)
        x_next = rk4_step(func, 0, x, dt)
        return x_next

def f_jax(*xs: jnp.ndarray) -> Union[jnp.ndarray, Tuple[jnp.ndarray, ...]]:
    x, u, f, p = xs
    y1 = x[..., 1]
    y2 = - (g / p[..., 0]) * (jnp.sin(x[..., 0]) + f[..., 0] + u[..., 0])
    return jnp.stack([y1, y2], axis=-1)
def f_rk4(x, u, f, p):
    func = lambda t, x: f_jax(x, u, f, p)
    x_next = rk4_step(func, 0, x, dt)
    return x_next

class DPJ(TemplateCorrAlg):
    CONT = False
    def extra_setup(self):
        self._jax_layer = JaxWrapper(f_rk4, jit=True)

    def base_dynamics(self, x: torch.Tensor, u: torch.Tensor, f: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return self._jax_layer(x, u, f, p)

mdl_kl = {
    "name" : 'res_model',
    "residual_layers" : 1,
    "hidden_dimension" : 32,
    "residual_dimension" : 1,
    "activation" : "none",
    "end_activation" : False,
    "weight_init" : "xavier_uniform",
    "gain" : 0.1,}

trn_nd = {
    "n_epochs": 100,
    "save_interval": 50,
    "load_checkpoint": False,
    "learning_rate": 5e-3,
    "decay_rate": 0.999,
    "sweep_lengths": [10],
    "sweep_epoch_step": 100,
    "chop_mode": "unfold",
    "chop_step": 0.5,
    "ode_method": "dopri5",
    "ode_args": {
        "rtol": 1.e-7,
        "atol": 1.e-9}
    }
config_path = 'res_model.yaml'

cfgs = [
    ('dp_nd', DPT, NODETrainer,     {"model": mdl_kl, "training" : trn_nd}),
    ('dj_nd', DPJ, NODETrainer,     {"model": mdl_kl, "training" : trn_nd}),
    ]

IDX = [0, 1]
labels = [cfgs[i][0] for i in IDX]

ifdat = 0
iftrn = 1
ifplt = 1
ifprd = 1

if ifdat:
    sampler = TrajectorySampler(f, config='res_data.yaml')
    ts, xs, us, ys, ps = sampler.sample(t_grid, batch=B, save='./data/res.npz')

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    for i in range(B):
        ax.plot(xs[i, :, 0], xs[i, :, 1], alpha=0.5)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title('2D Trajectories')
    plt.tight_layout()

    fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    for i in range(B):
        axs[0].plot(ts[i], xs[i, :, 0], alpha=0.5)
        axs[1].plot(ts[i], xs[i, :, 1], alpha=0.5)
        axs[2].plot(ts[i], us[i, :, 0], alpha=0.5)
    axs[0].set_ylabel('x1')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('x2')
    axs[2].set_ylabel('u')
    plt.tight_layout()

if iftrn:
    for i in IDX:
        mdl, MDL, Trainer, opt = cfgs[i]
        opt["model"]["name"] = f"res_{mdl}"
        trainer = Trainer(config_path, MDL, config_mod=opt)
        trainer.train()

if ifplt:
    npz_files = [f'res_{l}' for l in labels]
    npzs = plot_summary(npz_files, labels=labels, ifclose=False)

    for lbl, npz in zip(labels, npzs):
        print(f"Epoch time {lbl}: {npz['avg_epoch_time']}")

if ifprd:
    sampler = TrajectorySampler(f, config='res_test.yaml')
    ts, xs, us, ys, ps = sampler.sample(t_grid, batch=5)
    x_data = xs
    u_data = us
    t_data = ts[0]
    p_data = ps

    res = [x_data]
    for i in IDX:
        mdl, MDL, _, _ = cfgs[i]
        _, prd_func = load_model(MDL, f'res_{mdl}.pt')
        with torch.no_grad():
            pred = prd_func(x_data, t_data, u=u_data, p=p_data)
        res.append(pred)

    plot_multi_trajs(
        np.array(res), t_data, "DP",
        labels=['Truth'] + labels, ifclose=False)

plt.show()