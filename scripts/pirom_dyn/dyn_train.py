import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Tuple, Union

from dymad.io import load_model, visualize_model
from dymad.models import TemplateCorrDif
from dymad.training import NODETrainer
from dymad.utils import JaxWrapper, plot_multi_trajs, TrajectorySampler

B = 32
N = 101
t_grid = np.linspace(0, 2, N)

gg = 9.81
def f(t, x, u, p=[1.0]):
    dtheta = x[1]
    domega = - (gg / p[0]) * (np.sin(x[0]) + 0.1 * x[2] + u[0])
    dbeta  = -5*x[2] + x[0]
    return np.array([dtheta, domega, dbeta])
def g(t, x, u, p=[1.0]):
    return x[..., :2]

class DPT(TemplateCorrDif):
    CONT = True
    def base_dynamics(self, x: torch.Tensor, u: torch.Tensor, f: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        _f = torch.zeros_like(x)
        _f[..., 0] = x[..., 1]
        _f[..., 1] = - (gg / p[..., 0]) * (torch.sin(x[..., 0]) + f[..., 0] + u[..., 0])
        return _f

    def encoder(self, w) -> torch.Tensor:
        return torch.cat([w.x, torch.zeros(*w.x.shape[:-1], 1, device=w.x.device, dtype=w.x.dtype)], dim=-1)

    def decoder(self, z, w) -> torch.Tensor:
        return z[..., :2]

def f_jax(*xs: jnp.ndarray) -> Union[jnp.ndarray, Tuple[jnp.ndarray, ...]]:
    x, u, f, p = xs
    y1 = x[..., 1]
    y2 = - (gg / p[..., 0]) * (jnp.sin(x[..., 0]) + f[..., 0] + u[..., 0])
    return jnp.stack([y1, y2], axis=-1)
class DPJ(TemplateCorrDif):
    CONT = True
    def extra_setup(self):
        self._jax_layer = JaxWrapper(f_jax, jit=True)

    def base_dynamics(self, x: torch.Tensor, u: torch.Tensor, f: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return self._jax_layer(x, u, f, p)

    def encoder(self, w) -> torch.Tensor:
        return torch.cat([w.x, torch.zeros(*w.x.shape[:-1], 1, device=w.x.device, dtype=w.x.dtype)], dim=-1)

    def decoder(self, z, w) -> torch.Tensor:
        return z[..., :2]

mdl_kl = {
    "name" : 'res_model',
    "encoder_layers" : 1,
    "decoder_layers" : 1,
    "residual_layers" : 1,
    "residual_dimension" : 1,
    "latent_layers" : 1,
    "latent_dimension" : 1,
    "hidden_dimension" : 32,
    "activation" : "none",
    "end_activation" : False,
    "weight_init" : "xavier_uniform",
    "gain" : 0.1,}

trn_nd = {
    "n_epochs": 500,
    "save_interval": 50,
    "load_checkpoint": False,
    "learning_rate": 5e-3,
    "decay_rate": 0.999,
    "sweep_lengths": [5, 10, 15, 20],
    "sweep_epoch_step": 100,
    "ode_method": "dopri5",
    "ode_args": {
        "rtol": 1.e-7,
        "atol": 1.e-9}
    }
config_path = 'dyn_model.yaml'

cfgs = [
    ('dp_nd', DPT, NODETrainer,     {"model": mdl_kl, "training" : trn_nd}),
    ('dj_nd', DPJ, NODETrainer,     {"model": mdl_kl, "training" : trn_nd}),
    ]

IDX = [0, 1]
labels = [cfgs[i][0] for i in IDX]

ifdat = 0
iftrn = 1
ifviz = 1
ifprd = 1

if ifdat:
    sampler = TrajectorySampler(f, g, config='dyn_data.yaml')
    ts, xs, us, ys, ps = sampler.sample(t_grid, batch=B)
    np.savez_compressed('data/dyn.npz', t=ts, x=ys, u=us, p=ps)

    plot_multi_trajs(
        np.array([xs]), ts[0], "DP", us=us,
        labels=['Truth'], ifclose=False)

if iftrn:
    for i in IDX:
        mdl, MDL, Trainer, opt = cfgs[i]
        opt["model"]["name"] = f"dyn_{mdl}"
        trainer = Trainer(config_path, MDL, config_mod=opt)
        trainer.train()

if ifviz:
    for i in IDX:
        mdl, MDL, _, _ = cfgs[i]
        model_graph = visualize_model(
            mdl_class=MDL, checkpoint_path=f'dyn_{mdl}.pt', ref_data='data/dyn.npz',
            depth=1, ifsave=True)

if ifprd:
    sampler = TrajectorySampler(f, g, config='dyn_test.yaml')
    ts, xs, us, ys, ps = sampler.sample(t_grid, batch=5)
    x_data = ys
    u_data = us
    t_data = ts[0]
    p_data = ps

    res = [x_data]
    for i in IDX:
        mdl, MDL, _, _ = cfgs[i]
        _, prd_func = load_model(MDL, f'dyn_{mdl}.pt')
        with torch.no_grad():
            pred = prd_func(x_data, t_data, u=u_data, p=p_data)
        res.append(pred)

    plot_multi_trajs(
        np.array(res), t_data, "DP", us=u_data,
        labels=['Truth'] + labels, ifclose=False)

plt.show()