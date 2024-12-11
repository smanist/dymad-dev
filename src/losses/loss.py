import copy
from typing import Callable, Union
import jax
import jax.numpy as jn
import numpy as np
import optax

def loss_wrapper(single_loss: Callable, keys: list):
    """
    single_loss(params: optax.Params, traj: jn.ndarray) -> jn.ndarray
    Convert a loss defined on one trajectory to batch of trajs with gradients
    """
    vslo = jax.vmap(single_loss, in_axes=(None,0))
    def vloss(params: optax.Params, batch: jn.ndarray) -> Union[jn.ndarray,dict]:
        Ls = vslo(params, batch)
        Ls = jn.mean(Ls, axis=0)
        L  = jn.sum(Ls)
        res = {'L' : L}
        res.update(**dict(zip(keys[1:], Ls)))
        return L, res

    vloss_vg = jax.value_and_grad(vloss, has_aux=True)
    def loss_func(params: optax.Params, batch: jn.ndarray) -> Union[dict,dict]:
        (_, losses), grads = vloss_vg(params, batch)
        return grads, losses

    return loss_func

def reset_wrapper(integral: Callable, tag: str):
    """
    integral(params: optax.Params, traj: jn.ndarray) -> Union[jn.ndarray,jn.ndarray]
    Integrate features defined on one trajectory to batch of LHS and RHS,
    needed in two-step optimization.
    """
    vint = jax.vmap(integral, in_axes=(None,0))

    def param_reset(params: optax.Params, trajs: jn.ndarray) -> jn.ndarray:
        dz, df = vint(params, trajs)
        L = dz.reshape(-1, dz.shape[-1])
        R = df.reshape(-1, df.shape[-1])
        _A = jn.linalg.lstsq(R, L)[0].T
        params.update(**{tag : _A})
        return params

    return param_reset
