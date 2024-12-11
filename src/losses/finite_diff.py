import copy
from typing import Callable, Union
import jax
import jax.numpy as jn
import numpy as np
import optax

from .loss import loss_wrapper, reset_wrapper
from ..models.model import ModelBase
from ..utils import weight_bdf

def make_finite_diff_loss(mdl: ModelBase, ordbdf, dt):
    """
    Applicable to CT dynamics.

    Approximate the time derivative by BDF.

    Lr: Reconstruction loss
    Ld: Equation loss
    """
    keys = ['L', 'Lr', 'Ld']
    W = weight_bdf(ordbdf)
    vfeat = jax.vmap(mdl.features, in_axes=(0,0,None))

    def single_loss(params: optax.Params, traj: jn.ndarray) -> jn.ndarray:
        # Variables
        xs = traj[:,:mdl.Nx]
        us = traj[:,mdl.Nx:]
        zs = mdl.encoder(xs, params)

        # Reconstruction
        xd = mdl.decoder(zs, params)
        Lr = jn.mean((xd-xs)**2)

        # Dynamics
        dz = jn.dot(W, zs)
        df = jn.dot(params['As'], mdl.features(xs[-1], us[-1], params))
        _d = dz - df*dt
        Ld = jn.mean(_d**2)

        return jn.array([Lr, Ld])

    finite_diff_loss = loss_wrapper(single_loss, keys)

    def derivative(params: optax.Params, traj: jn.ndarray) -> jn.ndarray:
        xs = traj[:,:mdl.Nx]
        us = traj[:,mdl.Nx:]
        zs = mdl.encoder(xs, params)
        dz = jn.dot(W, zs).squeeze()
        df = mdl.features(xs[-1], us[-1], params).squeeze() * dt
        return dz, df

    param_reset = reset_wrapper(derivative, 'As')

    return {
        'loss' : finite_diff_loss,
        'reset' : param_reset,
        'keys' : keys}
