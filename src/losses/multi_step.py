import copy
from typing import Callable, Union
import jax
import jax.numpy as jn
import numpy as np
import optax

from ..models.model import ModelBase
from .loss import loss_wrapper, reset_wrapper

def make_multi_step_loss(mdl: ModelBase):
    """
    Applicable to DT dynamics.
    Computes cummulative error on the roll out prediction over a horizon.
    Length of horizon is given by the length of sample.
    Lr: Reconstruction loss
    Lz: Traj loss in z space
    Lx: Traj loss in x space
    """
    keys = ['L', 'Lr', 'Lz', 'Lx']
    vfeat = jax.vmap(mdl.features, in_axes=(0,0,None))

    def single_loss(params: optax.Params, data: jn.ndarray) -> jn.ndarray:
        # Variables
        xs = data[:,:mdl.Nx]
        us = data[:,mdl.Nx:]
        zs = mdl.encoder(xs, params)

        # Reconstruction
        xd = mdl.decoder(zs, params)
        Lr = jn.mean((xd-xs)**2)

        # Dynamics
        zp, xp = mdl.predict(xs[0], us, params)

        # Losses
        Lz = jn.mean((zp-zs)**2)
        Lx = jn.mean((xp-xs)**2)

        return jn.array([Lr, Lz, Lx])

    multi_step_loss = loss_wrapper(single_loss, keys)

    def integral(params: optax.Params, traj: jn.ndarray) -> jn.ndarray:
        xs = traj[:,:mdl.Nx]
        us = traj[:,mdl.Nx:]
        zs = mdl.encoder(xs, params)
        df = vfeat(xs, us, params)[:-1]
        dz = zs[1:]
        return dz, df

    param_reset = reset_wrapper(integral, 'As')

    return {
        'loss' : multi_step_loss,
        'reset' : param_reset,
        'keys' : keys}
