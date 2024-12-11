import copy
from typing import Callable, Union
import jax
import jax.numpy as jn
import numpy as np
import optax

from .loss import loss_wrapper, reset_wrapper
from ..models.model import ModelBase
from ..utils import weight_nc, get_base, get_der1

def make_wf_smpl_loss(mdl: ModelBase, ltraj, ordint, dt):
    """
    Applicable to CT dynamics.

    int_a^b \phi w \dot{x} dt = int_a^b \phi w f(x,u) dt
    LHS = (\phi w x)|_a^b - int_a^b \dot{\phi w} x dt
    Discretize to `CX = DF`.

    Here we choose \phi w = 1.

    Lr: Reconstruction loss
    Ld: Weak form loss
    """
    keys = ['L', 'Lr', 'Ld']
    W = weight_nc(ltraj, ordint, dt)
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
        fs = vfeat(xs, us, params)
        dz = jn.dot(W, jn.dot(fs, params['As'].T))
        _d = zs[-1]-zs[0]-dz
        Ld = jn.mean(_d**2) / ltraj

        return jn.array([Lr, Ld])

    wf_smpl_loss = loss_wrapper(single_loss, keys)

    def integral(params: optax.Params, traj: jn.ndarray) -> jn.ndarray:
        xs = traj[:,:mdl.Nx]
        us = traj[:,mdl.Nx:]
        zs = mdl.encoder(xs, params)
        fs = vfeat(xs, us, params)
        df = jn.dot(W, fs).squeeze()
        dz = zs[-1]-zs[0]
        return dz, df

    param_reset = reset_wrapper(integral, 'As')

    return {
        'loss' : wf_smpl_loss,
        'reset' : param_reset,
        'keys' : keys}


def make_wf_jaco_loss(mdl: ModelBase, ltraj, ordpol, ordint, dt):
    """
    Here we choose \phi to be Jacobi(1,1) polynomials.
    """
    keys = ['L', 'Lr', 'Ld']

    # Weight calculation
    L = (ltraj-1)*dt/2
    h = np.linspace(-1, 1, ltraj)
    P0 = get_base(ordpol, h)
    P1 = get_der1(ordpol, h) / L
    w0 = 1-h**2
    w1 = -2*h / L
    W = weight_nc(ltraj, ordint, dt)
    C = -(P1*w0+P0*w1)*W     # Integral with x for x_dot
    D = P0*w0*W              # Integral with f(x)

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
        fs = jn.dot(vfeat(xs, us, params), params['As'].T)
        _d = jn.dot(C, zs) - jn.dot(D, fs)
        Ld = jn.mean(_d**2) / ltraj

        return jn.array([Lr, Ld])

    wf_jaco_loss = loss_wrapper(single_loss, keys)

    def integral(params: optax.Params, traj: jn.ndarray) -> jn.ndarray:
        xs = traj[:,:mdl.Nx]
        us = traj[:,mdl.Nx:]
        df = jn.dot(D, vfeat(xs, us, params))
        dz = jn.dot(C, mdl.encoder(xs, params))
        return dz, df

    param_reset = reset_wrapper(integral, 'As')

    return {
        'loss' : wf_jaco_loss,
        'reset' : param_reset,
        'keys' : keys}
