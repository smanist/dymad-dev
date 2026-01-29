import logging
import numpy as np
import torch
from torchdiffeq import odeint
from typing import Union

from dymad.io import DynData
from dymad.numerics import expm_low_rank, expm_full_rank
from dymad.utils import ControlInterpolator

logger = logging.getLogger(__name__)

def _prepare_data(x0, ts, ws, device):
    # Initial conditions
    # Determines batch size
    is_batch = x0.ndim == 2
    _x0 = x0.clone().detach().to(device)
    if not is_batch:
        _x0 = _x0.unsqueeze(0)
    _Nb = _x0.shape[0]

    # Time stations
    _ts, _Nt = None, None
    if ts is not None:
        if isinstance(ts, np.ndarray):
            _ts = torch.from_numpy(ts).float()
        else:
            _ts = ts.float()
        if is_batch:
            if _ts.ndim == 1:
                _ts = _ts.unsqueeze(0).repeat(_Nb, 1)  # (batch_size, n_steps)
            elif _ts.ndim != 2:
                raise ValueError(f"Batch mode: ts must be 1D or 2D. Got ts: {_ts.shape}")
            if _ts.shape[0] != _Nb:
                raise ValueError(f"Batch mode: ts first dimension must match batch size. Got ts: {_ts.shape}, x0: {_x0.shape}")
        else:
            if _ts.ndim != 1:
                raise ValueError(f"Single mode: ts must be 1D. Got ts: {_ts.shape}")
            _ts = _ts.unsqueeze(0)
        _ts = _ts.to(device)
        _Nt = _ts.shape[-1]

    # Inputs
    _ws, _Nw = None, None
    if ws is not None:
        if is_batch:
            if ws.batch_size == 1 and _Nb > 1:
                _ws = DynData.collate([ws for _ in range(_Nb)])
            elif ws.batch_size == _Nb:
                _ws = ws
            else:
                raise ValueError(f"Batch mode: ws batch size must be 1 or match x0. Got ws: {ws.batch_size}, x0: {_Nb}")

            if _ws._has_graph:
                # Graph mode, batch always 1
                # Need to flatten x0
                _x0 = _x0.view(1, -1)
        else:
            if ws.batch_size is not None and ws.batch_size != 1:
                raise ValueError(f"Single mode: ws batch size must be 1. Got ws: {ws.batch_size}")
            _ws = ws
        _ws = _ws.to(device)
        _Nw = _ws.n_steps
    else:
        _ws = DynData().to(device)

    # Check step consistency
    if _Nt is None:
        if _Nw is None:
            raise ValueError("Either ts or ws must be provided to determine time steps.")
        n_steps = _Nw
    else:
        if _Nw is not None:
            if _Nt != _Nw:
                raise ValueError(f"ts and ws must have the same number of time steps. Got ts: {_Nt}, ws: {_Nw}")
        n_steps = _Nt

    return _x0, _ts, _ws, n_steps, is_batch

def _proc_ztraj(z_traj, model, ws, n_steps, is_batch):
    if ws._has_graph:
        # after stack: z_traj (n_steps, batch_size, node, z_dim)
        tmp = z_traj.permute(1, 0, 2, 3)  # (batch_size, n_steps, node, z_dim)
        x_traj = model.decoder(tmp, ws)
    else:
        x_traj = model.decoder(z_traj.view(-1, z_traj.shape[-1]), None)
        x_traj = x_traj.view(n_steps, z_traj.shape[1], -1).transpose(0, 1)

    if not is_batch:
        x_traj = x_traj.squeeze(0)
    return x_traj

# ------------------
# Continuous-time case
# ------------------

def predict_continuous(
    model,
    x0: torch.Tensor,
    ts: Union[np.ndarray, torch.Tensor],
    ws: DynData = None,
    method: str = 'dopri5',
    order: str = 'cubic',
    **kwargs
) -> torch.Tensor:
    """
    Predict trajectory(ies) for continuous-time models with batch support.

    Args:
        model: Model with encoder, decoder, and dynamics methods.
        x0 (torch.Tensor): Initial state(s).

            - Single: shape (n_features,)
            - Batch: shape (batch_size, n_features)

        ts (Union[np.ndarray, torch.Tensor]): Time points (n_steps,).
        ws: Dataclass containing additional information, e.g., u, p, ei, ew, etc.
        method (str): ODE solver method (default: 'dopri5').
        order (str): Interpolation method for control inputs ('zoh', 'linear', or 'cubic').

    Returns:
        torch.Tensor: Predicted trajectory(ies).

            - Single: shape (n_steps, n_features)
            - Batch: shape (batch_size, n_steps, n_features)
    """
    _x0, _ts, _ws, n_steps, is_batch = _prepare_data(x0, ts, ws, x0.device)
    _ts = _ts[0]

    def bucket(t):
        return torch.searchsorted(_ts, t).clamp(1, _ts.numel()-1)

    _has_u = _ws.u is not None
    if _has_u:
        logger.debug(f"predict_continuous: {'Batch' if is_batch else 'Single'} mode (controlled)")
        u_intp = ControlInterpolator(_ts, _ws.u, order=order)
    else:
        logger.debug(f"predict_continuous: {'Batch' if is_batch else 'Single'} mode (autonomous)")
        u_intp = lambda t: None

    z0 = model.encoder(_ws.get_step(0).set_x(_x0))
    def ode_func(t, z):
        _tk  = bucket(t)
        wtmp = _ws.get_step(_tk)
        u    = u_intp(t)
        x    = model.decoder(z, wtmp.set_u(u))
        wtmp = wtmp.set_x(x)
        z    = model.encoder(wtmp)
        z_dot = model.dynamics(z, wtmp)
        return z_dot

    logger.debug(f"predict_continuous: Starting ODE integration with shape {z0.shape}, method {method}, and interpolation order {order if _has_u else 'N/A'}")
    z_traj = odeint(ode_func, z0, _ts, method=method, **kwargs)
    logger.debug(f"predict_continuous: Completed integration, trajectory shape: {z_traj.shape}")

    x_traj = _proc_ztraj(z_traj, model, _ws, n_steps, is_batch)

    logger.debug(f"predict_continuous: Final trajectory shape {x_traj.shape}")
    return x_traj

def predict_continuous_np(
    model,
    x0: torch.Tensor,
    ts: Union[np.ndarray, torch.Tensor],
    ws: DynData = None,
    method: str = 'dopri5',
    order: str = 'cubic',
    **kwargs
) -> torch.Tensor:
    """
    Predict trajectory(ies) for continuous-time models with batch support.

    No-projection version, meaning during ODE integration, we do not decode
    back to the observation space and encode back; the decoding happens only at the end.
    """
    _x0, _ts, _ws, n_steps, is_batch = _prepare_data(x0, ts, ws, x0.device)
    _ts = _ts[0]

    def bucket(t):
        return torch.searchsorted(_ts, t).clamp(1, _ts.numel()-1)

    _has_u = _ws.u is not None
    if _has_u:
        logger.debug(f"predict_continuous_np: {'Batch' if is_batch else 'Single'} mode (controlled)")
        u_intp = ControlInterpolator(_ts, _ws.u, order=order)
    else:
        logger.debug(f"predict_continuous_np: {'Batch' if is_batch else 'Single'} mode (autonomous)")
        u_intp = lambda t: None

    z0 = model.encoder(_ws.get_step(0).set_x(_x0))
    def ode_func(t, z):
        _tk  = bucket(t)
        wtmp = _ws.get_step(_tk)
        u    = u_intp(t)
        z_dot = model.dynamics(z, wtmp.set_u(u))
        return z_dot

    logger.debug(f"predict_continuous_np: Starting ODE integration with shape {z0.shape}, method {method}, and interpolation order {order if _has_u else 'N/A'}")
    z_traj = odeint(ode_func, z0, _ts, method=method, **kwargs)
    logger.debug(f"predict_continuous_np: Completed integration, trajectory shape: {z_traj.shape}")

    x_traj = _proc_ztraj(z_traj, model, _ws, n_steps, is_batch)

    logger.debug(f"predict_continuous_np: Final trajectory shape {x_traj.shape}")
    return x_traj

def predict_continuous_exp(
    model,
    x0: torch.Tensor,
    ts: Union[np.ndarray, torch.Tensor],
    ws: DynData = None,
    **kwargs
) -> torch.Tensor:
    """
    Predict trajectory(ies) for continuous-time models with batch support.

    Autonomous case using matrix exponential.  In continuous-time, we compute exp(A*dt).

    The step size is assumed to be constant, so when there are batched time series,
    we just take the first one to compute dt.

    Currently only for KBF-type models with linear dynamics.
    """
    _x0, _, _ws, n_steps, is_batch = _prepare_data(x0, ts, ws, x0.device)
    if _ws is not None:
        assert _ws.u is None, "predict_discrete_exp only supports autonomous case."

    # Get the system matrix
    if model.processor_net.mode == "full":
        W = (model.processor_net.weight, )
    else:
        U = model.processor_net.U
        V = model.processor_net.V
        W = (U, V)

    logger.debug(f"predict_continuous_exp: {'Batch' if is_batch else 'Single'} mode (autonomous)")
    z0 = model.encoder(_ws.get_step(0).set_x(_x0))

    logger.debug(f"predict_continuous_exp: Starting ODE integration with shape {z0.shape}")
    if ts.dim() == 2:
        dt = ts[0] - ts[0,0]
    else:
        dt = ts - ts[0]  # (n_steps,)
    if len(W) == 1:
        z_traj = expm_full_rank(W[0].T, dt, z0)
    elif len(W) == 2:
        # Low-rank case: use a specialized function to exponentiate in reduced space
        z_traj = expm_low_rank(W[1], W[0], dt, z0)
    logger.debug(f"predict_continuous_exp: Completed integration, trajectory shape: {z_traj.shape}")

    x_traj = _proc_ztraj(z_traj, model, _ws, n_steps, is_batch)

    logger.debug(f"predict_continuous_exp: Final trajectory shape {x_traj.shape}")
    return x_traj

def predict_continuous_fenc(
    model,
    x0: torch.Tensor,
    ts: Union[np.ndarray, torch.Tensor],
    ws: DynData = None,
    **kwargs
) -> torch.Tensor:
    """
    Predict trajectory(ies) using First-order Euler with Normal Correction (FENC).

    Currently only for kernel machine with tangent kernel.
    """
    _x0, _ts, _ws, n_steps, is_batch = _prepare_data(x0, ts, ws, x0.device)

    logger.debug(f"predict_continuous_fenc: {'Batch' if is_batch else 'Single'} mode")

    z0 = model.encoder(_ws.get_step(0).set_x(_x0))

    # Discrete-time forward pass
    logger.debug(f"predict_continuous_fenc: Starting forward iterations with shape {z0.shape}")
    z_traj = [z0]
    for k in range(n_steps - 1):
        wtmp = _ws.get_step(k)
        z_next = model.fenc_step(z_traj[-1], wtmp, _ts[...,k+1]-_ts[...,k])
        z_traj.append(z_next)

    z_traj = torch.stack(z_traj, dim=0)  # (n_steps, batch_size, z_dim)
    logger.debug(f"predict_continuous_fenc: Completed integration, trajectory shape: {z_traj.shape}")

    x_traj = _proc_ztraj(z_traj, model, _ws, n_steps, is_batch)

    logger.debug(f"predict_continuous_fenc: Final trajectory shape {x_traj.shape}")
    return x_traj

# ------------------
# Discrete-time case
# ------------------

def predict_discrete(
    model,
    x0: torch.Tensor,
    ts: Union[np.ndarray, torch.Tensor],
    ws: DynData = None,
    **kwargs
) -> torch.Tensor:
    """
    Predict trajectory(ies) for discrete-time models with batch support.

    Args:
        model: Model with encoder, decoder, and dynamics methods
        x0: Initial state(s):

            - Single: (n_features,)
            - Batch: (batch_size, n_features)

        ts (Union[np.ndarray, torch.Tensor]): Time points (n_steps,).
        ws: Dataclass containing additional information, e.g., u, p, ei, ew, etc.

    Returns:
        torch.Tensor:
            Predicted trajectory(ies)

            - Single: (n_steps, n_features)
            - Batch: (batch_size, n_steps, n_features)
    """
    _x0, _, _ws, n_steps, is_batch = _prepare_data(x0, ts, ws, x0.device)

    wtmp = _ws.get_step(0).set_x(_x0)
    ztmp = model.encoder(wtmp)            # The initial condition for dynamics
    x0   = model.decoder(ztmp, wtmp)      # The first decoded observation

    x_traj = [x0]
    for k in range(n_steps - 1):
        wtmp = _ws.get_step(k).set_x(x_traj[-1])
        _z   = model.encoder(wtmp)
        ztmp = model.dynamics(_z, wtmp)
        x_k  = model.decoder(ztmp, wtmp)
        x_traj.append(x_k)
    x_traj = torch.stack(x_traj, dim=1)   # (batch_size, n_steps, z_dim)

    if not is_batch:
        x_traj = x_traj.squeeze(0)

    logger.debug(f"predict_discrete: Final trajectory shape {x_traj.shape}")
    return x_traj

def predict_discrete_exp(
    model,
    x0: torch.Tensor,
    ts: Union[np.ndarray, torch.Tensor],
    ws: DynData = None,
    **kwargs
) -> torch.Tensor:
    """
    Predict trajectory(ies) for discrete-time models with batch support.

    In discrete-time, this is equivalent to repeated application of the dynamics.
    """
    _x0, _, _ws, n_steps, is_batch = _prepare_data(x0, ts, ws, x0.device)

    logger.debug(f"predict_discrete: {'Batch' if is_batch else 'Single'} mode")

    # Initial state preparation
    z0 = model.encoder(_ws.get_step(0).set_x(_x0))

    # Discrete-time forward pass
    logger.debug(f"predict_discrete_exp: Starting forward iterations with shape {z0.shape}")
    z_traj = [z0]
    for k in range(n_steps - 1):
        tmp = _ws.get_step(k)
        z_next = model.dynamics(z_traj[-1], tmp)
        z_traj.append(z_next)
    z_traj = torch.stack(z_traj, dim=0)  # (n_steps, batch_size, z_dim)
    logger.debug(f"predict_discrete_exp: Completed integration, trajectory shape: {z_traj.shape}")

    x_traj = _proc_ztraj(z_traj, model, _ws, n_steps, is_batch)

    logger.debug(f"predict_discrete_exp: Final trajectory shape {x_traj.shape}")
    return x_traj
