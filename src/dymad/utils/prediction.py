import logging
import numpy as np
import scipy.interpolate as sp_inter
import torch
from torchdiffeq import odeint
from typing import Union

from dymad.data import DynData, DynGeoData
from dymad.utils import ControlInterpolator

logger = logging.getLogger(__name__)

def _prepare_data(x0, ts, us, device, edge_index=None):
    is_batch = x0.ndim == 2

    # Initial conditions
    if is_batch:
        if x0.ndim != 2:
            raise ValueError(f"Batch mode: x0 must be 2D. Got x0: {x0.shape}")
        _x0 = x0.clone().detach().to(device)
    else:
        if x0.ndim != 1:
            raise ValueError(f"Single mode: x0 must be 1D. Got x0: {x0.shape}")
        _x0 = x0.clone().detach().to(device).unsqueeze(0)

    # Time stations
    if isinstance(ts, np.ndarray):
        ts = torch.from_numpy(ts).float().to(device)
    else:
        ts = ts.float().to(device)
    n_steps = len(ts)

    # Inputs
    _us = None
    if us is not None:
        if is_batch:
            if us.ndim != 3:
                raise ValueError(f"Batch mode: us must be 3D. Got us: {us.shape}")
            _us = us.clone().detach().to(device)
        else:
            if us.ndim != 2:
                raise ValueError(f"Single mode: us must be 2D. Got us: {us.shape}")
            _us = us.clone().detach().to(device).unsqueeze(0)
        if _us.shape[1] != n_steps:
            raise ValueError(f"us time dimension ({_us.shape[1]}) must match time steps ({n_steps})")

    # Edge indices
    _ei = None
    if edge_index is not None:
        if edge_index.ndim == 2:
            _ei = edge_index.clone().detach().to(device).unsqueeze(0)
        elif edge_index.ndim == 3:
            _ei = edge_index.clone().detach().to(device)
        else:
            raise ValueError(f"edge_index must be 2D or 3D tensor. Got {edge_index.shape}")

    return _x0, ts, _us, n_steps, is_batch, _ei

# ------------------
# Continuous-time case
# ------------------

def predict_continuous(
    model,
    x0: torch.Tensor,
    ts: Union[np.ndarray, torch.Tensor],
    us: torch.Tensor = None,
    method: str = 'dopri5',
    order: str = 'cubic',
    **kwargs
) -> torch.Tensor:
    """
    Predict trajectory(ies) for regular (non-graph) models with batch support.

    Args:
        model: Model with encoder, decoder, and dynamics methods.
        x0 (torch.Tensor): Initial state(s).

            - Single: shape (n_features,)
            - Batch: shape (batch_size, n_features)

        ts (Union[np.ndarray, torch.Tensor]): Time points (n_steps,).
        us (torch.Tensor, optional): Control trajectory(ies).

            - Single: shape (n_steps, n_controls)
            - Batch: shape (batch_size, n_steps, n_controls)

        method (str): ODE solver method (default: 'dopri5').
        order (str): Interpolation method for control inputs ('zoh', 'linear', or 'cubic').

    Returns:
        torch.Tensor: Predicted trajectory(ies).

            - Single: shape (n_steps, n_features)
            - Batch: shape (n_steps, batch_size, n_features)

    Raises:
        ValueError: If input dimensions do not match requirements.
    """
    device = x0.device
    _x0, ts, _us, n_steps, is_batch, _ = _prepare_data(x0, ts, us, device)

    if _us is not None:
        logger.debug(f"predict_continuous: {'Batch' if is_batch else 'Single'} mode (controlled)")
        u0 = _us[:, 0, :]
        z0 = model.encoder(DynData(_x0, u0))
        interp = ControlInterpolator(ts, _us, order=order)
        def ode_func(t, z):
            x = model.decoder(z, None)
            u = interp(t)
            _, z_dot, _ = model(DynData(x, u))
            return z_dot
    else:
        logger.debug(f"predict_continuous: {'Batch' if is_batch else 'Single'} mode (autonomous)")
        z0 = model.encoder(DynData(_x0, None))
        def ode_func(t, z):
            x = model.decoder(z, None)
            _, z_dot, _ = model(DynData(x, None))
            return z_dot

    logger.debug(f"predict_continuous: Starting ODE integration with shape {z0.shape}, method {method}, and interpolation order {order if _us is not None else 'N/A'}")
    z_traj = odeint(ode_func, z0, ts, method=method)
    logger.debug(f"predict_continuous: Completed integration, trajectory shape: {z_traj.shape}")

    x_traj = model.decoder(z_traj.view(-1, z_traj.shape[-1]), None).view(n_steps, z_traj.shape[1], -1)
    if not is_batch:
        x_traj = x_traj.squeeze(1)

    logger.debug(f"predict_continuous: Final trajectory shape {x_traj.shape}")
    return x_traj

def predict_graph_continuous(
    model,
    x0: torch.Tensor,
    ts: Union[np.ndarray, torch.Tensor],
    edge_index: torch.Tensor,
    us: torch.Tensor = None,
    method: str = 'dopri5',
    order: str = 'cubic',
    **kwargs
) -> torch.Tensor:
    """
    Predict trajectory(ies) for (graph) models with batch support.

    Args:
        model: Model with encoder, decoder, and dynamics methods.
        x0 (torch.Tensor): Initial state(s).

            - Single: shape (n_features,)
            - Batch: shape (batch_size, n_features)

        ts (Union[np.ndarray, torch.Tensor]): Time points (n_steps,).
        edge_index (torch.Tensor): Edge indices for the graph.
        us (torch.Tensor, optional): Control trajectory(ies).

            - Single: shape (n_steps, n_controls)
            - Batch: shape (batch_size, n_steps, n_controls)

        method (str): ODE solver method (default: 'dopri5').
        order (str): Interpolation method for control inputs ('zoh', 'linear', or 'cubic').

    Returns:
        torch.Tensor: Predicted trajectory(ies).

            - Single: shape (n_steps, n_features)
            - Batch: shape (n_steps, batch_size, n_features)

    Raises:
        ValueError: If input dimensions do not match requirements.
    """
    device = x0.device
    _x0, ts, _us, n_steps, is_batch, _ei = _prepare_data(x0, ts, us, device, edge_index=edge_index)
    _data = DynGeoData(None, None, _ei)

    if _us is not None:
        logger.debug(f"predict_graph_continuous: {'Batch' if is_batch else 'Single'} mode (controlled)")
        u0 = _us[:, 0, :]
        z0 = model.encoder(DynGeoData(_x0, u0, _ei))
        interp = ControlInterpolator(ts, _us, order=order)
        def ode_func(t, z):
            x = model.decoder(z, _data)
            u = interp(t)
            _, z_dot, _ = model(DynGeoData(x, u, _ei))
            return z_dot
    else:
        logger.debug(f"predict_graph_continuous: {'Batch' if is_batch else 'Single'} mode (autonomous)")
        z0 = model.encoder(DynGeoData(_x0, None, _ei))
        def ode_func(t, z):
            x = model.decoder(z, _data)
            _, z_dot, _ = model(DynGeoData(x, None, _ei))
            return z_dot

    logger.debug(f"predict_graph_continuous: Starting ODE integration with shape {z0.shape}, method {method}, and interpolation order {order if _us is not None else 'N/A'}")
    z_traj = odeint(ode_func, z0, ts, method=method)
    logger.debug(f"predict_graph_continuous: Completed integration, trajectory shape: {z_traj.shape}")

    tmp = z_traj.permute(1, 0, 2)  # (batch_size, n_steps, n_features)
    x_traj = model.decoder(tmp, _data).permute(1, 0, 2)

    if not is_batch:
        x_traj = x_traj.squeeze(1)

    logger.debug(f"predict_graph_continuous: Final trajectory shape {x_traj.shape}")
    return x_traj

# ------------------
# Discrete-time case
# ------------------

def predict_discrete(
    model,
    x0: torch.Tensor,
    us: torch.Tensor,
    **kwargs
) -> torch.Tensor:
    """
    Predict trajectory(ies) for regular (non-graph) models with batch support.

    Args:
        model: Model with encoder, decoder, and dynamics methods
        x0: Initial state(s):

            - Single: (n_features,)
            - Batch: (batch_size, n_features)

        us: Control trajectory(ies):

            - Single: (n_steps, n_controls)
            - Batch: (batch_size, n_steps, n_controls)

    Returns:
        torch.Tensor:
            Predicted trajectory(ies)

            - Single: (n_steps, n_features)
            - Batch: (n_steps, batch_size, n_features)

    Raises:
        ValueError: If input dimensions don't match requirements
    """
    device = x0.device
    # Use _prepare_data for consistency
    _x0, _, _us, n_steps, is_batch, _ = _prepare_data(x0, None, us, device)

    logger.debug(f"predict_discrete: {'Batch' if is_batch else 'Single'} mode")

    # Initial state preparation
    u0 = _us[:, 0, :]
    z0 = model.encoder(DynData(_x0, u0))

    # Discrete-time forward pass
    logger.debug(f"predict_discrete: Starting forward iterations with shape {z0.shape}")
    z_traj = [z0]
    for k in range(n_steps - 1):
        x_k = model.decoder(z_traj[-1], None)
        u_k = _us[:, k, :]
        _, z_next, _ = model(DynData(x_k, u_k))
        z_traj.append(z_next)
    z_traj = torch.stack(z_traj, dim=0)  # (n_steps, batch_size, z_dim)
    logger.debug(f"predict_discrete: Completed integration, trajectory shape: {z_traj.shape}")

    x_traj = model.decoder(z_traj.view(-1, z_traj.shape[-1]), None).view(n_steps, z_traj.shape[1], -1)
    if not is_batch:
        x_traj = x_traj.squeeze(1)

    logger.debug(f"predict_discrete: Final trajectory shape {x_traj.shape}")
    return x_traj
