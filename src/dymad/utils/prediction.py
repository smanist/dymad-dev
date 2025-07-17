import logging
import numpy as np
import scipy.interpolate as sp_inter
import torch
from torchdiffeq import odeint
from typing import Union

from dymad.data import DynData, DynGeoData
from dymad.utils import ControlInterpolator

logger = logging.getLogger(__name__)

def predict_continuous(
    model,
    x0: torch.Tensor,
    us: torch.Tensor,
    ts: Union[np.ndarray, torch.Tensor],
    method: str = 'dopri5',
    order: str = 'cubic',
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

        ts: Time points (n_steps,)
        method: ODE solver method
        order: Interpolation method for control inputs ('zoh', 'linear' or 'cubic')

    Returns:
        np.ndarray:
            Predicted trajectory(ies)

            - Single: (n_steps, n_features)
            - Batch: (n_steps, batch_size, n_features)

    Raises:
        ValueError: If input dimensions don't match requirements
    """
    device = x0.device
    is_batch = x0.ndim == 2

    if is_batch:
        if x0.ndim != 2 or us.ndim != 3:
            raise ValueError(f"Batch mode: x0 must be 2D, us must be 3D. Got x0: {x0.shape}, us: {us.shape}")
        _x0 = x0.clone().detach().to(device)
        _us = us.clone().detach().to(device)
    else:
        if x0.ndim != 1 or us.ndim != 2:
            raise ValueError(f"Single mode: x0 must be 1D, us must be 2D. Got x0: {x0.shape}, us: {us.shape}")
        _x0 = x0.clone().detach().to(device).unsqueeze(0)
        _us = us.clone().detach().to(device).unsqueeze(0)

    # Convert ts to tensor
    if isinstance(ts, np.ndarray):
        ts = torch.from_numpy(ts).float().to(device)
    else:
        ts = ts.float().to(device)

    n_steps = len(ts)
    if _us.shape[1] != n_steps:
        raise ValueError(f"us time dimension ({_us.shape[1]}) must match time steps ({n_steps})")

    logger.debug(f"predict_continuous: {'Batch' if is_batch else 'Single'} mode")

    # Initial state preparation
    u0 = _us[:, 0, :]
    z0 = model.encoder(DynData(_x0, u0))

    interp = ControlInterpolator(ts, _us, order=order)
    def ode_func(t, z):
        x = model.decoder(z, None)
        u = interp(t)
        _, z_dot, _ = model(DynData(x, u))
        return z_dot

    # Integrate
    logger.debug(f"predict_continuous: Starting ODE integration with shape {z0.shape}, method {method}, and interpolation order {order}")
    z_traj = odeint(ode_func, z0, ts, method=method)
    logger.debug(f"predict_continuous: Completed integration, trajectory shape: {z_traj.shape}")

    x_traj = model.decoder(z_traj.view(-1, z_traj.shape[-1]), None).view(n_steps, z_traj.shape[1], -1)
    if not is_batch:
        x_traj = x_traj.squeeze(1)

    logger.debug(f"predict_continuous: Final trajectory shape {x_traj.shape}")
    return x_traj

def predict_continuous_auto(
    model,
    x0: torch.Tensor,
    ts: Union[np.ndarray, torch.Tensor],
    method: str = 'dopri5',
    **kwargs
) -> torch.Tensor:
    """
    Predict trajectory(ies) for regular (non-graph) models with batch support.

    Autonomous case.

    Args:
        model: Model with encoder, decoder, and dynamics methods
        x0: Initial state(s):

            - Single: (n_features,)
            - Batch: (batch_size, n_features)

        ts: Time points (n_steps,)
        method: ODE solver method
        order: Interpolation method for control inputs ('zoh', 'linear' or 'cubic')

    Returns:
        np.ndarray:
            Predicted trajectory(ies)

            - Single: (n_steps, n_features)
            - Batch: (n_steps, batch_size, n_features)

    Raises:
        ValueError: If input dimensions don't match requirements
    """
    device = x0.device
    is_batch = x0.ndim == 2

    if is_batch:
        if x0.ndim != 2:
            raise ValueError(f"Batch mode: x0 must be 2D. Got x0: {x0.shape}")
        _x0 = x0.clone().detach().to(device)
    else:
        if x0.ndim != 1:
            raise ValueError(f"Single mode: x0 must be 1D. Got x0: {x0.shape}")
        _x0 = x0.clone().detach().to(device).unsqueeze(0)

    # Convert ts to tensor
    if isinstance(ts, np.ndarray):
        ts = torch.from_numpy(ts).float().to(device)
    else:
        ts = ts.float().to(device)

    n_steps = len(ts)

    logger.debug(f"predict_continuous_auto: {'Batch' if is_batch else 'Single'} mode")

    # Initial state preparation
    z0 = model.encoder(DynData(_x0, None))

    def ode_func(t, z):
        x = model.decoder(z, None)
        _, z_dot, _ = model(DynData(x, None))
        return z_dot

    # Integrate
    logger.debug(f"predict_continuous_auto: Starting ODE integration with shape {z0.shape} and method {method}")
    z_traj = odeint(ode_func, z0, ts, method=method)
    logger.debug(f"predict_continuous_auto: Completed integration, trajectory shape: {z_traj.shape}")

    x_traj = model.decoder(z_traj.view(-1, z_traj.shape[-1]), None).view(n_steps, z_traj.shape[1], -1)
    if not is_batch:
        x_traj = x_traj.squeeze(1)

    logger.debug(f"predict_continuous_auto: Final trajectory shape {x_traj.shape}")
    return x_traj

def predict_graph_continuous(
    model,
    x0: torch.Tensor,
    us: torch.Tensor,
    ts: Union[np.ndarray, torch.Tensor],
    edge_index: torch.Tensor,
    method: str = 'dopri5',
    order: str = 'cubic',
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

            For autonomous systems, use zero-valued controls

        ts: Time points (n_steps,)
        method: ODE solver method
        order: Interpolation method for control inputs ('zoh', 'linear' or 'cubic')

    Returns:
        np.ndarray:
            Predicted trajectory(ies)

            - Single: (n_steps, n_features)
            - Batch: (n_steps, batch_size, n_features)

    Raises:
        ValueError: If input dimensions don't match requirements
    """
    device = x0.device
    is_batch = x0.ndim == 2

    if is_batch:
        if x0.ndim != 2 or us.ndim != 3:
            raise ValueError(f"Batch mode: x0 must be 2D, us must be 3D. Got x0: {x0.shape}, us: {us.shape}")
        _x0 = x0.clone().detach().to(device)
        _us = us.clone().detach().to(device)
    else:
        if x0.ndim != 1 or us.ndim != 2:
            raise ValueError(f"Single mode: x0 must be 1D, us must be 2D. Got x0: {x0.shape}, us: {us.shape}")
        _x0 = x0.clone().detach().to(device).unsqueeze(0)
        _us = us.clone().detach().to(device).unsqueeze(0)

    if edge_index.ndim == 2:
        _ei = edge_index.clone().detach().to(device).unsqueeze(0)
    elif edge_index.ndim == 3:
        _ei = edge_index.clone().detach().to(device)
    else:
        raise ValueError(f"edge_index must be 2D or 3D tensor. Got {edge_index.shape}")

    # Convert ts to tensor
    if isinstance(ts, np.ndarray):
        ts = torch.from_numpy(ts).float().to(device)
    else:
        ts = ts.float().to(device)

    n_steps = len(ts)
    if _us.shape[1] != n_steps:
        raise ValueError(f"us time dimension ({_us.shape[1]}) must match time steps ({n_steps})")

    logger.debug(f"predict_continuous: {'Batch' if is_batch else 'Single'} mode")

    # Initial state preparation
    u0 = _us[:, 0, :]
    z0 = model.encoder(DynGeoData(_x0, u0, _ei))

    interp = ControlInterpolator(ts, _us, order=order)
    _data  = DynGeoData(None, None, _ei)
    def ode_func(t, z):
        x = model.decoder(z, _data)
        u = interp(t)
        _, z_dot, _ = model(DynGeoData(x, u, _ei))
        return z_dot

    # Integrate
    logger.debug(f"predict_continuous: Starting ODE integration with shape {z0.shape}, method {method}, and interpolation order {order}")
    z_traj = odeint(ode_func, z0, ts, method=method)
    logger.debug(f"predict_continuous: Completed integration, trajectory shape: {z_traj.shape}")

    # x_traj = model.decoder(z_traj.view(-1, z_traj.shape[-1]), _data).view(n_steps, z_traj.shape[1], -1)
    tmp = z_traj.permute(1, 0, 2)  # (batch_size, n_steps, n_features)
    x_traj = model.decoder(tmp, _data).permute(1, 0, 2)

    if not is_batch:
        x_traj = x_traj.squeeze(1)

    logger.debug(f"predict_continuous: Final trajectory shape {x_traj.shape}")
    return x_traj
