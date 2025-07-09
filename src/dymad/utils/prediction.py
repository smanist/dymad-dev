import logging
import numpy as np
import scipy.interpolate as sp_inter
import torch
from torchdiffeq import odeint
from typing import Union

from dymad.utils.modules import ControlInterpolator, DynData

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
        x = model.decoder(z)
        u = interp(t)
        _, z_dot, _ = model(DynData(x, u))
        return z_dot

    # Integrate
    logger.debug(f"predict_continuous: Starting ODE integration with shape {z0.shape}, method {method}, and interpolation order {order}")
    z_traj = odeint(ode_func, z0, ts, method=method)
    logger.debug(f"predict_continuous: Completed integration, trajectory shape: {z_traj.shape}")

    x_traj = model.decoder(z_traj.view(-1, z_traj.shape[-1])).view(n_steps, z_traj.shape[1], -1)
    if not is_batch:
        x_traj = x_traj.squeeze(1)

    logger.debug(f"predict_continuous: Final trajectory shape {x_traj.shape}")
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
    Predict single trajectory for graph models.

    Args:
        model: Graph model with encoder, decoder, and dynamics methods
        x0: Initial node states (n_nodes, n_features)
        us: Control trajectory:

            - Constant control: (n_controls,) or (1, n_controls)
            - Time-varying control: (n_steps, n_controls)

            For autonomous systems, use zero-valued controls

        ts: Time points (n_steps,)
        edge_index: Graph connectivity tensor (2, n_edges)
        method: ODE solver method
        order: Interpolation method for control inputs ('zoh', 'linear' or 'cubic')

    Returns:
        np.ndarray: Predicted trajectory (n_steps, n_nodes, n_features)

    Raises:
        ValueError: If input dimensions don't match requirements
    """

    device = x0.device

    logger.debug(f"predict_graph_continuous: Single graph mode - x0 shape: {x0.shape}, us shape: {us.shape}")

    # Dimension checking
    if x0.ndim != 2:
        raise ValueError(f"For graph models, x0 must be 2D (n_nodes, n_features), got shape {x0.shape}")
    if us.ndim < 1 or us.ndim > 2:
        raise ValueError(f"For graph models, us must be 1D (n_controls,) or 2D (n_steps, n_controls), got shape {us.shape}")

    # Check node count if model has n_nodes attribute
    if hasattr(model, 'n_nodes') and x0.shape[0] != model.n_nodes:
        raise ValueError(f"x0 must have {model.n_nodes} nodes, got {x0.shape[0]}")

    # Convert ts to tensor if needed
    if isinstance(ts, np.ndarray):
        ts = torch.from_numpy(ts).float().to(device)
    else:
        ts = ts.float().to(device)

    n_steps = len(ts)
    logger.debug(f"predict_graph_continuous: Integrating over {n_steps} time steps using {method} solver")

    # # Handle control inputs
    # if (us.ndim == 1) or (us.ndim == 2 and us.shape[0] == 1):
    #     # Constant control
    #     logger.debug(f"predict_graph_continuous: Using constant control")
    #     u_func = lambda t: us.to(device)
    # else:
    #     # Time-varying control: interpolate
    #     logger.debug(f"predict_graph_continuous: Using time-varying control with interpolation")
    #     u_np = us.cpu().detach().numpy()
    #     ts_np = ts.cpu().detach().numpy()
    #     u_interp = sp_inter.interp1d(ts_np[:len(u_np)], u_np, axis=0, fill_value='extrapolate')
    #     u_func = lambda t: torch.tensor(u_interp(t.cpu().detach().numpy()),
    #                                   dtype=us.dtype).to(device)

    interp = ControlInterpolator(ts, us, order=order)

    # Initial encoding
    t0 = torch.tensor(ts[0]).to(device)
    u0 = interp(t0)
    w0 = model.encoder(x0, u0, edge_index)
    w0 = w0.T.flatten().detach()
    logger.debug(f"predict_graph_continuous: Encoded initial state to latent dimension {w0.shape}")

    def ode_func(t, w):
        # Reshape latent vector to (n_nodes, latent_dim)
        w_reshaped = w.reshape(-1, model.n_nodes).T
        u_t = interp(t)
        # Get dynamics
        w_dot = model.dynamics(w_reshaped, u_t)[1]  # Get w_dot from dynamics
        return w_dot.squeeze().detach()

    # ODE integration
    logger.debug(f"predict_graph_continuous: Starting ODE integration with initial latent state shape {w0.shape}")
    w_traj = odeint(ode_func, w0.squeeze(), ts, method=method)
    logger.debug(f"predict_graph_continuous: Completed ODE integration, latent trajectory shape: {w_traj.shape}")

    # Reshape and decode trajectory
    w_traj = w_traj.reshape(len(ts), -1, model.n_nodes).permute(0, 2, 1)
    z_pred = [model.decoder(w, edge_index) for w in w_traj]
    x_traj = torch.stack(z_pred)

    logger.debug(f"predict_graph_continuous: Successfully completed prediction with final shape {x_traj.shape}")
    return x_traj
