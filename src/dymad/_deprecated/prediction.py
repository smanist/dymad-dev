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

    # # Dimension checking
    # if x0.ndim != 2:
    #     raise ValueError(f"For graph models, x0 must be 2D (n_nodes, n_features), got shape {x0.shape}")
    # if us.ndim < 1 or us.ndim > 2:
    #     raise ValueError(f"For graph models, us must be 1D (n_controls,) or 2D (n_steps, n_controls), got shape {us.shape}")

    # # Check node count if model has n_nodes attribute
    # if hasattr(model, 'n_nodes') and x0.shape[0] != model.n_nodes:
    #     raise ValueError(f"x0 must have {model.n_nodes} nodes, got {x0.shape[0]}")

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

    # # Initial encoding
    # t0 = torch.tensor(ts[0]).to(device)
    # u0 = interp(t0)
    z0 = model.encoder(DynGeoData([x0], [us[0]], [edge_index]))
    # w0 = w0.T.flatten().detach()
    logger.debug(f"predict_graph_continuous: Encoded initial state to latent dimension {z0.shape}")

    # def ode_func(t, w):
    #     # # Reshape latent vector to (n_nodes, latent_dim)
    #     # w_reshaped = w.reshape(-1, model.n_nodes).T
    #     u_t = interp(t)
    #     # Get dynamics
    #     w, w_dot = model.dynamics(w, u_t)
    #     return w_dot.squeeze().detach()

    def ode_func(t, z):
        u = interp(t)
        x = model.decoder(z, DynGeoData([u], [u], [edge_index]))
        _, z_dot, _ = model(DynGeoData([x], [u], [edge_index]))
        return z_dot

    # ODE integration
    logger.debug(f"predict_graph_continuous: Starting ODE integration with initial latent state shape {z0.shape}")
    z_traj = odeint(ode_func, z0, ts, method=method)
    logger.debug(f"predict_graph_continuous: Completed ODE integration, latent trajectory shape: {z_traj.shape}")

    print(z_traj.shape)

    # Reshape and decode trajectory
    z_traj = z_traj.reshape(len(ts), -1, model.n_nodes).permute(0, 2, 1)
    z_pred = [model.decoder(z, edge_index) for z in z_traj]
    x_traj = torch.stack(z_pred)

    logger.debug(f"predict_graph_continuous: Successfully completed prediction with final shape {x_traj.shape}")
    return x_traj
