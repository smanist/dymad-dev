import numpy as np
import torch
from typing import Tuple, Dict, Callable

from dymad.utils.weak import generate_weak_weights

def generate_weak_form_params(metadata, dtype, device) -> None:
    if len(metadata["dt_and_n_steps"]) > 1:
        raise ValueError("Weak form generation is not currently supported for trajectories with different lengths.")

    N      = metadata["config"]["training"]["weak_form_params"]["N"]
    dN     = metadata["config"]["training"]["weak_form_params"]["dN"]
    ordpol = metadata["config"]["training"]["weak_form_params"]["ordpol"]
    ordint = metadata["config"]["training"]["weak_form_params"]["ordint"]
    alpha  = metadata["config"]["training"]["weak_form_params"].get("alpha", 1.0)

    # Call the generate_weak_weights function to get C, D, and K.
    C, D, K = generate_weak_weights(
        dt=metadata["dt_and_n_steps"][0][0],
        n_steps=metadata["dt_and_n_steps"][0][1],
        n_integration_points=N,
        integration_stride=dN,
        poly_order=ordpol,
        int_rule_order=ordint,
    )

    # Convert weights to torch tensors and store in the weak_dyn_param dictionary.
    weak_dyn_param = {
        "C": torch.tensor(C, dtype=dtype, device=device),
        "D": torch.tensor(D, dtype=dtype, device=device),
        "K": K,
        "N": N,
        "dN": dN,
        "ordPoly": ordpol,
        "ordInt": ordint,
        "alpha": alpha,
    }
    return weak_dyn_param

def weak_form_loss(truth: torch.Tensor, pred: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                  weak_dyn_param: Dict, criterion: Callable,
                  reconstruction_weight: float = 1.0, dynamics_weight: float = 1.0) -> torch.Tensor:
    """Compute weak form loss for a single trajectory.

    Mathematical foundation:
    - For dynamical systems dx/dt = f(x), we enforce the weak form:
      ∫ φ(t) * dx/dt dt = ∫ φ(t) * f(x) dt  (where φ are test functions)
    - In matrix form: C*z ≈ D*z_dot
      where C contains test functions and D contains their derivatives

    Implementation details:
    1. For each trajectory, applies sliding windows of size N with stride dN
    2. Enforces C*z ≈ D*z_dot for each window (weak form consistency)
    3. Also enforces reconstruction accuracy via decoder loss
    4. Optimized to reuse C and D matrices and minimize tensor operations

    Args:
        truth: Ground truth states of shape (n_steps, n_states)
        pred: Tuple of (latent states, latent dynamics, reconstructed states)
        weak_dyn_param: Dictionary containing weak form parameters (C, D, N, dN, K)
        criterion: Loss function to use (e.g., MSE)
        reconstruction_weight: Weight for reconstruction loss component
        dynamics_weight: Weight for weak form dynamics loss component

    Returns:
        Weighted combination of weak form and reconstruction loss
    """
    # Extract weak form parameters
    C, D = weak_dyn_param['C'], weak_dyn_param['D']
    N, dN, K = weak_dyn_param['N'], weak_dyn_param['dN'], weak_dyn_param['K']
    z, z_dot, x_hat = pred

    # Create sliding windows for weak form computation
    z_windows = z.unfold(0, N, dN).permute(0, 2, 1)  # (n_windows, N, n_states)
    z_dot_windows = z_dot.unfold(0, N, dN).permute(0, 2, 1)  # (n_windows, N, n_states)

    # Limit number of windows if needed
    K = min(K, len(z_windows))

    # Expand weak form matrices for batch computation
    C_expanded = C.unsqueeze(0).expand(K, -1, -1)  # (K, 2, N)
    D_expanded = D.unsqueeze(0).expand(K, -1, -1)  # (K, 2, N)

    # Compute weak form loss (dynamics consistency)
    truth_weak = torch.bmm(C_expanded, z_windows).view(-1, z.shape[1])
    pred_weak = torch.bmm(D_expanded, z_dot_windows).view(-1, z_dot.shape[1])
    weak_loss = criterion(pred_weak, truth_weak)

    # Compute reconstruction loss
    recon_loss = criterion(truth, x_hat)

    # Return weighted combination
    return dynamics_weight * weak_loss + reconstruction_weight * recon_loss

def weak_form_loss_batch(batch: torch.Tensor, pred_batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                        n_total_state_features: int, weak_dyn_param: Dict, criterion: Callable,
                        reconstruction_weight: float = 1.0, dynamics_weight: float = 1.0) -> torch.Tensor:
    """Compute weak form loss for a batch of trajectories.

    Args:
        batch: Batch of trajectories of shape (batch_size, n_steps, n_features)
        pred_batch: Tuple of (latent states, latent dynamics, reconstructed states) for the batch
        metadata: Dictionary containing metadata including weak form parameters
        criterion: Loss function to use (e.g., MSE)
        reconstruction_weight: Weight for reconstruction loss component
        dynamics_weight: Weight for weak form dynamics loss component

    Returns:
        Mean weighted loss across the batch
    """
    z_batch, z_dot_batch, x_hat_batch = pred_batch

    # Compute loss for each trajectory in the batch
    losses = [
        weak_form_loss(
            traj[..., :n_total_state_features],  # Extract state features
            [z, z_dot, x_hat],
            weak_dyn_param,
            criterion,
            reconstruction_weight=reconstruction_weight,
            dynamics_weight=dynamics_weight
        )
        for traj, z, z_dot, x_hat in zip(
            batch,
            z_batch,
            z_dot_batch,
            x_hat_batch
        )
    ]

    return torch.stack(losses).mean()