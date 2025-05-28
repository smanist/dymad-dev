import numpy as np
import torch
from typing import Tuple, Dict, Callable

def weak_form_loss(truth: torch.Tensor, pred: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], 
                  weak_dyn_param: Dict, criterion: Callable) -> torch.Tensor:
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
        
    Returns:
        Combined weak form and reconstruction loss
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
    
    # Compute weak form loss
    truth_weak = torch.bmm(C_expanded, z_windows).view(-1, z.shape[1])
    pred_weak = torch.bmm(D_expanded, z_dot_windows).view(-1, z_dot.shape[1])
    weak_loss = criterion(pred_weak, truth_weak)
    
    # Compute reconstruction loss
    recon_loss = criterion(truth, x_hat)
    
    return weak_loss + recon_loss

def weak_form_loss_batch(batch: torch.Tensor, pred_batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], 
                        metadata: Dict, criterion: Callable) -> torch.Tensor:
    """Compute weak form loss for a batch of trajectories.
    
    Args:
        batch: Batch of trajectories of shape (batch_size, n_steps, n_features)
        pred_batch: Tuple of (latent states, latent dynamics, reconstructed states) for the batch
        metadata: Dictionary containing metadata including weak form parameters
        criterion: Loss function to use (e.g., MSE)
        
    Returns:
        Mean loss across the batch
    """
    z_batch, z_dot_batch, x_hat_batch = pred_batch
    n_states = metadata['n_state_features']
    
    # Compute loss for each trajectory in the batch
    losses = [
        weak_form_loss(
            traj[..., :n_states],  # Extract state features
            [z, z_dot, x_hat],
            metadata['weak_dyn_param'],
            criterion
        )
        for traj, z, z_dot, x_hat in zip(
            batch,
            z_batch,
            z_dot_batch,
            x_hat_batch
        )
    ]
    
    return torch.stack(losses).mean()

# def clean_weak_form_loss(batch, predBatch, metadata, criterion):
#     # TODO: check this for various trajectory lengths

#     """
#     Computes the weak form loss for a batch of trajectories.
    
#     This implements a batch-optimized version of the Galerkin method for neural ODEs.
#     The weak form approach reformulates differential equations into integral form using
#     test functions, which provides better numerical properties for learning dynamics.
    
#     Mathematical foundation:
#     - For dynamical systems dx/dt = f(x), we enforce the weak form:
#       ∫ φ(t) * dx/dt dt = ∫ φ(t) * f(x) dt  (where φ are test functions)
#     - In matrix form: C*w ≈ D*w_dot
#       where C contains test functions and D contains their derivatives
    
#     Args:
#         batch: Tensor containing batch of trajectories [batch_size, time_steps, features]
#         predBatch: Tuple of (w_batch, wDot_batch, xHat_batch) containing:
#                   - w_batch: Embedded states [batch_size, time_steps, embed_dim]
#                   - wDot_batch: Derivatives [batch_size, time_steps, embed_dim]
#                   - xHat_batch: Reconstructed states [batch_size, time_steps, state_dim]
#         dataMeta: Dictionary with metadata and weak form parameters:
#                  - n_state_features: Number of state features
#                  - weakDynParam: Dictionary containing:
#                    * C: Test function matrix
#                    * D: Test function derivative matrix
#                    * N: Window size for sliding windows
#                    * dN: Stride between windows
#                    * K: Maximum number of windows
#         criterion: Loss function to use (e.g., MSELoss)
        
#     Returns:
#         Mean loss value across the batch (weak form consistency + reconstruction loss)
    
#     """
#     w_batch, wDot_batch, xHat_batch = predBatch
#     states = batch[..., :metadata['n_state_features']]
#     batch_size = states.shape[0]
    
#     C, D = metadata['weakDynParam']['C'], metadata['weakDynParam']['D']
#     N, dN, K = metadata['weakDynParam']['N'], metadata['weakDynParam']['dN'], metadata['weakDynParam']['K']
#     device = w_batch.device
    
#     # Move to correct device once
#     if C.device != device:
#         C = C.to(device)
#     if D.device != device:
#         D = D.to(device)
    
#     # Pre-reshape C and D for broadcasting
#     C_reshaped = C.unsqueeze(0)  # [1, C.shape[0], C.shape[1]]
#     D_reshaped = D.unsqueeze(0)  # [1, D.shape[0], D.shape[1]]
    
#     # Initialize loss accumulator
#     total_loss = 0.0
    
#     # Process each trajectory but batch operations
#     for i in range(batch_size):
#         w, wDot, xHat = w_batch[i], wDot_batch[i], xHat_batch[i]
#         truth = states[i]
        
#         # Compute sliding windows 
#         wR = w.unfold(0, N, dN).transpose(1, 2)
#         wDotR = wDot.unfold(0, N, dN).transpose(1, 2)
        
#         # Get actual number of windows
#         actual_K = min(K, wR.size(0))
#         if actual_K <= 0:
#             total_loss += criterion(xHat, truth)
#             continue
        
#         # Compute weak form loss using broadcasting
#         truth_pred = torch.bmm(C_reshaped.expand(actual_K, -1, -1), wR)
#         dynamics_pred = torch.bmm(D_reshaped.expand(actual_K, -1, -1), wDotR)
        
#         # Compute loss
#         wf_loss = criterion(
#             dynamics_pred.reshape(-1, wDot.shape[1]),
#             truth_pred.reshape(-1, w.shape[1])
#         )
#         de_loss = criterion(xHat, truth)
        
#         total_loss += wf_loss + de_loss
    
#     return total_loss / batch_size