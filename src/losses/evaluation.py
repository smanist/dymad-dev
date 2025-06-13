import numpy as np
import torch
from typing import Union, List, Optional
from src.utils.plot import plot_trajectory

def prediction_rmse(model, 
                   truth: torch.Tensor,
                   ts: Union[np.ndarray, torch.Tensor],
                   metadata: dict,
                   model_name: str,
                   method: str = 'dopri5',
                   plot: bool = False) -> float:
    """
    Calculate RMSE between model predictions and ground truth for regular models
    
    Args:
        model: The model to evaluate (any model with a predict method)
        truth: Ground truth trajectory tensor [time, features]
        ts: Time points for the trajectory
        model_name: Name of the model to save the plot
        metadata: Metadata dictionary with n_state_features and n_control_features
        method: ODE solver method (for models that use ODE solvers)
        plot: Whether to plot the predicted vs ground truth trajectories
        
    Returns:
        float: Root mean squared error between predictions and ground truth
    """
    with torch.no_grad():
        # Extract states and controls
        x_truth = truth[:, :metadata['n_state_features']].detach().cpu().numpy()
        x0 = truth[0, :metadata['n_state_features']]
        us = truth[:, -metadata['n_control_features']:]
        
        # Make prediction
        x_pred = model.predict(x0, us, ts, method=method).detach().cpu().numpy()
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean((x_pred - x_truth)**2))
        
        if plot:
            plot_trajectory(np.array([x_pred, x_truth]), ts, model_name)
            
        return rmse

def prediction_rmse_graph(model, 
                         truth: List[torch.Tensor],
                         ts: Union[np.ndarray, torch.Tensor],
                         metadata: dict,
                         model_name: str,
                         method: str = 'dopri5',
                         plot: bool = False) -> float:
    """
    Calculate RMSE between model predictions and ground truth for graph-based models
    
    Args:
        model: The model to evaluate (graph model with predict method)
        truth: List of Data objects containing ground truth trajectories
        ts: Time points for the trajectory
        metadata: Metadata dictionary with graph structure info
        model_name: Name of the model to save the plot
        method: ODE solver method (for models that use ODE solvers)
        plot: Whether to plot the predicted vs ground truth trajectories
        
    Returns:
        float: Root mean squared error between predictions and ground truth
    """
    with torch.no_grad():
        device = next(model.parameters()).device
        
        # Extract initial conditions and trajectory data
        x0 = truth[0].x.to(device)
        edge_index = truth[0].edge_index.to(device)
        
        # Extract control trajectory and true states
        us = []
        z_truth = []
        for step in truth:
            us.append(step.u.to(device))
            z_truth.append(step.x.to(device))
        
        us = torch.stack(us)  # (n_steps, n_controls)
        z_truth = torch.stack(z_truth)  # (n_steps, n_nodes, n_features)
        
        # Make prediction using graph-specific prediction function
        z_pred = model.predict(x0, us, ts, edge_index, method=method)
        
        # Convert to numpy for RMSE calculation and plotting
        z_pred_np = z_pred.detach().cpu().numpy()
        z_truth_np = z_truth.detach().cpu().numpy()
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean((z_pred_np - z_truth_np)**2))
        
        if plot: # Delay embedding may exist, so we plot the first state
            plot_trajectory(np.array([z_pred_np[..., 0], z_truth_np[..., 0]]), ts, model_name)
            
        return rmse