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
    Calculate RMSE between model predictions and ground truth
    
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
        u0 = truth[0, -metadata['n_control_features']:]
        # Make prediction
        x_pred = model.predict(x0, u0, ts, method=method).detach().cpu().numpy()
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
        model: The model to evaluate (any model with a predict method)
        truth: List of Data objects containing ground truth trajectories
        ts: Time points for the trajectory
        metadata: Metadata dictionary with nNodes, T, nodeType, etc.
        model_name: Name of the model to save the plot
        method: ODE solver method (for models that use ODE solvers)
        plot: Whether to plot the predicted vs ground truth trajectories
        
    Returns:
        float: Root mean squared error between predictions and ground truth
    """
    with torch.no_grad():
        device = next(model.parameters()).device
        
        # Extract initial conditions
        x0 = truth[0].x.to(device)
        edge_index = truth[0].edge_index.to(device)
        
        # Get initial latent state
        w0 = model.encoder(x0, edge_index)
        w0 = w0.T.flatten().detach()
        
        # Extract true trajectory
        z_true = []
        u_traj = []
        for step in truth:
            z_true.append(step.x[:, :-1].T.cpu().detach().numpy())  # exclude node type
            u_traj.append(step.u.cpu().detach().numpy())
        z_true = np.array(z_true)
        u_traj = np.array(u_traj)
        
        # Make prediction using model's predict method
        z_pred = model.predict(x0, u_traj, ts, edge_index=edge_index, method=method)
        z_pred = z_pred.detach().cpu().numpy()
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean((z_pred - z_true)**2))
        
        if plot:
            plot_trajectory(np.array([z_true, z_pred]), ts, model_name)
            
        return rmse
    
def prediction_rmse_graph(model, 
                   truth: torch.Tensor,
                   ts: Union[np.ndarray, torch.Tensor],
                   metadata: dict,
                   model_name: str,
                   method: str = 'dopri5',
                   plot: bool = False,
                   edge_index: Optional[torch.Tensor] = None) -> float:
    """
    Calculate RMSE between model predictions and ground truth
    
    Args:
        model: The model to evaluate (any model with a predict method)
        truth: Ground truth trajectory tensor [time, features]
        ts: Time points for the trajectory
        model_name: Name of the model to save the plot
        metadata: Metadata dictionary with n_state_features and n_control_features
        method: ODE solver method (for models that use ODE solvers)
        plot: Whether to plot the predicted vs ground truth trajectories
        edge_index: Optional graph connectivity tensor for graph-based models
        
    Returns:
        float: Root mean squared error between predictions and ground truth
    """
    with torch.no_grad():
        device = next(model.parameters()).device
        # Extract states and controls
        x0 = truth[0].x.to(device)
        edge_index = truth[0].edge_index.to(device)
        us, z_truth = [], []
        for step in truth:
            us.append(step.u.to(device))
            z_truth.append(step.x.to(device))
        us = torch.stack(us)
        z_truth = torch.stack(z_truth).cpu().detach().numpy()

        # Make prediction
        z_pred = model.predict(x0, us, ts, edge_index=edge_index, method=method).detach().cpu().numpy()

        # Calculate RMSE
        rmse = np.sqrt(np.mean((z_pred - z_truth)**2))
        
        if plot: # Delay embedding may exist, so we plot the first state
            plot_trajectory(np.array([z_pred[...,0], z_truth[...,0]]), ts, model_name)

        return rmse