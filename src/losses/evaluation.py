import numpy as np
import torch
from typing import Union, List, Optional
from src.utils.plot import plot_trajectory

def prediction_rmse(model, 
                   truth: torch.Tensor,
                   ts: Union[np.ndarray, torch.Tensor],
                   data_meta: dict,
                   model_name: str,
                   method: str = 'dopri5',
                   plot: bool = False,
                   model_type: Optional[str] = None) -> float:
    """
    Calculate RMSE between model predictions and ground truth
    
    Args:
        model: The model to evaluate (any model with a predict method)
        truth: Ground truth trajectory tensor [time, features]
        ts: Time points for the trajectory
        model_name: Name of the model to save the plot
        data_meta: Metadata dictionary with n_state_features and n_control_features
        method: ODE solver method (for models that use ODE solvers)
        plot: Whether to plot the predicted vs ground truth trajectories
        model_type: Optional type of model to customize evaluation behavior
        
    Returns:
        float: Root mean squared error between predictions and ground truth
    """
    with torch.no_grad():
        # Extract states and controls
        x_truth = truth[:, :data_meta['n_state_features']].detach().cpu().numpy()
        x0 = truth[0, :data_meta['n_state_features']]
        u0 = truth[0, -data_meta['n_control_features']:]
        # Make prediction
        x_pred = model.predict(x0, u0, ts, method=method).detach().cpu().numpy()
        # Calculate RMSE
        rmse = np.sqrt(np.mean((x_pred - x_truth)**2))
        if plot:
            plot_trajectory(np.array([x_pred, x_truth]), ts, model_name)
        return rmse
