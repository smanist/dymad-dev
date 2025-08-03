import numpy as np
import torch
from typing import Union

from dymad.data import DynData, DynGeoData
from dymad.utils import plot_trajectory

def prediction_rmse(model,
                   truth: Union[DynData, DynGeoData],
                   ts: Union[np.ndarray, torch.Tensor],
                   metadata: dict,
                   model_name: str,
                   method: str = 'dopri5',
                   plot: bool = False,
                   prefix: str = '.') -> float:
    """
    Calculate RMSE between model predictions and ground truth for regular models

    Args:
        model (torch.nn.Module): The model to evaluate (any model with a predict method)
        truth (DynData): Ground truth trajectory data
        ts (Union[np.ndarray, torch.Tensor]): Time points for the trajectory
        model_name (str): Name of the model to save the plot
        metadata (dict): Metadata dictionary with n_state_features and n_control_features
        method (str): ODE solver method (for models that use ODE solvers)
        plot (bool): Whether to plot the predicted vs ground truth trajectories
        prefix (str): Path prefix for saving plots

    Returns:
        float: Root mean squared error between predictions and ground truth
    """
    with torch.no_grad():
        # Extract states and controls
        x_truth = truth.x
        x0 = truth.x[0]
        us = truth.u

        # Make prediction
        x_pred = model.predict(x0, truth, ts, method=method)

        x_truth = x_truth.detach().cpu().numpy()
        x_pred = x_pred.detach().cpu().numpy()
        # Calculate RMSE
        rmse = np.sqrt(np.mean((x_pred - x_truth)**2))

        if plot:
            _us = None if us is None else us.detach().cpu().numpy()
            plot_trajectory(np.array([x_truth, x_pred]), ts, model_name, metadata,
                            us=_us, labels=['Truth', 'Prediction'], prefix=prefix)

        return rmse

def prediction_rmse_lstm(model,
                        truth: torch.Tensor,
                        ts: Union[np.ndarray, torch.Tensor],
                        metadata: dict,
                        model_name: str,
                        plot: bool = False) -> float:
    """
    Calculate RMSE between LSTM model predictions and ground truth
    The input truth is a single trajectory, not a list of trajectories

    Args:
        model (torch.nn.Module): The LSTM model to evaluate
        truth (torch.Tensor): Ground truth trajectory tensor [time, state_features + control_features]
        ts (Union[np.ndarray, torch.Tensor]): Time points for the trajectory
        metadata (dict): Metadata dictionary with n_state_features and n_control_features
        model_name (str): Name of the model to save the plot
        plot (bool): Whether to plot the predicted vs ground truth trajectories

    Returns:
        float: Root mean squared error between predictions and ground truth
    """
    with torch.no_grad():
        # Extract configuration
        n_state_features = metadata['n_state_features']
        n_control_features = metadata['n_control_features']
        seq_len = metadata['delay'] + 1

        # Extract states and controls from truth trajectory
        states_truth = truth[:, :n_state_features]  # [time, state_features]
        us = truth[:, -n_control_features:]  # [time, control_features]

        # Get initial state for rollout prediction (single initial state, not sequence)
        x0 = states_truth[:seq_len]  # [seq_len, state_features]

        # Make prediction using LSTM's predict method (rollout from initial state)
        x_pred = model.predict(x0, us, ts).detach().cpu().numpy()

        # Convert truth to numpy for comparison
        x_truth = states_truth.detach().cpu().numpy()

        # Calculate RMSE
        rmse = np.sqrt(np.mean((x_pred - x_truth)**2))

        if plot:
            plot_trajectory(np.array([x_pred, x_truth]), ts, model_name, metadata,
                            us=us.cpu(), labels=['Truth', 'Prediction'])

        return rmse