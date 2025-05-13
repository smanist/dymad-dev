import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Union, List, Optional
from .model_base import ModelBase

class LSTM(ModelBase):
    """
    LSTM model for dynamical systems following the ModelBase interface.
    
    This model uses an LSTM architecture to predict future states of a dynamical system.
    Unlike NODE or wMLP models which use continuous-time formulations,
    LSTM operates as a discrete-time model using Euler-like rollout prediction.
    """
    def __init__(self, model_config: Dict, data_meta: Dict):
        super(LSTM, self).__init__()
        
        # Extract configuration parameters
        self.n_state_features = data_meta.get('n_state_features')
        self.n_control_features = data_meta.get('n_control_features')
        self.n_total_features = self.n_state_features + self.n_control_features
        
        # LSTM specific parameters
        self.hidden_dimension = model_config.get('hidden_dimension', 64)
        self.num_layers = model_config.get('num_layers', 2)
        
        # Define LSTM and output layers
        self.lstm = nn.LSTM(
            input_size=self.n_total_features,
            hidden_size=self.hidden_dimension,
            num_layers=self.num_layers,
            batch_first=True
        )
        
        # Linear layer to map from hidden state to predicted states
        self.linear = nn.Linear(self.hidden_dimension, self.n_state_features)
        self.activation = nn.PReLU()
        
        # Initialize parameters
        self.init_params()
    
    def init_params(self):
        """Initialize model parameters."""
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    
    def init_hidden(self, batch_size: int, device: torch.device = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize hidden and cell states for the LSTM.
        
        Args:
            batch_size: Size of the batch
            device: Device to create tensors on
            
        Returns:
            Tuple of (hidden_state, cell_state)
        """
        if device is None:
            device = next(self.parameters()).device
            
        hidden = (
            torch.zeros(self.num_layers, batch_size, self.hidden_dimension, device=device),
            torch.zeros(self.num_layers, batch_size, self.hidden_dimension, device=device)
        )
        return hidden
    
    def features(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Combine state and control features.
        
        Args:
            x: State tensor
            u: Control input tensor
            
        Returns:
            Combined features tensor
        """
        return torch.cat([x, u], dim=-1)
    
    def encoder(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Encoder function - required by ModelBase but has different meaning in LSTM.
        For LSTM, encoding is the feature concatenation.
        
        Args:
            x: State tensor
            u: Control input tensor
            
        Returns:
            Combined features
        """
        return self.features(x, u)
    
    def decoder(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode function - required by ModelBase but less relevant for LSTM.
        
        Args:
            z: Latent tensor
            
        Returns:
            Predicted states
        """
        # This is a placeholder to satisfy ModelBase interface
        # Real decoding happens in the forward/predict_next methods
        return z
    
    def dynamics(self, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Not directly used in LSTM model, but required by ModelBase.
        
        LSTM uses discrete-time transitions rather than continuous dynamics.
        """
        raise NotImplementedError(
            "LSTM uses discrete-time transitions, not continuous dynamics. "
            "Use predict_next() or predict() instead."
        )
    
    def forward(self, x: torch.Tensor, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass that satisfies ModelBase interface but uses LSTM for prediction.
        
        Args:
            x: Sequence of states [batch, seq_len, state_features]
            u: Sequence of controls [batch, seq_len, control_features]
            
        Returns:
            Tuple of (z, z_dot, x_hat) where:
            - z: Hidden state (placeholder for latent)
            - z_dot: Empty tensor (not used in discrete models)
            - x_hat: Predicted next state
        """
        # Combine states and controls
        features = self.features(x, u)
        batch_size = features.shape[0]
        
        # Initialize hidden state
        hidden = self.init_hidden(batch_size, features.device)
        
        # Process sequence through LSTM
        lstm_out, hidden = self.lstm(features, hidden)
        
        # Predict next state using the last LSTM output
        next_state = self.linear(lstm_out[:, -1, :])
        next_state = self.activation(next_state)
        
        # For compatibility with ModelBase interface
        z = hidden[0][-1]  # Last layer's hidden state as latent
        z_dot = torch.zeros_like(next_state)  # Not used in discrete model
        x_hat = next_state  # Predicted next state
        
        return z, z_dot, x_hat
    
    def predict_next(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Predict the next state given the current state and control.
        
        Args:
            x: Current state tensor [batch, state_features]
            u: Current control input [batch, control_features]
            
        Returns:
            Predicted next state [batch, state_features]
        """
        # Add sequence dimension
        x_seq = x.unsqueeze(1) if x.dim() == 2 else x
        u_seq = u.unsqueeze(1) if u.dim() == 2 else u
        
        # Forward pass
        _, _, next_state = self.forward(x_seq, u_seq)
        
        return next_state
    
    def predict(self, x0: torch.Tensor, us: torch.Tensor, ts: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Predict trajectory using discrete-time Euler-like rollout.
        
        This implements the discrete-time prediction by:
        1. Starting with an initial condition
        2. Predicting the next state
        3. Using the prediction as the new input state
        4. Repeating for all time steps
        
        Args:
            x0: Initial state tensor [batch, state_features]
            us: Control inputs for all time steps [batch, time_steps, control_features]
                or [batch, control_features] for constant control
            ts: Time points (used only to determine the number of steps)
            
        Returns:
            Predicted trajectory [time_steps, batch, state_features]
        """
        # Ensure proper dimensionality
        batch_size = x0.shape[0] if x0.dim() > 1 else 1
        x0 = x0.unsqueeze(0) if x0.dim() == 1 else x0
        
        # Handle constant control case
        if us.dim() == 2:  # [batch, control_features]
            us = us.unsqueeze(1).expand(-1, len(ts), -1)
        
        # Store initial state as first prediction
        predictions = [x0]
        current_state = x0
        
        # Perform rollout prediction (Euler-like discrete-time stepping)
        for t in range(1, len(ts)):
            # Get control at current time step
            current_control = us[:, t-1]
            
            # Predict next state using current state and control
            next_state = self.predict_next(current_state, current_control)
            
            # Store prediction
            predictions.append(next_state)
            
            # Update current state for next step
            current_state = next_state
            
        # Stack all predictions and transpose to [time_steps, batch, state_features]
        trajectory = torch.stack(predictions)
        
        return trajectory