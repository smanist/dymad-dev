import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Union
from .model_base import ModelBase
from src.utils.prediction import predict_continuous

class LDM(ModelBase):
    """
    Latent Dynamics Model (LDM) that can be trained with either:
    - NODE trainer (direct ODE integration)
    - Weak form trainer (weak form loss)
    
    This model combines the functionality of both NODE and weakFormLDM.
    """
    def __init__(self, model_config: Dict, data_meta: Dict):
        super(LDM, self).__init__()
        self.n_state_features = data_meta.get('n_state_features')
        self.n_control_features = data_meta.get('n_control_features')
        self.n_total_features = self.n_state_features + self.n_control_features
        self.latent_dimension = model_config.get('latent_dimension', 64)
        
        # Track training mode to determine prediction method
        self.training_mode = None  # Will be set by trainer: 'node' or 'weak_form'
        
        # Get layer depths from config
        enc_depth = model_config.get('encoder_layers', 2)
        proc_depth = model_config.get('processor_layers', 2)
        dec_depth = model_config.get('decoder_layers', 2)
        
        # Build network components
        self.encoder_net = self.build_mlp(
            input_dimension=self.n_total_features,
            latent_dimension=self.latent_dimension,
            output_dimension=self.latent_dimension,
            num_layers=enc_depth
        )
        
        self.dynamics_net = self.build_mlp(
            input_dimension=self.latent_dimension,
            latent_dimension=self.latent_dimension,
            output_dimension=self.latent_dimension,
            num_layers=proc_depth
        )
        
        self.decoder_net = self.build_mlp(
            input_dimension=self.latent_dimension,
            latent_dimension=self.latent_dimension,
            output_dimension=self.n_state_features,
            num_layers=dec_depth
        )

    def init_params(self):
        """Initialize model parameters with Xavier uniform initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def features(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Create features by concatenating state and control.
        
        Args:
            x: State tensor of shape (batch_size, n_state_features)
            u: Control tensor of shape (batch_size, n_control_features)
            
        Returns:
            Combined feature tensor
        """
        return torch.cat([x, u], dim=-1)

    def encoder(self, w: torch.Tensor) -> torch.Tensor:
        """
        Map features to latent space.
        
        Args:
            w: Raw features (state and control concatenated)
            
        Returns:
            Latent representation
        """
        return self.encoder_net(w)

    def decoder(self, z: torch.Tensor) -> torch.Tensor:
        """
        Map from latent space back to state space.
        
        Args:
            z: Latent state
            
        Returns:
            Reconstructed state
        """
        return self.decoder_net(z)

    def dynamics(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute latent dynamics (derivative).
        
        Args:
            z: Latent state
            
        Returns:
            Latent state derivative
        """
        return self.dynamics_net(z)

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            x: State tensor
            u: Control tensor
            
        Returns:
            Tuple of (latent, latent_derivative, reconstruction)
        """
        w = self.features(x, u)
        z = self.encoder(w)
        z_dot = self.dynamics(z)
        x_hat = self.decoder(z)
        return z, z_dot, x_hat
    
    def ode_function(self, t, z):
        """
        ODE function for direct integration.
        For NODE training, this should work like the original NODE implementation:
        - Take full state+control vector
        - Use all layers (encoder->dynamics->decoder) 
        - Return state derivatives with zeroed control derivatives
        
        Args:
            t: Time parameter (required by odeint but unused)
            z: Combined state+control tensor (n_total_features,)
            
        Returns:
            Full derivative vector with control derivatives zeroed
        """
        # Split state and control parts
        x = z[..., :self.n_state_features]  # State part
        u = z[..., self.n_state_features:]  # Control part
        
        # Use full forward pass: encoder -> dynamics -> decoder
        w = self.features(x, u)  # Combine state and control
        z_latent = self.encoder(w)  # Encode to latent space
        z_dot_latent = self.dynamics(z_latent)  # Latent dynamics
        x_dot = self.decoder(z_dot_latent)  # Decode to state derivatives
        
        # Create full derivative vector: state derivatives + zero control derivatives
        full_derivatives = torch.zeros_like(z)
        full_derivatives[..., :self.n_state_features] = x_dot
        # Control derivatives remain zero (not making predictions on control)
        
        return full_derivatives
    
    def predict(self, x0: torch.Tensor, us: torch.Tensor, ts: Union[np.ndarray, torch.Tensor], 
                method: str = 'dopri5') -> torch.Tensor:
        """
        Predict trajectory using continuous-time integration.
        
        Args:
            x0: Initial state tensor(s):
                - Single: (n_state_features,) 
                - Batch: (batch_size, n_state_features)
            us: Control inputs:
                - Single: (time_steps, n_control_features)
                - Batch: (batch_size, time_steps, n_control_features)
                For autonomous systems, use zero-valued controls
            ts: Time points for prediction
            method: ODE solver method
            
        Returns:
            Predicted trajectory tensor(s):
            - Single: (time_steps, n_state_features)
            - Batch: (time_steps, batch_size, n_state_features)
        """
        return predict_continuous(self, x0, us, ts, method=method) 