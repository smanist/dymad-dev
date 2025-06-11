import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Union
from .model_base import ModelBase
from torchdiffeq import odeint

class NODE(ModelBase):
    """
    Neural ODE model following the ModelBase interface.
    """
    def __init__(self, model_config: Dict, data_meta: Dict):
        super(NODE, self).__init__()
        self.n_state_features = data_meta.get('n_state_features')
        self.n_control_features = data_meta.get('n_control_features')
        self.n_total_features = self.n_state_features + self.n_control_features
        self.latent_dimension = model_config.get('latent_dimension', 64)
        num_layers = model_config.get('num_layers', 3)

        # Build the MLP for the ODE function
        self.ode_net = self.build_mlp(
            input_dimension=self.n_total_features,
            latent_dimension=self.latent_dimension,
            output_dimension=self.n_total_features,
            num_layers=num_layers
        )

        # Custom initialization (as in your original code)
        for m in self.ode_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, 0)

    def init_params(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def features(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, u], dim=-1)

    def encoder(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        # For NODE, the encoder is identity (no latent space)
        return self.features(x, u)

    def decoder(self, z: torch.Tensor) -> torch.Tensor:
        # For NODE, the decoder just returns the state part
        return z[..., :self.n_state_features]

    def dynamics(self, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        # z: (batch, n_total_features)
        out = self.ode_net(z)
        # Zero out the control derivatives
        out[..., self.n_state_features:] = 0.0
        return out

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder(x, u)
        z_dot = self.dynamics(z, u)
        x_hat = self.decoder(z)
        return z, z_dot, x_hat
    
    def ode_function(self, t, z):
        """
        ODE function compatible with torchdiffeq.odeint
        t: time parameter (required by odeint but unused)
        z: state tensor including control inputs
        """
        # Return dynamics output with zeroed control part
        out = self.ode_net(z)
        out[..., self.n_state_features:] = 0.0
        return out
    
    def predict(self, x0: torch.Tensor, us: torch.Tensor, ts: Union[np.ndarray, torch.Tensor], 
                method: str = 'dopri5') -> torch.Tensor:
        """
        Predict trajectory using continuous-time integration.
        
        Args:
            x0: Initial state tensor with shape (batch_size, n_state_features)
            us: Control input tensor, assumed constant for now
            ts: Time points for prediction
            method: ODE solver method
            
        Returns:
            Predicted trajectory tensor (time_steps, batch_size, n_state_features)
        """
        device = x0.device
        
        # Combine initial state with control input
        z0 = self.encoder(x0, us)
        
        # Create a function for odeint that matches the expected signature
        def ode_func(t, z):
            return self.ode_function(t, z)
        
        # Handle ts conversion properly to avoid tensor warning
        if isinstance(ts, torch.Tensor):
            ts_tensor = ts.clone().detach().to(device)
        else:
            ts_tensor = torch.tensor(ts, dtype=torch.float32).to(device)
        
        # Integrate the ODE
        z_trajectory = odeint(ode_func, z0, ts_tensor, method=method)
        
        # Decode the trajectory to get the predicted states
        x_trajectory = self.decoder(z_trajectory)
        
        return x_trajectory

    