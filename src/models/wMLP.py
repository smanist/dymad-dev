import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Union
from .model_base import ModelBase

class weakFormMLP(ModelBase):
    def __init__(self, model_config: Dict, data_meta: Dict):
        super(weakFormMLP, self).__init__()
        self.n_state_features = data_meta.get('n_state_features')
        self.n_control_features = data_meta.get('n_control_features')
        self.n_total_features = data_meta.get('n_total_features')        
        self.latent_dimension = model_config.get('latent_dimension', 64)
        enc_depth = model_config.get('encoder_layers', 2)
        proc_depth = model_config.get('processor_layers', 2)
        dec_depth = model_config.get('decoder_layers', 2)
        
        # Build network components
        self.encoder_net = self._build_mlp(
            input_dimension=self.n_total_features,
            latent_dimension=self.latent_dimension,
            output_dimension=self.latent_dimension,
            num_layers=enc_depth
        )
        
        self.dynamics_net = self._build_mlp(
            input_dimension=self.latent_dimension,
            latent_dimension=self.latent_dimension,
            output_dimension=self.latent_dimension,
            num_layers=proc_depth
        )
        
        self.decoder_net = self._build_mlp(
            input_dimension=self.latent_dimension,
            latent_dimension=self.latent_dimension,
            output_dimension=self.n_state_features,
            num_layers=dec_depth
        )

    def _build_mlp(self, input_dimension: int, latent_dimension: int, output_dimension: int, num_layers: int) -> nn.Sequential:
        if num_layers == 1:
            return nn.Sequential(
                nn.Linear(input_dimension, output_dimension),
                nn.PReLU()
            )
        
        layers = [nn.Linear(input_dimension, latent_dimension), nn.PReLU()]
        
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(latent_dimension, latent_dimension), nn.PReLU()])
        
        layers.append(nn.Linear(latent_dimension, output_dimension))
        layers.append(nn.PReLU())       

        return nn.Sequential(*layers)
    
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
            w: Raw features
            (state and control concatenated)
            
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
    
    def predict(self, x0: torch.Tensor, us: torch.Tensor, ts: Union[np.ndarray, torch.Tensor], 
            method: str = 'dopri5') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict SINGLE trajectory using continuous-time integration.
        
        Args:
            x0: Initial state tensor (n_state_features)
            us: Control inputs, either:
                - Constant control (n_control_features)
                - Full trajectory (time_steps, n_control_features)
            ts: Time points for prediction (required).
            method: ODE solver method
            
        Returns:
            Tuple of (predicted_states, true_states if available, else None)
        """
        from torchdiffeq import odeint
        
        device = x0.device
        
        # For this model we assume constant control
        # TODO: should be able to change this to non-constant control
        if len(us.shape) == 2:  # Full trajectory provided
            u0 = us[0]  # Just use the first control
        else:
            u0 = us  # Constant control already provided
            
        # Initial latent state
        w0 = self.encoder(self.features(x0, u0))
        # Projection-based dynamics
        def _func(t_val, w):
            # Decode w to x at every step
            x = self.decoder(w).detach()
            # Combine with control input and get derivative in latent space
            _, w_dot, _ = self(x, u0)
            return w_dot.squeeze().detach()
    
        # Integrate using ODE solver
        w_traj = odeint(_func, w0.squeeze(), torch.DoubleTensor(ts).to(device), method=method)
        
        # Decode trajectory
        x_pred = self.decoder(w_traj)
        
        return x_pred