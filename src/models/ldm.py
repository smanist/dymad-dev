import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Dict, Union

from .model_base import ModelBase
from ...src.utils import MLP, predict_continuous

class LDM(ModelBase):
    """
    Latent Dynamics Model (LDM) that can be trained with either:
    - NODE trainer (direct ODE integration)
    - Weak form trainer (weak form loss)

    This model combines the functionality of both NODE and weakFormLDM.

    In the minimal case,
        Encoder: z = (x,u)
        Dynamics: z_dot = f(z)
        Decoder: x_hat = TakeFirst(z)
    """
    def __init__(self, model_config: Dict, data_meta: Dict):
        super(LDM, self).__init__()
        self.n_total_state_features = data_meta.get('n_total_state_features')
        self.n_control_features = data_meta.get('n_control_features')
        self.n_total_features = data_meta.get('n_total_features')
        self.latent_dimension = model_config.get('latent_dimension', 64)

        # Method for input handling
        self.input_order = model_config.get('input_order', 'cubic')

        # Get layer depths from config
        enc_depth = model_config.get('encoder_layers', 2)
        proc_depth = model_config.get('processor_layers', 2)
        dec_depth = model_config.get('decoder_layers', 2)

        # Determine dimensions
        enc_out_dim = self.latent_dimension if enc_depth > 0 else self.n_total_features
        dec_inp_dim = self.latent_dimension if dec_depth > 0 else self.n_total_features

        # Determine other options for MLP layers
        opts = {
            'activation'     : model_config.get('activation', 'prelu'),
            'weight_init'    : model_config.get('weight_init', 'xavier_uniform'),
            'bias_init'      : model_config.get('bias_init', 'zeros'),
            'gain'           : model_config.get('gain', 1.0),
            'end_activation' : model_config.get('end_activation', True)
        }

        # Build network components
        self.encoder_net = MLP(
            input_dim  = self.n_total_features,
            latent_dim = self.latent_dimension,
            output_dim = enc_out_dim,
            n_layers   = enc_depth,
            **opts
        )

        self.dynamics_net = MLP(
            input_dim  = enc_out_dim,
            latent_dim = self.latent_dimension,
            output_dim = dec_inp_dim,
            n_layers   = proc_depth,
            **opts
        )

        self.decoder_net = MLP(
            input_dim  = dec_inp_dim,
            latent_dim = self.latent_dimension,
            output_dim = self.n_total_state_features,
            n_layers   = dec_depth,
            **opts
        )

    def features(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Create features by concatenating state and control.

        Args:
            x: State tensor of shape (batch_size, n_total_state_features)
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

    def predict(self, x0: torch.Tensor, us: torch.Tensor, ts: Union[np.ndarray, torch.Tensor],
                method: str = 'dopri5') -> torch.Tensor:
        """
        Predict trajectory using continuous-time integration.

        Args:
            x0: Initial state tensor(s):
                - Single: (n_total_state_features,)
                - Batch: (batch_size, n_total_state_features)
            us: Control inputs:
                - Single: (time_steps, n_control_features)
                - Batch: (batch_size, time_steps, n_control_features)
                For autonomous systems, use zero-valued controls
            ts: Time points for prediction
            method: ODE solver method

        Returns:
            Predicted trajectory tensor(s):
            - Single: (time_steps, n_total_state_features)
            - Batch: (time_steps, batch_size, n_total_state_features)
        """
        return predict_continuous(self, x0, us, ts, method=method, order=self.input_order)