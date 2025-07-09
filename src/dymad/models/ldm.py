import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Dict, Union

from dymad.models.model_base import ModelBase
from dymad.utils import DynData, MLP, predict_continuous

class LDM(ModelBase):
    """Latent Dynamics Model (LDM)

    The encoder, dynamics, and decoder networks are implemented as MLPs.
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

    def diagnostic_info(self) -> str:
        model_info = super(LDM, self).diagnostic_info()
        model_info += f"Encoder: {self.encoder_net.diagnostic_info()}\n"
        model_info += f"Dynamics: {self.dynamics_net.diagnostic_info()}\n"
        model_info += f"Decoder: {self.decoder_net.diagnostic_info()}\n"
        model_info += f"Input order: {self.input_order}"
        return model_info

    def encoder(self, w: DynData) -> torch.Tensor:
        """
        Map features to latent space.

        Args:
            w (DynData): Raw features

        Returns:
            torch.Tensor: Latent representation
        """
        return self.encoder_net(torch.cat([w.x, w.u], dim=-1))

    def decoder(self, z: torch.Tensor, w: DynData) -> torch.Tensor:
        """
        Map from latent space back to state space.

        Args:
            z (torch.Tensor): Latent state

        Returns:
            torch.Tensor: Reconstructed state
        """
        return self.decoder_net(z)

    def dynamics(self, z: torch.Tensor, w: DynData) -> torch.Tensor:
        """
        Compute latent dynamics (derivative).

        Args:
            z (torch.Tensor): Latent state

        Returns:
            torch.Tensor: Latent state derivative
        """
        return self.dynamics_net(z)

    def forward(self, w: DynData) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            w (DynData): Input data containing state and control tensors

        Returns:
            Tuple of (latent, latent_derivative, reconstruction)
        """
        z = self.encoder(w)
        z_dot = self.dynamics(z, w)
        x_hat = self.decoder(z, w)
        return z, z_dot, x_hat

    def predict(self, x0: torch.Tensor, us: torch.Tensor, ts: Union[np.ndarray, torch.Tensor],
                method: str = 'dopri5') -> torch.Tensor:
        """
        Predict trajectory using continuous-time integration.

        Args:
            x0 (torch.Tensor): Initial state tensor(s):

                - Single: (n_total_state_features,)
                - Batch: (batch_size, n_total_state_features)

            us (torch.Tensor): Control inputs:

                - Single: (time_steps, n_control_features)
                - Batch: (batch_size, time_steps, n_control_features)

                For autonomous systems, use zero-valued controls

            ts (Union[np.ndarray, torch.Tensor]): Time points for prediction
            method (str): ODE solver method

        Returns:
            torch.Tensor: Predicted trajectory tensor(s):

                - Single: (time_steps, n_total_state_features)
                - Batch: (time_steps, batch_size, n_total_state_features)
        """
        return predict_continuous(self, x0, us, ts, method=method, order=self.input_order)