import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Dict, Union

from dymad.data import DynData, DynGeoData
from dymad.models import ModelBase
from dymad.utils import make_autoencoder, MLP, predict_continuous, predict_graph_continuous

class LDM(ModelBase):
    """Latent Dynamics Model (LDM)

    The encoder, dynamics, and decoder networks are implemented as MLPs.
    """
    GRAPH = False

    def __init__(self, model_config: Dict, data_meta: Dict):
        super(LDM, self).__init__()
        self.n_total_state_features = data_meta.get('n_total_state_features')
        self.n_total_control_features = data_meta.get('n_total_control_features')
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
        aec_type = model_config.get('autoencoder_type', 'smp')

        # Build encoder/decoder networks
        self.encoder_net, self.decoder_net = make_autoencoder(
            type="mlp_"+aec_type,
            input_dim=self.n_total_features,
            latent_dim=self.latent_dimension,
            hidden_dim=enc_out_dim,
            enc_depth=enc_depth,
            dec_depth=dec_depth,
            output_dim=self.n_total_state_features,
            **opts
        )

        self.dynamics_net = MLP(
            input_dim  = enc_out_dim,
            latent_dim = self.latent_dimension,
            output_dim = dec_inp_dim,
            n_layers   = proc_depth,
            **opts
        )

        if self.n_total_control_features == 0:
            self.encoder = self._encoder_auto
        else:
            self.encoder = self._encoder_ctrl

    def diagnostic_info(self) -> str:
        model_info = super(LDM, self).diagnostic_info()
        model_info += f"Encoder: {self.encoder_net.diagnostic_info()}\n"
        model_info += f"Dynamics: {self.dynamics_net.diagnostic_info()}\n"
        model_info += f"Decoder: {self.decoder_net.diagnostic_info()}\n"
        model_info += f"Input order: {self.input_order}"
        return model_info

    def _encoder_ctrl(self, w: DynData) -> torch.Tensor:
        """
        Map features to latent space for systems with inputs.

        Args:
            w (DynData): Raw features

        Returns:
            torch.Tensor: Latent representation
        """
        return self.encoder_net(torch.cat([w.x, w.u], dim=-1))

    def _encoder_auto(self, w: DynData) -> torch.Tensor:
        """
        Map features to latent space for autonomous systems.

        Args:
            w (DynData): Raw features

        Returns:
            torch.Tensor: Latent representation
        """
        return self.encoder_net(w.x)

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

    def predict(self, x0: torch.Tensor, w: DynData, ts: Union[np.ndarray, torch.Tensor],
                method: str = 'dopri5') -> torch.Tensor:
        """
        Predict trajectory using continuous-time integration.

        Args:
            x0 (torch.Tensor): Initial state tensor(s):

                - Single: (n_total_state_features,)
                - Batch: (batch_size, n_total_state_features)

            us (torch.Tensor): Control inputs:

                - Single: (time_steps, n_total_control_features)
                - Batch: (batch_size, time_steps, n_total_control_features)

                For autonomous systems, use zero-valued controls

            ts (Union[np.ndarray, torch.Tensor]): Time points for prediction
            method (str): ODE solver method

        Returns:
            torch.Tensor: Predicted trajectory tensor(s):

                - Single: (time_steps, n_total_state_features)
                - Batch: (time_steps, batch_size, n_total_state_features)
        """
        return predict_continuous(self, x0, ts, us=w.u, method=method, order=self.input_order)

class GLDM(ModelBase):
    """Graph Latent Dynamics Model (GLDM).

    Uses GNN for encoder/decoder and MLP for dynamics.
    """
    GRAPH = True

    def __init__(self, model_config: Dict, data_meta: Dict):
        super(GLDM, self).__init__()
        self.n_total_state_features = data_meta.get('n_total_state_features')
        self.n_total_control_features = data_meta.get('n_total_control_features')
        self.n_total_features = data_meta.get('n_total_features')
        self.latent_dimension = model_config.get('latent_dimension', 64)

        # Graph specific parameters
        self.n_nodes = data_meta['config']['data']['n_nodes']

        # Method for input handling
        self.input_order = model_config.get('input_order', 'cubic')

        # Get layer depths from config
        enc_depth = model_config.get('encoder_layers', 2)
        proc_depth = model_config.get('processor_layers', 2)
        dec_depth = model_config.get('decoder_layers', 2)

        # Determine dimensions
        enc_out_dim = self.latent_dimension #if enc_depth > 0 else self.n_total_features
        dec_inp_dim = self.latent_dimension #if dec_depth > 0 else self.n_total_features

        # Determine other options for MLP and GNN layers
        opts_mlp = {
            'activation'     : model_config.get('activation', 'prelu'),
            'weight_init'    : model_config.get('weight_init', 'xavier_uniform'),
            'bias_init'      : model_config.get('bias_init', 'zeros'),
            'gain'           : model_config.get('gain', 1.0),
            'end_activation' : model_config.get('end_activation', True)
        }
        opts_gnn = opts_mlp.copy()
        opts_gnn.update({
            'n_nodes'        : self.n_nodes,
            'gcl'            : model_config.get('gcl', 'sage'),
        })
        aec_type = model_config.get('autoencoder_type', 'smp')

        # Build encoder/decoder networks
        self.encoder_net, self.decoder_net = make_autoencoder(
            type="gnn_"+aec_type,
            input_dim=self.n_total_features,
            latent_dim=self.latent_dimension,
            hidden_dim=enc_out_dim,
            enc_depth=enc_depth,
            dec_depth=dec_depth,
            output_dim=self.n_total_state_features,
            **opts_gnn
        )

        self.dynamics_net = MLP(
            input_dim  = enc_out_dim * self.n_nodes,
            latent_dim = self.latent_dimension,
            output_dim = dec_inp_dim * self.n_nodes,
            n_layers   = proc_depth,
            **opts_mlp
        )

        if self.n_total_control_features == 0:
            self.encoder = self._encoder_auto
        else:
            self.encoder = self._encoder_ctrl

    def diagnostic_info(self) -> str:
        model_info = super(GLDM, self).diagnostic_info()
        model_info += f"Encoder: {self.encoder_net.diagnostic_info()}\n"
        model_info += f"Dynamics: {self.dynamics_net.diagnostic_info()}\n"
        model_info += f"Decoder: {self.decoder_net.diagnostic_info()}\n"
        model_info += f"Input order: {self.input_order}"
        return model_info

    def _encoder_ctrl(self, w: DynGeoData) -> torch.Tensor:
        x_shape = w.x.shape[:-1] + (self.n_nodes, -1)
        u_shape = w.u.shape[:-1] + (self.n_nodes, -1)

        x_reshaped = w.x.view(*x_shape)
        u_reshaped = w.u.view(*u_shape)

        xu_cat = torch.cat([x_reshaped, u_reshaped], dim=-1)
        xu_flat = xu_cat.view(*w.x.shape[:-1], self.n_total_state_features + self.n_total_control_features)

        return self.encoder_net(xu_flat, w.edge_index)

    def _encoder_auto(self, w: DynGeoData) -> torch.Tensor:
        return self.encoder_net(w.x, w.edge_index)

    def decoder(self, z: torch.Tensor, w: DynGeoData) -> torch.Tensor:
        return self.decoder_net(z, w.edge_index)

    def dynamics(self, z: torch.Tensor, w: DynGeoData) -> torch.Tensor:
        return self.dynamics_net(z)

    def forward(self, w: DynGeoData) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder(w)
        z_dot = self.dynamics(z, w)
        x_hat = self.decoder(z, w)
        return z, z_dot, x_hat

    def predict(self, x0: torch.Tensor, w: DynGeoData, ts: Union[np.ndarray, torch.Tensor], method: str = 'dopri5') -> torch.Tensor:
        return predict_graph_continuous(self, x0, ts, w.edge_index, us=w.u, method=method, order=self.input_order)
