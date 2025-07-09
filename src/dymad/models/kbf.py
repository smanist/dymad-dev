import numpy as np
import torch
import torch.nn as nn
try:
    from torch_geometric.nn import SAGEConv
except:
    SAGEConv = None
from typing import Dict, Union, Tuple

from dymad.models.model_base import ModelBase
from dymad.utils import DynData, GNN, MLP, predict_continuous, predict_graph_continuous

class KBF(ModelBase):
    """
    Koopman Bilinear Form (KBF) model - standard version.
    Uses MLP encoder/decoder and KBF operators for dynamics.

    - z = encoder(x)
    - z_dot = Az + sum(B_i * u_i * z)
    - x_hat = decoder(z)
    """

    def __init__(self, model_config: Dict, data_meta: Dict):
        super(KBF, self).__init__()
        self.n_total_state_features = data_meta.get('n_total_state_features')
        self.n_total_control_features = data_meta.get('n_total_control_features')
        self.latent_dimension = model_config.get('latent_dimension', 64)
        self.koopman_dimension = model_config.get('koopman_dimension', 16)
        self.const_term = model_config.get('const_term', True)

        # Method for input handling
        self.input_order = model_config.get('input_order', 'cubic')

        # Get layer depths from config
        enc_depth = model_config.get('encoder_layers', 2)
        dec_depth = model_config.get('decoder_layers', 2)

        if self.n_total_state_features != self.koopman_dimension:
            if enc_depth == 0 or dec_depth == 0:
                raise ValueError(f"Encoder depth {enc_depth}, decoder depth {dec_depth}: "
                                 f"but n_total_state_features ({self.n_total_state_features}) "
                                 f"must match koopman_dimension ({self.koopman_dimension})")

        # Determine other options for MLP layers
        opts = {
            'activation'     : model_config.get('activation', 'prelu'),
            'weight_init'    : model_config.get('weight_init', 'xavier_uniform'),
            'bias_init'      : model_config.get('bias_init', 'zeros'),
            'gain'           : model_config.get('gain', 1.0),
            'end_activation' : model_config.get('end_activation', True)
        }

        # Build MLP encoder: maps input features to Koopman space
        self.encoder_net = MLP(
            input_dim  = self.n_total_state_features,
            latent_dim = self.latent_dimension,
            output_dim = self.koopman_dimension,
            n_layers   = enc_depth,
            **opts
        )

        # Create KBF operators: first for autonomous dynamics (A) then one per control (B_i)
        tmp = [
            nn.Linear(self.koopman_dimension, self.koopman_dimension, bias=False)
            for _ in range(self.n_total_control_features + 1)]
        if self.const_term:
            tmp.append(nn.Linear(self.n_total_control_features, self.koopman_dimension, bias=False))
        self.operators = nn.ModuleList(tmp)

        # Build MLP decoder: maps Koopman space back to output features
        self.decoder_net = MLP(
            input_dim  = self.koopman_dimension,
            latent_dim = self.latent_dimension,
            output_dim = self.n_total_state_features,
            n_layers   = dec_depth,
            **opts
        )

    def diagnostic_info(self) -> str:
        model_info = super(KBF, self).diagnostic_info()
        model_info += f"Encoder: {self.encoder_net.diagnostic_info()}\n"
        model_info += f"Decoder: {self.decoder_net.diagnostic_info()}\n"
        model_info += f"Input order: {self.input_order}"
        return model_info

    def encoder(self, w: DynData) -> torch.Tensor:
        """Encode combined features to Koopman space."""
        return self.encoder_net(w.x)

    def decoder(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from Koopman space back to state space."""
        # Apply decoder layers (now nn.Sequential or nn.Identity/nn.Linear)
        return self.decoder_net(z)

    def dynamics(self, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Compute dynamics in Koopman space using bilinear form."""
        # Autonomous part: A @ z
        z_dot = self.operators[0](z)

        # Add control-dependent terms: sum(u_i * B_i @ z)
        for i in range(self.n_total_control_features):
            control_i = u[..., i].unsqueeze(-1)  # Extract control i and add dimension for broadcasting
            z_dot = z_dot + control_i * self.operators[i + 1](z)

        # Add constant term if enabled
        if self.const_term:
            z_dot = z_dot + self.operators[-1](u)

        return z_dot

    def predict(self, x0: torch.Tensor, us: torch.Tensor, ts: Union[np.ndarray, torch.Tensor],
                method: str = 'dopri5') -> torch.Tensor:
        """Predict trajectory using continuous-time integration.

        Args:
            x0: Initial state tensor(s):

                - Single: (n_state_features,)

            us: Control inputs:

                - Single: (time_steps, n_control_features)

            ts: Time points for prediction
            method: ODE solver method (default: 'dopri5')

        Returns:
            Predicted trajectory tensor(s):

                - Single: (time_steps, n_state_features)
                - Batch: (time_steps, batch_size, n_state_features)
        """
        return predict_continuous(self, x0, us, ts, method=method, order=self.input_order)

    def forward(self, w: DynData) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for KBF model.

        Args:
            x: State features tensor
            u: Control inputs tensor

        Returns:
            Tuple of (latent, latent_derivative, reconstruction)
        """
        z = self.encoder(w)
        z_dot = self.dynamics(z, w.u)
        x_hat = self.decoder(z)
        return z, z_dot, x_hat

class GKBF(ModelBase):
    """Graph Koopman Bilinear Form (GKBF) model - graph-specific version.
    Uses GNN encoder/decoder and KBF operators for dynamics.

    Koopman dimension is defined per node.
    """
    def __init__(self, model_config: Dict, data_meta: Dict):
        super(GKBF, self).__init__()
        self.n_total_state_features = data_meta.get('n_total_state_features')
        self.n_total_control_features = data_meta.get('n_total_control_features')
        self.latent_dimension = model_config.get('latent_dimension', 64)
        self.koopman_dimension = model_config.get('koopman_dimension', 16)
        self.const_term = model_config.get('const_term', True)

        # Graph specific parameters
        self.n_nodes = data_meta['config']['data']['n_nodes']
        self.system_dimension = self.n_nodes * self.koopman_dimension

        # Method for input handling
        self.input_order = model_config.get('input_order', 'cubic')

        # Get layer depths from config
        enc_depth = model_config.get('encoder_layers', 2)
        dec_depth = model_config.get('decoder_layers', 2)

        # Determine other options for GNN layers
        opts = {
            'gcl'            : model_config.get('gcl', 'sage'),
            'activation'     : model_config.get('activation', 'prelu'),
            'weight_init'    : model_config.get('weight_init', 'xavier_uniform'),
            'bias_init'      : model_config.get('bias_init', 'zeros'),
            'gain'           : model_config.get('gain', 1.0),
            'end_activation' : model_config.get('end_activation', True)
        }

        # Build GNN encoder: maps input features to Koopman space
        self.encoder_net = GNN(
            input_dim=self.n_total_state_features,
            latent_dim=self.latent_dimension,
            output_dim=self.koopman_dimension,
            n_layers=enc_depth,
            **opts
        )

        # KBF operators for graph system
        tmp = [
            nn.Linear(self.system_dimension, self.system_dimension, bias=False)
            for _ in range(self.n_total_control_features + 1)
        ]
        if self.const_term:
            tmp.append(nn.Linear(self.n_total_control_features, self.system_dimension, bias=False))
        self.operators = nn.ModuleList(tmp)

        # Build GNN decoder: maps Koopman space back to output features
        self.decoder_net = GNN(
            input_dim=self.koopman_dimension,
            latent_dim=self.latent_dimension,
            output_dim=self.n_total_state_features,
            n_layers=dec_depth,
            **opts
        )

    def encoder(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder_net(x, edge_index)

    def decoder(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.decoder_net(z, edge_index)

    def features(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return x

    def dynamics(self, z: torch.Tensor, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Reshape z into (batch_size, system_dim)
        n_steps = z.shape[0] // self.n_nodes
        w = z.reshape(n_steps, self.n_nodes, -1).permute(0, 2, 1)
        w = w.reshape(n_steps, -1)
        # Autonomous part: A @ w
        w_dot = self.operators[0](w)
        # Add control-dependent terms: sum(u_i * B_i @ w)
        u_reshaped = u.reshape(n_steps, -1)
        for i in range(self.n_total_control_features):
            w_dot = w_dot + u_reshaped[:, i].unsqueeze(-1) * self.operators[i + 1](w)
        if self.const_term:
            w_dot = w_dot + self.operators[-1](u_reshaped)
        return w, w_dot

    def predict(self, x0: torch.Tensor, us: torch.Tensor, ts: Union[np.ndarray, torch.Tensor],
                edge_index: torch.Tensor, method: str = 'dopri5') -> torch.Tensor:
        return predict_graph_continuous(self, x0, us, ts, edge_index, method=method, order=self.input_order)

    def forward(self, x: torch.Tensor, u: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        Gz = self.encoder(x, edge_index)
        z, z_dot = self.dynamics(Gz, u)
        x_hat = self.decoder(Gz, edge_index)
        return z, z_dot, x_hat
