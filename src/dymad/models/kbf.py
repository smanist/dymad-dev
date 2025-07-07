import numpy as np
import torch
import torch.nn as nn
try:
    from torch_geometric.nn import SAGEConv
except:
    SAGEConv = None
from typing import Dict, Union, Tuple

from dymad.models.model_base import ModelBase
from dymad.utils import MLP, predict_continuous, predict_graph_continuous

class KBF(ModelBase):
    """
    Koopman Bilinear Form (KBF) model - non-graph version.
    Uses MLP encoder/decoder and KBF operators for dynamics.
    """

    def __init__(self, model_config: Dict, data_meta: Dict):
        super(KBF, self).__init__()
        self.n_total_state_features = data_meta.get('n_total_state_features')
        self.n_control_features = data_meta.get('n_control_features')
        self.n_total_features = data_meta.get('n_total_features')
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
            for _ in range(self.n_control_features + 1)]
        if self.const_term:
            tmp.append(nn.Linear(self.n_control_features, self.koopman_dimension, bias=False))
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
        """
        Return diagnostic information about the model.

        Returns:
            String with model details
        """
        model_info = super(KBF, self).diagnostic_info()
        model_info += f"Encoder: {self.encoder_net.diagnostic_info()}\n"
        model_info += f"Decoder: {self.decoder_net.diagnostic_info()}\n"
        model_info += f"Input order: {self.input_order}"
        return model_info

    def encoder(self, w: torch.Tensor) -> torch.Tensor:
        """Encode combined features to Koopman space."""
        return self.encoder_net(w)

    def decoder(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from Koopman space back to state space."""
        # Apply decoder layers (now nn.Sequential or nn.Identity/nn.Linear)
        return self.decoder_net(z)

    def features(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return x

    def dynamics(self, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Compute dynamics in Koopman space using bilinear form."""
        # Autonomous part: A @ z
        z_dot = self.operators[0](z)

        # Add control-dependent terms: sum(u_i * B_i @ z)
        for i in range(self.n_control_features):
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
        return predict_continuous(self, x0, us, ts, method=method)

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for KBF model.

        Args:
            x: State features tensor
            u: Control inputs tensor

        Returns:
            Tuple of (latent, latent_derivative, reconstruction)
        """
        z = self.encoder(x)
        z_dot = self.dynamics(z, u)
        x_hat = self.decoder(z)
        return z, z_dot, x_hat

class GKBF(ModelBase):
    def __init__(self, model_config: Dict, data_meta: Dict):
        super(GKBF, self).__init__()

        # Initialize base parameters
        self.n_state_features = data_meta.get('n_state_features') # delay is handled differently from graph data
        self.n_control_features = data_meta.get('n_control_features')
        self.n_total_features = data_meta.get('n_total_features')
        self.n_nodes = data_meta['config']['data']['n_nodes']

        self.in_out_dimension = data_meta['delay'] + 1
        self.latent_dimension = model_config.get('latent_dimension', 64)
        self.koopman_dimension = model_config.get('koopman_dimension', 16)
        self.n_encoder_layers = model_config.get('encoder_layers', 2)
        self.n_decoder_layers = model_config.get('decoder_layers', 2)
        self.system_dimension = self.n_nodes * self.koopman_dimension

        # Create KBF operators: first for autonomous dynamics (A) then one per control (B_i)
        self.operators = nn.ModuleList([
            nn.Linear(self.system_dimension, self.system_dimension, bias=False)
            for _ in range(self.n_control_features + 1)
        ])

        # Build GNN encoder: maps node features to Koopman space
        self.encoder_layers = self._build_gnn(
            input_dimension=self.in_out_dimension,
            latent_dimension=self.latent_dimension,
            output_dimension=self.koopman_dimension,
            num_layers=self.n_encoder_layers
        )
        # Build GNN decoder: maps Koopman space back to node features
        self.decoder_layers = self._build_gnn(
            input_dimension=self.koopman_dimension,
            latent_dimension=self.latent_dimension,
            output_dimension=self.in_out_dimension,
            num_layers=self.n_decoder_layers
        )

    def _build_gnn(self, input_dimension: int, latent_dimension: int, output_dimension: int, num_layers: int) -> nn.ModuleList:
        layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_dimension if i == 0 else latent_dimension
            out_dim = output_dimension if i == num_layers - 1 else latent_dimension
            layers.append(SAGEConv(in_dim, out_dim))
            layers.append(nn.PReLU(out_dim))
        return layers

    def init_params(self):
        for module in self.modules():
            if isinstance(module, nn.Linear) or isinstance(module, SAGEConv):
                nn.init.xavier_uniform_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encoder(self, x: torch.Tensor, u: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for layer in self.encoder_layers:
            x = layer(x, edge_index) if hasattr(layer, 'aggr') else layer(x)
        return x

    def decoder(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = z.clone()
        for i, layer in enumerate(self.decoder_layers):
            x = layer(x, edge_index) if (i % 2 == 0) else layer(x)
        return x

    def features(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return x

    def dynamics(self, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        n_steps = z.shape[0]//self.n_nodes
        """Compute dynamics in latent space."""
        # Reshape z into (batch_size, system_dim)
        w = z.reshape(n_steps, self.n_nodes, -1).permute(0, 2, 1)
        w = w.reshape(n_steps, -1)
        # Autonomous part: A @ w
        w_dot = self.operators[0](w)
        # Add control-dependent terms: sum(u_i * B_i @ w)
        u_reshaped = u.reshape(n_steps, -1)
        for i in range(self.n_control_features):
            w_dot = w_dot + u_reshaped[:, i].unsqueeze(-1) * self.operators[i + 1](w)
        return w, w_dot

    def predict(self, x0: torch.Tensor, us: torch.Tensor, ts: Union[np.ndarray, torch.Tensor],
                edge_index: torch.Tensor, method: str = 'dopri5') -> torch.Tensor:
        """Predict single graph trajectory using continuous-time integration.

        Args:
            x0: Initial node states (n_nodes, n_features)
            us: Control trajectory (n_steps, n_controls)

                - For autonomous systems, use zero-valued controls

            ts: Time points for prediction:

                - numpy array of shape (n_steps,)
                - torch tensor of shape (n_steps,)

            edge_index: Graph connectivity tensor of shape (2, n_edges)
            method: ODE solver method (default: 'dopri5')

        Returns:
            Predicted trajectory tensor (n_steps, n_nodes, n_features)
        """
        return predict_graph_continuous(self, x0, us, ts, edge_index, method=method)

    def forward(self, x: torch.Tensor, u: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for graph-based model.

        Args:
            x: Node features tensor
            u: Control inputs tensor
            edge_index: Graph connectivity tensor
        """
        Gz = self.encoder(x, u, edge_index)
        z, z_dot = self.dynamics(Gz, u)
        x_hat = self.decoder(Gz, edge_index)
        return z, z_dot, x_hat