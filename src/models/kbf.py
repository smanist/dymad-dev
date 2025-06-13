from .model_base import ModelBase
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.nn import SAGEConv
from typing import Dict, Union, Tuple
from src.utils.prediction import predict_graph_continuous

class KBF(ModelBase): 
    """
    TODO: KBF class not implemented yet.
    This will be a non-graph version of the Koopman Bilinear Form model, trained with weak form.
    """
    pass

class GKBF(ModelBase):
    def __init__(self, model_config: Dict, data_meta: Dict):
        super(GKBF, self).__init__()
        
        # Initialize base parameters
        self.n_state_features = data_meta.get('n_state_features')
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
        """Transform input features to latent space."""
        return torch.cat([x, u], dim=-1)

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
                For autonomous systems, use zero-valued controls
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