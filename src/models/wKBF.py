from .model_base import weakKBFBase
import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv
from typing import Dict

class weakKBF(weakKBFBase):
    def encoder(self, z: torch.Tensor, u: torch.Tensor, **kwargs) -> torch.Tensor:
        # For the vanilla wKBF, use identity mapping (i.e. no transformation).
        return z

    def decoder(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        # For the vanilla wKBF, use identity mapping.
        return z
class weakGraphKBF(weakKBFBase):
    def __init__(self, config: Dict):
        super(weakGraphKBF, self).__init__(config)
        self.latent_dimension = config.get('latent_dimension', 64)
        self.n_gnn_layers = config.get('n_gnn_layers', 2)
        # Build GNN encoder: maps node features to Koopman space
        self.encoder_layers = self._build_gnn(
            input_dimension=self.n_state_features,
            latent_dimension=self.latent_dimension,
            output_dimension=self.koopman_dimension,
            num_layers=self.n_gnn_layers
        )
        # Build GNN decoder: maps Koopman space back to node features
        self.decoder_layers = self._build_gnn(
            input_dimension=self.koopman_dimension,
            latent_dimension=self.latent_dimension,
            output_dimension=self.n_state_features,
            num_layers=self.n_gnn_layers
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

    def encoder(self, x: torch.Tensor, u: torch.Tensor, edge_index: torch.Tensor = None) -> torch.Tensor:
        if edge_index is None:
            raise ValueError("edge_index is required for graph-based models")
        for layer in self.encoder_layers:
            x = layer(x, edge_index) if hasattr(layer, 'aggr') else layer(x)
        return x

    def decoder(self, z: torch.Tensor, edge_index: torch.Tensor = None) -> torch.Tensor:
        if edge_index is None:
            raise ValueError("edge_index is required for graph-based models")
        x = z
        for i, layer in enumerate(self.decoder_layers):
            x = layer(x, edge_index) if (i % 2 == 0) else layer(x)
        return x