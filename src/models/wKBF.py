from .model_base import ModelBase
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.nn import SAGEConv
from typing import Dict, Union, Tuple

class weakKBF(ModelBase):
    def encoder(self, z: torch.Tensor, u: torch.Tensor, **kwargs) -> torch.Tensor:
        # For the vanilla wKBF, use identity mapping (i.e. no transformation).
        return z

    def decoder(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        # For the vanilla wKBF, use identity mapping.
        return z

class weakGraphKBF(ModelBase):
    def __init__(self, model_config: Dict, data_meta: Dict):
        super(weakGraphKBF, self).__init__()
        
        # Initialize base parameters
        self.n_state_features = data_meta.get('n_state_features')
        self.n_control_features = data_meta.get('n_control_features')
        self.n_total_features = data_meta.get('n_total_features')        
        self.n_nodes = data_meta['config']['data']['n_nodes']
        
        self.in_out_dimension = data_meta['delay'] + 1
        self.latent_dimension = model_config.get('latent_dimension', 64)
        self.koopman_dimension = model_config.get('koopman_dimension', 16)
        self.system_dimension = self.n_nodes * self.koopman_dimension

        # Create KBF operators: first for autonomous dynamics (A) then one per control (B_i)
        self.operators = nn.ModuleList([
            nn.Linear(self.system_dimension, self.system_dimension, bias=False)
            for _ in range(self.n_control_features + 1)
        ])

        self.n_gnn_layers = model_config.get('n_gnn_layers', 2)

        # Build GNN encoder: maps node features to Koopman space
        self.encoder_layers = self._build_gnn(
            input_dimension=self.in_out_dimension,
            latent_dimension=self.latent_dimension,
            output_dimension=self.koopman_dimension,
            num_layers=self.n_gnn_layers
        )
        # Build GNN decoder: maps Koopman space back to node features
        self.decoder_layers = self._build_gnn(
            input_dimension=self.koopman_dimension,
            latent_dimension=self.latent_dimension,
            output_dimension=self.in_out_dimension,
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
        """Predict trajectory using continuous-time integration.
        
        Args:
            x0: Initial state tensor of shape (n_nodes, n_features)
            us: Control inputs tensor. Can be:
                - Constant control: shape (n_controls,) or (1, n_controls)
                - Time-varying control: shape (n_timesteps, n_controls)
            ts: Time points for prediction. Can be:
                - numpy array of shape (n_timesteps,)
                - torch tensor of shape (n_timesteps,)
            edge_index: Graph connectivity tensor of shape (2, n_edges)
            method: ODE solver method (default: 'dopri5')
            
        Returns:
            Predicted trajectory tensor of shape (n_timesteps, n_nodes, n_features)
        """
        from torchdiffeq import odeint
        import scipy.interpolate as sp_inter
        device = x0.device

        if x0.shape[0] != self.n_nodes:
            raise ValueError(f"x0 must have {self.n_nodes} nodes, got {x0.shape[0]}")

        # Convert ts to tensor if needed
        if isinstance(ts, np.ndarray):
            ts = torch.from_numpy(ts).to(device)

        # Handle control inputs
        if (us.ndim == 1) or (us.ndim == 2 and us.shape[0] == 1):
            # Constant control
            u_func = lambda t: us.to(device)
        else:
            # Time-varying control: interpolate
            u_np = us.cpu().detach().numpy()
            ts_np = ts.cpu().detach().numpy()
            u_interp = sp_inter.interp1d(ts_np[:len(u_np)], u_np, axis=0, fill_value='extrapolate')
            u_func = lambda t: torch.tensor(u_interp(t.cpu().detach().numpy()),
                                          dtype=us.dtype).to(device)
        t0 = torch.tensor(0.0).to(device)
        u0 = u_func(t0)
        # Encode initial state
        w0 = self.encoder(x0, u0, edge_index)
        w0 = w0.T.flatten().detach()
        
        def ode_func(t, w):
            # Reshape latent vector to (n_nodes, latent_dim)
            w_reshaped = w.reshape(-1, self.n_nodes).T
            u_t = u_func(t)
            # Get dynamics
            w_dot = self.dynamics(w_reshaped, u_t)[1]  # Get w_dot from dynamics
            return w_dot.squeeze().detach()

        # ODE integration
        w_traj = odeint(ode_func, w0.squeeze(), ts, method=method)

        # Reshape and decode trajectory
        w_traj = w_traj.reshape(len(ts), -1, self.n_nodes).permute(0, 2, 1)
        z_pred = [self.decoder(w, edge_index) for w in w_traj]
        
        return torch.stack(z_pred)

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