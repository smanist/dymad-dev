import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Union
from abc import ABC, abstractmethod

class ModelBase(nn.Module, ABC):
    """
    Base class for dynamic models.

    Notation:
      x: Physical state space.
      z: Embedding (latent) space.

    Discrete-time model:
      z_k = encoder(x_k)
      z_k+1 = dynamics(z_k, u_k)
      x_k+1 = decoder(z_k+1)

    Continuous-time model:
      z = encoder(x)
      z_dot = dynamics(z, u)
      x = decoder(z)
    """
    def __init__(self):
        super(ModelBase, self).__init__()

    @abstractmethod
    def init_params(self):
        raise NotImplementedError("This is the base class.")

    @abstractmethod
    def encoder(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("This is the base class.")

    @abstractmethod
    def decoder(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("This is the base class.")

    @abstractmethod
    def features(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("This is the base class.")

    @abstractmethod
    def dynamics(self, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("This is the base class.")

    @abstractmethod
    def predict(self, x0: torch.Tensor, us: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("This is the base class.")
    
    @staticmethod
    def build_mlp(input_dimension: int, latent_dimension: int, output_dimension: int, num_layers: int) -> nn.Sequential:
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

class weakKBFBase(ModelBase):
    def __init__(self, model_config: Dict, data_meta: Dict):
        
        super(weakKBFBase, self).__init__()
        self.n_state_features = data_meta.get('n_state_features')
        self.n_control_features = data_meta.get('n_control_features')
        self.n_total_features = data_meta.get('n_total_features')        
        self.latent_dimension = model_config.get('latent_dimension', 64)
        self.n_state_features = model_config['n_state_features']
        self.n_control_features = model_config['n_control_features']
        self.n_nodes = model_config.get('n_nodes', 1)
        self.koopman_dimension = model_config.get('koopman_dimension', 16)
        self.system_dimension = self.n_nodes * self.koopman_dimension # TODO: this needs another look-at for non-graph systems

        # Create KBF operators: first for autonomous dynamics (A) then one per control (B_i)
        self.operators = nn.ModuleList([
            nn.Linear(self.system_dimension, self.system_dimension, bias=False)
            for _ in range(self.n_control_features + 1)
        ])

    def init_params(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def features(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, u], dim=-1)

    def dynamics(self, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        batch_size = z.shape[0] if z.dim() > 1 else 1
        # Reshape z into (batch_size, system_dim)
        w = z.reshape(batch_size, self.n_nodes, -1).permute(0, 2, 1)
        w = w.reshape(batch_size, -1)
        # Autonomous part: A @ w
        w_dot = self.operators[0](w)
        # Add control-dependent terms: sum(u_i * B_i @ w)
        u_reshaped = u.reshape(batch_size, -1)
        for i in range(self.n_control_features):
            w_dot = w_dot + u_reshaped[:, i].unsqueeze(-1) * self.operators[i + 1](w)
        return w_dot

    def forward(self, x: torch.Tensor, u: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder(x, u, **kwargs)
        z_dot = self.dynamics(z, u)
        x_hat = self.decoder(z, **kwargs)
        return z, z_dot, x_hat
    
    def predict(self, x0: torch.Tensor, us: torch.Tensor, ts: Union[np.ndarray, torch.Tensor], 
                edge_index: torch.Tensor = None, node_type: torch.Tensor = None, 
                method: str = 'dopri5') -> torch.Tensor:
        """
        Predict trajectory using continuous-time integration.
        For graph-based models (edge_index provided), the encoder and decoder will use the graph structure.
        For non-graph models, edge_index is ignored.
        
        Args:
            x0: Initial state (or node features) tensor.
            us: Control inputs, either:
                - Constant control (1D tensor or 2D tensor with one row)
                - Control trajectory (time_steps x n_control_features)
            ts: Time points for prediction (required).
            edge_index: Graph edge indices (for graph-based models).
            node_type: Optional node type tensor (used in graph-based models if provided).
            method: ODE solver method.
            
        Returns:
            Predicted trajectory tensor.
        """
        from torchdiffeq import odeint
        device = x0.device

        # Handle control inputs: decide if constant or trajectory.
        if (us.ndim == 1) or (us.ndim == 2 and us.shape[0] == 1):
            u_func = lambda t: us.to(device)
            u_const = us.to(device)
        else:
            # For control trajectories, interpolate over time.
            import scipy.interpolate as sp_inter
            u_np = us.cpu().detach().numpy()
            ts_np = ts.cpu().detach().numpy()
            u_interp = sp_inter.interp1d(ts_np[:len(u_np)], u_np, axis=0, fill_value='extrapolate')
            u_func = lambda t: torch.tensor(u_interp(t.cpu().detach().numpy()),
                                              dtype=us.dtype).to(device)
            u_const = u_func(ts[0])

        # Set initial latent state based on model type.
        if edge_index is not None:
            # Graph-based model: use encoder with edge_index.
            w0 = self.encoder(x0, None, edge_index)
            # Flatten latent state as needed.
            w0 = w0.T.flatten().detach()
            def ode_func(t, w):
                # Reshape latent vector to (n_nodes, latent_dim)
                w_reshaped = w.reshape(-1, self.n_nodes).T
                u_t = u_func(t)
                if node_type is not None:
                    # Decode, concatenate with node type, and compute derivative via forward pass.
                    z = self.decoder(w_reshaped, edge_index)
                    z_with_type = torch.cat([z, node_type.to(device)], dim=1)
                    _, w_dot, _ = self(z_with_type, u_t, edge_index)
                else:
                    # Use dynamics directly.
                    w_dot = self.dynamics(w_reshaped, u_t)
                return w_dot.squeeze().detach()
        else:
            # Non-graph model: encoder and decoder do not use edge_index.
            w0 = self.encoder(x0, u_const)
            def ode_func(t, w):
                u_t = u_func(t)
                w_dot = self.dynamics(w, u_t)
                return w_dot.squeeze().detach()

        # ODE integration.
        w_traj = odeint(ode_func, w0.squeeze(), torch.DoubleTensor(ts).to(device), method=method)

        if edge_index is not None:
            # For graph-based models, reshape and decode each latent state.
            w_traj = w_traj.reshape(len(ts), -1, self.n_nodes).permute(0, 2, 1)
            z_pred = []
            for w in w_traj:
                z_pred.append(self.decoder(w, edge_index))
            return torch.stack(z_pred)
        else:
            # For non-graph models, decode the integrated trajectory.
            return self.decoder(w_traj)