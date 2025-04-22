import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv
from abc import ABC, abstractmethod
from typing import Tuple, Dict


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
    
class weakFormMLP(nn.Module):
    def __init__(self, config: Dict):
        """
        Args:
            config: Dictionary containing model parameters:
                - state_dim: Dimension of the state space
                - control_dim: Dimension of the control input
                - hidden_dim: Dimension of the latent space (default: 64)
                - encoder_layers: Number of encoder layers (default: 2)
                - processor_layers: Number of processor layers (default: 2)
                - decoder_layers: Number of decoder layers (default: 2)
        """
        super(weakFormMLP, self).__init__()
        
        # Extract configuration parameters
        self.state_dim = config['state_dim']
        self.control_dim = config['control_dim']
        self.hidden_dim = config.get('hidden_dim', 64)
        enc_depth = config.get('encoder_layers', 2)
        proc_depth = config.get('processor_layers', 2)
        dec_depth = config.get('decoder_layers', 2)
        
        # Build network components
        self.encoder_net = self._build_mlp(
            input_dim=self.state_dim + self.control_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            num_layers=enc_depth
        )
        
        self.dynamics_net = self._build_mlp(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            num_layers=proc_depth
        )
        
        self.decoder_net = self._build_mlp(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.state_dim,
            num_layers=dec_depth
        )

    def _build_mlp(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> nn.Sequential:
        """
        Helper function to build a multi-layer perceptron.
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension
            output_dim: Output dimension
            num_layers: Number of layers
            
        Returns:
            Sequential network
        """
        if num_layers == 1:
            return nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.PReLU()
            )
        
        layers = [nn.Linear(input_dim, hidden_dim), nn.PReLU()]
        
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.PReLU()])
        
        layers.append(nn.Linear(hidden_dim, output_dim), nn.PReLU())
            
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
            x: State tensor of shape (batch_size, state_dim)
            u: Control tensor of shape (batch_size, control_dim)
            
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
    
    def predict(self, x0: torch.Tensor, us: torch.Tensor, ts: torch.Tensor = None, 
            method: str = 'dopri5', proj: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict trajectory using continuous-time integration.
        
        Args:
            x0: Initial state tensor (batch_size, state_dim)
            us: Control inputs, either:
                - Constant control (batch_size, control_dim)
                - Full trajectory (time_steps, batch_size, control_dim)
            ts: Time points for prediction (required)
            method: ODE solver method
            proj: Whether to use projection-based dynamics
            
        Returns:
            Tuple of (predicted_states, true_states if available, else None)
        """
        from torchdiffeq import odeint
        
        device = x0.device
        
        # Ensure ts is a tensor
        if ts is None:
            raise ValueError("Time points (ts) must be provided for continuous prediction")
        
        if isinstance(ts, list):
            ts = torch.tensor(ts, dtype=torch.double).to(device)
        
        # For this model we assume constant control
        if len(us.shape) == 3:  # Full trajectory provided
            u0 = us[0]  # Just use the first control
        else:
            u0 = us  # Constant control already provided
            
        # Initial latent state
        w0 = self.encoder(x0, u0)
        
        if proj:
            # Projection-based dynamics
            def _func(t, w):
                # Decode w to x at every step
                x = self.decoder(w).detach()
                # Combine with control input and get derivative in latent space
                _, w_dot, _ = self(x, u0)
                return w_dot.squeeze().detach()
        else:
            # Pure latent dynamics
            def _func(t, w):
                # Evolve dynamics in latent space
                w_dot = self.dynamics(w, u0)
                return w_dot.squeeze().detach()
        
        # Integrate using ODE solver
        w_traj = odeint(_func, w0.squeeze(), ts, method=method)
        
        # Decode trajectory
        x_pred = self.decoder(w_traj)
        
        return x_pred
    
class WeakKBFBase(ModelBase):
    def __init__(self, config: Dict):
        super(WeakKBFBase, self).__init__()
        self.state_dim = config['state_dim']
        self.control_dim = config['control_dim']
        self.n_nodes = config.get('n_nodes', 1)
        self.koopman_dim = config.get('koopman_dim', 16)
        self.system_dim = self.n_nodes * self.koopman_dim

        # Create KBF operators: first for autonomous dynamics (A) then one per control (B_i)
        self.operators = nn.ModuleList([
            nn.Linear(self.system_dim, self.system_dim, bias=False)
            for _ in range(self.control_dim + 1)
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
        for i in range(self.control_dim):
            w_dot = w_dot + u_reshaped[:, i].unsqueeze(-1) * self.operators[i + 1](w)
        return w_dot

    def forward(self, x: torch.Tensor, u: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder(x, u, **kwargs)
        z_dot = self.dynamics(z, u)
        x_hat = self.decoder(z, **kwargs)
        return z, z_dot, x_hat
    
    def predict(self, x0: torch.Tensor, us: torch.Tensor, ts: torch.Tensor = None, 
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
                - Control trajectory (time_steps x control_dim)
            ts: Time points for prediction (required).
            edge_index: Graph edge indices (for graph-based models).
            node_type: Optional node type tensor (used in graph-based models if provided).
            method: ODE solver method.
            
        Returns:
            Predicted trajectory tensor.
        """
        from torchdiffeq import odeint
        device = x0.device

        if ts is None:
            raise ValueError("Time points (ts) must be provided for continuous prediction")
        if isinstance(ts, list):
            ts = torch.tensor(ts, dtype=torch.double).to(device)

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
        w_traj = odeint(ode_func, w0.squeeze(), ts, method=method)

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
class weakKBF(WeakKBFBase):
    def encoder(self, z: torch.Tensor, u: torch.Tensor, **kwargs) -> torch.Tensor:
        # For the vanilla wKBF, use identity mapping (i.e. no transformation).
        return z

    def decoder(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        # For the vanilla wKBF, use identity mapping.
        return z
class weakGraphKBF(WeakKBFBase):
    def __init__(self, config: Dict):
        super(weakGraphKBF, self).__init__(config)
        self.hidden_dim = config.get('hidden_dim', 64)
        self.n_gnn_layers = config.get('n_gnn_layers', 2)
        # Build GNN encoder: maps node features to Koopman space
        self.encoder_layers = self._build_gnn(
            input_dim=self.state_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.koopman_dim,
            num_layers=self.n_gnn_layers
        )
        # Build GNN decoder: maps Koopman space back to node features
        self.decoder_layers = self._build_gnn(
            input_dim=self.koopman_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.state_dim,
            num_layers=self.n_gnn_layers
        )

    def _build_gnn(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> nn.ModuleList:
        layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = output_dim if i == num_layers - 1 else hidden_dim
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

    
