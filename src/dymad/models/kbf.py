import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Union, Tuple

from dymad.data import DynData, DynGeoData
from dymad.models import ModelBase
from dymad.utils import FlexLinear, make_autoencoder, predict_continuous, predict_discrete, \
    predict_graph_continuous, predict_graph_discrete

class KBF(ModelBase):
    """
    Koopman Bilinear Form (KBF) model - standard version.
    Uses MLP encoder/decoder and KBF operators for dynamics.

    - z = encoder(x)
    - z_dot = Az + sum(B_i * u_i * z)
    - x_hat = decoder(z)
    """
    GRAPH = False

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
        aec_type = model_config.get('autoencoder_type', 'smp')

        # Build encoder/decoder networks
        self.encoder_net, self.decoder_net = make_autoencoder(
            type="mlp_"+aec_type,
            input_dim=self.n_total_state_features,
            latent_dim=self.latent_dimension,
            hidden_dim=self.koopman_dimension,
            enc_depth=enc_depth,
            dec_depth=dec_depth,
            output_dim=self.n_total_state_features,
            **opts
        )

        # Create KBF operators: first for autonomous dynamics (A) then one per control (B_i)
        if self.n_total_control_features > 0:
            if self.const_term:
                dyn_dim = self.koopman_dimension * (self.n_total_control_features + 1) + self.n_total_control_features
            else:
                dyn_dim = self.koopman_dimension * (self.n_total_control_features + 1)
        else:
            dyn_dim = self.koopman_dimension
        self.dynamics_net = FlexLinear(dyn_dim, self.koopman_dimension, bias=False)

        if self.n_total_control_features == 0:
            self._zu_cat = self._zu_cat_auto
        else:
            self._zu_cat = self._zu_cat_ctrl

    def diagnostic_info(self) -> str:
        model_info = super(KBF, self).diagnostic_info()
        model_info += f"Encoder: {self.encoder_net.diagnostic_info()}\n"
        model_info += f"Decoder: {self.decoder_net.diagnostic_info()}\n"
        model_info += f"Input order: {self.input_order}"
        return model_info

    def encoder(self, w: DynData) -> torch.Tensor:
        """Encode combined features to Koopman space."""
        return self.encoder_net(w.x)

    def decoder(self, z: torch.Tensor, w: DynData) -> torch.Tensor:
        """Decode from Koopman space back to state space."""
        return self.decoder_net(z)

    def dynamics(self, z: torch.Tensor, w: DynData) -> torch.Tensor:
        """Compute dynamics in Koopman space using bilinear form."""
        return self.dynamics_net(self._zu_cat(z, w))

    def _zu_cat_ctrl(self, z: torch.Tensor, w: DynData) -> torch.Tensor:
        """Concatenate state and control inputs."""
        z_u = (z.unsqueeze(-1) * w.u.unsqueeze(-2)).reshape(*z.shape[:-1], -1)
        if self.const_term:
            return torch.cat([z, z_u, w.u], dim=-1)
        return torch.cat([z, z_u], dim=-1)

    def _zu_cat_auto(self, z: torch.Tensor, w: DynData) -> torch.Tensor:
        """Concatenate state and control inputs."""
        return z

    def forward(self, w: DynData) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for KBF model.

        Args:
            w: DynData obejct, containing state (x) and control (u) tensors.

        Returns:
            Tuple of (latent, latent_derivative, reconstruction)
        """
        z = self.encoder(w)
        z_dot = self.dynamics(z, w)
        x_hat = self.decoder(z, w)
        return z, z_dot, x_hat

    def predict(self, x0: torch.Tensor, w: DynData, ts: Union[np.ndarray, torch.Tensor],
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
        return predict_continuous(self, x0, ts, us=w.u, method=method, order=self.input_order)

class DKBF(KBF):
    """Discrete Koopman Bilinear Form (DKBF) model - discrete-time version.

    In this case, the forward pass effectively does the following:

    ```
    z_n = self.encoder(w_n)
    z_{n+1} = self.dynamics(z_n, w_n)
    x_hat_n = self.decoder(z_n, w_n)
    ```
    """
    GRAPH = False

    def __init__(self, model_config: Dict, data_meta: Dict):
        super(DKBF, self).__init__(model_config, data_meta)

    def linear_features(self, w: DynData) -> torch.Tensor:
        """Compute linear features for training."""
        z = self.encoder(w)
        return self._zu_cat(z, w)[..., :-1, :]

    def linear_targets(self, w: DynData) -> torch.Tensor:
        """Compute linear targets for training."""
        z = self.encoder(w)
        return z[..., 1:, :]

    def linear_eval(self, z: torch.Tensor) -> torch.Tensor:
        """Evaluate linear features using the learned weights."""
        return self.dynamics_net(z)

    def set_linear_weights(self, W: torch.Tensor) -> None:
        """Set weights for the dynamics network."""
        if W.shape != self.dynamics_net.weight.shape:
            raise ValueError(f"Weight shape mismatch: expected {self.dynamics_net.weight.shape}, got {W.shape}")
        self.dynamics_net.weight.data = W

    def predict(self, x0: torch.Tensor, w: DynData, ts: Union[np.ndarray, torch.Tensor], **kwargs) -> torch.Tensor:
        """Predict trajectory using discrete-time iterations."""
        return predict_discrete(self, x0, ts, us=w.u)

class GKBF(ModelBase):
    """Graph Koopman Bilinear Form (GKBF) model - graph-specific version.
    Uses GNN encoder/decoder and KBF operators for dynamics.

    Koopman dimension is defined per node.
    """
    GRAPH = True

    def __init__(self, model_config: Dict, data_meta: Dict):
        super(GKBF, self).__init__()
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

        # Determine other options for GNN layers
        opts = {
            'gcl'            : model_config.get('gcl', 'sage'),
            'activation'     : model_config.get('activation', 'prelu'),
            'weight_init'    : model_config.get('weight_init', 'xavier_uniform'),
            'bias_init'      : model_config.get('bias_init', 'zeros'),
            'gain'           : model_config.get('gain', 1.0),
            'end_activation' : model_config.get('end_activation', True)
        }
        aec_type = model_config.get('autoencoder_type', 'smp')

        # Build encoder/decoder networks
        self.encoder_net, self.decoder_net = make_autoencoder(
            type="gnn_"+aec_type,
            input_dim=self.n_total_state_features,
            latent_dim=self.latent_dimension,
            hidden_dim=self.koopman_dimension,
            enc_depth=enc_depth,
            dec_depth=dec_depth,
            output_dim=self.n_total_state_features,
            **opts
        )

        # KBF operators for graph system
        tmp = [
            nn.Linear(self.koopman_dimension, self.koopman_dimension, bias=False)
            for _ in range(self.n_total_control_features + 1)
        ]
        if self.const_term and self.n_total_control_features > 0:
            tmp.append(nn.Linear(self.n_total_control_features, self.koopman_dimension, bias=False))
        self.operators = nn.ModuleList(tmp)

        if self.n_total_control_features == 0:
            self.dynamics = self._dynamics_auto
        else:
            self.dynamics = self._dynamics_ctrl

    def diagnostic_info(self) -> str:
        model_info = super(GKBF, self).diagnostic_info()
        model_info += f"Encoder: {self.encoder_net.diagnostic_info()}\n"
        model_info += f"Decoder: {self.decoder_net.diagnostic_info()}\n"
        model_info += f"Input order: {self.input_order}"
        return model_info

    def encoder(self, w: DynGeoData) -> torch.Tensor:
        return self.encoder_net(w.xg, w.edge_index)

    def decoder(self, z: torch.Tensor, w: DynGeoData) -> torch.Tensor:
        return self.decoder_net(w.g(z), w.edge_index)

    def _dynamics_ctrl(self, z: torch.Tensor, w: DynGeoData) -> Tuple[torch.Tensor, torch.Tensor]:
        z_reshaped = w.g(z)
        u_reshaped = w.ug

        # Autonomous part: A @ z
        z_dot = self.operators[0](z_reshaped)

        # Add control-dependent terms: sum(u_i * B_i @ z)
        for i in range(self.n_total_control_features):
            control_i = u_reshaped[..., i].unsqueeze(-1)  # Extract control i and add dimension for broadcasting
            z_dot = z_dot + control_i * self.operators[i + 1](z_reshaped)

        # Add constant term if enabled
        if self.const_term:
            z_dot = z_dot + self.operators[-1](u_reshaped)

        return w.G(z_dot)

    def _dynamics_auto(self, z: torch.Tensor, w: DynGeoData) -> Tuple[torch.Tensor, torch.Tensor]:
        # Autonomous part: A @ z
        z_dot = self.operators[0](w.g(z))
        return w.G(z_dot)

    def forward(self, w: DynGeoData) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder(w)
        z_dot = self.dynamics(z, w)
        x_hat = self.decoder(z, w)
        return z, z_dot, x_hat

    def predict(self, x0: torch.Tensor, w: DynGeoData, ts: Union[np.ndarray, torch.Tensor], method: str = 'dopri5') -> torch.Tensor:
        return predict_graph_continuous(self, x0, ts, w.edge_index, us=w.u, method=method, order=self.input_order)

class DGKBF(GKBF):
    """Discrete Graph Koopman Bilinear Form (DGKBF) model - discrete-time version.

    Same idea as DKBF vs KBF.
    """
    GRAPH = True

    def __init__(self, model_config: Dict, data_meta: Dict):
        super(DGKBF, self).__init__(model_config, data_meta)

    def predict(self, x0: torch.Tensor, w: DynGeoData, ts: Union[np.ndarray, torch.Tensor], **kwargs) -> torch.Tensor:
        """Predict trajectory using discrete-time iterations."""
        return predict_graph_discrete(self, x0, ts, w.edge_index, us=w.u)