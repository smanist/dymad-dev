import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Union, Tuple

from dymad.data import DynData, DynGeoData
from dymad.models import ModelBase
from dymad.utils import FlexLinear, make_autoencoder, predict_continuous, predict_continuous_exp, \
    predict_discrete, predict_discrete_exp, \
    predict_graph_continuous, predict_graph_discrete

class KBF(ModelBase):
    """
    Koopman Bilinear Form (KBF) model - standard version.
    Uses MLP encoder/decoder and KBF operators for dynamics.

    - z = encoder(x)
    - z_dot = Az + sum(B_i * u_i * z) + Bu
    - x_hat = decoder(z)
    """
    GRAPH = False
    CONT  = True

    def __init__(self, model_config: Dict, data_meta: Dict, dtype=None, device=None):
        super(KBF, self).__init__()
        self.n_total_state_features = data_meta.get('n_total_state_features')
        self.n_total_control_features = data_meta.get('n_total_control_features')
        self.latent_dimension = model_config.get('latent_dimension', 64)
        self.koopman_dimension = model_config.get('koopman_dimension', 16)
        self.const_term = model_config.get('const_term', True)

        self._predictor_type = model_config.get('predictor_type', 'ode')
        if self.n_total_control_features > 0:
            if self._predictor_type == "exp":
                raise ValueError("Exponential predictor is not supported for KBF with control inputs.")

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
            'end_activation' : model_config.get('end_activation', True),
            'dtype'          : dtype,
            'device'         : device
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

        # Create KBF operators, concatenated
        if self.n_total_control_features > 0:
            if self.const_term:
                dyn_dim = self.koopman_dimension * (self.n_total_control_features + 1) + self.n_total_control_features
            else:
                dyn_dim = self.koopman_dimension * (self.n_total_control_features + 1)
        else:
            dyn_dim = self.koopman_dimension
        self.dynamics_net = FlexLinear(dyn_dim, self.koopman_dimension, bias=False, dtype=dtype, device=device)

        if self.n_total_control_features == 0:
            self._zu_cat = self._zu_cat_auto
        else:
            self._zu_cat = self._zu_cat_ctrl

        self.set_linear_weights = self.dynamics_net.set_weights

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
                method: str = 'dopri5', **kwargs) -> torch.Tensor:
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
        if self._predictor_type == "exp":
            return predict_continuous_exp(self, x0, ts, **kwargs)
        return predict_continuous(self, x0, ts, us=w.u, method=method, order=self.input_order, **kwargs)

    def linear_features(self, w: DynData) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute linear features, f, and outputs, dz, for (D)KBF model.

        dz = Af

        For KBF, f contains the bilinear terms, z, z*u, u;
        dz is the output of KBF dynamics, z_dot for cont-time, z_next for disc-time.
        """
        z = self.encoder(w)
        return self._zu_cat(z, w), z

    def linear_eval(self, w: DynData) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute linear evaluation, dz, and states, z, for (D)KBF model.

        dz = Af

        For KBF, dz is the output of KBF dynamics.
        z is the encoded state, which will be used to compute the expected output.
        """
        z = self.encoder(w)
        z_dot = self.dynamics(z, w)
        return z_dot, z

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
    CONT  = False

    def __init__(self, model_config: Dict, data_meta: Dict, dtype=None, device=None):
        super(DKBF, self).__init__(model_config, data_meta, dtype=dtype, device=device)

    def predict(self, x0: torch.Tensor, w: DynData, ts: Union[np.ndarray, torch.Tensor], **kwargs) -> torch.Tensor:
        """Predict trajectory using discrete-time iterations."""
        if self._predictor_type == "exp":
            return predict_discrete_exp(self, x0, ts, **kwargs)
        return predict_discrete(self, x0, ts, us=w.u)

class GKBF(ModelBase):
    """Graph Koopman Bilinear Form (GKBF) model - graph-specific version.
    Uses GNN encoder/decoder and KBF operators for dynamics.

    Koopman dimension is defined per node.
    """
    GRAPH = True
    CONT  = True

    def __init__(self, model_config: Dict, data_meta: Dict, dtype=None, device=None):
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
            'end_activation' : model_config.get('end_activation', True),
            'dtype'          : dtype,
            'device'         : device
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

        # Create KBF operators, concatenated
        if self.n_total_control_features > 0:
            if self.const_term:
                dyn_dim = self.koopman_dimension * (self.n_total_control_features + 1) + self.n_total_control_features
            else:
                dyn_dim = self.koopman_dimension * (self.n_total_control_features + 1)
        else:
            dyn_dim = self.koopman_dimension
        self.dynamics_net = FlexLinear(dyn_dim, self.koopman_dimension, bias=False, dtype=dtype, device=device)

        if self.n_total_control_features == 0:
            self._zu_cat = self._zu_cat_auto
        else:
            self._zu_cat = self._zu_cat_ctrl

        self.set_linear_weights = self.dynamics_net.set_weights

    def diagnostic_info(self) -> str:
        model_info = super(GKBF, self).diagnostic_info()
        model_info += f"Encoder: {self.encoder_net.diagnostic_info()}\n"
        model_info += f"Decoder: {self.decoder_net.diagnostic_info()}\n"
        model_info += f"Input order: {self.input_order}"
        return model_info

    def encoder(self, w: DynGeoData) -> torch.Tensor:
        # The GNN implementation outputs flattened features
        # Here internal dynamics are node-wise, so we need to reshape
        # the features to node*features_per_node again
        return w.g(self.encoder_net(w.xg, w.edge_index))

    def decoder(self, z: torch.Tensor, w: DynGeoData) -> torch.Tensor:
        # Since the decoder outputs to the original space,
        # which is assumed to be flattened, we can use the GNN decoder directly
        # Note: the input, though, is still node-wise
        return self.decoder_net(z, w.edge_index)

    def dynamics(self, z: torch.Tensor, w: DynGeoData) -> torch.Tensor:
        """Compute dynamics in Koopman space using bilinear form."""
        return self.dynamics_net(self._zu_cat(z, w))

    def _zu_cat_ctrl(self, z: torch.Tensor, w: DynGeoData) -> torch.Tensor:
        """Concatenate state and control inputs."""
        u_reshaped = w.ug
        z_u = (z.unsqueeze(-1) * u_reshaped.unsqueeze(-2)).reshape(*z.shape[:-1], -1)
        if self.const_term:
            return torch.cat([z, z_u, u_reshaped], dim=-1)
        return torch.cat([z, z_u], dim=-1)

    def _zu_cat_auto(self, z: torch.Tensor, w: DynGeoData) -> torch.Tensor:
        """Concatenate state and control inputs."""
        return z

    def forward(self, w: DynGeoData) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder(w)
        z_dot = self.dynamics(z, w)
        x_hat = self.decoder(z, w)
        return z, z_dot, x_hat

    def predict(self, x0: torch.Tensor, w: DynGeoData, ts: Union[np.ndarray, torch.Tensor], method: str = 'dopri5', **kwargs) -> torch.Tensor:
        return predict_graph_continuous(self, x0, ts, w.edge_index, us=w.u, method=method, order=self.input_order, **kwargs)

    def linear_features(self, w: DynGeoData) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute linear features, f, and outputs, dz, for (D)GKBF model.

        Main difference with KBF: the middle two dimensions are permuted, so that
        the time dimension is the second last dimension, this is needed in
        linear trainer to match the expected shape.
        """
        z = self.encoder(w)
        f = self._zu_cat(z, w)
        return f.permute(0, 2, 1, 3), z.permute(0, 2, 1, 3)

    def linear_eval(self, w: DynGeoData) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute linear evaluation, dz, and states, z, for (D)GKBF model.

        Same idea as in linear_features about the permutation.
        """
        z = self.encoder(w)
        z_dot = self.dynamics(z, w)
        return z_dot.permute(0, 2, 1, 3), z.permute(0, 2, 1, 3)

class DGKBF(GKBF):
    """Discrete Graph Koopman Bilinear Form (DGKBF) model - discrete-time version.

    Same idea as DKBF vs KBF.
    """
    GRAPH = True
    CONT  = False

    def __init__(self, model_config: Dict, data_meta: Dict, dtype=None, device=None):
        super(DGKBF, self).__init__(model_config, data_meta, dtype=dtype, device=device)

    def predict(self, x0: torch.Tensor, w: DynGeoData, ts: Union[np.ndarray, torch.Tensor], **kwargs) -> torch.Tensor:
        """Predict trajectory using discrete-time iterations."""
        return predict_graph_discrete(self, x0, ts, w.edge_index, us=w.u)