import numpy as np
import textwrap as tw
import torch
import torch.nn as nn
from typing import Any, Callable, Dict, Tuple, Union

from dymad.io import DynData


Encoder = Callable[[nn.Module, DynData], torch.Tensor]
"""encoder(net, w) -> x"""

Features = Callable[[torch.Tensor, DynData], torch.Tensor]
"""features(z, w) -> s"""

Composer = Callable[[nn.Module, torch.Tensor, torch.Tensor, DynData], torch.Tensor]
"""composer(net, s, z, w) -> r"""

Decoder = Callable[[nn.Module, torch.Tensor, DynData], torch.Tensor]
"""decoder(net, z, w) -> x"""

Predictor = Callable[[torch.Tensor, DynData, Union[np.ndarray, torch.Tensor], Any], Tuple[torch.Tensor, torch.Tensor]]
r"""predict(x0, w, ts, \*\*kwargs) -> (x_pred, z_pred)"""


class ComposedDynamics(nn.Module):
    r"""Base class for dynamic models.

    Notation:

    - x: Physical state/observation space; can be time-delayed
    - u: Control input; can be time-delayed
    - z: Embedded space, where dynamics is learned.
    - s: Features for dynamics, composing z and u as needed
    - r: Output of processor, might be lower dimensional than z
    - z': next step of z (discrete-time) or z_dot (continuous-time)

    Full model:

    - z = encoder(x, u)
    - z' = dynamics(z, u)
    - x = decoder(z)

    Details in dynamics:

    - s = features(z, u); e.g., concatenation of linear or bilinear terms
    - r = processor(s, u); e.g., NN or linear transform
    - z' = composition(s, r); e.g., Direct or Skip-Connection

    In the above, `encoder`, `features`, `composer`, and `decoder` are functions
    that should be hooked to the model instance, while `processor` is a nn.Module.

    Linear training assumes:

    - processor is linear
    - linear_targets = r = W @ features(z, u)
    - and fits W only

    Signature for predict:

    - `predict(x0: torch.Tensor, w: DynData, ts: Union[np.ndarray, torch.Tensor], **kwargs) -> Tuple[torch.Tensor, torch.Tensor]`
    - Usually comes from `dymad/models/prediction`

    For mathematical formulation, see `theory/architecture` in the documentation.

    The class can be used in two ways:

    - Through predefined models and :func:`~dymad.models.helpers.build_model` function.
      User defines :func:`~dymad.models.model_base.ComposedDynamics.build_core` class method,
      as needed by :func:`~dymad.models.helpers.build_model`.
    - By directly instantiating the class and hooking the functions and networks.  User needs to
      define all components manually with an initializer like:

      `def __init__(self, model_config: Dict, data_meta: Dict, dtype=None, device=None)`

    Args:
        encoder (Encoder): Encoder function
        dynamics (Tuple[Features, Composer]): Features function and composer function for dynamics
        decoder (Decoder): Decoder function
        predict (Tuple[Predictor, str]): Prediction function and input order
        model_config (Dict, optional): Model configuration dictionary
        dims (Dict, optional): Dimensions dictionary, usually generated from :func:`~dymad.models.helpers.get_dims`
    """
    GRAPH = None   # True for graph compatible models
    CONT  = None   # True for continuous-time models, otherwise discrete-time

    def __init__(
            self,
            encoder: Encoder | None = None,
            dynamics: Tuple[Features, Composer] | None = None,
            decoder: Decoder | None = None,
            predict: Tuple[Predictor, str] | None = None,
            model_config: Dict | None = None,
            dims: Dict | None = None):
        super().__init__()

        self._encoder = encoder    # Hooked encoder function
        self._decoder = decoder    # Hooked decoder function
        if dynamics is not None:   # Hooked feature and composer functions
            self.features, self.composer = dynamics
        if predict is not None:    # Hooked prediction function and input order
            self._predict, self.input_order = predict

        if dims is not None:
            self.n_total_state_features = dims['x']
            self.latent_dimension = dims['z']
            self.seq_len = dims['seq']
        else:
            self.n_total_state_features = -1
            self.latent_dimension = -1
            self.seq_len = -1

        # To be assigned
        self.encoder_net   = None  # Network to be used by self._encoder
        self.processor_net = None  # Network to be used inside self.dynamics
        self.decoder_net   = None  # Network to be used by self._decoder
        self._linear_eval  = None  # Functions for linear solver, to be hooked
        self._linear_features = None

    @classmethod
    def build_core(cls, model_config, dtype, device, ifgnn=False):
        """
        Typically used together with predefined models, and the unified build_model function.

        Should return:

        - dims: class-specific dimension dictionary
        - (enc_type, fzu_type, dec_type, prd_type): finalized type strings
        - processor_net: network for the dynamics processor
        - input_order: input order string for the predictor

        See :func:`~dymad.models.helpers.build_model` for details.
        """
        raise NotImplementedError("This is the base class.")

    def diagnostic_info(self) -> str:
        """
        Return diagnostic information about the model.

        Returns:
            str: String with model details
        """
        ind = "          "
        fin = lambda net: tw.indent(f"{net}", ind)
        return f"Model parameters: {sum(p.numel() for p in self.parameters())}\n" + \
               f"Encoder:  {self._encoder.__name__}\n{fin(self.encoder_net)}\n" + \
               f"Dynamics: {self.features.__name__}\n" + \
               f"{fin(self.processor_net)}\n" + \
               f"{ind}{self.composer.__name__}\n" + \
               f"Decoder:  {self._decoder.__name__}\n{fin(self.decoder_net)}\n" + \
               f"Prediction: {self._predict.__name__}\n" + \
               f"Continuous-time: {self.CONT}, Graph-compatible: {self.GRAPH}, " + \
               f"Sequence length: {self.seq_len}\n"

    def forward(self, t=None, x=None, u=None, p=None, ei=None, ew=None, ea=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the full model: encode, dynamics, decode.

        Unified across most of the models, but this can be overridden if needed.

        Note:
            The forward pass should not be used directly.
            This interface is provided for model inspection and analysis.
        """
        # ei, ew, ea are tuples of values and offsets, so that torchview can handle them
        # For DynData, we need to convert them back to nested tensors
        _x = x if ei is None else x.unsqueeze(0)   # Graph case always use batch size 1
        w = DynData(
            t=t, u=u, p=p,
            ei=torch.nested.nested_tensor_from_jagged(*ei) if ei is not None else None,
            ew=torch.nested.nested_tensor_from_jagged(*ew) if ew is not None else None,
            ea=torch.nested.nested_tensor_from_jagged(*ea) if ea is not None else None
            ).get_step(0).set_x(_x)
        z = self.encoder(w)
        z_dot = self.dynamics(z, w)
        x_hat = self.decoder(z, w)
        return z, z_dot, x_hat

    def encoder(self, w: DynData) -> torch.Tensor:
        """Encode the inputs into latent states."""
        return self._encoder(self.encoder_net, w)

    def dynamics(self, z: torch.Tensor, w: DynData) -> torch.Tensor:
        """
        Compute the dynamics output given latent states and inputs.

        Note this uses three components: features, processor, and composer.
        """
        return self.composer(self.processor_net, self.features(z, w), z, w)

    def decoder(self, z: torch.Tensor, w: DynData) -> torch.Tensor:
        """Decode the latent states into outputs."""
        return self._decoder(self.decoder_net, z, w)

    def linear_eval(self, w: DynData) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute linear evaluation, dz, and states, z, for the model.

        dz = Af(z)

        z is the encoded state, which will be used to compute the expected output.
        """
        return self._linear_eval(self, w)

    def linear_features(self, w: DynData) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute linear features, f, and outputs, dz, for the model.

        dz = Af(z)

        dz is the output of the dynamics, z_dot for cont-time, z_next for disc-time.
        """
        return self._linear_features(self, w)

    def set_linear_weights(self,
        W: torch.Tensor | None = None, b: torch.Tensor | None = None,
        U: torch.Tensor | None = None, V: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Set the weights of the linear dynamics module."""
        return self.processor_net.set_weights(W, b, U, V)

    def linear_solve(self, inp: torch.Tensor, out: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Solve for linear dynamics weights given input-output pairs.

        The solution depends on the specific model and processor used.
        """
        raise NotImplementedError("This is the base class.")

    def predict(self, x0: torch.Tensor, w: DynData, ts: Union[np.ndarray, torch.Tensor],
                method= 'dopri5', **kwargs) -> torch.Tensor:
        """
        Predict trajectory using specified method.
        
        This function essentially determines whether the model is continuous-time or discrete-time.
        """
        return self._predict(self, x0, ts, w, method=method, order=self.input_order, **kwargs)
