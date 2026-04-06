import textwrap as tw
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from dymad.core.model_context import materialize_model_base_forward_payload
from dymad.models.runtime_view import ComponentInputPayload

Encoder = Callable[[nn.Module, ComponentInputPayload], torch.Tensor]
"""encoder(net, w) -> x"""

Features = Callable[[torch.Tensor, ComponentInputPayload], torch.Tensor]
"""features(z, w) -> s"""

Composer = Callable[[nn.Module, torch.Tensor, torch.Tensor, ComponentInputPayload], torch.Tensor]
"""composer(net, s, z, w) -> r"""

Decoder = Callable[[nn.Module, torch.Tensor, ComponentInputPayload], torch.Tensor]
"""decoder(net, z, w) -> x"""

Predictor = Callable[
    [torch.Tensor, ComponentInputPayload, np.ndarray | torch.Tensor, Any],
    tuple[torch.Tensor, torch.Tensor],
]
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

    - `predict(x0: torch.Tensor, w: ComponentInputPayload, ts: Union[np.ndarray, torch.Tensor], **kwargs) -> Tuple[torch.Tensor, torch.Tensor]`
    - Usually comes from `dymad/models/prediction`

    For mathematical formulation, see `theory/architecture` in the documentation.

    The class can be used in two ways:

    - Through predefined models and :func:`~dymad.models.helpers.build_model` function.
      User defines :func:`~dymad.models.model_base.ComposedDynamics.resolve_spec` class method,
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

    GRAPH = None  # True for graph compatible models
    CONT = None  # True for continuous-time models, otherwise discrete-time

    def __init__(
        self,
        encoder: Encoder | None = None,
        dynamics: tuple[Features, Composer] | None = None,
        decoder: Decoder | None = None,
        predict: tuple[Predictor, str] | None = None,
        model_config: dict | None = None,
        dims: dict | None = None,
    ):
        super().__init__()

        self._encoder = encoder  # Hooked encoder function
        self._decoder = decoder  # Hooked decoder function
        if dynamics is not None:  # Hooked feature and composer functions
            self.features, self.composer = dynamics
        if predict is not None:  # Hooked prediction function and input order
            self._predict, self.input_order = predict

        if dims is not None:
            self.n_total_state_features = dims["x"]
            self.latent_dimension = dims["z"]
            self.seq_len = dims["seq"]
        else:
            self.n_total_state_features = -1
            self.latent_dimension = -1
            self.seq_len = -1

        # To be assigned
        self.encoder_net = None  # Network to be used by self._encoder
        self.processor_net = None  # Network to be used inside self.dynamics
        self.decoder_net = None  # Network to be used by self._decoder
        self._linear_eval = None  # Functions for linear solver, to be hooked
        self._linear_features = None

    @classmethod
    def resolve_spec(cls, model_spec, model_config, data_meta, dtype, device):
        """
        Resolve a typed model spec into concrete build-time components.
        """
        raise NotImplementedError("This is the base class.")

    def diagnostic_info(self) -> str:
        """
        Return diagnostic information about the model.

        Returns:
            str: String with model details
        """
        ind = "          "

        def fin(net):
            return tw.indent(f"{net}", ind)

        return (
            f"Model parameters: {sum(p.numel() for p in self.parameters())}\n"
            + f"Encoder:  {self._encoder.__name__}\n{fin(self.encoder_net)}\n"
            + f"Dynamics: {self.features.__name__}\n"
            + f"{fin(self.processor_net)}\n"
            + f"{ind}{self.composer.__name__}\n"
            + f"Decoder:  {self._decoder.__name__}\n{fin(self.decoder_net)}\n"
            + f"Prediction: {self._predict.__name__}\n"
            + f"Continuous-time: {self.CONT}, Graph-compatible: {self.GRAPH}, "
            + f"Sequence length: {self.seq_len}\n"
        )

    def forward(
        self, t=None, x=None, u=None, p=None, ei=None, ew=None, ea=None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the full model: encode, dynamics, decode.

        Unified across most of the models, but this can be overridden if needed.

        Note:
            The forward pass should not be used directly.
            This interface is provided for model inspection and analysis.
        """
        w = materialize_model_base_forward_payload(t=t, x=x, u=u, p=p, ei=ei, ew=ew, ea=ea)
        if hasattr(w, "to_runtime"):
            w = w.to_runtime().get_step(0)
        z = self.encoder(w)
        z_dot = self.dynamics(z, w)
        x_hat = self.decoder(z, w)
        return z, z_dot, x_hat

    def encoder(self, w: ComponentInputPayload) -> torch.Tensor:
        """Encode the inputs into latent states."""
        return self._encoder(self.encoder_net, w)

    def dynamics(self, z: torch.Tensor, w: ComponentInputPayload) -> torch.Tensor:
        """
        Compute the dynamics output given latent states and inputs.

        Note this uses three components: features, processor, and composer.
        """
        return self.composer(self.processor_net, self.features(z, w), z, w)

    def decoder(self, z: torch.Tensor, w: ComponentInputPayload) -> torch.Tensor:
        """Decode the latent states into outputs."""
        return self._decoder(self.decoder_net, z, w)

    def linear_eval(self, w: ComponentInputPayload) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute linear evaluation, dz, and states, z, for the model.

        dz = Af(z)

        z is the encoded state, which will be used to compute the expected output.
        """
        return self._linear_eval(self, w)

    def linear_features(self, w: ComponentInputPayload) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute linear features, f, and outputs, dz, for the model.

        dz = Af(z)

        dz is the output of the dynamics, z_dot for cont-time, z_next for disc-time.
        """
        return self._linear_features(self, w)

    def set_linear_weights(
        self,
        W: torch.Tensor | None = None,
        b: torch.Tensor | None = None,
        U: torch.Tensor | None = None,
        V: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Set the weights of the linear dynamics module."""
        return self.processor_net.set_weights(W, b, U, V)

    def linear_solve(
        self, inp: torch.Tensor, out: torch.Tensor, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Solve for linear dynamics weights given input-output pairs.

        The solution depends on the specific model and processor used.
        """
        raise NotImplementedError("This is the base class.")

    def predict(
        self,
        x0: torch.Tensor,
        w: ComponentInputPayload,
        ts: np.ndarray | torch.Tensor,
        method="dopri5",
        **kwargs,
    ) -> torch.Tensor:
        """
        Predict trajectory using specified method.

        This function essentially determines whether the model is continuous-time or discrete-time.
        """
        return self._predict(self, x0, ts, w, method=method, order=self.input_order, **kwargs)
