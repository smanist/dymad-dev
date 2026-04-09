from __future__ import annotations

import textwrap as tw
from collections.abc import Callable
from typing import Any, Protocol, cast

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


class Predictor(Protocol):
    def __call__(
        self,
        model: nn.Module,
        x0: torch.Tensor,
        ts: np.ndarray | torch.Tensor,
        ws: Any = None,
        /,
        **kwargs: Any,
    ) -> torch.Tensor: ...


class SupportsSetWeights(Protocol):
    def set_weights(
        self,
        W: torch.Tensor | None = None,
        b: torch.Tensor | None = None,
        U: torch.Tensor | None = None,
        V: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]: ...


LinearHelper = Callable[
    ["ComposedDynamics", ComponentInputPayload], tuple[torch.Tensor, torch.Tensor]
]


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
        self.features: Features | None = None
        self.composer: Composer | None = None
        self.input_order: str | None = None
        self._predict: Predictor | None = None
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
        self.encoder_net: nn.Module | None = None  # Network to be used by self._encoder
        self.processor_net: nn.Module | None = None  # Network to be used inside self.dynamics
        self.decoder_net: nn.Module | None = None  # Network to be used by self._decoder
        self._linear_eval: LinearHelper | None = None  # Functions for linear solver, to be hooked
        self._linear_features: LinearHelper | None = None
        self.dtype: torch.dtype | None = None
        self.device: torch.device | str | None = None

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

        encoder = self._require_encoder()
        decoder = self._require_decoder()
        features = self._require_features()
        composer = self._require_composer()
        predictor = self._require_predictor()

        def fin(net: nn.Module) -> str:
            return tw.indent(f"{net}", ind)

        return (
            f"Model parameters: {sum(p.numel() for p in self.parameters())}\n"
            + f"Encoder:  {self._callable_name(encoder)}\n{fin(self._require_encoder_net())}\n"
            + f"Dynamics: {self._callable_name(features)}\n"
            + f"{fin(self._require_processor_net())}\n"
            + f"{ind}{self._callable_name(composer)}\n"
            + f"Decoder:  {self._callable_name(decoder)}\n{fin(self._require_decoder_net())}\n"
            + f"Prediction: {self._callable_name(predictor)}\n"
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
        z = self.encoder(w)
        z_dot = self.dynamics(z, w)
        x_hat = self.decoder(z, w)
        return z, z_dot, x_hat

    def encoder(self, w: ComponentInputPayload) -> torch.Tensor:
        """Encode the inputs into latent states."""
        return self._require_encoder()(self._require_encoder_net(), w)

    def dynamics(self, z: torch.Tensor, w: ComponentInputPayload) -> torch.Tensor:
        """
        Compute the dynamics output given latent states and inputs.

        Note this uses three components: features, processor, and composer.
        """
        processor = self._require_processor_net()
        features = self._require_features()
        composer = self._require_composer()
        return composer(processor, features(z, w), z, w)

    def decoder(self, z: torch.Tensor, w: ComponentInputPayload) -> torch.Tensor:
        """Decode the latent states into outputs."""
        return self._require_decoder()(self._require_decoder_net(), z, w)

    def linear_eval(self, w: ComponentInputPayload) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute linear evaluation, dz, and states, z, for the model.

        dz = Af(z)

        z is the encoded state, which will be used to compute the expected output.
        """
        return self._require_linear_eval()(self, w)

    def linear_features(self, w: ComponentInputPayload) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute linear features, f, and outputs, dz, for the model.

        dz = Af(z)

        dz is the output of the dynamics, z_dot for cont-time, z_next for disc-time.
        """
        return self._require_linear_features()(self, w)

    def set_linear_weights(
        self,
        W: torch.Tensor | None = None,
        b: torch.Tensor | None = None,
        U: torch.Tensor | None = None,
        V: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Set the weights of the linear dynamics module."""
        processor = self._require_processor_with_set_weights()
        return cast(tuple[torch.Tensor, torch.Tensor], processor.set_weights(W=W, b=b, U=U, V=V))

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
        predictor = self._require_predictor()
        return predictor(self, x0, ts, w, method=method, order=self.input_order, **kwargs)

    @staticmethod
    def _callable_name(fn: Callable[..., Any]) -> str:
        return getattr(fn, "__name__", type(fn).__name__)

    def _require_encoder(self) -> Encoder:
        if self._encoder is None:
            raise RuntimeError("Encoder hook is not initialized.")
        return self._encoder

    def _require_decoder(self) -> Decoder:
        if self._decoder is None:
            raise RuntimeError("Decoder hook is not initialized.")
        return self._decoder

    def _require_predictor(self) -> Predictor:
        if self._predict is None:
            raise RuntimeError("Predictor hook is not initialized.")
        return self._predict

    def _require_features(self) -> Features:
        if self.features is None:
            raise RuntimeError("Features hook is not initialized.")
        return self.features

    def _require_composer(self) -> Composer:
        if self.composer is None:
            raise RuntimeError("Composer hook is not initialized.")
        return self.composer

    def _require_encoder_net(self) -> nn.Module:
        if self.encoder_net is None:
            return cast(nn.Module, self)
        return self.encoder_net

    def _require_processor_net(self) -> nn.Module:
        if self.processor_net is None:
            return cast(nn.Module, self)
        return self.processor_net

    def _require_processor_with_set_weights(self) -> SupportsSetWeights:
        processor = self._require_processor_net()
        if not hasattr(processor, "set_weights"):
            raise RuntimeError("Processor does not support linear weight assignment.")
        return cast(SupportsSetWeights, processor)

    def _require_decoder_net(self) -> nn.Module:
        if self.decoder_net is None:
            return cast(nn.Module, self)
        return self.decoder_net

    def _require_linear_eval(self) -> LinearHelper:
        if self._linear_eval is None:
            raise RuntimeError("Linear evaluation hook is not initialized.")
        return self._linear_eval

    def _require_linear_features(self) -> LinearHelper:
        if self._linear_features is None:
            raise RuntimeError("Linear feature hook is not initialized.")
        return self._linear_features
