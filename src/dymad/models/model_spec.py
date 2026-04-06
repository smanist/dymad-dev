"""Typed model-spec contracts for predefined-model construction."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

GraphMode = Literal["none", "graph", "node"]
TimeDomain = Literal["continuous", "discrete"]
RecipeKind = Literal["ldm", "sdm", "lfm", "km", "kmsk", "kmm"]
EncoderKind = Literal["smpl", "raw", "graph", "node", "node_raw", "smpl_auto", "graph_auto"]
FeatureKind = Literal["none", "cat", "blin", "graph_cat", "graph_blin"]
DynamicsKind = Literal["direct", "skip", "graph_direct", "graph_skip"]
DecoderKind = Literal["auto", "graph", "node"]
PredictorKey = Literal[
    "continuous",
    "continuous_np",
    "continuous_exp",
    "continuous_fenc",
    "discrete",
    "discrete_exp",
]


class ModelSpecValidationError(ValueError):
    """Raised when a typed model spec cannot be resolved safely."""


@dataclass(frozen=True)
class RecipeSpec:
    kind: RecipeKind
    model_cls: object


@dataclass(frozen=True)
class EncoderSpec:
    kind: EncoderKind

    @property
    def family(self) -> EncoderKind:
        return self.kind


@dataclass(frozen=True)
class FeatureSpec:
    kind: FeatureKind

    @property
    def family(self) -> FeatureKind:
        return self.kind


@dataclass(frozen=True)
class DynamicsSpec:
    kind: DynamicsKind

    @property
    def family(self) -> DynamicsKind:
        return self.kind


@dataclass(frozen=True)
class DecoderSpec:
    kind: DecoderKind

    @property
    def family(self) -> DecoderKind:
        return self.kind


@dataclass(frozen=True)
class RolloutSpec:
    family: str
    default_predictor: PredictorKey
    allowed_predictors: tuple[PredictorKey, ...]
    supports_control_inputs: bool = True


@dataclass(frozen=True)
class MemorySpec:
    family: str
    latent_state: str
    requires_delay_window: bool


@dataclass(frozen=True)
class ModelSpec:
    """Authoritative typed model specification for model construction."""

    recipe: RecipeSpec
    time_domain: TimeDomain
    graph_mode: GraphMode
    encoder: EncoderSpec
    feature: FeatureSpec
    dynamics: DynamicsSpec
    decoder: DecoderSpec
    rollout: RolloutSpec
    memory: MemorySpec | None = None
    name: str | None = None

    @property
    def continuous_time(self) -> bool:
        return self.time_domain == "continuous"

    @property
    def model_cls(self) -> object:
        return self.recipe.model_cls


@dataclass(frozen=True)
class ResolvedModelSpec:
    """Normalized construction plan produced from a :class:`ModelSpec`."""

    model_spec: ModelSpec
    dims: dict[str, Any]
    encoder_key: str
    feature_key: str
    dynamics_key: str
    decoder_key: str
    predictor_key: PredictorKey
    predictor: Callable[..., Any]
    input_order: str | None
    processor_net: object
    graph_mode: GraphMode
    linear_mode: Literal["smpl", "graph"]
    continuous_time: bool
