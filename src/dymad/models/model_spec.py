"""Typed model-spec compatibility objects for predefined model entrypoints."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class EncoderSpec:
    family: str


@dataclass(frozen=True)
class FeatureSpec:
    family: str


@dataclass(frozen=True)
class DynamicsSpec:
    family: str


@dataclass(frozen=True)
class DecoderSpec:
    family: str


@dataclass(frozen=True)
class RolloutSpec:
    family: str
    predictor: str
    supports_control_inputs: bool


@dataclass(frozen=True)
class MemorySpec:
    family: str
    latent_state: str
    requires_delay_window: bool


@dataclass(frozen=True)
class ModelSpec:
    """Typed model specification used by predefined-model compatibility adapters."""

    continuous_time: bool
    encoder: EncoderSpec
    feature: FeatureSpec
    dynamics: DynamicsSpec
    decoder: DecoderSpec
    model_cls: object
    rollout: Optional[RolloutSpec] = None
    memory: Optional[MemorySpec] = None
    name: Optional[str] = None

    @property
    def graph_mode(self) -> str:
        """Return whether the spec represents a graph-aware model."""
        fields = (
            self.encoder.family,
            self.decoder.family,
            self.dynamics.family,
        )
        if any("graph" in field or "node" in field for field in fields):
            return "graph"
        return "none"

    def to_legacy_tuple(self) -> tuple[bool, str, str, str, str, object]:
        """Expose the temporary legacy helper contract."""
        return (
            self.continuous_time,
            self.encoder.family,
            self.feature.family,
            self.dynamics.family,
            self.decoder.family,
            self.model_cls,
        )


class LegacyPredefinedModelAdapter:
    """Compatibility adapter from legacy predefined names to typed ModelSpec."""

    @staticmethod
    def from_legacy_parts(
        *,
        continuous_time: bool,
        encoder: str,
        feature: str,
        dynamics: str,
        decoder: str,
        model_cls: object,
        rollout: Optional[RolloutSpec] = None,
        memory: Optional[MemorySpec] = None,
        name: Optional[str] = None,
    ) -> ModelSpec:
        return ModelSpec(
            continuous_time=continuous_time,
            encoder=EncoderSpec(family=encoder),
            feature=FeatureSpec(family=feature),
            dynamics=DynamicsSpec(family=dynamics),
            decoder=DecoderSpec(family=decoder),
            model_cls=model_cls,
            rollout=rollout,
            memory=memory,
            name=name,
        )
