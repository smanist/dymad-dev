"""Narrow runtime-view adapter for model helper/component functions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import torch

from dymad.core.model_context import GraphModelContext, LegacyRuntimeCollection, RegularModelContext
from dymad.core.runtime import GraphRuntimeStep, RegularRuntimeStep, TypedRuntime, TypedRuntimeStep
from dymad.io.legacy_runtime import LegacyRuntimeBatch


@dataclass(frozen=True)
class ComponentInputView:
    """Expose runtime fields needed by model helpers without indexing LegacyRuntimeBatch directly."""

    runtime: LegacyRuntimeBatch | TypedRuntime | TypedRuntimeStep

    @classmethod
    def build(cls, payload: ComponentInputPayload) -> "ComponentInputView":
        if isinstance(payload, ComponentInputView):
            return payload
        if isinstance(payload, RegularModelContext):
            return cls(payload.to_runtime().get_step(0))
        if isinstance(payload, GraphModelContext):
            return cls(payload.to_runtime().get_step(0))
        if isinstance(payload, (LegacyRuntimeBatch, RegularRuntimeStep, GraphRuntimeStep)):
            return cls(payload)
        if hasattr(payload, "get_step") or hasattr(payload, "x"):
            return cls(payload)
        raise TypeError(f"Unsupported component payload type: {type(payload)!r}")

    @property
    def state(self) -> torch.Tensor:
        return self.runtime.x

    @property
    def control(self) -> torch.Tensor | None:
        return self.runtime.u

    @property
    def graph_state(self) -> torch.Tensor:
        return self.runtime.xg

    @property
    def graph_control(self) -> torch.Tensor | None:
        return self.runtime.ug

    @property
    def edge_index(self) -> torch.Tensor:
        return self.runtime.ei

    @property
    def edge_weight(self) -> torch.Tensor | None:
        return self.runtime.ew

    @property
    def edge_attr(self) -> torch.Tensor | None:
        return self.runtime.ea

    def unflatten_nodes(self, value: torch.Tensor) -> torch.Tensor:
        return self.runtime.g(value)

    def flatten_nodes(self, value: torch.Tensor) -> torch.Tensor:
        return self.runtime.G(value)


def build_component_input_view(payload: ComponentInputPayload) -> ComponentInputView:
    """Build the narrow runtime-view adapter for model helpers."""

    return ComponentInputView.build(payload)


ComponentInputPayload: TypeAlias = LegacyRuntimeBatch | TypedRuntime | TypedRuntimeStep | RegularModelContext | GraphModelContext | ComponentInputView
