"""Narrow runtime-view adapter for model helper/component functions."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from dymad.core.model_context import GraphModelContext, RegularModelContext
from dymad.core.runtime import (
    GraphRuntimeStep,
    RaggedGraphRuntime,
    RegularRuntimeStep,
    TypedRuntime,
    TypedRuntimeStep,
    UniformGraphRuntime,
)


@dataclass(frozen=True)
class ComponentInputView:
    """Expose runtime fields needed by model helpers through typed runtime views."""

    runtime: TypedRuntime | TypedRuntimeStep

    @classmethod
    def build(cls, payload: ComponentInputPayload) -> ComponentInputView:
        if isinstance(payload, ComponentInputView):
            return payload
        if isinstance(payload, RegularModelContext):
            return cls(payload.to_runtime().get_step(0))
        if isinstance(payload, GraphModelContext):
            return cls(payload.to_runtime().get_step(0))
        if isinstance(payload, (RegularRuntimeStep, GraphRuntimeStep)):
            return cls(payload)
        if hasattr(payload, "get_step") and hasattr(payload, "is_graph"):
            return cls(payload)
        raise TypeError(f"Unsupported component payload type: {type(payload)!r}")

    @property
    def state(self) -> torch.Tensor:
        state = self.runtime.x
        if state is None:
            raise ValueError("Runtime payload does not contain state.")
        return state

    @property
    def control(self) -> torch.Tensor | None:
        return self.runtime.u

    @property
    def graph_state(self) -> torch.Tensor:
        if not isinstance(
            self.runtime, (GraphRuntimeStep, UniformGraphRuntime, RaggedGraphRuntime)
        ):
            raise TypeError("Graph state requested from a regular runtime payload.")
        state = self.runtime.xg
        if state is None:
            raise ValueError("Graph runtime payload does not contain node state.")
        return state

    @property
    def graph_control(self) -> torch.Tensor | None:
        if not isinstance(
            self.runtime, (GraphRuntimeStep, UniformGraphRuntime, RaggedGraphRuntime)
        ):
            raise TypeError("Graph control requested from a regular runtime payload.")
        return self.runtime.ug

    @property
    def edge_index(self) -> torch.Tensor:
        if not isinstance(
            self.runtime, (GraphRuntimeStep, UniformGraphRuntime, RaggedGraphRuntime)
        ):
            raise TypeError("Edge index requested from a regular runtime payload.")
        edge_index = self.runtime.ei
        if edge_index is None:
            raise ValueError("Graph runtime payload does not contain edge indices.")
        return edge_index

    @property
    def edge_weight(self) -> torch.Tensor | None:
        if not isinstance(
            self.runtime, (GraphRuntimeStep, UniformGraphRuntime, RaggedGraphRuntime)
        ):
            raise TypeError("Edge weight requested from a regular runtime payload.")
        return self.runtime.ew

    @property
    def edge_attr(self) -> torch.Tensor | None:
        if not isinstance(
            self.runtime, (GraphRuntimeStep, UniformGraphRuntime, RaggedGraphRuntime)
        ):
            raise TypeError("Edge attributes requested from a regular runtime payload.")
        return self.runtime.ea

    def unflatten_nodes(self, value: torch.Tensor) -> torch.Tensor:
        if not isinstance(
            self.runtime, (GraphRuntimeStep, UniformGraphRuntime, RaggedGraphRuntime)
        ):
            raise TypeError("Node unflatten requested from a regular runtime payload.")
        return self.runtime.g(value)

    def flatten_nodes(self, value: torch.Tensor) -> torch.Tensor:
        if not isinstance(
            self.runtime, (GraphRuntimeStep, UniformGraphRuntime, RaggedGraphRuntime)
        ):
            raise TypeError("Node flatten requested from a regular runtime payload.")
        return self.runtime.G(value)


def build_component_input_view(payload: ComponentInputPayload) -> ComponentInputView:
    """Build the narrow runtime-view adapter for model helpers."""

    return ComponentInputView.build(payload)


type ComponentInputPayload = (
    TypedRuntime | TypedRuntimeStep | RegularModelContext | GraphModelContext | ComponentInputView
)
