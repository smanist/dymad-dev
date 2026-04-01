"""Narrow runtime-view adapter for model helper/component functions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import torch

from dymad.core.model_context import GraphModelContext, LegacyRuntimeCollection, RegularModelContext
from dymad.io.legacy_runtime import LegacyRuntimeBatch


@dataclass(frozen=True)
class ComponentInputView:
    """Expose runtime fields needed by model helpers without indexing LegacyRuntimeBatch directly."""

    runtime: LegacyRuntimeBatch

    @classmethod
    def build(cls, payload: ComponentInputPayload) -> "ComponentInputView":
        if isinstance(payload, ComponentInputView):
            return payload
        if isinstance(payload, RegularModelContext):
            runtime = payload.to_legacy_runtime()
            if isinstance(runtime, LegacyRuntimeCollection):
                if len(runtime) != 1:
                    raise ValueError("ComponentInputView requires a single runtime payload, not a ragged batch.")
                runtime = runtime.items[0]
            return cls(runtime.get_step(0))
        if isinstance(payload, GraphModelContext):
            runtime = payload.to_legacy_runtime()
            if isinstance(runtime, LegacyRuntimeCollection):
                if len(runtime) != 1:
                    raise ValueError("ComponentInputView requires a single runtime payload, not a ragged batch.")
                runtime = runtime.items[0]
            return cls(runtime.get_step(0))
        if isinstance(payload, LegacyRuntimeBatch):
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


ComponentInputPayload: TypeAlias = LegacyRuntimeBatch | RegularModelContext | GraphModelContext | ComponentInputView
