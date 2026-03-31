"""Typed model-runtime context adapters built from typed series objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from dymad.core.graph_series import GraphSeries, GraphSeriesBatch
from dymad.core.series import RegularSeries, RegularSeriesBatch

if TYPE_CHECKING:
    from dymad.io.data import DynData


@dataclass(frozen=True)
class RegularModelContext:
    """Typed regular runtime context for prediction/model helper entrypoints."""

    batch: RegularSeriesBatch

    @classmethod
    def from_series(cls, series: RegularSeries) -> "RegularModelContext":
        return cls(RegularSeriesBatch.collate([series]))

    @classmethod
    def from_batch(cls, batch: RegularSeriesBatch) -> "RegularModelContext":
        return cls(batch)

    @property
    def batch_size(self) -> int:
        return len(self.batch)

    @property
    def n_steps(self) -> tuple[int, ...]:
        return tuple(int(item.time.shape[0]) for item in self.batch)

    def initial_state_tensor(self, *, squeeze_single: bool = False) -> torch.Tensor:
        state = torch.stack([item.state[0] for item in self.batch], dim=0)
        if squeeze_single and state.shape[0] == 1:
            return state[0]
        return state

    def to_legacy_runtime(self) -> "DynData":
        from dymad.io.data import DynData
        from dymad.io.series_adapter import DynDataAdapter

        payloads = [DynDataAdapter.from_regular_series(item) for item in self.batch]
        return DynData.collate(payloads)


@dataclass(frozen=True)
class GraphModelContext:
    """Typed graph runtime context for prediction/model helper entrypoints."""

    batch: GraphSeriesBatch

    @classmethod
    def from_series(cls, series: GraphSeries) -> "GraphModelContext":
        return cls(GraphSeriesBatch.collate([series]))

    @classmethod
    def from_batch(cls, batch: GraphSeriesBatch) -> "GraphModelContext":
        return cls(batch)

    @property
    def batch_size(self) -> int:
        return len(self.batch)

    @property
    def n_steps(self) -> tuple[int, ...]:
        return tuple(int(item.time.shape[0]) for item in self.batch)

    @property
    def n_nodes(self) -> tuple[int, ...]:
        return tuple(int(item.node_state.shape[1]) for item in self.batch)

    def initial_state_tensor(self, *, squeeze_single: bool = False) -> torch.Tensor:
        state = torch.stack([item.node_state[0].reshape(-1) for item in self.batch], dim=0)
        if squeeze_single and state.shape[0] == 1:
            return state[0]
        return state

    def to_legacy_runtime(self) -> "DynData":
        from dymad.io.data import DynData
        from dymad.io.series_adapter import DynDataAdapter

        payloads = [DynDataAdapter.from_graph_series(item) for item in self.batch]
        return DynData.collate(payloads)


def build_model_context(
    batch: RegularSeries | RegularSeriesBatch | GraphSeries | GraphSeriesBatch,
) -> RegularModelContext | GraphModelContext:
    """Build a typed model-runtime context from typed regular or graph series payloads."""

    if isinstance(batch, RegularSeries):
        return RegularModelContext.from_series(batch)
    if isinstance(batch, RegularSeriesBatch):
        return RegularModelContext.from_batch(batch)
    if isinstance(batch, GraphSeries):
        return GraphModelContext.from_series(batch)
    if isinstance(batch, GraphSeriesBatch):
        return GraphModelContext.from_batch(batch)
    raise TypeError(f"Unsupported model-context payload type: {type(batch)!r}")
