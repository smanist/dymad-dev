"""Typed graph-series primitives for the module-first data migration."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterable

import torch


def _move_tensor(
    tensor: torch.Tensor | tuple[torch.Tensor, ...] | None,
    *,
    device: torch.device | str | None,
    dtype: torch.dtype | None,
    index_dtype: torch.dtype = torch.long,
):
    if tensor is None:
        return None
    if isinstance(tensor, tuple):
        return tuple(_move_tensor(item, device=device, dtype=dtype, index_dtype=index_dtype) for item in tensor)
    if tensor.dtype in (torch.int32, torch.int64):
        return tensor.to(device=device, dtype=index_dtype)
    return tensor.to(device=device, dtype=dtype)


def _slice_edge_sequence(
    tensor: torch.Tensor | tuple[torch.Tensor, ...] | None,
    start: int,
    end: int,
):
    if tensor is None:
        return None
    if isinstance(tensor, tuple):
        return tensor[start:end]
    return tensor


@dataclass(frozen=True)
class GraphSeries:
    """One graph trajectory with explicit node and edge payload semantics."""

    time: torch.Tensor
    node_state: torch.Tensor
    edge_index: torch.Tensor | tuple[torch.Tensor, ...]
    control: torch.Tensor | None = None
    target: torch.Tensor | None = None
    params: torch.Tensor | None = None
    edge_weight: torch.Tensor | tuple[torch.Tensor, ...] | None = None
    edge_attr: torch.Tensor | tuple[torch.Tensor, ...] | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def fixed_topology(self) -> bool:
        return isinstance(self.edge_index, torch.Tensor)

    def slice_steps(self, start: int, end: int) -> "GraphSeries":
        return replace(
            self,
            time=self.time[start:end],
            node_state=self.node_state[start:end],
            control=self.control[start:end] if self.control is not None else None,
            target=self.target[start:end] if self.target is not None else None,
            edge_index=_slice_edge_sequence(self.edge_index, start, end),
            edge_weight=_slice_edge_sequence(self.edge_weight, start, end),
            edge_attr=_slice_edge_sequence(self.edge_attr, start, end),
            meta=dict(self.meta),
        )

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "GraphSeries":
        return replace(
            self,
            time=self.time.to(device=device, dtype=dtype),
            node_state=self.node_state.to(device=device, dtype=dtype),
            control=_move_tensor(self.control, device=device, dtype=dtype),
            target=_move_tensor(self.target, device=device, dtype=dtype),
            params=_move_tensor(self.params, device=device, dtype=dtype),
            edge_index=_move_tensor(self.edge_index, device=device, dtype=dtype),
            edge_weight=_move_tensor(self.edge_weight, device=device, dtype=dtype),
            edge_attr=_move_tensor(self.edge_attr, device=device, dtype=dtype),
            meta=dict(self.meta),
        )

    def to_flat_node_features(self) -> torch.Tensor:
        return self.node_state.reshape(self.node_state.shape[0], -1)


@dataclass(frozen=True)
class FixedGraphSeries(GraphSeries):
    """Graph series with one topology reused across all time steps."""

    edge_index: torch.Tensor

    def __post_init__(self) -> None:
        if self.edge_index.ndim != 2:
            raise ValueError("FixedGraphSeries.edge_index must have shape [2, n_edges]")


@dataclass(frozen=True)
class VariableEdgeGraphSeries(GraphSeries):
    """Graph series with per-step edge topology."""

    edge_index: tuple[torch.Tensor, ...]

    def __post_init__(self) -> None:
        if len(self.edge_index) != self.time.shape[0]:
            raise ValueError("VariableEdgeGraphSeries.edge_index must align with time steps")


@dataclass(frozen=True)
class GraphSeriesBatch:
    """Minimal explicit batch wrapper for graph trajectories."""

    items: tuple[GraphSeries, ...]

    @classmethod
    def collate(cls, items: Iterable[GraphSeries]) -> "GraphSeriesBatch":
        return cls(tuple(items))

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def __getitem__(self, index: int) -> GraphSeries:
        return self.items[index]

    def slice_batch(self, indices: Iterable[int]) -> "GraphSeriesBatch":
        return GraphSeriesBatch.collate(self.items[index] for index in indices)

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "GraphSeriesBatch":
        return GraphSeriesBatch.collate(item.to(device=device, dtype=dtype) for item in self.items)
