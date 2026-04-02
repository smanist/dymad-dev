"""Trainer-facing typed batch wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from dymad.core.graph_series import GraphSeries, GraphSeriesBatch
from dymad.core.runtime import (
    GraphRuntime,
    RegularRuntime,
    to_padded_graph_runtime,
    to_padded_regular_runtime,
)
from dymad.core.series import RegularSeries, RegularSeriesBatch


def _stack_optional(items: Iterable[torch.Tensor | None]) -> torch.Tensor | None:
    values = tuple(items)
    present = tuple(value for value in values if value is not None)
    if not present:
        return None
    ref = present[0]
    filled = tuple(
        value if value is not None else torch.zeros_like(ref)
        for value in values
    )
    return torch.stack(filled)


@dataclass(frozen=True)
class RegularTrainerBatch:
    """Trainer-facing wrapper for a batch of regular typed runtimes."""

    runtime: RegularRuntime
    series: RegularSeriesBatch | None = None

    def __post_init__(self) -> None:
        if isinstance(self.runtime, RegularSeries):
            series = RegularSeriesBatch.collate([self.runtime])
            object.__setattr__(self, "runtime", to_padded_regular_runtime(series))
            object.__setattr__(self, "series", series)
        elif isinstance(self.runtime, RegularSeriesBatch):
            object.__setattr__(self, "series", self.runtime)
            object.__setattr__(self, "runtime", to_padded_regular_runtime(self.runtime))

    @classmethod
    def collate_series(cls, items: Iterable[RegularSeries]) -> "RegularTrainerBatch":
        series = RegularSeriesBatch.collate(items)
        return cls(runtime=to_padded_regular_runtime(series), series=series)

    @classmethod
    def from_runtime(
        cls,
        runtime: RegularRuntime,
        *,
        series: RegularSeriesBatch | None = None,
    ) -> "RegularTrainerBatch":
        return cls(runtime=runtime, series=series)

    def __len__(self) -> int:
        return self.runtime.batch_size

    @property
    def is_ragged(self) -> bool:
        return not self.runtime.is_uniform_length

    def iter_single_batches(self):
        if self.series is not None:
            for item in self.series:
                yield RegularTrainerBatch.collate_series([item])
            return
        for item in self.runtime.iter_series():
            yield RegularTrainerBatch.from_runtime(item)

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "RegularTrainerBatch":
        runtime = self.runtime.to(device=device, dtype=dtype)
        series = self.series.to(device=device, dtype=dtype) if self.series is not None else None
        return RegularTrainerBatch(runtime=runtime, series=series)

    def truncate(self, num_steps: int) -> "RegularTrainerBatch":
        runtime = self.runtime.truncate(num_steps)
        series = None
        if self.series is not None:
            series = RegularSeriesBatch.collate(item.slice_steps(0, num_steps) for item in self.series)
        return RegularTrainerBatch(runtime=runtime, series=series)

    def window(self, window: int, stride: int) -> "RegularTrainerBatch":
        runtime = self.runtime.window(window, stride)
        return RegularTrainerBatch.from_runtime(runtime)

    def initial_state(self) -> torch.Tensor:
        return self.runtime.initial_state()

    def time_tensor(self) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if self.runtime.is_uniform_length:
            return self.runtime.t
        return tuple(item.t[0] for item in self.runtime.iter_series())

    def state_tensor(self) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if self.runtime.is_uniform_length:
            return self.runtime.x
        return tuple(item.x[0] for item in self.runtime.iter_series())

    def control_tensor(self) -> torch.Tensor | None:
        if self.runtime.is_uniform_length:
            return self.runtime.u
        return _stack_optional(item.u[0] if item.u is not None else None for item in self.runtime.iter_series())

    def target_tensor(self) -> torch.Tensor | None:
        if self.runtime.is_uniform_length:
            return self.runtime.y
        return _stack_optional(item.y[0] if item.y is not None else None for item in self.runtime.iter_series())

    def params_tensor(self) -> torch.Tensor | None:
        return self.runtime.p


@dataclass(frozen=True)
class GraphTrainerBatch:
    """Trainer-facing wrapper for a batch of graph typed runtimes."""

    runtime: GraphRuntime
    series: GraphSeriesBatch | None = None

    def __post_init__(self) -> None:
        if isinstance(self.runtime, GraphSeries):
            series = GraphSeriesBatch.collate([self.runtime])
            object.__setattr__(self, "runtime", to_padded_graph_runtime(series))
            object.__setattr__(self, "series", series)
        elif isinstance(self.runtime, GraphSeriesBatch):
            object.__setattr__(self, "series", self.runtime)
            object.__setattr__(self, "runtime", to_padded_graph_runtime(self.runtime))

    @classmethod
    def collate_series(cls, items: Iterable[GraphSeries]) -> "GraphTrainerBatch":
        series = GraphSeriesBatch.collate(items)
        return cls(runtime=to_padded_graph_runtime(series), series=series)

    @classmethod
    def from_runtime(
        cls,
        runtime: GraphRuntime,
        *,
        series: GraphSeriesBatch | None = None,
    ) -> "GraphTrainerBatch":
        return cls(runtime=runtime, series=series)

    def __len__(self) -> int:
        return self.runtime.batch_size

    @property
    def is_ragged(self) -> bool:
        return not self.runtime.is_uniform_length

    def iter_single_batches(self):
        if self.series is not None:
            for item in self.series:
                yield GraphTrainerBatch.collate_series([item])
            return
        for item in self.runtime.iter_series():
            yield GraphTrainerBatch.from_runtime(item)

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "GraphTrainerBatch":
        runtime = self.runtime.to(device=device, dtype=dtype)
        series = self.series.to(device=device, dtype=dtype) if self.series is not None else None
        return GraphTrainerBatch(runtime=runtime, series=series)

    def truncate(self, num_steps: int) -> "GraphTrainerBatch":
        runtime = self.runtime.truncate(num_steps)
        series = None
        if self.series is not None:
            series = GraphSeriesBatch.collate(item.slice_steps(0, num_steps) for item in self.series)
        return GraphTrainerBatch(runtime=runtime, series=series)

    def window(self, window: int, stride: int) -> "GraphTrainerBatch":
        runtime = self.runtime.window(window, stride)
        return GraphTrainerBatch.from_runtime(runtime)

    def initial_state(self) -> torch.Tensor:
        return self.runtime.initial_state()

    def time_tensor(self) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if self.runtime.is_uniform_length:
            return self.runtime.t
        return tuple(item.t[0] for item in self.runtime.iter_series())

    def node_state_tensor(self) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if self.runtime.is_uniform_length:
            return self.runtime.xg
        return tuple(item.xg[0] for item in self.runtime.iter_series())

    def control_tensor(self) -> torch.Tensor | None:
        if self.runtime.is_uniform_length:
            return self.runtime.ug
        return _stack_optional(item.ug[0] if item.ug is not None else None for item in self.runtime.iter_series())

    def edge_index_payload(self) -> tuple[torch.Tensor | tuple[torch.Tensor, ...], ...]:
        if self.series is not None:
            return tuple(item.edge_index for item in self.series)
        return tuple(item.ei[0] for item in self.runtime.iter_series())

    def edge_weight_payload(self) -> tuple[torch.Tensor | tuple[torch.Tensor, ...] | None, ...]:
        if self.series is not None:
            return tuple(item.edge_weight for item in self.series)
        return tuple(item.ew[0] if item.ew is not None else None for item in self.runtime.iter_series())

    def edge_attr_payload(self) -> tuple[torch.Tensor | tuple[torch.Tensor, ...] | None, ...]:
        if self.series is not None:
            return tuple(item.edge_attr for item in self.series)
        return tuple(item.ea[0] if item.ea is not None else None for item in self.runtime.iter_series())


TrainerBatch = RegularTrainerBatch | GraphTrainerBatch
