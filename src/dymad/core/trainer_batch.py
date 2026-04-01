"""Trainer-facing typed batch wrappers used during LegacyRuntimeBatch retirement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from dymad.core.graph_series import GraphSeries, GraphSeriesBatch, UniformLengthGraphSeriesBatch
from dymad.core.model_context import GraphModelContext, RegularModelContext, build_model_context
from dymad.core.series import RegularSeries, RegularSeriesBatch, UniformLengthRegularSeriesBatch


def _stack_optional(items: Iterable[torch.Tensor | None]) -> torch.Tensor | None:
    values = tuple(items)
    if not values or values[0] is None:
        return None
    return torch.stack(values)


@dataclass(frozen=True)
class RegularTrainerBatch:
    """Trainer-facing wrapper for a batch of regular typed series."""

    series: RegularSeriesBatch

    @classmethod
    def collate_series(cls, items: Iterable[RegularSeries]) -> "RegularTrainerBatch":
        return cls(RegularSeriesBatch.collate(items))

    @property
    def runtime(self) -> RegularModelContext:
        return build_model_context(self.series)

    def __len__(self) -> int:
        return len(self.series)

    @property
    def is_ragged(self) -> bool:
        return not self.series.is_uniform_length

    def iter_single_batches(self):
        for item in self.series:
            yield RegularTrainerBatch.collate_series([item])

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "RegularTrainerBatch":
        return RegularTrainerBatch(self.series.to(device=device, dtype=dtype))

    def truncate(self, num_steps: int) -> "RegularTrainerBatch":
        return RegularTrainerBatch.collate_series(
            item.slice_steps(0, num_steps) for item in self.series
        )

    def window(self, window: int, stride: int) -> "RegularTrainerBatch":
        items: list[RegularSeries] = []
        for item in self.series:
            items.extend(item.window(window, stride))
        return RegularTrainerBatch.collate_series(items)

    def initial_state(self) -> torch.Tensor:
        return self.runtime.initial_state_tensor()

    def time_tensor(self) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if isinstance(self.series, UniformLengthRegularSeriesBatch):
            return self.series.stacked_time()
        return tuple(item.time for item in self.series)

    def state_tensor(self) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if isinstance(self.series, UniformLengthRegularSeriesBatch):
            return self.series.stacked_state()
        return tuple(item.state for item in self.series)

    def control_tensor(self) -> torch.Tensor | None:
        return _stack_optional(item.control for item in self.series)

    def target_tensor(self) -> torch.Tensor | None:
        return _stack_optional(item.target for item in self.series)

    def params_tensor(self) -> torch.Tensor | None:
        return _stack_optional(item.params for item in self.series)


@dataclass(frozen=True)
class GraphTrainerBatch:
    """Trainer-facing wrapper for a batch of graph typed series."""

    series: GraphSeriesBatch

    @classmethod
    def collate_series(cls, items: Iterable[GraphSeries]) -> "GraphTrainerBatch":
        return cls(GraphSeriesBatch.collate(items))

    @property
    def runtime(self) -> GraphModelContext:
        return build_model_context(self.series)

    def __len__(self) -> int:
        return len(self.series)

    @property
    def is_ragged(self) -> bool:
        return not self.series.is_uniform_length

    def iter_single_batches(self):
        for item in self.series:
            yield GraphTrainerBatch.collate_series([item])

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "GraphTrainerBatch":
        return GraphTrainerBatch(self.series.to(device=device, dtype=dtype))

    def truncate(self, num_steps: int) -> "GraphTrainerBatch":
        return GraphTrainerBatch.collate_series(
            item.slice_steps(0, num_steps) for item in self.series
        )

    def window(self, window: int, stride: int) -> "GraphTrainerBatch":
        if window <= 0:
            raise ValueError("window must be positive")
        if stride <= 0:
            raise ValueError("stride must be positive")

        items: list[GraphSeries] = []
        for item in self.series:
            if item.time.size(0) < window:
                continue
            for start in range(0, item.time.size(0) - window + 1, stride):
                items.append(item.slice_steps(start, start + window))
        return GraphTrainerBatch.collate_series(items)

    def initial_state(self) -> torch.Tensor:
        return self.runtime.initial_state_tensor()

    def time_tensor(self) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if isinstance(self.series, UniformLengthGraphSeriesBatch):
            return self.series.stacked_time()
        return tuple(item.time for item in self.series)

    def node_state_tensor(self) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if isinstance(self.series, UniformLengthGraphSeriesBatch):
            return self.series.stacked_node_state()
        return tuple(item.node_state for item in self.series)

    def control_tensor(self) -> torch.Tensor | None:
        return _stack_optional(item.control for item in self.series)

    def edge_index_payload(self) -> tuple[torch.Tensor | tuple[torch.Tensor, ...], ...]:
        return tuple(item.edge_index for item in self.series)

    def edge_weight_payload(self) -> tuple[torch.Tensor | tuple[torch.Tensor, ...] | None, ...]:
        return tuple(item.edge_weight for item in self.series)

    def edge_attr_payload(self) -> tuple[torch.Tensor | tuple[torch.Tensor, ...] | None, ...]:
        return tuple(item.edge_attr for item in self.series)
