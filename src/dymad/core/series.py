"""Typed regular-series primitives for the first migration seam."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterable

import torch


@dataclass(frozen=True)
class RegularSeries:
    """One regular, non-graph trajectory with explicit semantic fields."""

    time: torch.Tensor
    state: torch.Tensor
    control: torch.Tensor | None = None
    target: torch.Tensor | None = None
    params: torch.Tensor | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def slice_steps(self, start: int, end: int) -> "RegularSeries":
        return replace(
            self,
            time=self.time[start:end],
            state=self.state[start:end],
            control=self.control[start:end] if self.control is not None else None,
            target=self.target[start:end] if self.target is not None else None,
            meta=dict(self.meta),
        )

    def with_state(self, new_state: torch.Tensor) -> "RegularSeries":
        return replace(self, state=new_state, meta=dict(self.meta))

    def with_control(self, new_control: torch.Tensor | None) -> "RegularSeries":
        return replace(self, control=new_control, meta=dict(self.meta))

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "RegularSeries":
        def _move(tensor: torch.Tensor | None) -> torch.Tensor | None:
            if tensor is None:
                return None
            return tensor.to(device=device, dtype=dtype)

        return replace(
            self,
            time=_move(self.time),
            state=_move(self.state),
            control=_move(self.control),
            target=_move(self.target),
            params=_move(self.params),
            meta=dict(self.meta),
        )

    def window(self, window: int, stride: int) -> "RegularSeriesBatch":
        if window <= 0:
            raise ValueError("window must be positive")
        if stride <= 0:
            raise ValueError("stride must be positive")
        if self.time.size(0) < window:
            return RegularSeriesBatch.collate([])

        items = []
        for start in range(0, self.time.size(0) - window + 1, stride):
            items.append(self.slice_steps(start, start + window))
        return RegularSeriesBatch.collate(items)


@dataclass(frozen=True)
class RegularSeriesBatch:
    """Minimal explicit batch wrapper for regular trajectories."""

    items: tuple[RegularSeries, ...]

    @classmethod
    def collate(cls, items: Iterable[RegularSeries]) -> "RegularSeriesBatch":
        items = tuple(items)
        if cls is not RegularSeriesBatch:
            return cls(items)
        if not items:
            return UniformLengthRegularSeriesBatch(items)

        lengths = {int(item.time.shape[0]) for item in items}
        if len(lengths) == 1:
            return UniformLengthRegularSeriesBatch(items)
        return RaggedRegularSeriesBatch(items)

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def __getitem__(self, index: int) -> RegularSeries:
        return self.items[index]

    @property
    def step_lengths(self) -> tuple[int, ...]:
        return tuple(int(item.time.shape[0]) for item in self.items)

    @property
    def is_uniform_length(self) -> bool:
        return len(set(self.step_lengths)) <= 1

    def slice_batch(self, indices: Iterable[int]) -> "RegularSeriesBatch":
        return RegularSeriesBatch.collate(self.items[index] for index in indices)

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "RegularSeriesBatch":
        return RegularSeriesBatch.collate(item.to(device=device, dtype=dtype) for item in self.items)


@dataclass(frozen=True)
class UniformLengthRegularSeriesBatch(RegularSeriesBatch):
    """Regular batch with equal step count across all trajectories."""

    def stacked_time(self) -> torch.Tensor:
        return torch.stack([item.time for item in self.items])

    def stacked_state(self) -> torch.Tensor:
        return torch.stack([item.state for item in self.items])


@dataclass(frozen=True)
class RaggedRegularSeriesBatch(RegularSeriesBatch):
    """Regular batch with varying step counts across trajectories."""

    @property
    def is_uniform_length(self) -> bool:
        return False
