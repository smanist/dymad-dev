"""Torch-first transform contracts over typed series objects."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Iterable, Literal, Sequence

import numpy as np
import torch
from torch import nn

from dymad.core.graph_series import GraphSeries, GraphSeriesBatch
from dymad.core.series import RegularSeries, RegularSeriesBatch
from dymad.transform.base import Transform

FieldName = Literal[
    "state",
    "control",
    "target",
    "params",
    "node_state",
    "edge_weight",
    "edge_attr",
]
Invertibility = Literal["exact", "approximate", "none"]
GradientSupport = Literal["true", "false", "approximate"]
SeriesItem = RegularSeries | GraphSeries
SeriesBatch = RegularSeriesBatch | GraphSeriesBatch

_TIME_VARYING_FIELDS = {"state", "control", "target", "node_state", "edge_weight", "edge_attr"}


def _slice_payload(payload, start: int):
    if payload is None or start <= 0:
        return payload
    if isinstance(payload, tuple):
        return payload[start:]
    return payload[start:]


def _replace_field(series: SeriesItem, field: FieldName, payload) -> SeriesItem:
    return replace(series, **{field: payload}, meta=dict(series.meta))


@dataclass(frozen=True)
class TransformMetadata:
    input_dim: int | None
    output_dim: int | None
    delay: int
    invertibility: Invertibility
    supports_gradients: GradientSupport


class TransformModule(nn.Module, ABC):
    """Base class for Torch-native fitted transforms."""

    def __init__(
        self,
        *,
        delay: int = 0,
        invertibility: Invertibility = "exact",
        supports_gradients: GradientSupport = "true",
    ) -> None:
        super().__init__()
        self.delay = int(delay)
        self.input_dim: int | None = None
        self.output_dim: int | None = None
        self.invertibility = invertibility
        self.supports_gradients = supports_gradients

    def fit(self, data: Sequence[torch.Tensor]) -> "TransformModule":
        return self

    @abstractmethod
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(f"{type(self).__name__} does not implement inverse(...)")

    def transform_batch(self, data: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        return [self(item) for item in data]

    def inverse_batch(self, data: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        return [self.inverse(item) for item in data]

    @property
    def metadata(self) -> TransformMetadata:
        return TransformMetadata(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            delay=self.delay,
            invertibility=self.invertibility,
            supports_gradients=self.supports_gradients,
        )


class LegacyTransformModuleAdapter(TransformModule):
    """Torch-facing adapter over a fitted legacy NumPy-list transform."""

    def __init__(
        self,
        legacy_transform: Transform,
        *,
        to_legacy=None,
        from_legacy=None,
        invertibility: Invertibility = "exact",
        supports_gradients: GradientSupport = "false",
    ) -> None:
        super().__init__(
            delay=int(getattr(legacy_transform, "delay", 0)),
            invertibility=invertibility,
            supports_gradients=supports_gradients,
        )
        self.legacy_transform = legacy_transform
        self._to_legacy = to_legacy or self._tensor_to_numpy
        self._from_legacy = from_legacy or self._numpy_to_tensor

    def fit(self, data: Sequence[torch.Tensor]) -> "LegacyTransformModuleAdapter":
        payloads = []
        for item in data:
            converted = self._to_legacy(item)
            if isinstance(converted, list):
                payloads.extend(converted)
            else:
                payloads.append(converted)
        if payloads:
            self.legacy_transform.fit(payloads)
            self.input_dim = int(getattr(self.legacy_transform, "_inp_dim", 0) or 0) or None
            self.output_dim = int(getattr(self.legacy_transform, "_out_dim", 0) or 0) or None
            self.delay = int(getattr(self.legacy_transform, "delay", 0))
        return self

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        converted = self._to_legacy(data)
        if isinstance(converted, list):
            output = self.legacy_transform.transform(converted)
        else:
            output = self.legacy_transform.transform([converted])[0]
        return self._from_legacy(output, reference=data)

    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        converted = self._to_legacy(data)
        if isinstance(converted, list):
            output = self.legacy_transform.inverse_transform(converted)
        else:
            output = self.legacy_transform.inverse_transform([converted])[0]
        return self._from_legacy(output, reference=data)

    @staticmethod
    def _tensor_to_numpy(data: torch.Tensor) -> np.ndarray:
        return data.detach().cpu().numpy()

    @staticmethod
    def _numpy_to_tensor(data, *, reference: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(data, dtype=reference.dtype, device=reference.device)


class FieldTransformModule(nn.Module):
    """Bind one tensor transform to one semantic series field."""

    def __init__(
        self,
        field: FieldName,
        transform: TransformModule,
        *,
        time_varying: bool | None = None,
    ) -> None:
        super().__init__()
        self.field = field
        self.transform = transform
        self.time_varying = field in _TIME_VARYING_FIELDS if time_varying is None else bool(time_varying)

    @property
    def delay(self) -> int:
        return self.transform.delay if self.time_varying else 0

    def fit(self, batch: SeriesBatch) -> "FieldTransformModule":
        payloads = []
        for series in batch:
            payload = getattr(series, self.field)
            if payload is None:
                continue
            if isinstance(payload, tuple):
                payloads.extend(payload)
            else:
                payloads.append(payload)
        if payloads:
            self.transform.fit(payloads)
        return self

    def apply_to_series(self, series: SeriesItem):
        payload = getattr(series, self.field)
        if payload is None:
            return series
        if isinstance(payload, tuple):
            payload = tuple(self.transform(item) for item in payload)
        else:
            payload = self.transform(payload)
        return _replace_field(series, self.field, payload)

    def inverse_payload(self, payload):
        if payload is None:
            return None
        if isinstance(payload, tuple):
            return tuple(self.transform.inverse(item) for item in payload)
        return self.transform.inverse(payload)


class SeriesTransformPipeline(nn.Module):
    """Canonical Torch-first transform pipeline over typed series batches."""

    def __init__(self, stages: Iterable[FieldTransformModule] | None = None) -> None:
        super().__init__()
        self.stages = nn.ModuleList(list(stages or []))

    @property
    def delay(self) -> int:
        delays = [stage.delay for stage in self.stages if stage.time_varying]
        return max(delays) if delays else 0

    def fit(self, batch: SeriesBatch) -> "SeriesTransformPipeline":
        for stage in self.stages:
            stage.fit(batch)
        return self

    def forward(self, batch: SeriesBatch) -> SeriesBatch:
        items: list[SeriesItem] = [series for series in batch]
        for stage in self.stages:
            items = [stage.apply_to_series(series) for series in items]
        items = [self._align_series(series) for series in items]
        if isinstance(batch, RegularSeriesBatch):
            return RegularSeriesBatch.collate(items)
        return GraphSeriesBatch.collate(items)

    def inverse_field(self, field: FieldName, payload):
        for stage in reversed(self.stages):
            if stage.field != field:
                continue
            payload = stage.inverse_payload(payload)
        return payload

    def _align_series(self, series: SeriesItem) -> SeriesItem:
        if self.delay <= 0:
            return replace(series, meta=dict(series.meta))

        field_delays = {
            stage.field: stage.delay
            for stage in self.stages
            if stage.time_varying and getattr(series, stage.field) is not None
        }
        if not field_delays:
            return replace(series, meta=dict(series.meta))

        aligned = replace(series, time=series.time[self.delay :], meta=dict(series.meta))
        for field, field_delay in field_delays.items():
            trim = self.delay - field_delay
            if trim <= 0:
                continue
            aligned = _replace_field(aligned, field, _slice_payload(getattr(aligned, field), trim))

        aligned.meta["delay"] = self.delay
        aligned.meta["field_delays"] = field_delays
        return aligned
