"""Typed regular-series transform pipeline for the first working slice."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

from dymad.core.series import RegularSeries, RegularSeriesBatch
from dymad.transform.base import Transform

FieldName = Literal["state", "control", "target", "params"]


def _to_numpy(tensor: torch.Tensor | None) -> np.ndarray | None:
    if tensor is None:
        return None
    return tensor.detach().cpu().numpy()


def _to_tensor(
    array: np.ndarray | None,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor | None:
    if array is None:
        return None
    return torch.as_tensor(array, dtype=dtype, device=device)


@dataclass(frozen=True)
class FieldTransform:
    """A typed field binding over an existing fitted legacy transform."""

    field: FieldName
    transform: Transform
    time_varying: bool = True

    @property
    def delay(self) -> int:
        return int(getattr(self.transform, "delay", 0))


@dataclass(frozen=True)
class RegularSeriesTransformPipeline:
    """Apply fitted field transforms to typed regular-series batches."""

    state: FieldTransform
    control: FieldTransform | None = None
    target: FieldTransform | None = None
    params: FieldTransform | None = None

    @classmethod
    def from_legacy(
        cls,
        *,
        state_transform: Transform,
        control_transform: Transform | None = None,
        target_transform: Transform | None = None,
        params_transform: Transform | None = None,
    ) -> RegularSeriesTransformPipeline:
        return cls(
            state=FieldTransform("state", state_transform, time_varying=True),
            control=FieldTransform("control", control_transform, time_varying=True)
            if control_transform is not None
            else None,
            target=FieldTransform("target", target_transform, time_varying=True)
            if target_transform is not None
            else None,
            params=FieldTransform("params", params_transform, time_varying=False)
            if params_transform is not None
            else None,
        )

    @property
    def delay(self) -> int:
        delays = [self.state.delay]
        for field in (self.control, self.target):
            if field is not None:
                delays.append(field.delay)
        return max(delays) if delays else 0

    def fit(self, batch: RegularSeriesBatch) -> RegularSeriesTransformPipeline:
        self.state.transform.fit([_to_numpy(series.state) for series in batch])
        if self.control is not None:
            controls = [_to_numpy(series.control) for series in batch if series.control is not None]
            if controls:
                self.control.transform.fit(controls)
        if self.target is not None:
            targets = [_to_numpy(series.target) for series in batch if series.target is not None]
            if targets:
                self.target.transform.fit(targets)
        if self.params is not None:
            params = [_to_numpy(series.params) for series in batch if series.params is not None]
            if params:
                self.params.transform.fit(params)
        return self

    def apply(self, batch: RegularSeriesBatch) -> RegularSeriesBatch:
        if len(batch) == 0:
            return batch

        state_out = self.state.transform.transform([_to_numpy(series.state) for series in batch])
        control_out = self._transform_optional(batch, self.control)
        target_out = self._transform_optional(batch, self.target)
        params_out = self._transform_optional(batch, self.params)

        items: list[RegularSeries] = []
        for index, series in enumerate(batch):
            meta = dict(series.meta)
            meta["delay"] = self.delay
            meta["field_delays"] = {
                "state": self.state.delay,
                "control": self.control.delay if self.control is not None else 0,
                "target": self.target.delay if self.target is not None else 0,
            }
            items.append(
                RegularSeries(
                    time=self._trim_time(series.time),
                    state=self._convert_time_varying(
                        state_out[index], series.state, self.state.delay
                    ),
                    control=self._convert_optional_time_varying(
                        control_out[index] if control_out is not None else None,
                        series.control,
                        self.control.delay if self.control is not None else 0,
                    ),
                    target=self._convert_optional_time_varying(
                        target_out[index] if target_out is not None else None,
                        series.target,
                        self.target.delay if self.target is not None else 0,
                    ),
                    params=self._convert_optional_static(
                        params_out[index] if params_out is not None else None,
                        series.params,
                    ),
                    meta=meta,
                )
            )
        return RegularSeriesBatch.collate(items)

    def inverse_state_arrays(self, payload: np.ndarray) -> np.ndarray:
        inverse = np.array(self.state.transform.inverse_transform(self._as_array_batch(payload)))
        if inverse.shape[0] == 1:
            return inverse[0]
        return inverse

    def _transform_optional(
        self,
        batch: RegularSeriesBatch,
        field: FieldTransform | None,
    ) -> list[np.ndarray | None] | None:
        if field is None:
            return None

        attr_name = field.field
        payloads = [getattr(series, attr_name) for series in batch]
        if all(item is None for item in payloads):
            return [None for _ in payloads]

        present = [item for item in payloads if item is not None]
        transformed_present = field.transform.transform([_to_numpy(item) for item in present])
        output: list[np.ndarray | None] = []
        cursor = 0
        for item in payloads:
            if item is None:
                output.append(None)
                continue
            output.append(transformed_present[cursor])
            cursor += 1
        return output

    def _trim_time(self, time: torch.Tensor) -> torch.Tensor:
        if self.delay <= 0:
            return time
        return time[self.delay :]

    def _convert_time_varying(
        self,
        array: np.ndarray,
        reference: torch.Tensor,
        field_delay: int,
    ) -> torch.Tensor:
        trim = self.delay - field_delay
        if trim > 0:
            array = array[trim:]
        return _to_tensor(array, dtype=reference.dtype, device=reference.device)

    def _convert_optional_time_varying(
        self,
        array: np.ndarray | None,
        reference: torch.Tensor | None,
        field_delay: int,
    ) -> torch.Tensor | None:
        if array is None or reference is None:
            return None
        return self._convert_time_varying(array, reference, field_delay)

    def _convert_optional_static(
        self,
        array: np.ndarray | None,
        reference: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if array is None or reference is None:
            return None
        return _to_tensor(array, dtype=reference.dtype, device=reference.device)

    @staticmethod
    def _as_array_batch(payload: np.ndarray) -> list[np.ndarray]:
        if payload.ndim == 1:
            return [np.expand_dims(payload, axis=0)]
        if payload.ndim == 2:
            return [payload]
        return [payload[index] for index in range(payload.shape[0])]
