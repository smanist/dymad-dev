"""Torch-first transform contracts over typed series objects."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal, cast

import numpy as np
import torch
from torch import nn

from dymad.core.graph_series import GraphSeries, GraphSeriesBatch
from dymad.core.series import RegularSeries, RegularSeriesBatch

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


def _module_device_dtype(module: nn.Module) -> tuple[torch.device, torch.dtype]:
    for tensor in module.parameters():
        return tensor.device, tensor.dtype
    for tensor in module.buffers():
        return tensor.device, tensor.dtype
    return torch.device("cpu"), torch.get_default_dtype()


def _ref_to_tensor(ref, *, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, bool]:
    if isinstance(ref, torch.Tensor):
        return ref.to(device=device, dtype=dtype), True
    return torch.as_tensor(ref, device=device, dtype=dtype), False


def _restore_like(value: torch.Tensor, *, as_tensor: bool):
    if as_tensor:
        return value
    return value.detach().cpu().numpy()


def _flatten_jacobian(
    jacobian: torch.Tensor, out_shape: torch.Size, in_shape: torch.Size
) -> torch.Tensor:
    return jacobian.reshape(int(np.prod(out_shape)), int(np.prod(in_shape)))


def _autograd_jacobian(
    fn,
    ref,
    *,
    device: torch.device,
    dtype: torch.dtype,
):
    tensor_ref, as_tensor = _ref_to_tensor(ref, device=device, dtype=dtype)
    flat_ref = tensor_ref.reshape(-1).detach().clone().requires_grad_(True)
    input_shape = tensor_ref.shape

    def _flat_fn(flat_input: torch.Tensor) -> torch.Tensor:
        payload = flat_input.reshape(input_shape)
        return fn(payload).reshape(-1)

    jacobian = torch.autograd.functional.jacobian(_flat_fn, flat_ref)
    output_shape = _flat_fn(flat_ref).shape
    flat_jac = _flatten_jacobian(jacobian, output_shape, flat_ref.shape)
    return _restore_like(flat_jac, as_tensor=as_tensor)


class _ExternalForwardAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, module: ExternalTransformModule, data: torch.Tensor):
        output = module._forward_external_tensor(data)
        ctx.module = module
        ctx.save_for_backward(data.detach(), output.detach())
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        module: ExternalTransformModule = ctx.module
        ref, _output = ctx.saved_tensors
        grad_input = module.forward_vjp(ref, grad_output)
        return None, grad_input.to(device=ref.device, dtype=ref.dtype)


class _ExternalInverseAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, module: ExternalTransformModule, data: torch.Tensor):
        output = module._inverse_external_tensor(data)
        ctx.module = module
        ctx.save_for_backward(data.detach(), output.detach())
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        module: ExternalTransformModule = ctx.module
        ref, _output = ctx.saved_tensors
        grad_input = module.inverse_vjp(ref, grad_output)
        return None, grad_input.to(device=ref.device, dtype=ref.dtype)


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
        self.invertibility: Invertibility = invertibility
        self.supports_gradients: GradientSupport = supports_gradients

    @property
    def _inp_dim(self) -> int | None:
        return self.input_dim

    @property
    def _out_dim(self) -> int | None:
        return self.output_dim

    @property
    def NT(self) -> int:
        return 1

    def fit(self, data: Sequence[torch.Tensor]) -> TransformModule:
        return self

    def _require_gradient_support(self, operation: str) -> None:
        if self.supports_gradients == "false":
            raise NotImplementedError(
                f"{type(self).__name__} does not support gradient-dependent operation "
                f"'{operation}'."
            )

    @abstractmethod
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(f"{type(self).__name__} does not implement inverse(...)")

    def append(self, transform: TransformModule) -> None:
        raise TypeError(f"{type(self).__name__} does not support append(...)")

    def transform_batch(
        self,
        data: Sequence[torch.Tensor],
        rng: list[int] | None = None,
    ) -> list[torch.Tensor]:
        return [self.forward_range(item, rng=rng) for item in data]

    def inverse_batch(
        self,
        data: Sequence[torch.Tensor],
        rng: list[int] | None = None,
    ) -> list[torch.Tensor]:
        return [self.inverse_range(item, rng=rng) for item in data]

    def _normalize_single_range(self, rng: list[int] | None) -> list[int]:
        if rng is None:
            return [0, 1]
        if len(rng) != 2 or rng != [0, 1]:
            raise ValueError(f"{type(self).__name__} only supports the full range [0, 1].")
        return rng

    def forward_range(self, data: torch.Tensor, rng: list[int] | None = None) -> torch.Tensor:
        self._normalize_single_range(rng)
        return self(data)

    def inverse_range(self, data: torch.Tensor, rng: list[int] | None = None) -> torch.Tensor:
        self._normalize_single_range(rng)
        return self.inverse(data)

    def transform(self, data: list[np.ndarray], rng: list[int] | None = None) -> list[np.ndarray]:
        return self.transform_arrays(data, rng=rng)

    def inverse_transform(
        self,
        data: list[np.ndarray],
        rng: list[int] | None = None,
    ) -> list[np.ndarray]:
        return self.inverse_transform_arrays(data, rng=rng)

    def transform_arrays(
        self,
        data: list[np.ndarray],
        rng: list[int] | None = None,
    ) -> list[np.ndarray]:
        device, dtype = _module_device_dtype(self)
        payloads = [torch.as_tensor(item, device=device, dtype=dtype) for item in data]
        outputs = self.transform_batch(payloads, rng=rng)
        return [item.detach().cpu().numpy() for item in outputs]

    def inverse_transform_arrays(
        self,
        data: list[np.ndarray],
        rng: list[int] | None = None,
    ) -> list[np.ndarray]:
        device, dtype = _module_device_dtype(self)
        payloads = [torch.as_tensor(item, device=device, dtype=dtype) for item in data]
        outputs = self.inverse_batch(payloads, rng=rng)
        return [item.detach().cpu().numpy() for item in outputs]

    def forward_jacobian(self, ref):
        self._require_gradient_support("forward_jacobian")
        device, dtype = _module_device_dtype(self)
        return _autograd_jacobian(self.forward, ref, device=device, dtype=dtype)

    def inverse_jacobian(self, ref):
        self._require_gradient_support("inverse_jacobian")
        device, dtype = _module_device_dtype(self)
        return _autograd_jacobian(self.inverse, ref, device=device, dtype=dtype)

    def forward_vjp(self, ref, cotangent):
        self._require_gradient_support("forward_vjp")
        device, dtype = _module_device_dtype(self)
        ref_tensor, ref_is_tensor = _ref_to_tensor(ref, device=device, dtype=dtype)
        cotangent_tensor, cot_is_tensor = _ref_to_tensor(cotangent, device=device, dtype=dtype)
        jacobian = self.forward_jacobian(ref_tensor)
        jacobian_tensor = torch.as_tensor(jacobian, device=device, dtype=dtype)
        cotangent_shape = (*jacobian_tensor.shape[:-2], jacobian_tensor.shape[-2])
        cotangent_tensor = cotangent_tensor.reshape(cotangent_shape)
        grad = jacobian_tensor.transpose(-2, -1).matmul(cotangent_tensor.unsqueeze(-1)).squeeze(-1)
        grad = grad.reshape(ref_tensor.shape)
        return _restore_like(grad, as_tensor=ref_is_tensor or cot_is_tensor)

    def inverse_vjp(self, ref, cotangent):
        self._require_gradient_support("inverse_vjp")
        device, dtype = _module_device_dtype(self)
        ref_tensor, ref_is_tensor = _ref_to_tensor(ref, device=device, dtype=dtype)
        cotangent_tensor, cot_is_tensor = _ref_to_tensor(cotangent, device=device, dtype=dtype)
        jacobian = self.inverse_jacobian(ref_tensor)
        jacobian_tensor = torch.as_tensor(jacobian, device=device, dtype=dtype)
        cotangent_shape = (*jacobian_tensor.shape[:-2], jacobian_tensor.shape[-2])
        cotangent_tensor = cotangent_tensor.reshape(cotangent_shape)
        grad = jacobian_tensor.transpose(-2, -1).matmul(cotangent_tensor.unsqueeze(-1)).squeeze(-1)
        grad = grad.reshape(ref_tensor.shape)
        return _restore_like(grad, as_tensor=ref_is_tensor or cot_is_tensor)

    def get_forward_modes(self, ref=None, rng: list[int] | None = None, **_kwargs) -> np.ndarray:
        if ref is None:
            raise ValueError("A reference point is required to compute forward modes.")
        module = self if rng is None else self._module_for_range(rng)
        module._require_gradient_support("get_forward_modes")
        return np.asarray(module.forward_jacobian(ref))

    def get_backward_modes(self, ref=None, rng: list[int] | None = None, **_kwargs) -> np.ndarray:
        if ref is None:
            raise ValueError("A reference point is required to compute backward modes.")
        module = self if rng is None else self._module_for_range(rng)
        module._require_gradient_support("get_backward_modes")
        jacobian = np.asarray(module.inverse_jacobian(ref))
        return jacobian.T

    def _module_for_range(self, rng: list[int]) -> TransformModule:
        self._normalize_single_range(rng)
        return self

    @property
    def metadata(self) -> TransformMetadata:
        return TransformMetadata(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            delay=self.delay,
            invertibility=self.invertibility,
            supports_gradients=self.supports_gradients,
        )


class ExternalTransformModule(TransformModule, ABC):
    """Typed wrapper over CPU / external transforms with explicit derivative contracts."""

    def __init__(
        self,
        *,
        delay: int = 0,
        invertibility: Invertibility = "approximate",
        supports_gradients: GradientSupport = "approximate",
        to_external=None,
        from_external=None,
    ) -> None:
        super().__init__(
            delay=delay,
            invertibility=invertibility,
            supports_gradients=supports_gradients,
        )
        self._to_external = to_external or self._tensor_to_external
        self._from_external = from_external or self._external_to_tensor

    def fit(self, data: Sequence[torch.Tensor]) -> ExternalTransformModule:
        payloads: list[Any] = []
        for item in data:
            converted = self._to_external(item)
            if isinstance(converted, list):
                payloads.extend(converted)
            else:
                payloads.append(converted)
        if payloads:
            self._fit_external(payloads)
        return self

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return _ExternalForwardAutograd.apply(self, data)

    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        return _ExternalInverseAutograd.apply(self, data)

    def forward_jacobian(self, ref):
        device, dtype = _module_device_dtype(self)
        tensor_ref, as_tensor = _ref_to_tensor(ref, device=device, dtype=dtype)
        jacobian = torch.as_tensor(
            self._forward_jacobian_external(self._to_external(tensor_ref)),
            device=device,
            dtype=dtype,
        )
        return _restore_like(jacobian, as_tensor=as_tensor)

    def inverse_jacobian(self, ref):
        device, dtype = _module_device_dtype(self)
        tensor_ref, as_tensor = _ref_to_tensor(ref, device=device, dtype=dtype)
        jacobian_t = torch.as_tensor(
            self._inverse_modes_external(self._to_external(tensor_ref)),
            device=device,
            dtype=dtype,
        )
        jacobian = jacobian_t.transpose(-2, -1)
        return _restore_like(jacobian, as_tensor=as_tensor)

    def get_backward_modes(self, ref=None, rng: list[int] | None = None, **_kwargs) -> np.ndarray:
        if ref is None:
            raise ValueError("A reference point is required to compute backward modes.")
        if rng is not None:
            self._normalize_single_range(rng)
        device, dtype = _module_device_dtype(self)
        tensor_ref, _ = _ref_to_tensor(ref, device=device, dtype=dtype)
        return np.asarray(self._inverse_modes_external(self._to_external(tensor_ref)))

    def _forward_external_tensor(self, data: torch.Tensor) -> torch.Tensor:
        converted = self._to_external(data)
        if isinstance(converted, list):
            output = self._forward_external(converted)
        else:
            output = self._forward_external([converted])[0]
        return self._from_external(output, reference=data)

    def _inverse_external_tensor(self, data: torch.Tensor) -> torch.Tensor:
        converted = self._to_external(data)
        if isinstance(converted, list):
            output = self._inverse_external(converted)
        else:
            output = self._inverse_external([converted])[0]
        return self._from_external(output, reference=data)

    @staticmethod
    def _tensor_to_external(data: torch.Tensor) -> np.ndarray:
        return data.detach().cpu().numpy()

    @staticmethod
    def _external_to_tensor(data, *, reference: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(data, dtype=reference.dtype, device=reference.device)

    @abstractmethod
    def _fit_external(self, data: list[Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def _forward_external(self, data: list[Any]) -> list[Any]:
        raise NotImplementedError

    @abstractmethod
    def _inverse_external(self, data: list[Any]) -> list[Any]:
        raise NotImplementedError

    @abstractmethod
    def _forward_jacobian_external(self, ref: Any) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def _inverse_modes_external(self, ref: Any) -> np.ndarray:
        raise NotImplementedError


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
        self.field: FieldName = field
        self.transform: TransformModule = transform
        self.time_varying: bool = (
            field in _TIME_VARYING_FIELDS if time_varying is None else bool(time_varying)
        )

    @property
    def delay(self) -> int:
        return self.transform.delay if self.time_varying else 0

    def fit(self, batch: SeriesBatch) -> FieldTransformModule:
        payloads = []
        for series in batch:
            payload = getattr(series, self.field)
            if payload is None:
                continue
            if self.field == "edge_weight":
                payloads.extend(self._edge_weight_payloads(payload))
            elif isinstance(payload, tuple):
                payloads.extend(payload)
            else:
                payloads.append(payload)
        if payloads:
            self.transform.fit(payloads)
        return self

    def apply_to_series(self, series: SeriesItem) -> SeriesItem:
        payload = getattr(series, self.field)
        if payload is None:
            return series
        if self.field == "edge_weight":
            payload = self._transform_edge_weight_payload(payload)
        elif isinstance(payload, tuple):
            payload = tuple(self.transform(item) for item in payload)
        else:
            payload = self.transform(payload)
        return _replace_field(series, self.field, payload)

    def inverse_payload(self, payload):
        if payload is None:
            return None
        if self.field == "edge_weight":
            return self._transform_edge_weight_payload(payload, inverse=True)
        if isinstance(payload, tuple):
            return tuple(self.transform.inverse(item) for item in payload)
        return self.transform.inverse(payload)

    @staticmethod
    def _edge_weight_step_to_features(step: torch.Tensor) -> torch.Tensor:
        if step.ndim == 1:
            return step.reshape(-1, 1)
        return step.reshape(-1, step.shape[-1])

    def _edge_weight_payloads(self, payload) -> list[torch.Tensor]:
        if isinstance(payload, tuple):
            return [self._edge_weight_step_to_features(step) for step in payload]
        if payload.ndim == 1:
            return [self._edge_weight_step_to_features(payload)]
        return [self._edge_weight_step_to_features(step) for step in payload]

    def _transform_edge_weight_step(
        self,
        step: torch.Tensor,
        *,
        inverse: bool = False,
    ) -> torch.Tensor:
        payload = self._edge_weight_step_to_features(step)
        transformed = self.transform.inverse(payload) if inverse else self.transform(payload)
        if transformed.ndim == 2 and transformed.shape[-1] == 1:
            return transformed.reshape(-1)
        return transformed

    def _transform_edge_weight_payload(self, payload, *, inverse: bool = False):
        if isinstance(payload, tuple):
            return tuple(
                self._transform_edge_weight_step(step, inverse=inverse) for step in payload
            )
        if payload.ndim == 1:
            return self._transform_edge_weight_step(payload, inverse=inverse)
        return torch.stack(
            [self._transform_edge_weight_step(step, inverse=inverse) for step in payload],
            dim=0,
        )


class SeriesTransformPipeline(nn.Module):
    """Canonical Torch-first transform pipeline over typed series batches."""

    def __init__(self, stages: Iterable[FieldTransformModule] | None = None) -> None:
        super().__init__()
        self.stages = nn.ModuleList(list(stages or []))

    def _typed_stages(self) -> list[FieldTransformModule]:
        return [cast(FieldTransformModule, stage) for stage in self.stages]

    @property
    def delay(self) -> int:
        delays = [stage.delay for stage in self._typed_stages() if stage.time_varying]
        return max(delays) if delays else 0

    def fit(self, batch: SeriesBatch) -> SeriesTransformPipeline:
        for stage in self._typed_stages():
            stage.fit(batch)
        return self

    def forward(self, batch: SeriesBatch) -> SeriesBatch:
        items: list[SeriesItem] = [series for series in batch]
        for stage in self._typed_stages():
            items = [stage.apply_to_series(series) for series in items]
        items = [self._align_series(series) for series in items]
        if isinstance(batch, RegularSeriesBatch):
            return RegularSeriesBatch.collate(cast(list[RegularSeries], items))
        return GraphSeriesBatch.collate(cast(list[GraphSeries], items))

    def inverse_field(self, field: FieldName, payload):
        for stage in reversed(self._typed_stages()):
            if stage.field != field:
                continue
            payload = stage.inverse_payload(payload)
        return payload

    def _align_series(self, series: SeriesItem) -> SeriesItem:
        if self.delay <= 0:
            return replace(series, meta=dict(series.meta))

        field_delays: dict[FieldName, int] = {
            stage.field: stage.delay
            for stage in self._typed_stages()
            if stage.time_varying and getattr(series, stage.field) is not None
        }
        if not field_delays:
            return replace(series, meta=dict(series.meta))

        aligned_updates = {"time": series.time[self.delay :], "meta": dict(series.meta)}
        pretrimmed_fields: set[FieldName] = set()
        if isinstance(series, GraphSeries):
            if isinstance(series.edge_index, tuple):
                aligned_updates["edge_index"] = series.edge_index[self.delay :]
            elif series.edge_index.ndim == 3 and series.edge_index.shape[0] == series.time.shape[0]:
                aligned_updates["edge_index"] = series.edge_index[self.delay :]
            graph_fields: tuple[tuple[FieldName, int], ...] = (("edge_weight", 2), ("edge_attr", 3))
            for graph_field, min_ndim in graph_fields:
                payload = getattr(series, graph_field)
                trim = self.delay - field_delays.get(graph_field, 0)
                if payload is None or trim <= 0:
                    continue
                if isinstance(payload, tuple):
                    aligned_updates[graph_field] = payload[trim:]
                    pretrimmed_fields.add(graph_field)
                elif payload.ndim >= min_ndim and payload.shape[0] == series.time.shape[0]:
                    aligned_updates[graph_field] = payload[trim:]
                    pretrimmed_fields.add(graph_field)
        aligned = replace(series, **aligned_updates)
        for field, field_delay in field_delays.items():
            if field in pretrimmed_fields:
                continue
            trim = self.delay - field_delay
            if trim <= 0:
                continue
            aligned = _replace_field(aligned, field, _slice_payload(getattr(aligned, field), trim))

        aligned.meta["delay"] = self.delay
        aligned.meta["field_delays"] = field_delays
        return aligned
