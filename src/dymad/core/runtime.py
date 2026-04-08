"""Native typed runtime payloads used by model execution and training."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any, cast

import torch

from dymad.core.graph_series import (
    GraphSeries,
    GraphSeriesBatch,
    RaggedGraphSeriesBatch,
    UniformLengthGraphSeriesBatch,
)
from dymad.core.series import (
    RaggedRegularSeriesBatch,
    RegularSeries,
    RegularSeriesBatch,
    UniformLengthRegularSeriesBatch,
)


def _stack_optional(items: Iterable[torch.Tensor | None]) -> torch.Tensor | None:
    values = tuple(items)
    if not values or any(value is None for value in values):
        return None
    return torch.stack(cast(tuple[torch.Tensor, ...], values), dim=0)


def _pad_optional(
    items: tuple[torch.Tensor | None, ...],
    *,
    max_steps: int,
    pad_shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor | None:
    if not items or items[0] is None:
        return None
    out = torch.zeros((len(items), max_steps, *pad_shape), dtype=dtype, device=device)
    for idx, item in enumerate(items):
        if item is None:
            continue
        out[idx, : item.shape[0]] = item
    return out


def _pad_regular_batch(
    batch: RaggedRegularSeriesBatch,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor,
]:
    items = tuple(batch)
    if not items:
        empty = torch.empty(0)
        return empty, empty, None, None, None, empty.bool()
    lengths = batch.step_lengths
    max_steps = max(lengths)
    state_dim = items[0].state.shape[-1]
    time = torch.zeros(
        (len(items), max_steps), dtype=items[0].time.dtype, device=items[0].time.device
    )
    state = torch.zeros(
        (len(items), max_steps, state_dim),
        dtype=items[0].state.dtype,
        device=items[0].state.device,
    )
    mask = torch.zeros((len(items), max_steps), dtype=torch.bool, device=items[0].state.device)
    for idx, item in enumerate(items):
        steps = item.time.shape[0]
        time[idx, :steps] = item.time
        state[idx, :steps] = item.state
        mask[idx, :steps] = True

    control = _pad_optional(
        tuple(item.control for item in items),
        max_steps=max_steps,
        pad_shape=items[0].control.shape[1:] if items[0].control is not None else (),
        dtype=items[0].control.dtype if items[0].control is not None else items[0].state.dtype,
        device=items[0].state.device,
    )
    target = _pad_optional(
        tuple(item.target for item in items),
        max_steps=max_steps,
        pad_shape=items[0].target.shape[1:] if items[0].target is not None else (),
        dtype=items[0].target.dtype if items[0].target is not None else items[0].state.dtype,
        device=items[0].state.device,
    )
    params = _stack_optional(item.params for item in items)
    return time, state, control, target, params, mask


def _graph_edge_steps(
    series: GraphSeries,
) -> tuple[
    tuple[torch.Tensor, ...],
    tuple[torch.Tensor, ...] | None,
    tuple[torch.Tensor, ...] | None,
]:
    n_steps = int(series.time.shape[0])
    if isinstance(series.edge_index, torch.Tensor):
        if series.edge_index.ndim == 3 and series.edge_index.shape[0] == n_steps:
            edge_index = tuple(
                step if step.shape[0] == 2 else step.transpose(0, 1) for step in series.edge_index
            )
        else:
            edge_index = tuple(series.edge_index for _ in range(n_steps))
    else:
        edge_index = series.edge_index

    if isinstance(series.edge_weight, torch.Tensor):
        if series.edge_weight.ndim >= 2 and series.edge_weight.shape[0] == n_steps:
            edge_weight = tuple(series.edge_weight[step] for step in range(n_steps))
        else:
            edge_weight = tuple(series.edge_weight for _ in range(n_steps))
    elif series.edge_weight is None:
        edge_weight = None
    else:
        edge_weight = series.edge_weight

    if isinstance(series.edge_attr, torch.Tensor):
        if series.edge_attr.ndim >= 3 and series.edge_attr.shape[0] == n_steps:
            edge_attr = tuple(series.edge_attr[step] for step in range(n_steps))
        else:
            edge_attr = tuple(series.edge_attr for _ in range(n_steps))
    elif series.edge_attr is None:
        edge_attr = None
    else:
        edge_attr = series.edge_attr

    return edge_index, edge_weight, edge_attr


def _stack_graph_steps(
    series_items: tuple[GraphSeries, ...],
    *,
    max_steps: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if not series_items:
        raise ValueError("series_items must not be empty")

    step_payloads = [_graph_edge_steps(series) for series in series_items]
    lengths = [len(item[0]) for item in step_payloads]
    if max_steps is None:
        max_steps = lengths[0]

    ref_dtype = step_payloads[0][0][0].dtype
    ref_device = step_payloads[0][0][0].device
    have_weight = any(weight is not None for _, weight, _ in step_payloads)
    edge_counts = [int(step.shape[1]) for edge_index, _, _ in step_payloads for step in edge_index]
    if len(set(edge_counts)) != 1 and not have_weight:
        raise ValueError(
            "Native graph runtime currently requires a consistent edge count unless edge weights are provided."
        )
    n_edges = max(edge_counts)
    edge_index = torch.zeros(
        (len(series_items), max_steps, n_edges, 2),
        dtype=ref_dtype,
        device=ref_device,
    )

    edge_weight = None
    if have_weight:
        ref_weight = next(weight[0] for _, weight, _ in step_payloads if weight is not None)
        edge_weight = torch.zeros(
            (len(series_items), max_steps, n_edges),
            dtype=ref_weight.dtype,
            device=ref_weight.device,
        )

    have_attr = any(attr is not None for _, _, attr in step_payloads)
    edge_attr = None
    if have_attr:
        ref_attr = next(attr[0] for _, _, attr in step_payloads if attr is not None)
        edge_attr = torch.zeros(
            (len(series_items), max_steps, n_edges, ref_attr.shape[-1]),
            dtype=ref_attr.dtype,
            device=ref_attr.device,
        )

    for batch_index, (series, (edge_index_steps, edge_weight_steps, edge_attr_steps)) in enumerate(
        zip(series_items, step_payloads, strict=False)
    ):
        valid_steps = int(series.time.shape[0])
        fill_edge = edge_index_steps[0]
        fill_weight = edge_weight_steps[0] if edge_weight_steps is not None else None
        fill_attr = edge_attr_steps[0] if edge_attr_steps is not None else None
        for step in range(max_steps):
            if step < valid_steps:
                edge_count = int(edge_index_steps[step].shape[1])
                edge_index[batch_index, step, :edge_count] = edge_index_steps[step].transpose(0, 1)
                if edge_weight is not None and edge_weight_steps is not None:
                    weight_step = edge_weight_steps[step]
                    if weight_step.ndim > 1 and weight_step.shape[-1] == 1:
                        weight_step = weight_step.squeeze(-1)
                    edge_weight[batch_index, step, :edge_count] = weight_step
                if edge_attr is not None and edge_attr_steps is not None:
                    edge_attr[batch_index, step, :edge_count] = edge_attr_steps[step]
            else:
                edge_count = int(fill_edge.shape[1])
                edge_index[batch_index, step, :edge_count] = fill_edge.transpose(0, 1)
                if edge_weight is not None and fill_weight is not None:
                    weight_step = fill_weight
                    if weight_step.ndim > 1 and weight_step.shape[-1] == 1:
                        weight_step = weight_step.squeeze(-1)
                    edge_weight[batch_index, step, :edge_count] = weight_step
                if edge_attr is not None and fill_attr is not None:
                    edge_attr[batch_index, step, :edge_count] = fill_attr

    return edge_index, edge_weight, edge_attr


def _stack_fixed_edge_index(series_items: tuple[GraphSeries, ...]) -> torch.Tensor:
    edge_index_tensors: list[torch.Tensor] = []
    for item in series_items:
        edge_index = item.edge_index
        if not isinstance(edge_index, torch.Tensor):
            raise TypeError("Fixed-topology graph batches require tensor edge indices.")
        edge_index_tensors.append(edge_index.transpose(0, 1))
    return torch.stack(
        edge_index_tensors,
        dim=0,
    )


def _normalize_time_varying_edge_index(edge_index: torch.Tensor) -> torch.Tensor:
    if edge_index.ndim != 3:
        raise TypeError("time-varying edge indices must be stored as rank-3 tensors")
    if edge_index.shape[1] == 2:
        return edge_index.transpose(1, 2)
    if edge_index.shape[2] == 2:
        return edge_index
    raise ValueError(
        "time-varying edge_index tensors must have shape [n_steps, 2, n_edges] "
        "or [n_steps, n_edges, 2]"
    )


def _stack_time_varying_edge_index(
    series_items: tuple[GraphSeries, ...],
    *,
    max_steps: int,
) -> torch.Tensor:
    tensors = [
        _normalize_time_varying_edge_index(item.edge_index)
        for item in series_items
        if isinstance(item.edge_index, torch.Tensor)
    ]
    if len(tensors) != len(series_items):
        raise TypeError("time-varying edge indices must be stored as rank-3 tensors")
    max_edges = max(tensor.shape[1] for tensor in tensors)
    ref = tensors[0]
    out = torch.zeros(
        (len(tensors), max_steps, max_edges, 2),
        dtype=ref.dtype,
        device=ref.device,
    )
    for idx, tensor in enumerate(tensors):
        steps, edges = tensor.shape[:2]
        out[idx, :steps, :edges] = tensor
        if steps < max_steps:
            out[idx, steps:, :edges] = tensor[0]
    return out


def _stack_time_varying_graph_field(
    items: tuple[torch.Tensor | None, ...],
    *,
    max_steps: int,
    time_ndim: int,
) -> torch.Tensor | None:
    present = tuple(item for item in items if item is not None)
    if not present:
        return None

    ref = present[0]
    max_edges = max(item.shape[1] if item.ndim == time_ndim else item.shape[0] for item in present)
    feature_shape = ref.shape[2:] if ref.ndim == time_ndim else ref.shape[1:]
    out = torch.zeros(
        (len(items), max_steps, max_edges, *feature_shape),
        dtype=ref.dtype,
        device=ref.device,
    )
    for idx, item in enumerate(items):
        if item is None:
            continue
        if item.ndim == time_ndim:
            steps = item.shape[0]
            edge_count = item.shape[1]
            out[idx, :steps, :edge_count] = item
            if steps < max_steps:
                out[idx, steps:, :edge_count] = item[0]
        else:
            edge_count = item.shape[0]
            out[idx, :, :edge_count] = item.unsqueeze(0).expand(
                max_steps, edge_count, *feature_shape
            )
    if out.shape[-1:] == (1,):
        return out.squeeze(-1)
    return out


def _pad_time_varying_edge_index(
    series_items: tuple[GraphSeries, ...],
    *,
    max_steps: int,
) -> torch.Tensor:
    return _stack_time_varying_edge_index(series_items, max_steps=max_steps)


def _stack_fixed_graph_field(
    items: tuple[torch.Tensor | None, ...],
    *,
    max_steps: int,
    time_ndim: int,
) -> torch.Tensor | None:
    present = tuple(item for item in items if item is not None)
    if not present:
        return None

    dynamic = any(item.ndim == time_ndim for item in present)
    ref = present[0]
    pad_shape = ref.shape[1:] if dynamic else ref.shape
    out_shape = (len(items), max_steps, *pad_shape) if dynamic else (len(items), *pad_shape)
    out = torch.zeros(out_shape, dtype=ref.dtype, device=ref.device)

    for idx, item in enumerate(items):
        if item is None:
            continue
        if dynamic:
            if item.ndim == time_ndim:
                steps = item.shape[0]
                out[idx, :steps] = item
            else:
                out[idx] = item.unsqueeze(0).expand(max_steps, *pad_shape)
        else:
            out[idx] = item

    return out


def _graph_step_tensor(
    tensor: torch.Tensor | None,
    *,
    step: int,
    time_ndim: int,
) -> torch.Tensor | None:
    if tensor is None:
        return None
    if tensor.ndim == time_ndim:
        return tensor[:, step]
    return tensor


def _graph_batch_tensor(
    tensor: torch.Tensor | None,
    *,
    index: int,
    length: int | None = None,
    time_ndim: int,
) -> torch.Tensor | None:
    if tensor is None:
        return None
    value = tensor[index : index + 1]
    if tensor.ndim == time_ndim and length is not None:
        value = value[:, :length]
    return value


def _require_tensor(name: str, tensor: torch.Tensor | None) -> torch.Tensor:
    if tensor is None:
        raise ValueError(f"{name} is required for graph runtime construction.")
    return tensor


def _graph_tensor_field(
    payload: torch.Tensor | tuple[torch.Tensor, ...] | None,
) -> torch.Tensor | None:
    if payload is None:
        return None
    if isinstance(payload, tuple):
        raise TypeError("Graph tensor payload must be materialized before runtime construction.")
    return payload


def _graph_truncate_tensor(
    tensor: torch.Tensor | None,
    *,
    num_steps: int,
    time_ndim: int,
) -> torch.Tensor | None:
    if tensor is None:
        return None
    if tensor.ndim == time_ndim:
        return tensor[:, :num_steps]
    return tensor


def _graph_window_tensor(
    tensor: torch.Tensor | None,
    *,
    n_windows: int,
    starts: list[int],
    window: int,
    time_ndim: int,
) -> torch.Tensor | None:
    if tensor is None:
        return None
    if tensor.ndim != time_ndim:
        return (
            tensor.unsqueeze(1)
            .expand(tensor.shape[0], n_windows, *tensor.shape[1:])
            .reshape(
                tensor.shape[0] * n_windows,
                *tensor.shape[1:],
            )
        )
    parts = [tensor[:, start : start + window] for start in starts]
    return torch.stack(parts, dim=1).reshape(tensor.shape[0] * n_windows, window, *tensor.shape[2:])


@dataclass(frozen=True)
class RegularRuntimeStep:
    time: torch.Tensor | None = None
    state: torch.Tensor | None = None
    control: torch.Tensor | None = None
    target: torch.Tensor | None = None
    params: torch.Tensor | None = None
    valid_mask: torch.Tensor | None = None
    meta: tuple[dict[str, Any], ...] = ()

    _has_graph: bool = False

    @property
    def batch_size(self) -> int:
        if self.state is not None:
            return int(self.state.shape[0])
        if self.time is not None:
            return int(self.time.shape[0])
        if self.params is not None:
            return int(self.params.shape[0])
        return 0

    @property
    def x(self) -> torch.Tensor | None:
        return self.state

    @property
    def t(self) -> torch.Tensor | None:
        return self.time

    @property
    def u(self) -> torch.Tensor | None:
        return self.control

    @property
    def y(self) -> torch.Tensor | None:
        return self.target

    @property
    def p(self) -> torch.Tensor | None:
        return self.params

    def set_x(self, value: torch.Tensor) -> RegularRuntimeStep:
        return replace(self, state=value)

    def set_u(self, value: torch.Tensor | None = None) -> RegularRuntimeStep:
        if value is None:
            return self
        return replace(self, control=value)

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
    ) -> RegularRuntimeStep:
        def _move(tensor: torch.Tensor | None) -> torch.Tensor | None:
            if tensor is None:
                return None
            return tensor.to(device=device, dtype=dtype, non_blocking=non_blocking)

        return replace(
            self,
            time=_move(self.time),
            state=_move(self.state),
            control=_move(self.control),
            target=_move(self.target),
            params=_move(self.params),
            valid_mask=_move(self.valid_mask),
        )


@dataclass(frozen=True)
class GraphRuntimeStep:
    time: torch.Tensor | None = None
    node_state: torch.Tensor | None = None
    control: torch.Tensor | None = None
    target: torch.Tensor | None = None
    params: torch.Tensor | None = None
    edge_index: torch.Tensor | None = None
    edge_weight: torch.Tensor | None = None
    edge_attr: torch.Tensor | None = None
    valid_mask: torch.Tensor | None = None
    meta: tuple[dict[str, Any], ...] = ()

    _has_graph: bool = True

    @property
    def batch_size(self) -> int:
        if self.node_state is not None:
            return int(self.node_state.shape[0])
        if self.time is not None:
            return int(self.time.shape[0])
        if self.params is not None:
            return int(self.params.shape[0])
        return 0

    @property
    def n_nodes(self) -> int:
        if self.node_state is None:
            return 0
        return int(self.node_state.shape[-2])

    @property
    def x(self) -> torch.Tensor | None:
        if self.node_state is None:
            return None
        return self.node_state.reshape(*self.node_state.shape[:-2], -1)

    @property
    def t(self) -> torch.Tensor | None:
        return self.time

    @property
    def u(self) -> torch.Tensor | None:
        if self.control is None:
            return None
        return self.control.reshape(*self.control.shape[:-2], -1)

    @property
    def y(self) -> torch.Tensor | None:
        if self.target is None:
            return None
        return self.target.reshape(*self.target.shape[:-2], -1)

    @property
    def p(self) -> torch.Tensor | None:
        return self.params

    @property
    def xg(self) -> torch.Tensor | None:
        return self.node_state

    @property
    def ug(self) -> torch.Tensor | None:
        return self.control

    @property
    def yg(self) -> torch.Tensor | None:
        return self.target

    @property
    def ei(self) -> torch.Tensor | None:
        return self.edge_index

    @property
    def ew(self) -> torch.Tensor | None:
        return self.edge_weight

    @property
    def ea(self) -> torch.Tensor | None:
        return self.edge_attr

    def g(self, value: torch.Tensor) -> torch.Tensor:
        return value.reshape(*value.shape[:-1], self.n_nodes, -1)

    def G(self, value: torch.Tensor) -> torch.Tensor:
        return value.reshape(*value.shape[:-2], -1)

    def set_x(self, value: torch.Tensor) -> GraphRuntimeStep:
        return replace(self, node_state=self.g(value) if value.ndim == 2 else value)

    def set_u(self, value: torch.Tensor | None = None) -> GraphRuntimeStep:
        if value is None:
            return self
        return replace(self, control=self.g(value) if value.ndim == 2 else value)

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
    ) -> GraphRuntimeStep:
        def _move(tensor: torch.Tensor | None) -> torch.Tensor | None:
            if tensor is None:
                return None
            if tensor.dtype in (torch.int32, torch.int64):
                return tensor.to(device=device, non_blocking=non_blocking)
            return tensor.to(device=device, dtype=dtype, non_blocking=non_blocking)

        return replace(
            self,
            time=_move(self.time),
            node_state=_move(self.node_state),
            control=_move(self.control),
            target=_move(self.target),
            params=_move(self.params),
            edge_index=_move(self.edge_index),
            edge_weight=_move(self.edge_weight),
            edge_attr=_move(self.edge_attr),
            valid_mask=_move(self.valid_mask),
        )


@dataclass(frozen=True)
class EmptyRegularRuntime:
    batch_size: int = 0

    _has_graph: bool = False
    is_graph: bool = False
    is_uniform_length: bool = True
    n_steps: None = None
    step_lengths: tuple[int, ...] = ()

    @property
    def valid_mask(self) -> torch.Tensor | None:
        return None

    @property
    def x(self) -> None:
        return None

    @property
    def t(self) -> None:
        return None

    @property
    def u(self) -> None:
        return None

    @property
    def y(self) -> None:
        return None

    @property
    def p(self) -> None:
        return None

    def initial_state(self) -> None:
        return None

    def time_payload(self) -> None:
        return None

    def control_payload(self) -> None:
        return None

    def params_payload(self) -> None:
        return None

    def get_step(self, step: int) -> RegularRuntimeStep:
        return RegularRuntimeStep(meta=tuple({} for _ in range(self.batch_size)))

    def iter_series(self):
        return iter(())

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
    ) -> EmptyRegularRuntime:
        return self


@dataclass(frozen=True)
class UniformRegularRuntime:
    time: torch.Tensor
    state: torch.Tensor
    control: torch.Tensor | None = None
    target: torch.Tensor | None = None
    params: torch.Tensor | None = None
    meta: tuple[dict[str, Any], ...] = ()

    _has_graph: bool = False
    is_graph: bool = False
    is_uniform_length: bool = True

    @property
    def batch_size(self) -> int:
        return int(self.state.shape[0])

    @property
    def n_steps(self) -> int:
        return int(self.state.shape[1])

    @property
    def step_lengths(self) -> tuple[int, ...]:
        return tuple(self.n_steps for _ in range(self.batch_size))

    @property
    def valid_mask(self) -> torch.Tensor:
        return torch.ones(
            self.state.shape[:2],
            dtype=torch.bool,
            device=self.state.device,
        )

    @property
    def x(self) -> torch.Tensor:
        return self.state

    @property
    def t(self) -> torch.Tensor:
        return self.time

    @property
    def u(self) -> torch.Tensor | None:
        return self.control

    @property
    def y(self) -> torch.Tensor | None:
        return self.target

    @property
    def p(self) -> torch.Tensor | None:
        return self.params

    def initial_state(self) -> torch.Tensor:
        return self.state[:, 0, :]

    def time_payload(self) -> torch.Tensor:
        return self.time

    def control_payload(self) -> torch.Tensor | None:
        return self.control

    def params_payload(self) -> torch.Tensor | None:
        return self.params

    def get_step(self, step: int) -> RegularRuntimeStep:
        return RegularRuntimeStep(
            time=self.time[:, step] if self.time is not None else None,
            state=self.state[:, step],
            control=self.control[:, step] if self.control is not None else None,
            target=self.target[:, step] if self.target is not None else None,
            params=self.params,
            valid_mask=self.valid_mask[:, step],
            meta=self.meta,
        )

    def truncate(self, num_steps: int) -> UniformRegularRuntime:
        return replace(
            self,
            time=self.time[:, :num_steps],
            state=self.state[:, :num_steps],
            control=self.control[:, :num_steps] if self.control is not None else None,
            target=self.target[:, :num_steps] if self.target is not None else None,
        )

    def window(self, window: int, stride: int) -> UniformRegularRuntime:
        if window <= 0:
            raise ValueError("window must be positive")
        if stride <= 0:
            raise ValueError("stride must be positive")
        if self.n_steps < window:
            return replace(
                self,
                time=self.time[:0],
                state=self.state[:0],
                control=self.control[:0] if self.control is not None else None,
                target=self.target[:0] if self.target is not None else None,
                params=self.params[:0] if self.params is not None else None,
                meta=(),
            )

        starts = list(range(0, self.n_steps - window + 1, stride))
        n_windows = len(starts)
        time = torch.stack([self.time[:, start : start + window] for start in starts], dim=1)
        state = torch.stack([self.state[:, start : start + window] for start in starts], dim=1)
        time = time.reshape(self.batch_size * n_windows, window)
        state = state.reshape(self.batch_size * n_windows, window, *self.state.shape[2:])
        control = None
        if self.control is not None:
            control = torch.stack(
                [self.control[:, start : start + window] for start in starts], dim=1
            )
            control = control.reshape(self.batch_size * n_windows, window, *self.control.shape[2:])
        target = None
        if self.target is not None:
            target = torch.stack(
                [self.target[:, start : start + window] for start in starts], dim=1
            )
            target = target.reshape(self.batch_size * n_windows, window, *self.target.shape[2:])
        params = None
        if self.params is not None:
            params = self.params.unsqueeze(1).expand(
                self.batch_size, n_windows, *self.params.shape[1:]
            )
            params = params.reshape(self.batch_size * n_windows, *self.params.shape[1:])
        meta = tuple(dict(self.meta[idx]) for idx in range(self.batch_size) for _ in starts)
        return UniformRegularRuntime(
            time=time,
            state=state,
            control=control,
            target=target,
            params=params,
            meta=meta,
        )

    def iter_series(self):
        for idx in range(self.batch_size):
            yield UniformRegularRuntime(
                time=self.time[idx : idx + 1],
                state=self.state[idx : idx + 1],
                control=self.control[idx : idx + 1] if self.control is not None else None,
                target=self.target[idx : idx + 1] if self.target is not None else None,
                params=self.params[idx : idx + 1] if self.params is not None else None,
                meta=(dict(self.meta[idx]),) if self.meta else (),
            )

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
    ) -> UniformRegularRuntime:
        def _move(tensor: torch.Tensor | None) -> torch.Tensor | None:
            if tensor is None:
                return None
            return tensor.to(device=device, dtype=dtype, non_blocking=non_blocking)

        return replace(
            self,
            time=_move(self.time),
            state=_move(self.state),
            control=_move(self.control),
            target=_move(self.target),
            params=_move(self.params),
        )


@dataclass(frozen=True)
class RaggedRegularRuntime:
    time: torch.Tensor
    state: torch.Tensor
    step_lengths: tuple[int, ...]
    valid_mask: torch.Tensor
    control: torch.Tensor | None = None
    target: torch.Tensor | None = None
    params: torch.Tensor | None = None
    meta: tuple[dict[str, Any], ...] = ()

    _has_graph: bool = False
    is_graph: bool = False
    is_uniform_length: bool = False

    @property
    def batch_size(self) -> int:
        return int(self.state.shape[0])

    @property
    def n_steps(self) -> int:
        return int(self.state.shape[1])

    @property
    def x(self) -> torch.Tensor:
        return self.state

    @property
    def t(self) -> torch.Tensor:
        return self.time

    @property
    def u(self) -> torch.Tensor | None:
        return self.control

    @property
    def y(self) -> torch.Tensor | None:
        return self.target

    @property
    def p(self) -> torch.Tensor | None:
        return self.params

    def initial_state(self) -> torch.Tensor:
        return self.state[:, 0, :]

    def time_payload(self) -> torch.Tensor:
        return self.time

    def control_payload(self) -> torch.Tensor | None:
        return self.control

    def params_payload(self) -> torch.Tensor | None:
        return self.params

    def get_step(self, step: int) -> RegularRuntimeStep:
        return RegularRuntimeStep(
            time=self.time[:, step] if self.time is not None else None,
            state=self.state[:, step],
            control=self.control[:, step] if self.control is not None else None,
            target=self.target[:, step] if self.target is not None else None,
            params=self.params,
            valid_mask=self.valid_mask[:, step],
            meta=self.meta,
        )

    def truncate(self, num_steps: int) -> RaggedRegularRuntime:
        step_lengths = tuple(min(length, num_steps) for length in self.step_lengths)
        return replace(
            self,
            time=self.time[:, :num_steps],
            state=self.state[:, :num_steps],
            step_lengths=step_lengths,
            valid_mask=self.valid_mask[:, :num_steps],
            control=self.control[:, :num_steps] if self.control is not None else None,
            target=self.target[:, :num_steps] if self.target is not None else None,
        )

    def window(self, window: int, stride: int) -> TypedRuntime:
        if window <= 0:
            raise ValueError("window must be positive")
        if stride <= 0:
            raise ValueError("stride must be positive")
        items = []
        for runtime in self.iter_series():
            if runtime.n_steps < window:
                continue
            items.append(runtime.window(window, stride))
        if not items:
            return EmptyRegularRuntime()
        stacked_time = torch.cat([item.time for item in items], dim=0)
        stacked_state = torch.cat([item.state for item in items], dim=0)
        stacked_control = None
        if any(item.control is not None for item in items):
            ref = next(item.control for item in items if item.control is not None)
            parts = []
            for item in items:
                if item.control is None:
                    parts.append(
                        torch.zeros(
                            (item.batch_size, item.n_steps, *ref.shape[2:]),
                            dtype=ref.dtype,
                            device=ref.device,
                        )
                    )
                else:
                    parts.append(item.control)
            stacked_control = torch.cat(parts, dim=0)
        stacked_target = None
        if any(item.target is not None for item in items):
            ref = next(item.target for item in items if item.target is not None)
            parts = []
            for item in items:
                if item.target is None:
                    parts.append(
                        torch.zeros(
                            (item.batch_size, item.n_steps, *ref.shape[2:]),
                            dtype=ref.dtype,
                            device=ref.device,
                        )
                    )
                else:
                    parts.append(item.target)
            stacked_target = torch.cat(parts, dim=0)
        stacked_params = None
        if any(item.params is not None for item in items):
            ref = next(item.params for item in items if item.params is not None)
            parts = []
            for item in items:
                if item.params is None:
                    parts.append(
                        torch.zeros(
                            (item.batch_size, *ref.shape[1:]), dtype=ref.dtype, device=ref.device
                        )
                    )
                else:
                    parts.append(item.params)
            stacked_params = torch.cat(parts, dim=0)
        meta = tuple(meta for item in items for meta in item.meta)
        return UniformRegularRuntime(
            time=stacked_time,
            state=stacked_state,
            control=stacked_control,
            target=stacked_target,
            params=stacked_params,
            meta=meta,
        )

    def iter_series(self):
        for idx, length in enumerate(self.step_lengths):
            yield UniformRegularRuntime(
                time=self.time[idx : idx + 1, :length],
                state=self.state[idx : idx + 1, :length],
                control=self.control[idx : idx + 1, :length] if self.control is not None else None,
                target=self.target[idx : idx + 1, :length] if self.target is not None else None,
                params=self.params[idx : idx + 1] if self.params is not None else None,
                meta=(dict(self.meta[idx]),) if self.meta else (),
            )

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
    ) -> RaggedRegularRuntime:
        def _move(tensor: torch.Tensor | None) -> torch.Tensor | None:
            if tensor is None:
                return None
            return tensor.to(device=device, dtype=dtype, non_blocking=non_blocking)

        return replace(
            self,
            time=_move(self.time),
            state=_move(self.state),
            valid_mask=_move(self.valid_mask),
            control=_move(self.control),
            target=_move(self.target),
            params=_move(self.params),
        )


@dataclass(frozen=True)
class UniformGraphRuntime:
    time: torch.Tensor
    node_state: torch.Tensor
    edge_index: torch.Tensor
    control: torch.Tensor | None = None
    target: torch.Tensor | None = None
    params: torch.Tensor | None = None
    edge_weight: torch.Tensor | None = None
    edge_attr: torch.Tensor | None = None
    meta: tuple[dict[str, Any], ...] = ()

    _has_graph: bool = True
    is_graph: bool = True
    is_uniform_length: bool = True

    @property
    def is_fixed_topology(self) -> bool:
        return self.edge_index.ndim == 3

    @property
    def batch_size(self) -> int:
        return int(self.node_state.shape[0])

    @property
    def n_steps(self) -> int:
        return int(self.node_state.shape[1])

    @property
    def step_lengths(self) -> tuple[int, ...]:
        return tuple(self.n_steps for _ in range(self.batch_size))

    @property
    def n_nodes(self) -> int:
        return int(self.node_state.shape[2])

    @property
    def valid_mask(self) -> torch.Tensor:
        return torch.ones(
            self.node_state.shape[:2],
            dtype=torch.bool,
            device=self.node_state.device,
        )

    @property
    def x(self) -> torch.Tensor:
        return self.node_state.reshape(self.batch_size, self.n_steps, -1)

    @property
    def t(self) -> torch.Tensor:
        return self.time

    @property
    def u(self) -> torch.Tensor | None:
        if self.control is None:
            return None
        return self.control.reshape(self.batch_size, self.n_steps, -1)

    @property
    def y(self) -> torch.Tensor | None:
        if self.target is None:
            return None
        return self.target.reshape(self.batch_size, self.n_steps, -1)

    @property
    def p(self) -> torch.Tensor | None:
        return self.params

    @property
    def xg(self) -> torch.Tensor:
        return self.node_state

    @property
    def ug(self) -> torch.Tensor | None:
        return self.control

    @property
    def yg(self) -> torch.Tensor | None:
        return self.target

    @property
    def ei(self) -> torch.Tensor:
        return self.edge_index

    @property
    def ew(self) -> torch.Tensor | None:
        return self.edge_weight

    @property
    def ea(self) -> torch.Tensor | None:
        return self.edge_attr

    def initial_state(self) -> torch.Tensor:
        return self.x[:, 0, :]

    def time_payload(self) -> torch.Tensor:
        return self.time

    def control_payload(self) -> torch.Tensor | None:
        return self.control

    def params_payload(self) -> torch.Tensor | None:
        return self.params

    def get_step(self, step: int) -> GraphRuntimeStep:
        return GraphRuntimeStep(
            time=self.time[:, step] if self.time is not None else None,
            node_state=self.node_state[:, step],
            control=self.control[:, step] if self.control is not None else None,
            target=self.target[:, step] if self.target is not None else None,
            params=self.params,
            edge_index=_graph_step_tensor(self.edge_index, step=step, time_ndim=4),
            edge_weight=_graph_step_tensor(self.edge_weight, step=step, time_ndim=3),
            edge_attr=_graph_step_tensor(self.edge_attr, step=step, time_ndim=4),
            valid_mask=self.valid_mask[:, step],
            meta=self.meta,
        )

    def truncate(self, num_steps: int) -> UniformGraphRuntime:
        return replace(
            self,
            time=self.time[:, :num_steps],
            node_state=self.node_state[:, :num_steps],
            control=self.control[:, :num_steps] if self.control is not None else None,
            target=self.target[:, :num_steps] if self.target is not None else None,
            edge_index=_graph_truncate_tensor(self.edge_index, num_steps=num_steps, time_ndim=4),
            edge_weight=_graph_truncate_tensor(self.edge_weight, num_steps=num_steps, time_ndim=3),
            edge_attr=_graph_truncate_tensor(self.edge_attr, num_steps=num_steps, time_ndim=4),
        )

    def window(self, window: int, stride: int) -> UniformGraphRuntime:
        if window <= 0:
            raise ValueError("window must be positive")
        if stride <= 0:
            raise ValueError("stride must be positive")
        if self.n_steps < window:
            return replace(
                self,
                time=self.time[:0],
                node_state=self.node_state[:0],
                edge_index=self.edge_index[:0] if self.is_fixed_topology else self.edge_index[:0],
                control=self.control[:0] if self.control is not None else None,
                target=self.target[:0] if self.target is not None else None,
                params=self.params[:0] if self.params is not None else None,
                edge_weight=self.edge_weight[:0] if self.edge_weight is not None else None,
                edge_attr=self.edge_attr[:0] if self.edge_attr is not None else None,
                meta=(),
            )

        starts = list(range(0, self.n_steps - window + 1, stride))
        n_windows = len(starts)
        time = torch.stack([self.time[:, start : start + window] for start in starts], dim=1)
        node_state = torch.stack(
            [self.node_state[:, start : start + window] for start in starts], dim=1
        )
        time = time.reshape(self.batch_size * n_windows, window)
        node_state = node_state.reshape(
            self.batch_size * n_windows, window, *self.node_state.shape[2:]
        )
        control = None
        if self.control is not None:
            control = torch.stack(
                [self.control[:, start : start + window] for start in starts], dim=1
            )
            control = control.reshape(self.batch_size * n_windows, window, *self.control.shape[2:])
        target = None
        if self.target is not None:
            target = torch.stack(
                [self.target[:, start : start + window] for start in starts], dim=1
            )
            target = target.reshape(self.batch_size * n_windows, window, *self.target.shape[2:])
        params = None
        if self.params is not None:
            params = self.params.unsqueeze(1).expand(
                self.batch_size, n_windows, *self.params.shape[1:]
            )
            params = params.reshape(self.batch_size * n_windows, *self.params.shape[1:])
        edge_index = _graph_window_tensor(
            self.edge_index, n_windows=n_windows, starts=starts, window=window, time_ndim=4
        )
        edge_weight = _graph_window_tensor(
            self.edge_weight, n_windows=n_windows, starts=starts, window=window, time_ndim=3
        )
        edge_attr = _graph_window_tensor(
            self.edge_attr, n_windows=n_windows, starts=starts, window=window, time_ndim=4
        )
        meta = tuple(dict(self.meta[idx]) for idx in range(self.batch_size) for _ in starts)
        return UniformGraphRuntime(
            time=time,
            node_state=node_state,
            edge_index=_require_tensor("edge_index", edge_index),
            control=control,
            target=target,
            params=params,
            edge_weight=edge_weight,
            edge_attr=edge_attr,
            meta=meta,
        )

    def iter_series(self):
        for idx in range(self.batch_size):
            yield UniformGraphRuntime(
                time=self.time[idx : idx + 1],
                node_state=self.node_state[idx : idx + 1],
                edge_index=_require_tensor(
                    "edge_index", _graph_batch_tensor(self.edge_index, index=idx, time_ndim=4)
                ),
                control=self.control[idx : idx + 1] if self.control is not None else None,
                target=self.target[idx : idx + 1] if self.target is not None else None,
                params=self.params[idx : idx + 1] if self.params is not None else None,
                edge_weight=_graph_batch_tensor(self.edge_weight, index=idx, time_ndim=3),
                edge_attr=_graph_batch_tensor(self.edge_attr, index=idx, time_ndim=4),
                meta=(dict(self.meta[idx]),) if self.meta else (),
            )

    def g(self, value: torch.Tensor) -> torch.Tensor:
        return value.reshape(*value.shape[:-1], self.n_nodes, -1)

    def G(self, value: torch.Tensor) -> torch.Tensor:
        return value.reshape(*value.shape[:-2], -1)

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
    ) -> UniformGraphRuntime:
        def _move(tensor: torch.Tensor | None) -> torch.Tensor | None:
            if tensor is None:
                return None
            if tensor.dtype in (torch.int32, torch.int64):
                return tensor.to(device=device, non_blocking=non_blocking)
            return tensor.to(device=device, dtype=dtype, non_blocking=non_blocking)

        return replace(
            self,
            time=_move(self.time),
            node_state=_move(self.node_state),
            edge_index=_move(self.edge_index),
            control=_move(self.control),
            target=_move(self.target),
            params=_move(self.params),
            edge_weight=_move(self.edge_weight),
            edge_attr=_move(self.edge_attr),
        )


@dataclass(frozen=True)
class RaggedGraphRuntime:
    time: torch.Tensor
    node_state: torch.Tensor
    edge_index: torch.Tensor
    step_lengths: tuple[int, ...]
    valid_mask: torch.Tensor
    control: torch.Tensor | None = None
    target: torch.Tensor | None = None
    params: torch.Tensor | None = None
    edge_weight: torch.Tensor | None = None
    edge_attr: torch.Tensor | None = None
    meta: tuple[dict[str, Any], ...] = ()

    _has_graph: bool = True
    is_graph: bool = True
    is_uniform_length: bool = False

    @property
    def is_fixed_topology(self) -> bool:
        return self.edge_index.ndim == 3

    @property
    def batch_size(self) -> int:
        return int(self.node_state.shape[0])

    @property
    def n_steps(self) -> int:
        return int(self.node_state.shape[1])

    @property
    def n_nodes(self) -> int:
        return int(self.node_state.shape[2])

    @property
    def x(self) -> torch.Tensor:
        return self.node_state.reshape(self.batch_size, self.n_steps, -1)

    @property
    def t(self) -> torch.Tensor:
        return self.time

    @property
    def u(self) -> torch.Tensor | None:
        if self.control is None:
            return None
        return self.control.reshape(self.batch_size, self.n_steps, -1)

    @property
    def y(self) -> torch.Tensor | None:
        if self.target is None:
            return None
        return self.target.reshape(self.batch_size, self.n_steps, -1)

    @property
    def p(self) -> torch.Tensor | None:
        return self.params

    @property
    def xg(self) -> torch.Tensor:
        return self.node_state

    @property
    def ug(self) -> torch.Tensor | None:
        return self.control

    @property
    def yg(self) -> torch.Tensor | None:
        return self.target

    @property
    def ei(self) -> torch.Tensor:
        return self.edge_index

    @property
    def ew(self) -> torch.Tensor | None:
        return self.edge_weight

    @property
    def ea(self) -> torch.Tensor | None:
        return self.edge_attr

    def initial_state(self) -> torch.Tensor:
        return self.x[:, 0, :]

    def time_payload(self) -> torch.Tensor:
        return self.time

    def control_payload(self) -> torch.Tensor | None:
        return self.control

    def params_payload(self) -> torch.Tensor | None:
        return self.params

    def get_step(self, step: int) -> GraphRuntimeStep:
        return GraphRuntimeStep(
            time=self.time[:, step] if self.time is not None else None,
            node_state=self.node_state[:, step],
            control=self.control[:, step] if self.control is not None else None,
            target=self.target[:, step] if self.target is not None else None,
            params=self.params,
            edge_index=_graph_step_tensor(self.edge_index, step=step, time_ndim=4),
            edge_weight=_graph_step_tensor(self.edge_weight, step=step, time_ndim=3),
            edge_attr=_graph_step_tensor(self.edge_attr, step=step, time_ndim=4),
            valid_mask=self.valid_mask[:, step],
            meta=self.meta,
        )

    def truncate(self, num_steps: int) -> RaggedGraphRuntime:
        step_lengths = tuple(min(length, num_steps) for length in self.step_lengths)
        return replace(
            self,
            time=self.time[:, :num_steps],
            node_state=self.node_state[:, :num_steps],
            edge_index=_graph_truncate_tensor(self.edge_index, num_steps=num_steps, time_ndim=4),
            step_lengths=step_lengths,
            valid_mask=self.valid_mask[:, :num_steps],
            control=self.control[:, :num_steps] if self.control is not None else None,
            target=self.target[:, :num_steps] if self.target is not None else None,
            edge_weight=_graph_truncate_tensor(self.edge_weight, num_steps=num_steps, time_ndim=3),
            edge_attr=_graph_truncate_tensor(self.edge_attr, num_steps=num_steps, time_ndim=4),
        )

    def window(self, window: int, stride: int) -> GraphRuntime:
        if window <= 0:
            raise ValueError("window must be positive")
        if stride <= 0:
            raise ValueError("stride must be positive")
        items = []
        for runtime in self.iter_series():
            if runtime.n_steps < window:
                continue
            items.append(runtime.window(window, stride))
        if not items:
            raise ValueError("window produced no graph trajectories")
        if len(items) == 1:
            return items[0]
        time = torch.cat([item.time for item in items], dim=0)
        node_state = torch.cat([item.node_state for item in items], dim=0)
        control = None
        if any(item.control is not None for item in items):
            ref = next(item.control for item in items if item.control is not None)
            control = torch.cat(
                [
                    item.control
                    if item.control is not None
                    else torch.zeros(
                        (item.batch_size, item.n_steps, *ref.shape[2:]),
                        dtype=ref.dtype,
                        device=ref.device,
                    )
                    for item in items
                ],
                dim=0,
            )
        target = None
        if any(item.target is not None for item in items):
            ref = next(item.target for item in items if item.target is not None)
            target = torch.cat(
                [
                    item.target
                    if item.target is not None
                    else torch.zeros(
                        (item.batch_size, item.n_steps, *ref.shape[2:]),
                        dtype=ref.dtype,
                        device=ref.device,
                    )
                    for item in items
                ],
                dim=0,
            )
        params = None
        if any(item.params is not None for item in items):
            ref = next(item.params for item in items if item.params is not None)
            params = torch.cat(
                [
                    item.params
                    if item.params is not None
                    else torch.zeros(
                        (item.batch_size, *ref.shape[1:]), dtype=ref.dtype, device=ref.device
                    )
                    for item in items
                ],
                dim=0,
            )
        edge_index = torch.cat([item.edge_index for item in items], dim=0)
        edge_weight = None
        if any(item.edge_weight is not None for item in items):
            ref = next(item.edge_weight for item in items if item.edge_weight is not None)
            edge_weight = torch.cat(
                [
                    item.edge_weight
                    if item.edge_weight is not None
                    else torch.zeros(
                        (item.batch_size, *ref.shape[1:]), dtype=ref.dtype, device=ref.device
                    )
                    for item in items
                ],
                dim=0,
            )
        edge_attr = None
        if any(item.edge_attr is not None for item in items):
            ref = next(item.edge_attr for item in items if item.edge_attr is not None)
            edge_attr = torch.cat(
                [
                    item.edge_attr
                    if item.edge_attr is not None
                    else torch.zeros(
                        (item.batch_size, *ref.shape[1:]), dtype=ref.dtype, device=ref.device
                    )
                    for item in items
                ],
                dim=0,
            )
        meta = tuple(meta for item in items for meta in item.meta)
        return UniformGraphRuntime(
            time=time,
            node_state=node_state,
            edge_index=edge_index,
            control=control,
            target=target,
            params=params,
            edge_weight=edge_weight,
            edge_attr=edge_attr,
            meta=meta,
        )

    def iter_series(self):
        for idx, length in enumerate(self.step_lengths):
            yield UniformGraphRuntime(
                time=self.time[idx : idx + 1, :length],
                node_state=self.node_state[idx : idx + 1, :length],
                edge_index=_require_tensor(
                    "edge_index",
                    _graph_batch_tensor(self.edge_index, index=idx, length=length, time_ndim=4),
                ),
                control=self.control[idx : idx + 1, :length] if self.control is not None else None,
                target=self.target[idx : idx + 1, :length] if self.target is not None else None,
                params=self.params[idx : idx + 1] if self.params is not None else None,
                edge_weight=_graph_batch_tensor(
                    self.edge_weight, index=idx, length=length, time_ndim=3
                ),
                edge_attr=_graph_batch_tensor(
                    self.edge_attr, index=idx, length=length, time_ndim=4
                ),
                meta=(dict(self.meta[idx]),) if self.meta else (),
            )

    def g(self, value: torch.Tensor) -> torch.Tensor:
        return value.reshape(*value.shape[:-1], self.n_nodes, -1)

    def G(self, value: torch.Tensor) -> torch.Tensor:
        return value.reshape(*value.shape[:-2], -1)

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
    ) -> RaggedGraphRuntime:
        def _move(tensor: torch.Tensor | None) -> torch.Tensor | None:
            if tensor is None:
                return None
            if tensor.dtype in (torch.int32, torch.int64):
                return tensor.to(device=device, non_blocking=non_blocking)
            return tensor.to(device=device, dtype=dtype, non_blocking=non_blocking)

        return replace(
            self,
            time=_move(self.time),
            node_state=_move(self.node_state),
            edge_index=_move(self.edge_index),
            valid_mask=_move(self.valid_mask),
            control=_move(self.control),
            target=_move(self.target),
            params=_move(self.params),
            edge_weight=_move(self.edge_weight),
            edge_attr=_move(self.edge_attr),
        )


type RegularRuntime = EmptyRegularRuntime | UniformRegularRuntime | RaggedRegularRuntime
type GraphRuntime = UniformGraphRuntime | RaggedGraphRuntime
type TypedRuntime = RegularRuntime | GraphRuntime
type TypedRuntimeStep = RegularRuntimeStep | GraphRuntimeStep


def to_padded_regular_runtime(batch: RegularSeriesBatch) -> RegularRuntime:
    if len(batch) == 0:
        return EmptyRegularRuntime()
    if isinstance(batch, UniformLengthRegularSeriesBatch):
        return UniformRegularRuntime(
            time=batch.stacked_time(),
            state=batch.stacked_state(),
            control=_stack_optional(item.control for item in batch),
            target=_stack_optional(item.target for item in batch),
            params=_stack_optional(item.params for item in batch),
            meta=tuple(dict(item.meta) for item in batch),
        )
    if not isinstance(batch, RaggedRegularSeriesBatch):
        batch = RaggedRegularSeriesBatch(tuple(batch))
    time, state, control, target, params, mask = _pad_regular_batch(batch)
    return RaggedRegularRuntime(
        time=time,
        state=state,
        step_lengths=batch.step_lengths,
        valid_mask=mask,
        control=control,
        target=target,
        params=params,
        meta=tuple(dict(item.meta) for item in batch),
    )


def to_padded_graph_runtime(batch: GraphSeriesBatch) -> GraphRuntime:
    if len(batch) == 0:
        raise ValueError("Graph runtime requires at least one series")

    items = tuple(batch)
    node_counts = {int(item.node_state.shape[1]) for item in items}
    if len(node_counts) != 1:
        raise ValueError("Native graph runtime requires a consistent node count across the batch.")

    if isinstance(batch, UniformLengthGraphSeriesBatch):
        n_steps = batch.step_lengths[0]
        fixed_topology = all(item.fixed_topology for item in items)
        if fixed_topology:
            edge_index = _stack_fixed_edge_index(items)
            edge_weight = _stack_fixed_graph_field(
                tuple(_graph_tensor_field(item.edge_weight) for item in items),
                max_steps=n_steps,
                time_ndim=2,
            )
            edge_attr = _stack_fixed_graph_field(
                tuple(_graph_tensor_field(item.edge_attr) for item in items),
                max_steps=n_steps,
                time_ndim=3,
            )
        elif all(
            isinstance(item.edge_index, torch.Tensor) and item.edge_index.ndim == 3
            for item in items
        ):
            edge_index = _stack_time_varying_edge_index(items, max_steps=n_steps)
            edge_weight = _stack_time_varying_graph_field(
                tuple(_graph_tensor_field(item.edge_weight) for item in items),
                max_steps=n_steps,
                time_ndim=2,
            )
            edge_attr = _stack_time_varying_graph_field(
                tuple(_graph_tensor_field(item.edge_attr) for item in items),
                max_steps=n_steps,
                time_ndim=3,
            )
        else:
            edge_index, edge_weight, edge_attr = _stack_graph_steps(items, max_steps=n_steps)
        return UniformGraphRuntime(
            time=batch.stacked_time(),
            node_state=batch.stacked_node_state(),
            edge_index=edge_index,
            control=_stack_optional(item.control for item in batch),
            target=_stack_optional(item.target for item in batch),
            params=_stack_optional(item.params for item in batch),
            edge_weight=edge_weight,
            edge_attr=edge_attr,
            meta=tuple(dict(item.meta) for item in batch),
        )

    if not isinstance(batch, RaggedGraphSeriesBatch):
        batch = RaggedGraphSeriesBatch(tuple(batch))

    max_steps = max(batch.step_lengths)
    ref = items[0]
    time = torch.zeros((len(items), max_steps), dtype=ref.time.dtype, device=ref.time.device)
    node_state = torch.zeros(
        (len(items), max_steps, *ref.node_state.shape[1:]),
        dtype=ref.node_state.dtype,
        device=ref.node_state.device,
    )
    mask = torch.zeros((len(items), max_steps), dtype=torch.bool, device=ref.node_state.device)
    for idx, item in enumerate(items):
        steps = item.time.shape[0]
        time[idx, :steps] = item.time
        node_state[idx, :steps] = item.node_state
        mask[idx, :steps] = True

    fixed_topology = all(item.fixed_topology for item in items)
    if fixed_topology:
        edge_index = _stack_fixed_edge_index(items)
        edge_weight = _stack_fixed_graph_field(
            tuple(_graph_tensor_field(item.edge_weight) for item in items),
            max_steps=max_steps,
            time_ndim=2,
        )
        edge_attr = _stack_fixed_graph_field(
            tuple(_graph_tensor_field(item.edge_attr) for item in items),
            max_steps=max_steps,
            time_ndim=3,
        )
    elif all(
        isinstance(item.edge_index, torch.Tensor) and item.edge_index.ndim == 3 for item in items
    ):
        edge_index = _pad_time_varying_edge_index(items, max_steps=max_steps)
        edge_weight = _stack_time_varying_graph_field(
            tuple(_graph_tensor_field(item.edge_weight) for item in items),
            max_steps=max_steps,
            time_ndim=2,
        )
        edge_attr = _stack_time_varying_graph_field(
            tuple(_graph_tensor_field(item.edge_attr) for item in items),
            max_steps=max_steps,
            time_ndim=3,
        )
    else:
        edge_index, edge_weight, edge_attr = _stack_graph_steps(items, max_steps=max_steps)
    control = _pad_optional(
        tuple(item.control for item in items),
        max_steps=max_steps,
        pad_shape=ref.control.shape[1:] if ref.control is not None else (),
        dtype=ref.control.dtype if ref.control is not None else ref.node_state.dtype,
        device=ref.node_state.device,
    )
    target = _pad_optional(
        tuple(item.target for item in items),
        max_steps=max_steps,
        pad_shape=ref.target.shape[1:] if ref.target is not None else (),
        dtype=ref.target.dtype if ref.target is not None else ref.node_state.dtype,
        device=ref.node_state.device,
    )
    return RaggedGraphRuntime(
        time=time,
        node_state=node_state,
        edge_index=edge_index,
        step_lengths=batch.step_lengths,
        valid_mask=mask,
        control=control,
        target=target,
        params=_stack_optional(item.params for item in items),
        edge_weight=edge_weight,
        edge_attr=edge_attr,
        meta=tuple(dict(item.meta) for item in items),
    )


def runtime_from_series(series: RegularSeries | GraphSeries) -> TypedRuntime:
    if isinstance(series, RegularSeries):
        return to_padded_regular_runtime(RegularSeriesBatch.collate([series]))
    if isinstance(series, GraphSeries):
        return to_padded_graph_runtime(GraphSeriesBatch.collate([series]))
    raise TypeError(f"Unsupported series type: {type(series)!r}")
