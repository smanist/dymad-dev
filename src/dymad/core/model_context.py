"""Typed model-runtime context adapters built from typed series objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import torch

from dymad.core.graph_series import (
    FixedGraphSeries,
    GraphSeries,
    GraphSeriesBatch,
    VariableEdgeGraphSeries,
)
from dymad.core.runtime import (
    EmptyRegularRuntime,
    GraphRuntime,
    RegularRuntime,
    TypedRuntime,
    UniformGraphRuntime,
    UniformRegularRuntime,
    to_padded_graph_runtime,
    to_padded_regular_runtime,
)
from dymad.core.series import (
    RegularSeries,
    RegularSeriesBatch,
)


@dataclass(frozen=True)
class RegularModelContext:
    """Typed regular runtime context for prediction/model helper entrypoints."""

    batch: RegularSeriesBatch

    @classmethod
    def from_series(cls, series: RegularSeries) -> RegularModelContext:
        return cls(RegularSeriesBatch.collate([series]))

    @classmethod
    def from_batch(cls, batch: RegularSeriesBatch) -> RegularModelContext:
        return cls(batch)

    @property
    def batch_size(self) -> int:
        return len(self.batch)

    @property
    def n_steps(self) -> tuple[int, ...]:
        return tuple(int(item.time.shape[0]) for item in self.batch)

    def initial_state_tensor(self, *, squeeze_single: bool = False) -> torch.Tensor:
        state = self.to_runtime().initial_state()
        if state is None:
            raise ValueError("RegularModelContext has no initial state for an empty batch.")
        if squeeze_single and state.shape[0] == 1:
            return state[0]
        return state

    def to_runtime(self) -> RegularRuntime:
        return to_padded_regular_runtime(self.batch)


@dataclass(frozen=True)
class GraphModelContext:
    """Typed graph runtime context for prediction/model helper entrypoints."""

    batch: GraphSeriesBatch

    @classmethod
    def from_series(cls, series: GraphSeries) -> GraphModelContext:
        return cls(GraphSeriesBatch.collate([series]))

    @classmethod
    def from_batch(cls, batch: GraphSeriesBatch) -> GraphModelContext:
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
        state = self.to_runtime().initial_state()
        if squeeze_single and state.shape[0] == 1:
            return state[0]
        return state

    def to_runtime(self) -> GraphRuntime:
        return to_padded_graph_runtime(self.batch)


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


type ModelRuntimePayload = "TypedRuntime | RegularModelContext | GraphModelContext"


def _expand_regular_context_for_prediction(
    context: RegularModelContext,
    *,
    batch_size: int,
    is_batch: bool,
) -> RegularModelContext:
    if not is_batch:
        if context.batch_size != 1:
            raise ValueError(f"Single mode: ws batch size must be 1. Got ws: {context.batch_size}")
        return context

    if context.batch_size == 1 and batch_size > 1:
        return RegularModelContext.from_batch(
            RegularSeriesBatch.collate(context.batch[0] for _ in range(batch_size))
        )
    if context.batch_size == batch_size:
        return context
    raise ValueError(
        f"Batch mode: ws batch size must be 1 or match x0. Got ws: {context.batch_size}, x0: {batch_size}"
    )


def _expand_graph_context_for_prediction(
    context: GraphModelContext,
    *,
    batch_size: int,
    is_batch: bool,
) -> GraphModelContext:
    if not is_batch:
        if context.batch_size != 1:
            raise ValueError(f"Single mode: ws batch size must be 1. Got ws: {context.batch_size}")
        return context

    if context.batch_size == 1 and batch_size > 1:
        return GraphModelContext.from_batch(
            GraphSeriesBatch.collate(context.batch[0] for _ in range(batch_size))
        )
    if context.batch_size == batch_size:
        return context
    raise ValueError(
        f"Batch mode: ws batch size must be 1 or match x0. Got ws: {context.batch_size}, x0: {batch_size}"
    )


def _regular_series_from_runtime(payload: UniformRegularRuntime) -> RegularSeries:
    return RegularSeries(
        time=payload.time[0],
        state=payload.state[0],
        control=payload.control[0] if payload.control is not None else None,
        target=payload.target[0] if payload.target is not None else None,
        params=payload.params[0] if payload.params is not None else None,
        meta=dict(payload.meta[0]) if payload.meta else {},
    )


def _graph_series_from_runtime(payload: UniformGraphRuntime) -> GraphSeries:
    if payload.edge_index.ndim == 3:
        edge_index: torch.Tensor | tuple[torch.Tensor, ...] = payload.edge_index[0].transpose(0, 1)
    else:
        edge_index_steps = payload.edge_index[0].transpose(1, 2)
        if all(torch.equal(edge_index_steps[0], step) for step in edge_index_steps[1:]):
            edge_index = edge_index_steps[0]
        else:
            edge_index = edge_index_steps

    edge_weight = None
    if payload.edge_weight is not None:
        if payload.edge_weight.ndim == 2:
            edge_weight = payload.edge_weight[0]
        else:
            edge_weight = payload.edge_weight[0]

    edge_attr = None
    if payload.edge_attr is not None:
        if payload.edge_attr.ndim == 3:
            edge_attr = payload.edge_attr[0]
        else:
            edge_attr = payload.edge_attr[0]

    cls = (
        FixedGraphSeries
        if isinstance(edge_index, torch.Tensor) and edge_index.ndim == 2
        else VariableEdgeGraphSeries
    )
    return cls(
        time=payload.time[0],
        node_state=payload.node_state[0],
        edge_index=edge_index,
        control=payload.control[0] if payload.control is not None else None,
        target=payload.target[0] if payload.target is not None else None,
        params=payload.params[0] if payload.params is not None else None,
        edge_weight=edge_weight,
        edge_attr=edge_attr,
        meta=dict(payload.meta[0]) if payload.meta else {},
    )


def _context_from_runtime(payload: TypedRuntime) -> RegularModelContext | GraphModelContext:
    if isinstance(payload, EmptyRegularRuntime):
        return RegularModelContext.from_batch(RegularSeriesBatch.collate([]))
    items = list(payload.iter_series())
    if not items:
        return RegularModelContext.from_batch(RegularSeriesBatch.collate([]))
    if payload.is_graph:
        graph_items = tuple(cast(UniformGraphRuntime, item) for item in items)
        return GraphModelContext.from_batch(
            GraphSeriesBatch.collate(_graph_series_from_runtime(item) for item in graph_items)
        )
    regular_items = tuple(cast(UniformRegularRuntime, item) for item in items)
    return RegularModelContext.from_batch(
        RegularSeriesBatch.collate(_regular_series_from_runtime(item) for item in regular_items)
    )


def _ensure_time_tensor(
    t: torch.Tensor | None,
    *,
    batch_size: int,
    n_steps: int,
    device: torch.device,
) -> torch.Tensor:
    if t is None:
        return torch.arange(n_steps, device=device, dtype=torch.get_default_dtype()).expand(
            batch_size, -1
        )
    if t.ndim == 0:
        return t.reshape(1, 1).expand(batch_size, n_steps)
    if t.ndim == 1:
        if t.shape[0] == n_steps:
            return t.reshape(1, -1).expand(batch_size, -1)
        if t.shape[0] == batch_size and n_steps == 1:
            return t.reshape(batch_size, 1)
        raise ValueError(
            f"Unsupported time shape {tuple(t.shape)} for batch_size={batch_size}, n_steps={n_steps}"
        )
    if t.ndim == 2:
        if t.shape == (batch_size, n_steps):
            return t
        if t.shape == (1, n_steps):
            return t.expand(batch_size, -1)
        if t.shape == (batch_size, 1):
            return t.expand(-1, n_steps)
        raise ValueError(
            f"Unsupported time shape {tuple(t.shape)} for batch_size={batch_size}, n_steps={n_steps}"
        )
    raise ValueError(f"Unsupported time shape {tuple(t.shape)}")


def _split_nested_payload(payload: Any) -> list[torch.Tensor] | None:
    if payload is None:
        return None
    if isinstance(payload, list):
        return [torch.as_tensor(item) for item in payload]
    if isinstance(payload, tuple):
        values, offsets = payload
        offsets = torch.as_tensor(offsets, dtype=torch.int64, device=values.device)
        return [values[offsets[idx] : offsets[idx + 1]] for idx in range(offsets.numel() - 1)]
    if getattr(payload, "is_nested", False):
        return [item for item in payload.unbind()]

    tensor = torch.as_tensor(payload)
    if tensor.ndim in (1, 2):
        return [tensor]
    if tensor.ndim == 3:
        return [item for item in tensor.unbind(0)]
    raise ValueError(f"Unsupported nested payload shape {tuple(tensor.shape)}")


def _infer_graph_nodes(edge_index: Any) -> int:
    tensor = torch.as_tensor(edge_index)
    if tensor.numel() == 0:
        raise ValueError("Cannot infer graph node count from an empty edge_index payload.")
    return int(tensor.max().item()) + 1


def materialize_model_base_forward_payload(
    *,
    t: torch.Tensor | None,
    x: torch.Tensor | None,
    u: torch.Tensor | None,
    p: torch.Tensor | None,
    ei: tuple[torch.Tensor, torch.Tensor] | None,
    ew: tuple[torch.Tensor, torch.Tensor] | None,
    ea: tuple[torch.Tensor, torch.Tensor] | None,
) -> RegularModelContext | GraphModelContext:
    """Build a typed model-runtime context from raw model-base forward inputs."""

    if x is None:
        raise ValueError("model_base.forward requires `x` to materialize runtime payload.")

    if ei is None:
        from dymad.io.series_adapter import SeriesAdapter

        if x.ndim == 1:
            state = x.reshape(1, 1, -1)
        elif x.ndim == 2:
            state = x.unsqueeze(1)
        elif x.ndim == 3:
            state = x
        else:
            raise ValueError(f"Unsupported regular forward input shape for x: {tuple(x.shape)}")
        batch_size, n_steps = state.shape[:2]
        time = _ensure_time_tensor(t, batch_size=batch_size, n_steps=n_steps, device=state.device)
        if u is None:
            control = None
        elif u.ndim == 1:
            control = u.reshape(1, 1, -1).expand(batch_size, n_steps, -1)
        elif u.ndim == 2:
            control = (
                u.unsqueeze(1)
                if u.shape[0] == batch_size
                else u.reshape(1, n_steps, -1).expand(batch_size, -1, -1)
            )
        elif u.ndim == 3:
            control = u
        else:
            raise ValueError(f"Unsupported regular forward input shape for u: {tuple(u.shape)}")
        if p is None:
            params = None
        elif p.ndim == 1:
            params = p.unsqueeze(0).expand(batch_size, -1)
        elif p.ndim == 2:
            params = p
        else:
            raise ValueError(f"Unsupported regular forward input shape for p: {tuple(p.shape)}")
        return RegularModelContext.from_batch(
            RegularSeriesBatch.collate(
                RegularSeries(
                    time=time[idx],
                    state=state[idx],
                    control=None if control is None else control[idx],
                    params=None if params is None else params[idx],
                )
                for idx in range(batch_size)
            )
        )

    from dymad.io.series_adapter import SeriesAdapter

    if x.ndim == 1:
        flat_state = x.reshape(1, 1, -1)
    elif x.ndim == 2:
        flat_state = x.unsqueeze(1)
    elif x.ndim == 3:
        flat_state = x
    else:
        raise ValueError(f"Unsupported graph forward input shape for x: {tuple(x.shape)}")

    edge_index_items = _split_nested_payload(ei)
    edge_weight_items = _split_nested_payload(ew)
    edge_attr_items = _split_nested_payload(ea)
    if edge_index_items is None or not edge_index_items:
        raise ValueError("Graph forward payload requires edge_index data.")

    batch_size, n_steps = flat_state.shape[:2]
    if len(edge_index_items) == 1 and batch_size > 1:
        edge_index_items = edge_index_items * batch_size
    if edge_weight_items is not None and len(edge_weight_items) == 1 and batch_size > 1:
        edge_weight_items = edge_weight_items * batch_size
    if edge_attr_items is not None and len(edge_attr_items) == 1 and batch_size > 1:
        edge_attr_items = edge_attr_items * batch_size
    if len(edge_index_items) != batch_size:
        raise ValueError("Graph edge_index payload batch size must match x.")

    time = _ensure_time_tensor(t, batch_size=batch_size, n_steps=n_steps, device=flat_state.device)
    items = []
    for idx in range(batch_size):
        edge_index_item = edge_index_items[idx]
        n_nodes = _infer_graph_nodes(edge_index_item)
        node_state = flat_state[idx].reshape(n_steps, n_nodes, -1)
        control = None
        if u is not None:
            if u.ndim == 1:
                control = u.reshape(1, n_nodes, -1).expand(n_steps, -1, -1)
            elif u.ndim == 2:
                control = u.reshape(n_steps, n_nodes, -1)
            elif u.ndim == 3:
                control = u[idx].reshape(n_steps, n_nodes, -1)
            else:
                raise ValueError(f"Unsupported graph forward input shape for u: {tuple(u.shape)}")
        params = None
        if p is not None:
            params = p[idx] if p.ndim > 1 else p
        edge_weight_item = None if edge_weight_items is None else edge_weight_items[idx]
        edge_attr_item = None if edge_attr_items is None else edge_attr_items[idx]
        items.append(
            SeriesAdapter.from_graph_arrays(
                time=time[idx],
                node_state=node_state,
                edge_index=edge_index_item,
                control=control,
                params=params,
                edge_weight=edge_weight_item,
                edge_attr=edge_attr_item,
                dtype=node_state.dtype,
                device=node_state.device,
            )
        )
    return GraphModelContext.from_batch(GraphSeriesBatch.collate(items))


def materialize_prediction_runtime(
    payload: ModelRuntimePayload | None,
    *,
    batch_size: int,
    is_batch: bool,
) -> TypedRuntime:
    """Materialize a prediction runtime payload through the typed-context boundary."""

    if payload is None:
        return EmptyRegularRuntime(batch_size=batch_size if is_batch else 1)

    if isinstance(payload, RegularModelContext):
        return _expand_regular_context_for_prediction(
            payload,
            batch_size=batch_size,
            is_batch=is_batch,
        ).to_runtime()

    if isinstance(payload, GraphModelContext):
        return _expand_graph_context_for_prediction(
            payload,
            batch_size=batch_size,
            is_batch=is_batch,
        ).to_runtime()

    if hasattr(payload, "batch_size") and hasattr(payload, "is_graph"):
        if not is_batch:
            if payload.batch_size != 1:
                raise ValueError(
                    f"Single mode: ws batch size must be 1. Got ws: {payload.batch_size}"
                )
            return payload
        if payload.batch_size == batch_size:
            return payload
        if payload.batch_size == 1:
            context = _context_from_runtime(payload)
            if isinstance(context, RegularModelContext):
                return _expand_regular_context_for_prediction(
                    context,
                    batch_size=batch_size,
                    is_batch=True,
                ).to_runtime()
            return _expand_graph_context_for_prediction(
                context,
                batch_size=batch_size,
                is_batch=True,
            ).to_runtime()
        if payload.batch_size != batch_size:
            raise ValueError(
                f"Batch mode: ws batch size must be 1 or match x0. Got ws: {payload.batch_size}, x0: {batch_size}"
            )

    raise TypeError(f"Unsupported runtime payload type: {type(payload)!r}")
