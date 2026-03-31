"""Compatibility adapters between typed series objects and legacy LegacyRuntimeBatch."""

from __future__ import annotations

from typing import Any

import torch

from dymad.core.graph_series import FixedGraphSeries, GraphSeries, VariableEdgeGraphSeries
from dymad.core.series import RegularSeries
from dymad.io.legacy_runtime import LegacyRuntimeBatch


class SeriesAdapter:
    """Build typed series objects from legacy-compatible payloads."""

    @staticmethod
    def from_regular_arrays(
        time: Any,
        state: Any,
        *,
        control: Any = None,
        target: Any = None,
        params: Any = None,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> RegularSeries:
        return RegularSeries(
            time=torch.as_tensor(time, dtype=dtype, device=device),
            state=torch.as_tensor(state, dtype=dtype, device=device),
            control=torch.as_tensor(control, dtype=dtype, device=device) if control is not None else None,
            target=torch.as_tensor(target, dtype=dtype, device=device) if target is not None else None,
            params=torch.as_tensor(params, dtype=dtype, device=device) if params is not None else None,
            meta=dict(meta or {}),
        )

    @staticmethod
    def from_graph_arrays(
        time: Any,
        node_state: Any,
        *,
        edge_index: Any,
        control: Any = None,
        target: Any = None,
        params: Any = None,
        edge_weight: Any = None,
        edge_attr: Any = None,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> GraphSeries:
        time_tensor = torch.as_tensor(time, dtype=dtype, device=device)
        node_state_tensor = torch.as_tensor(node_state, dtype=dtype, device=device)
        control_tensor = torch.as_tensor(control, dtype=dtype, device=device) if control is not None else None
        target_tensor = torch.as_tensor(target, dtype=dtype, device=device) if target is not None else None
        params_tensor = torch.as_tensor(params, dtype=dtype, device=device) if params is not None else None

        edge_index_payload = SeriesAdapter._graph_edge_index_payload(edge_index, device=device)
        edge_weight_payload = SeriesAdapter._graph_optional_payload(edge_weight, dtype=dtype, device=device)
        edge_attr_payload = SeriesAdapter._graph_optional_payload(edge_attr, dtype=dtype, device=device)

        cls = FixedGraphSeries if isinstance(edge_index_payload, torch.Tensor) else VariableEdgeGraphSeries
        return cls(
            time=time_tensor,
            node_state=node_state_tensor,
            edge_index=edge_index_payload,
            control=control_tensor,
            target=target_tensor,
            params=params_tensor,
            edge_weight=edge_weight_payload,
            edge_attr=edge_attr_payload,
            meta=dict(meta or {}),
        )

    @staticmethod
    def from_dyndata(data: LegacyRuntimeBatch) -> RegularSeries | GraphSeries:
        if data._has_graph:
            return SeriesAdapter.from_graph_dyndata(data)

        return RegularSeries(
            time=SeriesAdapter._squeeze_batch(data.t),
            state=SeriesAdapter._squeeze_batch(data.x),
            control=SeriesAdapter._squeeze_batch(data.u),
            target=SeriesAdapter._squeeze_batch(data.y),
            params=SeriesAdapter._squeeze_batch(data.p),
            meta=dict(data.meta[0]) if data.meta else {"source": "LegacyRuntimeBatch"},
        )

    @staticmethod
    def from_graph_dyndata(data: LegacyRuntimeBatch) -> GraphSeries:
        if not data._has_graph:
            raise ValueError("non-graph LegacyRuntimeBatch is not supported by the graph-series adapter")

        time = SeriesAdapter._squeeze_batch(data.t)
        n_steps = int(time.shape[0])
        n_nodes = int(data.n_nodes)

        node_state = SeriesAdapter._reshape_graph_payload(SeriesAdapter._squeeze_batch(data.x), n_nodes)
        control = SeriesAdapter._reshape_graph_payload(SeriesAdapter._squeeze_batch(data.u), n_nodes)
        target = SeriesAdapter._reshape_graph_payload(SeriesAdapter._squeeze_batch(data.y), n_nodes)
        params = data.p[0].reshape(n_nodes, -1) if data.p is not None else None

        edge_index_steps = tuple(step.transpose(0, 1) for step in data.ei.unbind())
        edge_weight_steps = tuple(step for step in data.ew.unbind()) if data.ew is not None else None
        edge_attr_steps = tuple(step for step in data.ea.unbind()) if data.ea is not None else None
        if len(edge_index_steps) != n_steps:
            raise ValueError("Graph LegacyRuntimeBatch edge_index steps must align with time")

        edge_index: torch.Tensor | tuple[torch.Tensor, ...]
        if all(torch.equal(edge_index_steps[0], step) for step in edge_index_steps[1:]):
            edge_index = edge_index_steps[0]
        else:
            edge_index = edge_index_steps

        return SeriesAdapter.from_graph_arrays(
            time=time,
            node_state=node_state,
            edge_index=edge_index,
            control=control,
            target=target,
            params=params,
            edge_weight=edge_weight_steps,
            edge_attr=edge_attr_steps,
            dtype=node_state.dtype,
            device=node_state.device,
            meta=dict(data.meta[0]) if data.meta else {"source": "LegacyRuntimeBatch"},
        )

    @staticmethod
    def _squeeze_batch(tensor: torch.Tensor | None) -> torch.Tensor | None:
        if tensor is None:
            return None
        if tensor.ndim == 0:
            return tensor
        if tensor.ndim >= 1 and tensor.shape[0] == 1:
            return tensor[0]
        raise ValueError(
            "LegacyRuntimeBatch batch_size > 1 is not supported by SeriesAdapter.from_dyndata; "
            "adapt one trajectory at a time"
        )

    @staticmethod
    def _reshape_graph_payload(tensor: torch.Tensor | None, n_nodes: int) -> torch.Tensor | None:
        if tensor is None:
            return None
        return tensor.reshape(tensor.shape[0], n_nodes, -1)

    @staticmethod
    def _graph_edge_index_payload(edge_index: Any, *, device: torch.device | str | None):
        if isinstance(edge_index, (list, tuple)):
            steps = tuple(SeriesAdapter._to_edge_index_tensor(item, device=device) for item in edge_index)
            if steps and all(torch.equal(steps[0], step) for step in steps[1:]):
                return steps[0]
            return steps
        tensor = torch.as_tensor(edge_index, dtype=torch.long, device=device)
        if tensor.ndim == 3:
            return SeriesAdapter._graph_edge_index_payload(tuple(tensor.unbind()), device=device)
        return SeriesAdapter._to_edge_index_tensor(tensor, device=device)

    @staticmethod
    def _to_edge_index_tensor(edge_index: Any, *, device: torch.device | str | None) -> torch.Tensor:
        tensor = torch.as_tensor(edge_index, dtype=torch.long, device=device)
        if tensor.ndim != 2:
            raise ValueError("edge_index tensors must have exactly two dimensions")
        if tensor.shape[0] == 2:
            return tensor
        if tensor.shape[1] == 2:
            return tensor.transpose(0, 1)
        raise ValueError("edge_index tensors must have shape [2, n_edges] or [n_edges, 2]")

    @staticmethod
    def _graph_optional_payload(
        payload: Any,
        *,
        dtype: torch.dtype | None,
        device: torch.device | str | None,
    ):
        if payload is None:
            return None
        if isinstance(payload, (list, tuple)):
            return tuple(torch.as_tensor(item, dtype=dtype, device=device) for item in payload)
        return torch.as_tensor(payload, dtype=dtype, device=device)


def regular_series_to_legacy_runtime(series: RegularSeries) -> LegacyRuntimeBatch:
    """Temporary deletion-stage bridge from typed regular series to LegacyRuntimeBatch."""

    return LegacyRuntimeBatch(
        t=series.time,
        x=series.state,
        y=series.target,
        u=series.control,
        p=series.params,
        meta=[dict(series.meta)] if series.meta else [],
    )


def graph_series_to_legacy_runtime(series: GraphSeries) -> LegacyRuntimeBatch:
    """Temporary deletion-stage bridge from typed graph series to LegacyRuntimeBatch."""

    n_steps = int(series.time.shape[0])
    return LegacyRuntimeBatch(
        t=series.time,
        x=series.node_state.reshape(n_steps, -1),
        y=series.target.reshape(n_steps, -1) if series.target is not None else None,
        u=series.control.reshape(n_steps, -1) if series.control is not None else None,
        p=series.params.reshape(-1) if series.params is not None else None,
        ei=_graph_edge_steps(series, n_steps),
        ew=_graph_optional_steps(series.edge_weight, n_steps),
        ea=_graph_optional_steps(series.edge_attr, n_steps),
        meta=[dict(series.meta)] if series.meta else [],
    )


def _graph_edge_steps(series: GraphSeries, n_steps: int) -> list[torch.Tensor]:
    if isinstance(series.edge_index, torch.Tensor):
        return [series.edge_index.transpose(0, 1) for _ in range(n_steps)]
    return [step.transpose(0, 1) for step in series.edge_index]


def _graph_optional_steps(payload, n_steps: int):
    if payload is None:
        return None
    if isinstance(payload, tuple):
        return list(payload)
    if isinstance(payload, torch.Tensor):
        if payload.ndim >= 2 and payload.shape[0] == n_steps:
            return [payload[step] for step in range(n_steps)]
        if payload.ndim == 1 and payload.shape[0] == n_steps:
            return [payload[step:step + 1] for step in range(n_steps)]
        return [payload for _ in range(n_steps)]
    return [payload for _ in range(n_steps)]
