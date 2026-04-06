"""Array-to-series adapters for typed regular and graph payloads."""

from __future__ import annotations

from typing import Any

import torch

from dymad.core.graph_series import FixedGraphSeries, GraphSeries, VariableEdgeGraphSeries
from dymad.core.series import RegularSeries


class SeriesAdapter:
    """Build typed series objects from array-like payloads."""

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
            control=torch.as_tensor(control, dtype=dtype, device=device)
            if control is not None
            else None,
            target=torch.as_tensor(target, dtype=dtype, device=device)
            if target is not None
            else None,
            params=torch.as_tensor(params, dtype=dtype, device=device)
            if params is not None
            else None,
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
        control_tensor = (
            torch.as_tensor(control, dtype=dtype, device=device) if control is not None else None
        )
        target_tensor = (
            torch.as_tensor(target, dtype=dtype, device=device) if target is not None else None
        )
        params_tensor = (
            torch.as_tensor(params, dtype=dtype, device=device) if params is not None else None
        )

        edge_index_payload = SeriesAdapter._graph_edge_index_payload(edge_index, device=device)
        edge_weight_payload = SeriesAdapter._graph_optional_payload(
            edge_weight, dtype=dtype, device=device
        )
        edge_attr_payload = SeriesAdapter._graph_optional_payload(
            edge_attr, dtype=dtype, device=device
        )

        cls = (
            FixedGraphSeries
            if isinstance(edge_index_payload, torch.Tensor)
            else VariableEdgeGraphSeries
        )
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
    def _graph_edge_index_payload(edge_index: Any, *, device: torch.device | str | None):
        if isinstance(edge_index, (list, tuple)):
            steps = tuple(
                SeriesAdapter._to_edge_index_tensor(item, device=device) for item in edge_index
            )
            if steps and all(torch.equal(steps[0], step) for step in steps[1:]):
                return steps[0]
            return steps
        tensor = torch.as_tensor(edge_index, dtype=torch.long, device=device)
        if tensor.ndim == 3:
            return SeriesAdapter._graph_edge_index_payload(tuple(tensor.unbind()), device=device)
        return SeriesAdapter._to_edge_index_tensor(tensor, device=device)

    @staticmethod
    def _to_edge_index_tensor(
        edge_index: Any, *, device: torch.device | str | None
    ) -> torch.Tensor:
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
