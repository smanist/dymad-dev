from __future__ import annotations

import torch

from dymad.core import FixedGraphSeries, GraphSeriesBatch, VariableEdgeGraphSeries


def test_fixed_graph_series_slice_and_device_dtype_move() -> None:
    series = FixedGraphSeries(
        time=torch.arange(5, dtype=torch.float32),
        node_state=torch.arange(30, dtype=torch.float32).reshape(5, 3, 2),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
        control=torch.arange(15, dtype=torch.float32).reshape(5, 3, 1),
        target=torch.arange(15, dtype=torch.float32).reshape(5, 3, 1),
        params=torch.arange(6, dtype=torch.float32).reshape(3, 2),
        edge_weight=torch.arange(15, dtype=torch.float32).reshape(5, 3),
        edge_attr=torch.arange(30, dtype=torch.float32).reshape(5, 3, 2),
        meta={"kind": "fixed"},
    )

    sliced = series.slice_steps(1, 4)
    moved = sliced.to(dtype=torch.float64)

    assert sliced.time.shape == (3,)
    assert sliced.node_state.shape == (3, 3, 2)
    assert sliced.edge_index.shape == (2, 3)
    assert moved.time.dtype == torch.float64
    assert moved.node_state.dtype == torch.float64
    assert moved.edge_index.dtype == torch.long
    assert moved.to_flat_node_features().shape == (3, 6)


def test_variable_edge_graph_series_batch_collation() -> None:
    series_a = VariableEdgeGraphSeries(
        time=torch.arange(4, dtype=torch.float32),
        node_state=torch.arange(24, dtype=torch.float32).reshape(4, 3, 2),
        edge_index=(
            torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            torch.tensor([[0, 2], [2, 1]], dtype=torch.long),
            torch.tensor([[1, 2], [2, 0]], dtype=torch.long),
            torch.tensor([[0, 1], [2, 0]], dtype=torch.long),
        ),
        edge_weight=(
            torch.tensor([1.0, 2.0]),
            torch.tensor([2.0, 3.0]),
            torch.tensor([3.0, 4.0]),
            torch.tensor([4.0, 5.0]),
        ),
        meta={"kind": "variable-a"},
    )
    series_b = VariableEdgeGraphSeries(
        time=torch.arange(4, dtype=torch.float32),
        node_state=torch.arange(24, 48, dtype=torch.float32).reshape(4, 3, 2),
        edge_index=(
            torch.tensor([[0], [1]], dtype=torch.long),
            torch.tensor([[1], [2]], dtype=torch.long),
            torch.tensor([[2], [0]], dtype=torch.long),
            torch.tensor([[0], [2]], dtype=torch.long),
        ),
        meta={"kind": "variable-b"},
    )

    batch = GraphSeriesBatch.collate([series_a, series_b])
    moved = batch.to(dtype=torch.float64)
    subset = batch.slice_batch([1])

    assert len(batch) == 2
    assert len(subset) == 1
    assert subset[0].meta["kind"] == "variable-b"
    assert moved[0].node_state.dtype == torch.float64
    assert moved[0].edge_index[0].dtype == torch.long
