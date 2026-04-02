from __future__ import annotations

import torch

from dymad.core import FixedGraphSeries
from dymad.io.series_adapter import SeriesAdapter
from dymad.io.trajectory_manager import TrajectoryManagerGraph


def _graph_metadata(path: str) -> dict:
    return {
        "data_key": "data",
        "config": {
            "data": {"path": path, "n_samples": 4, "n_steps": 10},
            "transform_x": [
                {"type": "Scaler", "mode": "01"},
                {"type": "delay", "delay": 2},
            ],
            "transform_u": {"type": "Scaler", "mode": "-11"},
            "transform_p": {"type": "Scaler", "mode": "std"},
            "transform_ew": {"type": "Scaler", "mode": "-11"},
        },
    }


def test_graph_series_adapter_builds_typed_series_from_arrays() -> None:
    series = SeriesAdapter.from_graph_arrays(
        time=[0.0, 1.0],
        node_state=[
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ],
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        control=[
            [[0.1], [0.2]],
            [[0.3], [0.4]],
        ],
        params=[9.0, 10.0],
        edge_weight=torch.tensor([1.0, 2.0]),
        dtype=torch.float64,
    )

    assert isinstance(series, FixedGraphSeries)
    assert series.time.dtype == torch.float64
    assert series.node_state.shape == (2, 2, 2)
    torch.testing.assert_close(series.control, torch.tensor([[[0.1], [0.2]], [[0.3], [0.4]]], dtype=torch.float64))
    torch.testing.assert_close(series.params, torch.tensor([9.0, 10.0], dtype=torch.float64))


def test_graph_trajectory_manager_uses_typed_series_pipeline(ltg_data) -> None:
    manager = TrajectoryManagerGraph(metadata=_graph_metadata(str(ltg_data)))
    manager.prepare_data()
    manager.set_data_index([0, 1])
    manager.apply_data_transformations()

    assert manager.typed_dataset is not None
    assert manager.dataset is manager.typed_dataset
    assert [type(series).__name__ for series in manager.typed_dataset] == [
        "FixedGraphSeries",
        "FixedGraphSeries",
    ]


def test_graph_series_dataset_exposes_typed_graph_objects(ltg_data) -> None:
    manager = TrajectoryManagerGraph(metadata=_graph_metadata(str(ltg_data)))
    manager.prepare_data()
    manager.set_data_index([0, 1])
    manager.apply_data_transformations()

    series_dataset = manager.create_graph_series_dataset()

    assert len(series_dataset) == len(manager.dataset) == 2
    first_series = series_dataset[0]
    assert isinstance(first_series, FixedGraphSeries)
    assert first_series.node_state.ndim == 3
    assert first_series.control is not None
    assert first_series.params is not None
