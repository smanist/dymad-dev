from __future__ import annotations

import numpy as np
import torch
import pytest

from dymad.core import FixedGraphSeries
from dymad.core.graph_series import VariableEdgeGraphSeries
from dymad.core.transform_module import (
    FieldTransformModule,
    SeriesTransformPipeline,
    TransformModule,
)
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
    torch.testing.assert_close(
        series.control, torch.tensor([[[0.1], [0.2]], [[0.3], [0.4]]], dtype=torch.float64)
    )
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


def test_graph_trajectory_manager_accepts_variable_edge_counts() -> None:
    manager = TrajectoryManagerGraph.__new__(TrajectoryManagerGraph)
    manager.metadata = {
        "n_aux_features": 0,
        "n_control_features": 1,
        "n_parameters": 0,
        "n_edge_weights": 1,
        "n_edge_features": 0,
    }
    manager.n_nodes = 2
    manager.dtype = torch.float32
    manager.device = torch.device("cpu")
    manager.t = [np.array([0.0, 1.0], dtype=np.float32)]
    manager.x = [np.array([[1.0, 0.0, 0.0, 1.0], [1.1, 0.1, 0.1, 1.1]], dtype=np.float32)]
    manager.y = [np.empty((2, 0), dtype=np.float32)]
    manager.u = [np.array([[0.0, 0.0], [0.2, 0.2]], dtype=np.float32)]
    manager.p = [np.empty((0,), dtype=np.float32)]
    manager.ei = [
        [
            np.array([[0, 1], [1, 0]], dtype=np.int64),
            np.array([[0], [1]], dtype=np.int64),
        ]
    ]
    manager.ew = [
        [
            np.array([1.0, 1.0], dtype=np.float32),
            np.array([0.5], dtype=np.float32),
        ]
    ]
    manager.ea = [[np.empty((2, 0), dtype=np.float32), np.empty((1, 0), dtype=np.float32)]]

    series = manager._create_raw_graph_series_by_index(torch.tensor([0]))[0]
    assert isinstance(series, VariableEdgeGraphSeries)
    assert isinstance(series.edge_weight, tuple)
    assert tuple(step.shape[0] for step in series.edge_weight) == (2, 1)


class _IdentityDelayTransform(TransformModule):
    def __init__(self, delay: int) -> None:
        super().__init__(delay=delay)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return data


def test_graph_transform_pipeline_aligns_variable_edge_sequences_with_delay() -> None:
    series = VariableEdgeGraphSeries(
        time=torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32),
        node_state=torch.tensor(
            [
                [[1.0], [2.0]],
                [[1.5], [2.5]],
                [[2.0], [3.0]],
            ],
            dtype=torch.float32,
        ),
        edge_index=(
            torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            torch.tensor([[0], [1]], dtype=torch.long),
            torch.tensor([[1], [0]], dtype=torch.long),
        ),
        edge_weight=(
            torch.tensor([1.0, 1.0], dtype=torch.float32),
            torch.tensor([0.5], dtype=torch.float32),
            torch.tensor([0.25], dtype=torch.float32),
        ),
        meta={},
    )
    pipeline = SeriesTransformPipeline(
        [FieldTransformModule("node_state", _IdentityDelayTransform(delay=1))]
    )

    aligned = pipeline._align_series(series)

    assert isinstance(aligned, VariableEdgeGraphSeries)
    assert aligned.time.shape[0] == 2
    assert len(aligned.edge_index) == 2
    assert len(aligned.edge_weight) == 2


@pytest.mark.parametrize("field", ["edge_weight", "edge_attr"])
def test_graph_transform_pipeline_scales_variable_edge_payloads(field: str) -> None:
    manager = TrajectoryManagerGraph.__new__(TrajectoryManagerGraph)
    manager.metadata = {
        "config": {
            "transform_x": None,
            "transform_u": None,
            "transform_p": None,
            "transform_ew": {"type": "Scaler", "mode": "-11"},
            "transform_ea": {"type": "Scaler", "mode": "01"},
        },
        "n_aux_features": 0,
        "n_control_features": 0,
        "n_parameters": 0,
        "n_edge_weights": 1,
        "n_edge_features": 2,
    }
    manager.n_nodes = 2
    manager.dtype = torch.float32
    manager.device = torch.device("cpu")
    manager.t = [np.array([0.0, 1.0], dtype=np.float32)]
    manager.x = [np.array([[1.0, 0.0], [2.0, 1.0]], dtype=np.float32)]
    manager.y = [np.empty((2, 0), dtype=np.float32)]
    manager.u = [np.empty((2, 0), dtype=np.float32)]
    manager.p = [np.empty((0,), dtype=np.float32)]
    manager.ei = [
        [
            np.array([[0, 1], [1, 0]], dtype=np.int64),
            np.array([[0], [1]], dtype=np.int64),
        ]
    ]
    manager.ew = [
        [
            np.array([1.0, 3.0], dtype=np.float32),
            np.array([0.5], dtype=np.float32),
        ]
    ]
    manager.ea = [
        [
            np.array([[1.0, 4.0], [2.0, 6.0]], dtype=np.float32),
            np.array([[3.0, 8.0]], dtype=np.float32),
        ]
    ]
    manager._init_transforms()

    raw_batch = manager._create_raw_graph_series_by_index(torch.tensor([0]))
    pipeline = manager._build_graph_transform_pipeline()
    pipeline.fit(raw_batch)
    transformed = list(pipeline(raw_batch))[0]

    if field == "edge_weight":
        assert isinstance(transformed.edge_weight, tuple)
        assert tuple(step.shape for step in transformed.edge_weight) == ((2,), (1,))
        recovered = pipeline.inverse_field("edge_weight", transformed.edge_weight)
        assert isinstance(recovered, tuple)
        assert tuple(step.shape for step in recovered) == ((2,), (1,))
    else:
        assert isinstance(transformed.edge_attr, tuple)
        assert tuple(step.shape for step in transformed.edge_attr) == ((2, 2), (1, 2))
        recovered = pipeline.inverse_field("edge_attr", transformed.edge_attr)
        assert isinstance(recovered, tuple)
        assert tuple(step.shape for step in recovered) == ((2, 2), (1, 2))
