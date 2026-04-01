from __future__ import annotations

import torch

from dymad.core import FixedGraphSeries
from dymad.io.series_adapter import SeriesAdapter, graph_series_to_legacy_runtime
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


def test_graph_series_seam_matches_legacy_graph_dataset(ltg_data) -> None:
    manager = TrajectoryManagerGraph(metadata=_graph_metadata(str(ltg_data)))
    manager.prepare_data()
    manager.set_data_index([0, 1])
    manager.apply_data_transformations()

    series_dataset = manager.create_graph_series_dataset()
    assert len(series_dataset) == len(manager.dataset) == 2

    first_series = series_dataset[0]
    first_legacy = manager.dataset[0]
    assert isinstance(first_series, FixedGraphSeries)

    roundtrip = graph_series_to_legacy_runtime(first_series)
    from_legacy = SeriesAdapter.from_dyndata(first_legacy)

    assert isinstance(from_legacy, FixedGraphSeries)
    torch.testing.assert_close(first_series.time, first_legacy.t.squeeze(0))
    torch.testing.assert_close(
        first_series.node_state,
        first_legacy.x.squeeze(0).reshape(first_legacy.n_steps, first_legacy.n_nodes, -1),
    )
    torch.testing.assert_close(first_series.control, first_legacy.u.squeeze(0).reshape(first_legacy.n_steps, first_legacy.n_nodes, -1))
    torch.testing.assert_close(first_series.params, first_legacy.p.squeeze(0))
    torch.testing.assert_close(roundtrip.x, first_legacy.x)
    torch.testing.assert_close(roundtrip.u, first_legacy.u)
    torch.testing.assert_close(roundtrip.p, first_legacy.p)
    torch.testing.assert_close(from_legacy.node_state, first_series.node_state)
    torch.testing.assert_close(from_legacy.control, first_series.control)
    torch.testing.assert_close(from_legacy.params, first_series.params)
    for expected, actual in zip(roundtrip.ei.unbind(), first_legacy.ei.unbind()):
        torch.testing.assert_close(expected, actual)


def test_graph_series_roundtrip_preserves_explicit_param_shape() -> None:
    series = FixedGraphSeries(
        time=torch.tensor([0.0, 1.0]),
        node_state=torch.arange(8, dtype=torch.float32).reshape(2, 2, 2),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        params=torch.arange(6, dtype=torch.float32).reshape(2, 3),
        meta={"kind": "manual"},
    )

    runtime = graph_series_to_legacy_runtime(series)
    restored = SeriesAdapter.from_dyndata(runtime)

    assert runtime.p.shape == (1, 6)
    torch.testing.assert_close(restored.params, series.params)


def test_graph_trajectory_manager_uses_typed_series_before_legacy_adaptation(monkeypatch, ltg_data) -> None:
    manager = TrajectoryManagerGraph(metadata=_graph_metadata(str(ltg_data)))
    manager.prepare_data()
    manager.set_data_index([0, 1])

    import dymad.io.trajectory_manager as trajectory_manager_module

    calls: list[str] = []
    original = trajectory_manager_module.graph_series_to_legacy_runtime

    def traced(series):
        calls.append(type(series).__name__)
        return original(series)

    monkeypatch.setattr(trajectory_manager_module, "graph_series_to_legacy_runtime", traced)

    manager.apply_data_transformations()

    assert calls == ["FixedGraphSeries", "FixedGraphSeries"]
