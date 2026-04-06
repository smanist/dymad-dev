from __future__ import annotations

import numpy as np
import torch

from dymad.core.series import RegularSeries
from dymad.io.series_adapter import SeriesAdapter
from dymad.io.trajectory_manager import TrajectoryManager
from dymad.transform import make_transform


def test_regular_series_adapter_builds_typed_series_from_arrays() -> None:
    series = SeriesAdapter.from_regular_arrays(
        time=[0.0, 1.0, 2.0],
        state=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        control=[[0.1], [0.2], [0.3]],
        params=[7.0],
        dtype=torch.float64,
    )

    assert isinstance(series, RegularSeries)
    assert series.time.dtype == torch.float64
    torch.testing.assert_close(
        series.state, torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float64)
    )
    torch.testing.assert_close(
        series.control, torch.tensor([[0.1], [0.2], [0.3]], dtype=torch.float64)
    )
    torch.testing.assert_close(series.params, torch.tensor([7.0], dtype=torch.float64))


def test_regular_series_pipeline_matches_transform_path(tmp_path) -> None:
    data_path = tmp_path / "toy_regular_transform.npz"
    t = np.stack(
        [
            np.linspace(0.0, 1.0, 6),
            np.linspace(0.0, 1.0, 6),
        ]
    )
    x = np.stack(
        [
            np.column_stack((np.linspace(0.0, 1.0, 6), np.linspace(1.0, 2.0, 6))),
            np.column_stack((np.linspace(2.0, 3.0, 6), np.linspace(3.0, 4.0, 6))),
        ]
    )
    u = np.stack(
        [
            np.linspace(0.0, 0.5, 6).reshape(-1, 1),
            np.linspace(0.5, 1.0, 6).reshape(-1, 1),
        ]
    )
    np.savez(data_path, t=t, x=x, u=u)

    config = {
        "data": {"path": str(data_path)},
        "transform_x": [
            {"type": "Scaler", "mode": "01"},
            {"type": "delay", "delay": 2},
        ],
        "transform_u": {"type": "Scaler", "mode": "-11"},
    }
    manager = TrajectoryManager(metadata={"data_key": "data", "config": config})
    manager.prepare_data()
    manager.set_data_index([0, 1])
    manager.apply_data_transformations()

    series_dataset = manager.create_regular_series_dataset()
    legacy_x = make_transform(config["transform_x"])
    legacy_u = make_transform(config["transform_u"])
    raw_x = [sample for sample in x]
    raw_u = [sample for sample in u]
    legacy_x.fit(raw_x)
    legacy_u.fit(raw_u)
    ref_x = legacy_x.transform(raw_x)
    ref_u = legacy_u.transform(raw_u)
    common_delay = max(legacy_x.delay, legacy_u.delay)

    for index, series in enumerate(series_dataset):
        expected_t = t[index][common_delay:]
        expected_x = ref_x[index][common_delay - legacy_x.delay :]
        expected_u = ref_u[index][common_delay - legacy_u.delay :]
        torch.testing.assert_close(
            series.time, torch.as_tensor(expected_t, dtype=series.time.dtype)
        )
        torch.testing.assert_close(
            series.state, torch.as_tensor(expected_x, dtype=series.state.dtype)
        )
        torch.testing.assert_close(
            series.control, torch.as_tensor(expected_u, dtype=series.control.dtype)
        )
        assert series.meta["delay"] == common_delay
