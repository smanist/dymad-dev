from __future__ import annotations

import numpy as np
import torch

from dymad.core.series import RegularSeries
from dymad.transform import make_transform
from dymad.io.series_adapter import DynDataAdapter, SeriesAdapter
from dymad.io.trajectory_manager import TrajectoryManager


def test_regular_series_seam_matches_legacy_regular_dataset(tmp_path) -> None:
    data_path = tmp_path / "toy_regular.npz"
    t = np.stack([
        np.linspace(0.0, 1.0, 6),
        np.linspace(0.0, 1.0, 6),
    ])
    x = np.stack([
        np.column_stack((np.linspace(0.0, 1.0, 6), np.linspace(1.0, 2.0, 6))),
        np.column_stack((np.linspace(2.0, 3.0, 6), np.linspace(3.0, 4.0, 6))),
    ])
    u = np.stack([
        np.linspace(0.0, 0.5, 6).reshape(-1, 1),
        np.linspace(0.5, 1.0, 6).reshape(-1, 1),
    ])
    np.savez(data_path, t=t, x=x, u=u)

    manager = TrajectoryManager(
        metadata={
            "data_key": "data",
            "config": {
                "data": {
                    "path": str(data_path),
                },
            },
        }
    )
    manager.prepare_data()
    manager.set_data_index([0, 1])
    manager.apply_data_transformations()

    series_dataset = manager.create_regular_series_dataset()

    assert len(series_dataset) == len(manager.dataset) == 2
    first_series = series_dataset[0]
    first_legacy = manager.dataset[0]
    assert isinstance(first_series, RegularSeries)

    roundtrip = DynDataAdapter.from_regular_series(first_series)
    from_legacy = SeriesAdapter.from_dyndata(first_legacy)

    torch.testing.assert_close(first_series.time, first_legacy.t.squeeze(0))
    torch.testing.assert_close(first_series.state, first_legacy.x.squeeze(0))
    torch.testing.assert_close(first_series.control, first_legacy.u.squeeze(0))
    torch.testing.assert_close(roundtrip.t.squeeze(0), first_legacy.t.squeeze(0))
    torch.testing.assert_close(roundtrip.x.squeeze(0), first_legacy.x.squeeze(0))
    torch.testing.assert_close(roundtrip.u.squeeze(0), first_legacy.u.squeeze(0))
    torch.testing.assert_close(from_legacy.time, first_series.time)
    torch.testing.assert_close(from_legacy.state, first_series.state)
    torch.testing.assert_close(from_legacy.control, first_series.control)


def test_regular_series_pipeline_matches_legacy_transform_path(tmp_path) -> None:
    data_path = tmp_path / "toy_regular_transform.npz"
    t = np.stack([
        np.linspace(0.0, 1.0, 6),
        np.linspace(0.0, 1.0, 6),
    ])
    x = np.stack([
        np.column_stack((np.linspace(0.0, 1.0, 6), np.linspace(1.0, 2.0, 6))),
        np.column_stack((np.linspace(2.0, 3.0, 6), np.linspace(3.0, 4.0, 6))),
    ])
    u = np.stack([
        np.linspace(0.0, 0.5, 6).reshape(-1, 1),
        np.linspace(0.5, 1.0, 6).reshape(-1, 1),
    ])
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
        torch.testing.assert_close(series.time, torch.as_tensor(expected_t, dtype=series.time.dtype))
        torch.testing.assert_close(series.state, torch.as_tensor(expected_x, dtype=series.state.dtype))
        torch.testing.assert_close(series.control, torch.as_tensor(expected_u, dtype=series.control.dtype))
        assert series.meta["delay"] == common_delay
