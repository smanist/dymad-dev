from __future__ import annotations

import numpy as np
import torch

from dymad.core.series import RegularSeries
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
