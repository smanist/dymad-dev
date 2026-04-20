from __future__ import annotations

import copy

from dymad.core import GraphSeries, GraphTrainerBatch, RegularSeries, RegularTrainerBatch
from dymad.io.trajectory_manager import TrajectoryManager, TrajectoryManagerGraph
from dymad.training.driver import _build_phase_context


def test_build_data_state_uses_regular_typed_batches_for_linear_only(tmp_path) -> None:
    import numpy as np

    data_path = tmp_path / "toy_regular_linear_driver.npz"
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

    cfg = {
        "data": {"path": str(data_path)},
        "dataloader": {"batch_size": 2, "shuffle": False},
        "phases": [{"name": "Linear", "trainer": "Linear"}],
    }
    train = TrajectoryManager(metadata={"data_key": "data", "config": copy.deepcopy(cfg)})
    valid = TrajectoryManager(metadata={"data_key": "data", "config": copy.deepcopy(cfg)})
    train.prepare_data()
    valid.prepare_data()
    train.set_data_index([0, 1])
    valid.set_data_index([0, 1])

    context = _build_phase_context(0, cfg, [train], [valid])
    batch = next(iter(context.train_loader))

    assert isinstance(batch, RegularTrainerBatch)
    assert isinstance(context.train_set[0], RegularSeries)


def test_build_data_state_uses_graph_typed_batches_for_linear_only(ltg_data) -> None:
    cfg = {
        "data": {"path": str(ltg_data), "n_samples": 4, "n_steps": 10},
        "dataloader": {"batch_size": 2, "shuffle": False},
        "transform_x": [
            {"type": "Scaler", "mode": "01"},
            {"type": "delay", "delay": 2},
        ],
        "transform_u": {"type": "Scaler", "mode": "-11"},
        "transform_p": {"type": "Scaler", "mode": "std"},
        "transform_ew": {"type": "Scaler", "mode": "-11"},
        "phases": [{"name": "Linear", "trainer": "Linear"}],
    }
    train = TrajectoryManagerGraph(metadata={"data_key": "data", "config": copy.deepcopy(cfg)})
    valid = TrajectoryManagerGraph(metadata={"data_key": "data", "config": copy.deepcopy(cfg)})
    train.prepare_data()
    valid.prepare_data()
    train.set_data_index([0, 1])
    valid.set_data_index([0, 1])

    context = _build_phase_context(0, cfg, [train], [valid])
    batch = next(iter(context.train_loader))

    assert isinstance(batch, GraphTrainerBatch)
    assert isinstance(context.train_set[0], GraphSeries)
