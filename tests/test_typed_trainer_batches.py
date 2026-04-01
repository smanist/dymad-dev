from __future__ import annotations

import numpy as np
import torch

from dymad.core import GraphTrainerBatch, RaggedRegularSeriesBatch, RegularTrainerBatch, RegularSeries, RegularSeriesBatch
from dymad.io.trajectory_manager import TrajectoryManager, TrajectoryManagerGraph


def test_regular_typed_dataloader_emits_regular_trainer_batch(tmp_path) -> None:
    data_path = tmp_path / "toy_regular_typed_loader.npz"
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
                "data": {"path": str(data_path)},
                "dataloader": {"batch_size": 2, "shuffle": False},
            },
        }
    )
    dataloader, dataset, metadata = manager.process_all(typed=True)
    batch = next(iter(dataloader))

    assert isinstance(batch, RegularTrainerBatch)
    assert len(dataset) == 2
    assert metadata["n_data"] == 2
    assert batch.time_tensor().shape == (2, 6)
    assert batch.state_tensor().shape == (2, 6, 2)
    assert batch.control_tensor().shape == (2, 6, 1)
    assert batch.initial_state().shape == (2, 2)


def test_graph_typed_dataloader_emits_graph_trainer_batch(ltg_data) -> None:
    metadata = {
        "data_key": "data",
        "config": {
            "data": {"path": str(ltg_data), "n_samples": 4, "n_steps": 10},
            "dataloader": {"batch_size": 2, "shuffle": False},
            "transform_x": [
                {"type": "Scaler", "mode": "01"},
                {"type": "delay", "delay": 2},
            ],
            "transform_u": {"type": "Scaler", "mode": "-11"},
            "transform_p": {"type": "Scaler", "mode": "std"},
            "transform_ew": {"type": "Scaler", "mode": "-11"},
        },
    }

    manager = TrajectoryManagerGraph(metadata=metadata)
    dataloader, dataset, metadata = manager.process_all(typed=True)
    batch = next(iter(dataloader))

    assert isinstance(batch, GraphTrainerBatch)
    assert len(dataset) == metadata["n_data"] == 4
    assert batch.time_tensor().shape[0] == 2
    assert batch.node_state_tensor().ndim == 4
    assert batch.initial_state().shape[0] == 2
    assert len(batch.edge_index_payload()) == 2
    moved = batch.to(dtype=torch.double)
    assert moved.node_state_tensor().dtype == torch.double


def test_regular_typed_batch_marks_ragged_and_preserves_singletons() -> None:
    batch = RegularTrainerBatch(
        RegularSeriesBatch.collate(
            [
                RegularSeries(
                    time=torch.tensor([0.0, 1.0]),
                    state=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                ),
                RegularSeries(
                    time=torch.tensor([0.0, 1.0, 2.0]),
                    state=torch.tensor([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]),
                ),
            ]
        )
    )

    assert isinstance(batch.series, RaggedRegularSeriesBatch)
    assert batch.is_ragged is True
    singleton_lengths = [len(single.series[0].time) for single in batch.iter_single_batches()]
    assert singleton_lengths == [2, 3]
