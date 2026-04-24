from pathlib import Path

import torch

from dymad.core import (
    FixedGraphSeries,
    GraphModelContext,
    GraphTrainerBatch,
    RaggedRegularRuntime,
    RegularModelContext,
    RegularSeries,
    RegularSeriesBatch,
    RegularTrainerBatch,
    UniformGraphRuntime,
    UniformRegularRuntime,
)
from dymad.models.prediction import predict_discrete, predict_discrete_exp
from dymad.models.runtime_view import build_component_input_view


class _IdentityDiscreteModel:
    def encoder(self, payload):
        return build_component_input_view(payload).state

    def dynamics(self, z, payload):
        return z + 1.0

    def decoder(self, z, payload):
        return z


class _CountingGraphExpModel:
    def __init__(self):
        self.decoder_calls = 0

    def encoder(self, payload):
        return build_component_input_view(payload).graph_state

    def dynamics(self, z, payload):
        return z + 1.0

    def decoder(self, z, payload):
        self.decoder_calls += 1
        return z


def test_regular_model_context_to_runtime_emits_uniform_runtime():
    batch = RegularSeriesBatch.collate(
        [
            RegularSeries(
                time=torch.tensor([0.0, 1.0, 2.0]),
                state=torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
                control=torch.tensor([[0.1], [0.2], [0.3]]),
                params=torch.tensor([7.0, 8.0]),
            ),
            RegularSeries(
                time=torch.tensor([0.0, 1.0, 2.0]),
                state=torch.tensor([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]]),
                control=torch.tensor([[0.4], [0.5], [0.6]]),
                params=torch.tensor([9.0, 10.0]),
            ),
        ]
    )

    runtime = RegularModelContext.from_batch(batch).to_runtime()

    assert isinstance(runtime, UniformRegularRuntime)
    assert runtime.is_uniform_length
    assert runtime.state.shape == (2, 3, 2)
    assert runtime.control.shape == (2, 3, 1)
    assert runtime.params.shape == (2, 2)
    assert torch.equal(runtime.get_step(1).state, torch.tensor([[3.0, 4.0], [1.0, 2.0]]))


def test_regular_model_context_to_runtime_emits_ragged_runtime():
    batch = RegularSeriesBatch.collate(
        [
            RegularSeries(
                time=torch.tensor([0.0, 1.0]),
                state=torch.tensor([[1.0], [2.0]]),
            ),
            RegularSeries(
                time=torch.tensor([0.0, 1.0, 2.0, 3.0]),
                state=torch.tensor([[3.0], [4.0], [5.0], [6.0]]),
            ),
        ]
    )

    runtime = RegularModelContext.from_batch(batch).to_runtime()

    assert isinstance(runtime, RaggedRegularRuntime)
    assert runtime.step_lengths == (2, 4)
    assert runtime.valid_mask.tolist() == [[True, True, False, False], [True, True, True, True]]
    assert torch.equal(runtime.state[0, 2:], torch.zeros(2, 1))


def test_predict_discrete_returns_padded_ragged_batch():
    batch = RegularSeriesBatch.collate(
        [
            RegularSeries(
                time=torch.tensor([0.0, 1.0, 2.0]),
                state=torch.tensor([[1.0], [2.0], [3.0]]),
            ),
            RegularSeries(
                time=torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0]),
                state=torch.tensor([[10.0], [11.0], [12.0], [13.0], [14.0]]),
            ),
        ]
    )
    runtime = RegularModelContext.from_batch(batch).to_runtime()
    model = _IdentityDiscreteModel()

    preds = predict_discrete(
        model,
        runtime.initial_state(),
        runtime.t,
        runtime,
    )

    assert preds.shape == (2, 5, 1)
    assert torch.equal(preds[0, :3, 0], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.equal(preds[0, 3:, 0], torch.tensor([0.0, 0.0]))
    assert torch.equal(preds[1, :, 0], torch.tensor([10.0, 11.0, 12.0, 13.0, 14.0]))


def test_native_hot_paths_do_not_route_through_legacy_runtime():
    root = Path(__file__).resolve().parents[1] / "src" / "dymad"
    files = [
        root / "models" / "prediction.py",
        root / "models" / "runtime_view.py",
        root / "training" / "phases.py",
        root / "training" / "phase_pipeline.py",
        root / "training" / "ls_update.py",
    ]

    for path in files:
        text = path.read_text()
        assert "batch_to_legacy_runtime" not in text
        assert ".to_legacy_runtime()" not in text


def test_fixed_topology_graph_runtime_keeps_shared_edge_storage():
    series = [
        FixedGraphSeries(
            time=torch.tensor([0.0, 1.0, 2.0]),
            node_state=torch.arange(18.0).reshape(3, 3, 2),
            edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            edge_weight=torch.tensor([0.5, 1.5]),
        ),
        FixedGraphSeries(
            time=torch.tensor([0.0, 1.0, 2.0]),
            node_state=torch.arange(18.0, 36.0).reshape(3, 3, 2),
            edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            edge_weight=torch.tensor([2.5, 3.5]),
        ),
    ]

    runtime = GraphModelContext.from_batch(
        GraphTrainerBatch.collate_series(series).series
    ).to_runtime()

    assert isinstance(runtime, UniformGraphRuntime)
    assert runtime.is_fixed_topology
    assert runtime.edge_index.shape == (2, 2, 2)
    assert runtime.edge_weight.shape == (2, 2)
    step = runtime.get_step(1)
    assert step.edge_index.shape == (2, 2, 2)
    assert torch.equal(step.edge_index[0], runtime.edge_index[0])


def test_runtime_native_truncate_and_window_keep_runtime_batches():
    regular_batch = RegularTrainerBatch.collate_series(
        [
            RegularSeries(
                time=torch.tensor([0.0, 1.0, 2.0, 3.0]),
                state=torch.arange(8.0).reshape(4, 2),
            )
        ]
    )
    truncated = regular_batch.truncate(3)
    windowed = regular_batch.window(2, 1)

    assert isinstance(truncated.runtime, UniformRegularRuntime)
    assert truncated.runtime.state.shape == (1, 3, 2)
    assert isinstance(windowed.runtime, UniformRegularRuntime)
    assert windowed.runtime.state.shape == (3, 2, 2)

    graph_batch = GraphTrainerBatch.collate_series(
        [
            FixedGraphSeries(
                time=torch.tensor([0.0, 1.0, 2.0, 3.0]),
                node_state=torch.arange(24.0).reshape(4, 3, 2),
                edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
                edge_weight=torch.tensor([1.0, 2.0]),
            )
        ]
    )
    g_truncated = graph_batch.truncate(3)
    g_windowed = graph_batch.window(2, 1)

    assert isinstance(g_truncated.runtime, UniformGraphRuntime)
    assert g_truncated.runtime.node_state.shape == (1, 3, 3, 2)
    assert g_truncated.runtime.edge_index.shape == (1, 2, 2)
    assert isinstance(g_windowed.runtime, UniformGraphRuntime)
    assert g_windowed.runtime.node_state.shape == (3, 2, 3, 2)
    assert g_windowed.runtime.edge_index.shape == (3, 2, 2)


def test_uniform_fixed_topology_graph_prediction_uses_vectorized_decode():
    batch = GraphTrainerBatch.collate_series(
        [
            FixedGraphSeries(
                time=torch.tensor([0.0, 1.0, 2.0]),
                node_state=torch.tensor(
                    [
                        [[1.0], [2.0]],
                        [[2.0], [3.0]],
                        [[3.0], [4.0]],
                    ]
                ),
                edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            )
        ]
    )
    runtime = batch.runtime
    model = _CountingGraphExpModel()

    preds = predict_discrete_exp(
        model,
        runtime.initial_state(),
        runtime.t,
        runtime,
    )

    assert preds.shape == (1, 3, 2, 1)
    assert model.decoder_calls == 1
