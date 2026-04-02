from pathlib import Path

import torch

from dymad.core import (
    RaggedRegularRuntime,
    RegularModelContext,
    RegularSeries,
    RegularSeriesBatch,
    UniformRegularRuntime,
)
from dymad.models.prediction import predict_discrete
from dymad.models.runtime_view import build_component_input_view


class _IdentityDiscreteModel:
    def encoder(self, payload):
        return build_component_input_view(payload).state

    def dynamics(self, z, payload):
        return z + 1.0

    def decoder(self, z, payload):
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
    root = Path("/Users/daninghuang/Repos/dymad-dev/src/dymad")
    files = [
        root / "models" / "prediction.py",
        root / "models" / "runtime_view.py",
        root / "training" / "opt_base.py",
        root / "training" / "opt_node.py",
        root / "training" / "opt_weak_form.py",
        root / "training" / "opt_linear.py",
        root / "training" / "ls_update.py",
    ]

    for path in files:
        text = path.read_text()
        assert "batch_to_legacy_runtime" not in text
        assert ".to_legacy_runtime()" not in text
