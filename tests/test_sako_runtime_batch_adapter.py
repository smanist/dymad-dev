import torch

from dymad.core import RegularSeries, RegularTrainerBatch, UniformRegularRuntime
from dymad.io.legacy_runtime import LegacyRuntimeBatch
from dymad.sako.base import encode_runtime_batch


class _DummyModel:
    def __init__(self):
        self.last_payload = None

    def encoder(self, payload):
        self.last_payload = payload
        return payload.x


def _build_regular_batch() -> RegularTrainerBatch:
    time = torch.arange(4, dtype=torch.float64)
    state = torch.tensor(
        [[1.0, 2.0], [2.0, 3.0], [3.0, 5.0], [5.0, 8.0]],
        dtype=torch.float64,
    )
    series = RegularSeries(time=time, state=state)
    return RegularTrainerBatch.collate_series([series])


def test_encode_runtime_batch_accepts_typed_regular_trainer_batch():
    model = _DummyModel()
    batch = _build_regular_batch()

    encoded = encode_runtime_batch(model, batch)

    assert isinstance(model.last_payload, UniformRegularRuntime)
    assert encoded.shape == (1, 4, 2)
    assert torch.equal(model.last_payload.x, batch.state_tensor())


def test_encode_runtime_batch_preserves_legacy_payload():
    model = _DummyModel()
    payload = LegacyRuntimeBatch(x=torch.ones(1, 3, 2, dtype=torch.float64))

    encoded = encode_runtime_batch(model, payload)

    assert model.last_payload is payload
    assert encoded.shape == (1, 3, 2)
