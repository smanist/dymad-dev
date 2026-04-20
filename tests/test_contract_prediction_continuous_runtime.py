import torch

from dymad.core.runtime import runtime_from_series
from dymad.core.series import RegularSeries
from dymad.models.prediction import predict_continuous, predict_continuous_np


class _RuntimeAwareModel:
    def __init__(self):
        self.seen_u = None

    def encoder(self, w):
        return w.x

    def decoder(self, z, w):
        if w is None:
            return z
        return z + w.u

    def dynamics(self, z, w):
        self.seen_u = w.u.detach().clone()
        return torch.zeros_like(z)


def test_predict_continuous_passes_interpolated_control_to_dynamics(monkeypatch):
    series = RegularSeries(
        time=torch.tensor([0.0, 1.0]),
        state=torch.tensor([[0.0], [0.0]]),
        control=torch.tensor([[0.0], [1.0]]),
    )
    runtime = runtime_from_series(series)
    model = _RuntimeAwareModel()

    def fake_odeint(func, z0, ts, method=None, **kwargs):
        _ = method, kwargs
        func(torch.tensor(0.5), z0)
        return z0.unsqueeze(0).expand(ts.shape[0], *z0.shape)

    monkeypatch.setattr("dymad.models.prediction.odeint", fake_odeint)

    predict_continuous(
        model,
        x0=series.state[0],
        ts=series.time,
        ws=runtime,
        order="linear",
    )

    assert model.seen_u is not None
    assert torch.allclose(model.seen_u, torch.tensor([[0.5]]))


def test_predict_continuous_np_retries_underflow_with_rk4(monkeypatch):
    series = RegularSeries(
        time=torch.tensor([0.0, 1.0]),
        state=torch.tensor([[0.0], [0.0]]),
        control=torch.tensor([[0.0], [0.0]]),
    )
    runtime = runtime_from_series(series)
    model = _RuntimeAwareModel()
    calls = []

    def fake_odeint(func, z0, ts, method=None, **kwargs):
        calls.append((method, kwargs))
        if len(calls) == 1:
            raise AssertionError("underflow in dt 0.0")
        return z0.unsqueeze(0).expand(ts.shape[0], *z0.shape)

    monkeypatch.setattr("dymad.models.prediction.odeint", fake_odeint)

    pred = predict_continuous_np(
        model,
        x0=series.state[0],
        ts=series.time,
        ws=runtime,
        method="dopri5",
        order="linear",
        rtol=1e-7,
        atol=1e-9,
    )

    assert pred.shape == (2, 1)
    assert calls[0][0] == "dopri5"
    assert calls[1][0] == "rk4"
    assert "rtol" not in calls[1][1]
    assert "atol" not in calls[1][1]
