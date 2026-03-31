import torch

import dymad.models.model_base as model_base_module
from dymad.core import RegularModelContext, RegularSeries
from dymad.models.model_base import ComposedDynamics
from dymad.models.runtime_view import build_component_input_view


def test_model_base_forward_routes_through_runtime_contract(monkeypatch):
    seam_trace: dict[str, object] = {}
    payload_trace: dict[str, object] = {}

    def fake_materialize_model_base_forward_payload(**kwargs):
        seam_trace["kwargs"] = kwargs
        return RegularModelContext.from_series(
            RegularSeries(
                time=torch.tensor([0.0]),
                state=torch.tensor([[1.0, 2.0]]),
                control=torch.tensor([[0.5]]),
            )
        )

    monkeypatch.setattr(
        model_base_module,
        "materialize_model_base_forward_payload",
        fake_materialize_model_base_forward_payload,
    )

    def encoder(_net, w):
        payload_trace["type"] = type(w)
        return build_component_input_view(w).state

    def features(z, _w):
        return z

    def composer(_net, s, _z, _w):
        return s

    def decoder(_net, z, _w):
        return z

    model = ComposedDynamics(
        encoder=encoder,
        dynamics=(features, composer),
        decoder=decoder,
    )

    z, z_dot, x_hat = model.forward(
        t=torch.tensor([0.0]),
        x=torch.tensor([[9.0, 8.0]]),
        u=torch.tensor([[0.9]]),
    )

    assert "kwargs" in seam_trace
    assert isinstance(payload_trace["type"], type)
    assert payload_trace["type"] is RegularModelContext
    assert torch.equal(z, torch.tensor([[1.0, 2.0]]))
    assert torch.equal(z_dot, z)
    assert torch.equal(x_hat, z)
