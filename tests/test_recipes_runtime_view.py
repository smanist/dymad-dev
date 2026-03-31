import torch

from dymad.core import RegularModelContext, RegularSeries
import dymad.models.recipes_corr as recipes_corr


class _EchoTail:
    def __init__(self, tail_dim: int):
        self._tail_dim = tail_dim

    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        return value[..., -self._tail_dim :]


def _regular_context() -> RegularModelContext:
    return RegularModelContext.from_series(
        RegularSeries(
            time=torch.tensor([0.0, 1.0]),
            state=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            control=torch.tensor([[0.5], [0.7]]),
            params=torch.tensor([2.0]),
        )
    )


def test_recipes_corr_encoder_helpers_accept_typed_context():
    context = _regular_context()

    class Dummy:
        net = _EchoTail(1)

    encoded_ctrl = recipes_corr.enc_corr_dif_ctrl(Dummy(), context)
    encoded_auto = recipes_corr.enc_corr_dif_auto(Dummy(), context)

    assert torch.equal(encoded_ctrl, torch.tensor([[1.0, 2.0, 0.5]]))
    assert torch.equal(encoded_auto, torch.tensor([[1.0, 2.0, 2.0]]))


def test_template_corr_alg_dynamics_accepts_typed_context():
    context = _regular_context()

    class DummyAlg:
        processor_net = _EchoTail(2)
        features = staticmethod(lambda z, w: torch.cat([z, z], dim=-1))

        @staticmethod
        def base_dynamics(x, u, f, p):
            assert u is not None
            assert p is not None
            return f

    z = torch.tensor([[1.0, 2.0]])
    out = recipes_corr.TemplateCorrAlg.dynamics(DummyAlg(), z, context)
    assert torch.equal(out, z)


def test_template_corr_dif_dynamics_accepts_typed_context():
    context = _regular_context()

    class DummyDif:
        n_total_state_features = 2
        processor_net = _EchoTail(2)
        latent_net = _EchoTail(1)
        features = staticmethod(lambda z, w: z)

        @staticmethod
        def base_dynamics(x, u, f, p):
            assert u is not None
            assert p is not None
            return f

    z = torch.tensor([[1.0, 2.0, 5.0]])
    out = recipes_corr.TemplateCorrDif.dynamics(DummyDif(), z, context)
    assert torch.equal(out, torch.tensor([[2.0, 5.0, 5.0]]))
