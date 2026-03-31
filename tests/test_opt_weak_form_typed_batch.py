import torch

from dymad.core.series import RegularSeries
from dymad.core.trainer_batch import RegularTrainerBatch
from dymad.training.opt_base import OptBase
from dymad.training.opt_weak_form import OptWeakForm


class _IdentityPredictModel:
    def encoder(self, runtime):
        return runtime.x

    def dynamics(self, z, runtime):
        return z

    def decoder(self, z, runtime):
        return z

    def predict(self, x0, runtime, ts, **kwargs):
        return runtime.x


def _build_regular_batch(n_steps: int = 6) -> RegularTrainerBatch:
    time = torch.linspace(0.0, 1.0, n_steps)
    state = torch.arange(n_steps * 2, dtype=torch.float32).reshape(n_steps, 2)
    control = torch.zeros(n_steps, 1)
    series = RegularSeries(time=time, state=state, control=control)
    return RegularTrainerBatch.collate_series([series])


def test_opt_weak_form_accepts_typed_regular_batch():
    opt = object.__new__(OptWeakForm)
    opt.device = torch.device("cpu")
    opt.model = _IdentityPredictModel()
    opt.criteria = [torch.nn.MSELoss()]
    opt.criteria_weights = [1.0]
    opt.criteria_names = ["dynamics"]
    opt.N = 3
    opt.dN = 1
    opt.C = torch.ones((3, 1), dtype=torch.float32)
    opt.D = torch.ones((3, 1), dtype=torch.float32)

    losses = opt._process_batch(_build_regular_batch())

    assert len(losses) == 1
    assert torch.isclose(losses[0], torch.tensor(0.0))


def test_opt_base_prediction_criterion_accepts_typed_regular_batch():
    opt = object.__new__(OptBase)
    opt.model = _IdentityPredictModel()
    opt.criteria = [torch.nn.MSELoss()]
    opt.criteria_names = ["mse"]
    opt.model_name = "typed-test"
    opt.results_prefix = "."
    opt.config = {}

    score = opt.evaluate_prediction_criterion_single(_build_regular_batch(), plot=False)

    assert score == 0.0
