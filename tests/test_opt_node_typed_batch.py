import torch

from dymad.core.series import RegularSeries
from dymad.core.trainer_batch import RegularTrainerBatch
from dymad.training.opt_node import OptNODE


class _SweepScheduler:
    def __init__(self, length=None):
        self._length = length

    def get_length(self):
        return self._length


class _IdentityPredictModel:
    def predict(self, init_states, runtime, ts, **kwargs):
        return runtime.x


def _build_opt_node(*, chop_mode: str, sweep_length=None) -> OptNODE:
    opt = object.__new__(OptNODE)
    opt.schedulers = [object(), _SweepScheduler(sweep_length)]
    opt.chop_mode = chop_mode
    opt.chop_step = 0.5
    opt.device = torch.device("cpu")
    opt.model = _IdentityPredictModel()
    opt.criteria = [torch.nn.MSELoss()]
    opt.criteria_weights = [1.0]
    opt.criteria_names = ["dynamics"]
    opt.ode_method = "dopri5"
    opt.ode_args = {}
    return opt


def _build_regular_batch(n_steps: int = 6) -> RegularTrainerBatch:
    time = torch.linspace(0.0, 1.0, n_steps)
    state = torch.arange(n_steps * 2, dtype=torch.float32).reshape(n_steps, 2)
    control = torch.zeros(n_steps, 1)
    series = RegularSeries(time=time, state=state, control=control)
    return RegularTrainerBatch.collate_series([series])


def test_opt_node_accepts_typed_regular_batch_initial_mode():
    opt = _build_opt_node(chop_mode="initial")
    batch = _build_regular_batch()

    losses = opt._process_batch(batch)

    assert len(losses) == 1
    assert torch.isclose(losses[0], torch.tensor(0.0))


def test_opt_node_accepts_typed_regular_batch_unfold_mode():
    opt = _build_opt_node(chop_mode="unfold", sweep_length=4)
    batch = _build_regular_batch(n_steps=6)

    losses = opt._process_batch(batch)

    assert len(losses) == 1
    assert torch.isclose(losses[0], torch.tensor(0.0))
