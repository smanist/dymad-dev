import torch

from dymad.core.series import RegularSeries
from dymad.core.trainer_batch import RegularTrainerBatch
from dymad.training.phase_runtime import OptimizerStateArtifact
from dymad.training.phases import NodeOptimizerPhase, OptimizerPhaseSpec


class _SweepScheduler:
    def __init__(self, length=None):
        self._length = length

    def get_length(self):
        return self._length


class _IdentityPredictModel:
    def __init__(self):
        self.calls = 0

    def predict(self, init_states, runtime, ts, **kwargs):
        _ = init_states, ts, kwargs
        self.calls += 1
        return runtime.x


def _build_opt_node(*, chop_mode: str, sweep_length=None) -> NodeOptimizerPhase:
    opt = object.__new__(NodeOptimizerPhase)
    opt.schedulers = [object(), _SweepScheduler(sweep_length)]
    opt.device = torch.device("cpu")
    opt.model = _IdentityPredictModel()
    opt.ode_method = "dopri5"
    opt.ode_args = {}
    opt.spec = OptimizerPhaseSpec(
        name="node",
        trainer="NODE",
        config={"chop_mode": chop_mode, "chop_step": 0.5},
    )
    return opt


def _build_regular_batch(n_steps: int = 6) -> RegularTrainerBatch:
    time = torch.linspace(0.0, 1.0, n_steps)
    state = torch.arange(n_steps * 2, dtype=torch.float32).reshape(n_steps, 2)
    control = torch.zeros(n_steps, 1)
    series = RegularSeries(time=time, state=state, control=control)
    return RegularTrainerBatch.collate_series([series])


def _build_ragged_regular_batch() -> RegularTrainerBatch:
    series_a = RegularSeries(
        time=torch.linspace(0.0, 1.0, 5),
        state=torch.arange(10, dtype=torch.float32).reshape(5, 2),
        control=torch.zeros(5, 1),
    )
    series_b = RegularSeries(
        time=torch.linspace(0.0, 0.75, 4),
        state=torch.arange(8, dtype=torch.float32).reshape(4, 2),
        control=torch.zeros(4, 1),
    )
    return RegularTrainerBatch.collate_series([series_a, series_b])


def test_opt_node_accepts_typed_regular_batch_initial_mode():
    opt = _build_opt_node(chop_mode="initial")
    batch = _build_regular_batch()
    optimizer_state = OptimizerStateArtifact(
        optimizer=torch.optim.Adam([torch.nn.Parameter(torch.tensor(0.0))]),
        schedulers=opt.schedulers,
        criteria=[torch.nn.MSELoss()],
        criteria_weights=[1.0],
        criteria_names=["dynamics"],
    )

    losses = opt._compute_losses(opt.model, optimizer_state, batch, opt.ode_method, opt.ode_args)

    assert len(losses) == 1
    assert torch.isclose(losses[0], torch.tensor(0.0))


def test_opt_node_accepts_typed_regular_batch_unfold_mode():
    opt = _build_opt_node(chop_mode="unfold", sweep_length=4)
    batch = _build_regular_batch(n_steps=6)
    optimizer_state = OptimizerStateArtifact(
        optimizer=torch.optim.Adam([torch.nn.Parameter(torch.tensor(0.0))]),
        schedulers=opt.schedulers,
        criteria=[torch.nn.MSELoss()],
        criteria_weights=[1.0],
        criteria_names=["dynamics"],
    )

    losses = opt._compute_losses(opt.model, optimizer_state, batch, opt.ode_method, opt.ode_args)

    assert len(losses) == 1
    assert torch.isclose(losses[0], torch.tensor(0.0))


def test_opt_node_keeps_ragged_batch_prediction_batched():
    opt = _build_opt_node(chop_mode="initial")
    batch = _build_ragged_regular_batch()
    optimizer_state = OptimizerStateArtifact(
        optimizer=torch.optim.Adam([torch.nn.Parameter(torch.tensor(0.0))]),
        schedulers=opt.schedulers,
        criteria=[torch.nn.MSELoss()],
        criteria_weights=[1.0],
        criteria_names=["dynamics"],
    )

    losses = opt._compute_losses(opt.model, optimizer_state, batch, opt.ode_method, opt.ode_args)

    assert len(losses) == 1
    assert torch.isclose(losses[0], torch.tensor(0.0))
    assert opt.model.calls == 1
