import torch

from dymad.core.series import RegularSeries
from dymad.core.trainer_batch import RegularTrainerBatch
from dymad.training.phase_runtime import OptimizerStateArtifact
from dymad.training.phases import NodeOptimizerPhase, OptimizerPhaseSpec, WeakFormOptimizerPhase


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
    opt = object.__new__(WeakFormOptimizerPhase)
    opt.device = torch.device("cpu")
    opt.model = _IdentityPredictModel()
    optimizer_state = OptimizerStateArtifact(
        optimizer=torch.optim.Adam([torch.nn.Parameter(torch.tensor(0.0))]),
        criteria=[torch.nn.MSELoss()],
        criteria_weights=[1.0],
        criteria_names=["dynamics"],
    )
    optimizer_state._weak_N = 3
    optimizer_state._weak_dN = 1
    optimizer_state._weak_C = torch.ones((3, 1), dtype=torch.float32)
    optimizer_state._weak_D = torch.ones((3, 1), dtype=torch.float32)

    losses = opt._compute_losses(opt.model, optimizer_state, _build_regular_batch(), "dopri5", {})

    assert len(losses) == 1
    assert torch.isclose(losses[0], torch.tensor(0.0))


def test_opt_base_prediction_criterion_accepts_typed_regular_batch():
    opt = object.__new__(NodeOptimizerPhase)
    opt.model = _IdentityPredictModel()
    optimizer_state = OptimizerStateArtifact(
        optimizer=torch.optim.Adam([torch.nn.Parameter(torch.tensor(0.0))]),
        criteria=[torch.nn.MSELoss(), torch.nn.MSELoss()],
        criteria_weights=[1.0],
        criteria_names=["dynamics", "mse"],
    )
    opt.spec = OptimizerPhaseSpec(name="node", trainer="NODE", config={})

    score = opt._evaluate_prediction_criterion_single(
        opt.model,
        optimizer_state,
        _build_regular_batch(),
        method="dopri5",
    )

    assert score == 0.0
