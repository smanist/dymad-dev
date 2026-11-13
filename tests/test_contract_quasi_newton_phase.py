import copy
import logging
from dataclasses import replace

import pytest
import torch
from torch.utils.data import DataLoader

from dymad.core import GraphSeries, RegularSeries, RegularTrainerBatch
from dymad.numerics.bfgs import BFGS, LBFGS
from dymad.training.execution_services import ExecutionServices
from dymad.training.phase_runtime import (
    ArtifactRegistry,
    ModelArtifact,
    OptimizerStateArtifact,
    PhaseContext,
    TrainerState,
    TrainingHistoryArtifact,
)
from dymad.training.phases import (
    BaseOptimizerPhase,
    NodeOptimizerPhase,
    OptimizerPhaseSpec,
    PhaseSpecValidationError,
    WeakFormOptimizerPhase,
)
from dymad.training.trainer_run import TrainerRun


class _LinearFlow(torch.nn.Module):
    """Tiny NODE contract model with an analytic controlled constant vector field."""

    def __init__(self, config=None, metadata=None, *, dtype=torch.float64, device="cpu"):
        super().__init__()
        self.rate = torch.nn.Parameter(torch.tensor(0.1, dtype=dtype, device=device))
        self.dropout = torch.nn.Dropout((config or {}).get("dropout", 0.0))

    def predict(self, initial, runtime, times, **kwargs):
        increments = (times - times[:, :1])[..., None] * runtime.u
        return initial[:, None, :] + self.dropout(self.rate * increments)

    def encoder(self, runtime):
        return runtime.x

    def decoder(self, latent, runtime):
        return (1 + self.rate) * latent


class _WeakFlow(_LinearFlow):
    def __init__(self, config=None, metadata=None, *, dtype=torch.float64, device="cpu"):
        super().__init__(config, metadata, dtype=dtype, device=device)
        self.scale = torch.nn.Parameter(torch.tensor(1.2, dtype=dtype, device=device))

    def encoder(self, runtime):
        return self.scale * runtime.x

    def dynamics(self, latent, runtime):
        return self.rate * latent

    def decoder(self, latent, runtime):
        return latent


def _context(batch_size=2):
    times = torch.linspace(0, 1, 5, dtype=torch.float64)
    data = [
        RegularSeries(
            time=times.clone(),
            state=(0.2 * index + 0.7 * (index + 1) * times)[:, None],
            control=torch.full((5, 1), float(index + 1), dtype=torch.float64),
        )
        for index in range(5)
    ]
    validation = copy.deepcopy(data)
    loader = lambda items: DataLoader(
        items, batch_size=batch_size, shuffle=True, collate_fn=RegularTrainerBatch.collate_series
    )
    return PhaseContext(
        train_set=data,
        valid_set=validation,
        train_loader=loader(data),
        valid_loader=loader(validation),
        train_md={"dt_and_n_steps": [(0.25, 5)]},
        valid_md={"dt_and_n_steps": [(0.25, 5)]},
    )


def _phase(tmp_path, kind="bfgs", *, criterion=None, trainer="NODE", **options):
    config = {
        "model": {"name": "tiny"},
        "path": {"checkpoint_prefix": str(tmp_path), "results_prefix": str(tmp_path)},
    }
    if criterion is not None:
        config["criterion"] = criterion
    phase_class = NodeOptimizerPhase if trainer == "NODE" else WeakFormOptimizerPhase
    if trainer == "Weak":
        options.setdefault("weak_form_params", {"N": 5, "dN": 1, "ordpol": 2, "ordint": 2})
    return phase_class(
        spec=OptimizerPhaseSpec(
            name=kind,
            trainer=trainer,
            config={"optimizer": kind, "n_epochs": 4, "save_interval": 100, **options},
        ),
        config=config,
        model_class=_LinearFlow if trainer == "NODE" else _WeakFlow,
        dtype=torch.float64,
        execution_services=ExecutionServices.from_config(config, torch.device("cpu")),
    )


def _prepare(phase, model, context, artifacts=None):
    artifacts = artifacts or ArtifactRegistry()
    state = phase._build_optimizer_artifact(model, artifacts)
    phase._customize_optimizer_artifact(state, model, context, logging.getLogger("test"))
    return state


@pytest.mark.parametrize("kind", ["bfgs", "lbfgs"])
def test_qn_full_objective_matches_full_batch_with_uneven_batches_and_multiple_mse(tmp_path, kind):
    criterion = {"dynamics": {"weight": 2.0}, "recon": {"weight": 0.3}}
    phase = _phase(tmp_path, kind, criterion=criterion)
    model, context = _LinearFlow({"dropout": 0.5}), _context(2)
    state = _prepare(phase, model, context)
    assert [len(batch) for batch in phase._qn_train_batches] == [2, 2, 1]
    assert state.criteria_names[:-1] == ["dynamics", "recon"]
    first, first_items = phase._full_objective(
        model, state, phase._qn_train_batches, "rk4", {}, backward=True
    )
    first_gradient = model.rate.grad.clone()
    repeated, repeated_items = phase._full_objective(
        model, state, phase._qn_train_batches, "rk4", {}, backward=True
    )
    torch.testing.assert_close(repeated, first, rtol=0, atol=0)
    torch.testing.assert_close(model.rate.grad, first_gradient, rtol=0, atol=0)
    assert repeated_items == first_items
    full_batch = RegularTrainerBatch.collate_series(context.train_set)
    state.optimizer.zero_grad(set_to_none=True)
    losses = phase._compute_losses(model, state, full_batch, "rk4", {})
    full = phase._aggregate_losses(losses, state.criteria_weights)
    full.backward()
    torch.testing.assert_close(first, full.detach(), rtol=1e-13, atol=1e-14)
    torch.testing.assert_close(first_gradient, model.rate.grad, rtol=1e-13, atol=1e-14)
    assert first_items == pytest.approx([float(loss.detach()) for loss in losses])
    assert model.training is False
    assert model.dropout.training is False
    value, items = phase._evaluate_loader(model, state, context.valid_loader, "rk4", {})
    assert value == pytest.approx(float(full.detach()))
    assert items == pytest.approx(first_items)


@pytest.mark.parametrize("kind", ["bfgs", "lbfgs"])
def test_qn_history_components_follow_accepted_iterate_and_stationary_cache(tmp_path, kind):
    phase, model, context = _phase(tmp_path, kind), _LinearFlow(), _context()
    state = _prepare(phase, model, context)
    for _ in range(5):
        loss, items, converged = phase._train_epoch(
            model, state, context, "rk4", {}, logging.getLogger("test")
        )
        measured, measured_items = phase._evaluate_loader(
            model, state, context.train_loader, "rk4", {}
        )
        assert loss == pytest.approx(measured, abs=1e-15)
        assert items == pytest.approx(measured_items, abs=1e-15)
        if converged:
            break
    assert converged
    before = copy.deepcopy(state.optimizer.state_dict())
    repeated = phase._train_epoch(model, state, context, "rk4", {}, logging.getLogger("test"))
    assert repeated == (loss, items, True)
    assert state.optimizer.state[model.rate]["func_evals"] == before["state"][0]["func_evals"]
    assert state.schedulers == []


@pytest.mark.parametrize("kind,optimizer_class", [("bfgs", BFGS), ("lbfgs", LBFGS)])
def test_qn_handoff_uses_terminal_adam_parameters_and_fresh_state(tmp_path, kind, optimizer_class):
    model, context = _LinearFlow(), _context()
    artifacts = ArtifactRegistry()
    adam = _phase(tmp_path, "adam", learning_rate=0.05)
    prior = _prepare(adam, model, context, artifacts)
    adam._train_epoch(model, prior, context, "rk4", {}, logging.getLogger("test"))
    terminal = model.rate.detach().clone()
    phase = _phase(tmp_path, kind)
    state = _prepare(phase, model, context, artifacts)
    assert isinstance(state.optimizer, optimizer_class)
    assert state.optimizer.state == {}
    torch.testing.assert_close(model.rate, terminal, rtol=0, atol=0)
    phase._train_epoch(model, state, context, "rk4", {}, logging.getLogger("test"))
    assert state.optimizer.state
    final = model.rate.detach().clone()
    fresh = phase._build_optimizer_artifact(model, artifacts)
    assert fresh.optimizer.state == {}
    torch.testing.assert_close(model.rate, final, rtol=0, atol=0)
    returned_adam = adam._build_optimizer_artifact(model, artifacts)
    assert type(returned_adam.optimizer) is torch.optim.Adam
    assert returned_adam.optimizer.state == {}


def test_qn_two_phase_workflow_rebaselines_prefix_best_and_phase_convergence(tmp_path, monkeypatch):
    monkeypatch.setattr(NodeOptimizerPhase, "_write_progress_plots", lambda *args, **kwargs: None)
    context, artifacts, state = _context(), ArtifactRegistry(), TrainerState(config={})
    adam = _phase(tmp_path, "adam", n_epochs=2, learning_rate=0.02, sweep_lengths=[2])
    adam.execute(
        trainer_state=state,
        phase_context=context,
        artifacts=artifacts,
        run_name="tiny",
        logger=logging.getLogger("test"),
    )
    model = artifacts.require("model", ModelArtifact).model
    history = artifacts.require("history", TrainingHistoryArtifact)
    terminal_adam = model.rate.detach().clone()
    # Simulate a converged short-prefix phase with a noncomparable best score.
    state.converged = True
    state.best_loss = {"valid_total": -1.0}
    history.best_model_state_dict = {"rate": torch.tensor(99.0, dtype=torch.float64)}
    baseline = _phase(tmp_path, "bfgs", n_epochs=0)
    baseline.execute(
        trainer_state=state,
        phase_context=context,
        artifacts=artifacts,
        run_name="tiny",
        logger=logging.getLogger("test"),
    )
    assert state.converged is False
    assert state.epoch == 2
    assert state.best_loss["valid_total"] > 0
    torch.testing.assert_close(history.best_model_state_dict["rate"], terminal_adam)
    refinement = _phase(tmp_path, "bfgs", n_epochs=4)
    refinement.execute(
        trainer_state=state,
        phase_context=context,
        artifacts=artifacts,
        run_name="tiny",
        logger=logging.getLogger("test"),
    )
    assert state.epoch > 2
    assert state.best_loss["valid_total"] < 1e-12
    assert state.converged is True
    assert history.hist[-1]["valid_total"][-1] < 1e-12


def test_qn_phase_boundary_checkpoint_preserves_parameter_binding(tmp_path):
    phase, model, context, artifacts = (
        _phase(tmp_path),
        _LinearFlow(),
        _context(),
        ArtifactRegistry(),
    )
    artifacts.put("model", ModelArtifact(model, {}, {}, {}, torch.float64))
    optimizer_state = _prepare(phase, model, context, artifacts)
    phase._train_epoch(model, optimizer_state, context, "rk4", {}, logging.getLogger("test"))
    run = TrainerRun(
        config={"model": {"name": "tiny"}, "phases": [{"trainer": "NODE"}]},
        model_class=_LinearFlow,
        device=torch.device("cpu"),
        dtype=torch.float64,
        run_name="tiny",
        checkpoint_prefix=str(tmp_path),
        results_prefix=str(tmp_path),
    )
    run.save_run_checkpoint(TrainerState(config=run.config, phase_cursor=1), artifacts)
    loaded_state, loaded = run.load_run_checkpoint()
    loaded_model = loaded.require("model", ModelArtifact).model
    loaded_optimizer = loaded.require("optimizer_state", OptimizerStateArtifact).optimizer
    assert loaded_state.phase_cursor == 1
    assert loaded_optimizer.param_groups[0]["params"][0] is loaded_model.rate
    assert loaded_optimizer.state[loaded_model.rate]["n_iter"] > 0
    fresh = phase._build_optimizer_artifact(loaded_model, loaded)
    assert fresh.optimizer.state == {}
    assert fresh.optimizer.param_groups[0]["params"][0] is loaded_model.rate


def test_qn_rebaselines_better_saved_checkpoint_without_changing_terminal_handoff(tmp_path):
    phase, model, context = _phase(tmp_path, n_epochs=0), _LinearFlow(), _context()
    terminal = model.rate.detach().clone()
    artifacts = ArtifactRegistry()
    artifacts.put("model", ModelArtifact(model, {}, {}, {}, torch.float64))
    history = artifacts.put(
        "history",
        TrainingHistoryArtifact(
            best_loss={"valid_total": 123.0},
            best_model_state_dict={"rate": torch.tensor(0.7, dtype=torch.float64)},
        ),
    )
    state = TrainerState(config={}, epoch=3, best_loss={"valid_total": 123.0})
    phase.execute(
        trainer_state=state,
        phase_context=context,
        artifacts=artifacts,
        run_name="tiny",
        logger=logging.getLogger("test"),
    )
    torch.testing.assert_close(model.rate, terminal, rtol=0, atol=0)
    assert state.best_loss["valid_total"] < 1e-25
    assert history.best_loss["valid_total"] < 1e-25
    assert float(history.best_model_state_dict["rate"]) == pytest.approx(0.7)
    optimizer = artifacts.require("optimizer_state", OptimizerStateArtifact).optimizer
    assert optimizer.state == {}
    assert optimizer.param_groups[0]["params"][0] is model.rate


@pytest.mark.parametrize("trainer", ["NODE", "Weak"])
@pytest.mark.parametrize("invalid", ["ragged", "graph", "non_mse", "sum_mse", "batchnorm"])
def test_qn_rejects_unsupported_effective_runtime_objectives(tmp_path, invalid, trainer):
    phase, model, context = _phase(tmp_path, trainer=trainer), _LinearFlow(), _context()
    if invalid == "ragged":
        context.train_set[-1] = context.train_set[-1].slice_steps(0, 3)
    elif invalid == "graph":
        context.train_set = [
            GraphSeries(
                time=torch.linspace(0, 1, 5),
                node_state=torch.zeros(5, 2, 1),
                edge_index=torch.tensor([[0], [1]]),
            )
        ]
    elif invalid == "batchnorm":
        model.normalization = torch.nn.BatchNorm1d(1, dtype=torch.float64)
    else:
        phase.config["criterion"] = {
            "dynamics": {
                "type": "wmse" if invalid == "non_mse" else "mse",
                "params": {} if invalid == "non_mse" else {"reduction": "sum"},
            }
        }
    with pytest.raises(PhaseSpecValidationError, match="BFGS/L-BFGS"):
        _prepare(phase, model, context)


@pytest.mark.parametrize("kind", ["bfgs", "lbfgs"])
def test_weak_qn_preserves_true_encoder_gradient_and_microbatch_objective(tmp_path, kind):
    phase = _phase(
        tmp_path,
        kind,
        trainer="Weak",
        criterion={"dynamics": {"weight": 1.0}, "recon": {"weight": 0.4}},
    )
    model, context = _WeakFlow(), _context()
    state = _prepare(phase, model, context)
    loss, items = phase._full_objective(
        model, state, phase._qn_train_batches, "rk4", {}, backward=True
    )
    parameters = list(model.parameters())
    gradient = torch.stack([parameter.grad.clone() for parameter in parameters])
    epsilon = 1e-6
    finite_difference = []
    for parameter in parameters:
        original = parameter.detach().clone()
        values = []
        for sign in (1, -1):
            with torch.no_grad():
                parameter.copy_(original + sign * epsilon)
            value, _ = phase._full_objective(
                model, state, phase._qn_train_batches, "rk4", {}, backward=False
            )
            values.append(value)
        with torch.no_grad():
            parameter.copy_(original)
        finite_difference.append((values[0] - values[1]) / (2 * epsilon))
    # The trainable encoder enters pred_weak AND true_weak; detaching the latter fails this check.
    torch.testing.assert_close(gradient, torch.stack(finite_difference), rtol=1e-6, atol=1e-8)
    assert gradient[1].abs() > 1e-3
    full = (RegularTrainerBatch.collate_series(context.train_set),)
    full_loss, full_items = phase._full_objective(model, state, full, "rk4", {}, backward=True)
    torch.testing.assert_close(full_loss, loss, rtol=1e-13, atol=1e-14)
    torch.testing.assert_close(
        torch.stack([parameter.grad for parameter in parameters]), gradient, rtol=1e-13, atol=1e-14
    )
    assert full_items == pytest.approx(items)
    before_weights = (state._weak_C.clone(), state._weak_D.clone())
    phase._train_epoch(model, state, context, "rk4", {}, logging.getLogger("test"))
    torch.testing.assert_close(state._weak_C, before_weights[0], rtol=0, atol=0)
    torch.testing.assert_close(state._weak_D, before_weights[1], rtol=0, atol=0)
    assert state.schedulers == []


def test_adam_to_weak_bfgs_runs_shared_training_path(tmp_path, monkeypatch):
    monkeypatch.setattr(BaseOptimizerPhase, "_write_progress_plots", lambda *args, **kwargs: None)
    context, artifacts, state = _context(), ArtifactRegistry(), TrainerState(config={})
    criterion = {"dynamics": {"weight": 1.0}, "recon": {"weight": 0.4}}
    adam = _phase(tmp_path, "adam", trainer="Weak", criterion=criterion, n_epochs=2)
    adam.execute(
        trainer_state=state,
        phase_context=context,
        artifacts=artifacts,
        run_name="weak",
        logger=logging.getLogger("test"),
    )
    model = artifacts.require("model", ModelArtifact).model
    old_rate = model.rate.detach().clone()
    old_best = state.best_loss["valid_total"]
    refinement = _phase(tmp_path, "bfgs", trainer="Weak", criterion=criterion, n_epochs=10)
    result = refinement.execute(
        trainer_state=state,
        phase_context=context,
        artifacts=artifacts,
        run_name="weak",
        logger=logging.getLogger("test"),
    )
    assert result.trainer_state.epoch > 2
    assert not torch.equal(model.rate.detach(), old_rate)
    assert state.best_loss["valid_total"] < old_best
    optimizer = artifacts.require("optimizer_state", OptimizerStateArtifact).optimizer
    assert isinstance(optimizer, BFGS)
    assert optimizer.state[model.rate]["n_iter"] > 0
    assert artifacts.require("history", TrainingHistoryArtifact).hist[-1]["train_recon"]


@pytest.mark.parametrize(
    "invalid",
    [
        "nonuniform_train",
        "nonuniform_validation",
        "large_offset_float32",
        "validation_dt",
        "metadata_steps",
        "metadata_dt",
        "short_window",
        "quadrature",
        "stride",
        "nonpositive_dt",
    ],
)
def test_weak_qn_requires_one_valid_quadrature_for_train_and_validation(tmp_path, invalid):
    phase, model, context = _phase(tmp_path, trainer="Weak"), _WeakFlow(), _context()
    if invalid in {"nonuniform_train", "nonuniform_validation"}:
        dataset = context.train_set if invalid == "nonuniform_train" else context.valid_set
        dataset[-1].time[2] += 0.02
    elif invalid == "large_offset_float32":
        for dataset in (context.train_set, context.valid_set):
            for index, series in enumerate(dataset):
                dataset[index] = replace(
                    series,
                    time=torch.tensor(
                        [1e6, 1e6 + 1, 1e6 + 2.5, 1e6 + 3.5, 1e6 + 4.5], dtype=torch.float32
                    ),
                )
        context.train_md["dt_and_n_steps"] = [(1.0, 5)]
        context.valid_md["dt_and_n_steps"] = [(1.0, 5)]
    elif invalid == "validation_dt":
        for series in context.valid_set:
            series.time.mul_(2)
        context.valid_md["dt_and_n_steps"] = [(0.5, 5)]
    elif invalid == "metadata_steps":
        context.valid_md["dt_and_n_steps"] = [(0.25, 6)]
    elif invalid == "metadata_dt":
        context.valid_md["dt_and_n_steps"] = [(0.3, 5)]
    elif invalid == "nonpositive_dt":
        context.train_md["dt_and_n_steps"] = [(0.0, 5)]
    else:
        key, value = {"short_window": ("N", 7), "quadrature": ("ordint", 3), "stride": ("dN", 0)}[
            invalid
        ]
        phase.spec.config["weak_form_params"][key] = value
    with pytest.raises(PhaseSpecValidationError, match="Weak BFGS/L-BFGS"):
        _prepare(phase, model, context)


def test_weak_qn_accepts_uniform_float32_time_grids(tmp_path):
    phase, model, context = _phase(tmp_path, trainer="Weak"), _WeakFlow(), _context()
    for dataset in (context.train_set, context.valid_set):
        for index in range(len(dataset)):
            dataset[index] = RegularSeries(
                time=torch.linspace(0, 6, 301, dtype=torch.float32),
                state=torch.zeros(301, 1, dtype=torch.float64),
                control=torch.ones(301, 1, dtype=torch.float64),
            )
    context.train_md["dt_and_n_steps"] = [(0.02, 301)]
    context.valid_md["dt_and_n_steps"] = [(0.02, 301)]
    _prepare(phase, model, context)
