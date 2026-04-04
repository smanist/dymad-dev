import torch

from dymad.training.helper import RunState
from dymad.training.phase_runtime import (
    compose_run_state,
    run_state_to_phase_context,
    run_state_to_trainer_state,
)
from dymad.training import phase_pipeline
from dymad.training import stacked_opt


def _build_data_state():
    marker = object()
    return RunState(
        config={"path": {"results_prefix": "."}},
        device=torch.device("cpu"),
        epoch=3,
        best_loss={"valid_total": 1.23},
        hist=[{"epoch": [0]}],
        crit=[[0, 1.0, 2.0]],
        epoch_times=[0.1],
        converged=False,
        train_set=marker,
        valid_set=marker,
        train_loader=marker,
        valid_loader=marker,
        train_md={"dt_and_n_steps": [(0.1, 5)]},
        valid_md={"dt_and_n_steps": [(0.1, 5)]},
    ), marker


def test_phase_runtime_round_trip_preserves_state_and_context():
    state, marker = _build_data_state()
    trainer_state = run_state_to_trainer_state(state)
    phase_context = run_state_to_phase_context(state)
    rebuilt = compose_run_state(trainer_state, phase_context)

    assert rebuilt.epoch == state.epoch
    assert rebuilt.best_loss == state.best_loss
    assert rebuilt.best_loss is not state.best_loss
    assert rebuilt.hist == state.hist
    assert rebuilt.crit == state.crit
    assert rebuilt.train_loader is marker
    assert rebuilt.valid_loader is marker
    assert rebuilt.train_set is marker
    assert rebuilt.valid_set is marker


def test_stacked_opt_uses_phase_runtime_adapters(monkeypatch):
    calls = {"trainer": 0, "context": 0, "compose": 0}
    original_to_trainer = phase_pipeline.run_state_to_trainer_state
    original_to_context = phase_pipeline.run_state_to_phase_context
    original_compose = phase_pipeline.compose_run_state

    def wrapped_to_trainer(state):
        calls["trainer"] += 1
        return original_to_trainer(state)

    def wrapped_to_context(state):
        calls["context"] += 1
        return original_to_context(state)

    def wrapped_compose(trainer_state, context):
        calls["compose"] += 1
        return original_compose(trainer_state, context)

    monkeypatch.setattr(phase_pipeline, "run_state_to_trainer_state", wrapped_to_trainer)
    monkeypatch.setattr(phase_pipeline, "run_state_to_phase_context", wrapped_to_context)
    monkeypatch.setattr(phase_pipeline, "compose_run_state", wrapped_compose)

    class _FakeTrainer:
        def __init__(self, config, config_phase, model_class, run_state, device, dtype):
            self.run_state = run_state
            self.hist = [{"phase": config_phase.get("name", "p0")}]

        def train(self):
            return self.run_state.epoch

        def export_run_state(self, epoch):
            return RunState(
                config=self.run_state.config,
                device=self.run_state.device,
                epoch=epoch + 1,
                best_loss=self.run_state.best_loss,
                hist=self.run_state.hist,
                crit=self.run_state.crit,
                epoch_times=self.run_state.epoch_times,
                converged=self.run_state.converged,
                train_set=self.run_state.train_set,
                valid_set=self.run_state.valid_set,
                train_loader=self.run_state.train_loader,
                valid_loader=self.run_state.valid_loader,
                train_md=self.run_state.train_md,
                valid_md=self.run_state.valid_md,
            )

    monkeypatch.setitem(phase_pipeline.OPT_REGISTRY, "Fake", _FakeTrainer)
    initial_state, marker = _build_data_state()
    initial_state.config = {
        "path": {"results_prefix": "."},
        "log": {"stdout": True, "level": "warning"},
        "phases": [{"name": "phase_0", "trainer": "Fake"}],
    }

    opt = stacked_opt.StackedOpt(
        config=initial_state.config,
        model_class=object,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    results = opt.run(initial_state=initial_state)

    assert calls["trainer"] >= 2
    assert calls["context"] >= 2
    assert calls["compose"] >= 2
    assert results[-1].run_state.epoch == 4
    assert results[-1].run_state.train_loader is marker


def test_stacked_opt_wraps_phase_pipeline(monkeypatch):
    calls = {"init": 0, "run": 0}

    class _FakePipeline:
        def __init__(self, config, model_class, device, dtype):
            calls["init"] += 1
            self.config = config
            self.phases = config["phases"]

        def run(self, initial_state):
            calls["run"] += 1
            return [initial_state]

    monkeypatch.setattr(stacked_opt, "PhasePipeline", _FakePipeline)

    state, _ = _build_data_state()
    state.config = {
        "path": {"results_prefix": "."},
        "log": {"stdout": True, "level": "warning"},
        "phases": [{"name": "phase_0", "trainer": "Linear"}],
    }

    opt = stacked_opt.StackedOpt(
        config=state.config,
        model_class=object,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    results = opt.run(initial_state=state)

    assert calls["init"] == 1
    assert calls["run"] == 1
    assert opt.phases == state.config["phases"]
    assert results == [state]
