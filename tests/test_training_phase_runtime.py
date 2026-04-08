import pytest
import torch

from dymad.training import driver
from dymad.training.execution_services import ExecutionServices
import dymad.training.phases as phases_module
from dymad.training.phase_runtime import (
    ArtifactRegistry,
    ModelArtifact,
    OptimizerStateArtifact,
    PhaseContext,
    PhaseRecord,
    PhaseResult,
    TrainerState,
    TrainingCheckpointError,
)
from dymad.training.phases import (
    AnalysisPhaseSpec,
    ExportPhaseSpec,
    LinearSolvePhaseSpec,
    OptimizerPhaseSpec,
    PhaseSpecValidationError,
    build_phase,
    normalize_phase_specs,
)
from dymad.training.trainer_run import TrainerRun
from dymad.utils import load_config


def _build_phase_context():
    marker = object()
    return (
        PhaseContext(
            train_set=[marker],
            valid_set=[marker],
            train_loader=marker,
            valid_loader=marker,
            train_md={"dt_and_n_steps": [(0.1, 5)]},
            valid_md={"dt_and_n_steps": [(0.1, 5)]},
        ),
        marker,
    )


def test_phase_result_get_metric_reads_typed_trainer_state():
    context, _ = _build_phase_context()
    trainer_state = TrainerState(config={}, best_loss={"valid_total": 0.456})
    result = PhaseResult(
        name="phase_0",
        kind="optimizer",
        trainer_state=trainer_state,
        phase_context=context,
        artifacts=ArtifactRegistry(),
    )

    assert result.get_metric("total") == 0.456


def test_normalize_phase_specs_expands_repeat_schedule():
    specs = normalize_phase_specs(
        {
            "model": {"name": "demo"},
            "phases": [
                {
                    "repeat": {
                        "times": 2,
                        "phases": [
                            {
                                "type": "linear_solve",
                                "name": "ls",
                                "method": "truncated",
                                "params": 2,
                            },
                            {
                                "type": "optimizer",
                                "name": "node",
                                "trainer": "NODE",
                                "n_epochs": 5,
                            },
                        ],
                    }
                }
            ],
        }
    )

    assert isinstance(specs[0], LinearSolvePhaseSpec)
    assert isinstance(specs[1], OptimizerPhaseSpec)
    assert isinstance(specs[2], LinearSolvePhaseSpec)
    assert isinstance(specs[3], OptimizerPhaseSpec)
    assert isinstance(specs[-4], AnalysisPhaseSpec)
    assert isinstance(specs[-3], ExportPhaseSpec)
    assert isinstance(specs[-2], ExportPhaseSpec)
    assert isinstance(specs[-1], ExportPhaseSpec)


def test_normalize_phase_specs_warns_for_analysis_and_export_inside_repeat():
    with pytest.warns(UserWarning, match="Repeat block 'cycle' contains an analysis phase"):
        with pytest.warns(UserWarning, match="Repeat block 'cycle' contains an export phase"):
            specs = normalize_phase_specs(
                {
                    "model": {"name": "demo"},
                    "phases": [
                        {
                            "repeat": {
                                "name": "cycle",
                                "times": 1,
                                "phases": [
                                    {"type": "analysis", "name": "inspect"},
                                    {
                                        "type": "export",
                                        "name": "save_model",
                                        "export_kind": "best_model",
                                    },
                                ],
                            }
                        }
                    ],
                }
            )

    assert isinstance(specs[0], AnalysisPhaseSpec)
    assert isinstance(specs[1], ExportPhaseSpec)


def test_normalize_phase_specs_rejects_legacy_ls_update():
    with pytest.raises(
        PhaseSpecValidationError, match="'ls_update' is deprecated and no longer supported"
    ):
        normalize_phase_specs(
            {
                "model": {"name": "demo"},
                "phases": [
                    {
                        "name": "node",
                        "trainer": "NODE",
                        "n_epochs": 5,
                        "ls_update": {"method": "truncated", "params": 2},
                    }
                ],
            }
        )


def test_validation_analysis_phase_uses_phase_solver_settings(monkeypatch):
    config = {
        "model": {"name": "demo"},
        "phases": [
            {
                "type": "optimizer",
                "name": "node",
                "trainer": "NODE",
                "ode_method": "rk4",
                "ode_args": {"step_size": 0.05},
            }
        ],
    }
    execution_services = ExecutionServices.from_config(config, default_device=torch.device("cpu"))
    phase = build_phase(
        AnalysisPhaseSpec(name="analysis"),
        config=config,
        model_class=object,
        dtype=torch.float32,
        execution_services=execution_services,
    )

    class _DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

    model = _DummyModel()
    optimizer_state = OptimizerStateArtifact(
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        criteria=[torch.nn.MSELoss(), torch.nn.MSELoss()],
        criteria_weights=[1.0],
        criteria_names=["dynamics", "mse"],
    )
    artifacts = ArtifactRegistry()
    artifacts.put(
        "model",
        ModelArtifact(
            model=model,
            config=config,
            train_md={},
            valid_md={},
            dtype=torch.float32,
        ),
    )
    artifacts.put("optimizer_state", optimizer_state)
    trainer_state = TrainerState(config=config, device=torch.device("cpu"))
    phase_context = PhaseContext(train_set=[object()], valid_set=[object()])

    captured: dict[str, object] = {}

    def traced_eval(
        self,
        model,
        optimizer_state,
        dataset,
        *,
        method,
        ode_args=None,
        evaluate_all=False,
    ):
        captured["method"] = method
        captured["ode_args"] = ode_args
        captured["dataset"] = dataset
        captured["evaluate_all"] = evaluate_all
        return 0.25

    monkeypatch.setattr(
        phases_module.NodeOptimizerPhase,
        "_evaluate_prediction_criterion",
        traced_eval,
    )

    result = phase.execute(
        trainer_state=trainer_state,
        phase_context=phase_context,
        artifacts=artifacts,
        run_name="demo",
        logger=execution_services.configure_logger("dymad.test.validation_analysis", prefix=""),
    )

    assert captured["method"] == "rk4"
    assert captured["ode_args"] == {"step_size": 0.05}
    assert captured["dataset"] is phase_context.valid_set
    assert captured["evaluate_all"] is False
    assert result.metrics == {"mse": 0.25}


def test_load_config_normalizes_legacy_training_alias(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model:
  name: demo
training:
  ode_method: rk4
  ode_args:
    step_size: 0.2
phases:
  - type: optimizer
    name: node
    trainer: NODE
    ode_method: dopri5
    ode_args:
      rtol: 1.0e-6
cv:
  param_grid:
    training.learning_rate: [0.1, 0.2]
""".strip(),
        encoding="utf-8",
    )

    config = load_config(str(config_path))

    assert "training" not in config
    assert config["phases"][0]["ode_method"] == "rk4"
    assert config["phases"][0]["ode_args"] == {"step_size": 0.2}
    assert config["cv"]["param_grid"] == {"phases.0.learning_rate": [0.1, 0.2]}

    execution_services = ExecutionServices.from_config(config, default_device=torch.device("cpu"))
    phase = build_phase(
        AnalysisPhaseSpec(name="analysis"),
        config=config,
        model_class=object,
        dtype=torch.float32,
        execution_services=execution_services,
    )

    assert phase._prediction_settings() == ("rk4", {"step_size": 0.2})


def test_run_cv_single_uses_trainer_run_with_typed_context(monkeypatch):
    calls = {"init": 0, "run": 0}
    expected_metric = 0.123
    phase_context = object()
    trainer_state = object()

    cfg = {
        "model": {"name": "demo-model"},
        "path": {"checkpoint_prefix": "/tmp/cp", "results_prefix": "/tmp/rp"},
        "phases": [{"name": "p0", "trainer": "Linear"}],
    }

    monkeypatch.setattr(
        driver,
        "_apply_combo_to_config",
        lambda combo_idx, fold_id, fold_cfg, combo, base_name, checkpoint_prefix, results_prefix: (
            cfg,
            "/tmp/model_prefix",
        ),
    )
    monkeypatch.setattr(
        driver, "_build_phase_context", lambda fold_id, cfg, train_sets, valid_sets: phase_context
    )
    monkeypatch.setattr(
        driver, "build_initial_trainer_state", lambda cfg, execution_services: trainer_state
    )

    class _FakePhaseResult:
        def get_metric(self, metric_name):
            assert metric_name == "total"
            return expected_metric

    class _FakeTrainerRun:
        def __init__(
            self,
            config,
            model_class,
            device,
            dtype,
            run_name,
            checkpoint_prefix,
            results_prefix,
            execution_services=None,
        ):
            calls["init"] += 1
            calls["run_name"] = run_name
            calls["checkpoint_prefix"] = checkpoint_prefix
            calls["results_prefix"] = results_prefix
            calls["execution_services"] = execution_services

        def run(self, *, initial_context, initial_state, artifacts=None):
            calls["run"] += 1
            calls["initial_context"] = initial_context
            calls["initial_state"] = initial_state
            return [_FakePhaseResult()]

    monkeypatch.setattr(driver, "TrainerRun", _FakeTrainerRun)

    class _FakeTrainSet:
        dtype = torch.float32

    args = {
        "combo_idx": 5,
        "fold_idx": 2,
        "fold_cfg": {"seed": 0},
        "combo": {"phases.0.lr": 0.1},
        "base_name": "base",
        "checkpoint_prefix": "/checkpoints",
        "results_prefix": "/results",
        "train_sets": [_FakeTrainSet()],
        "valid_sets": [_FakeTrainSet()],
        "model_class": object,
        "device": torch.device("cpu"),
        "metric": "total",
    }

    result = driver.run_cv_single(args)

    assert calls["init"] == 1
    assert calls["run"] == 1
    assert calls["initial_context"] is phase_context
    assert calls["initial_state"] is trainer_state
    assert calls["run_name"] == "demo-model"
    assert calls["checkpoint_prefix"] == "/tmp/cp"
    assert calls["results_prefix"] == "/tmp/rp"
    assert calls["execution_services"] is not None
    assert result["metric_value"] == expected_metric


def test_trainer_run_rejects_legacy_resume_checkpoints(tmp_path):
    checkpoint_path = tmp_path / "legacy.pt"
    torch.save({"config": {"model": {"name": "demo"}}}, checkpoint_path)

    trainer_run = TrainerRun(
        config={"model": {"name": "demo"}, "phases": [{"name": "p0", "trainer": "Linear"}]},
        model_class=object,
        device=torch.device("cpu"),
        dtype=torch.float32,
        run_name="demo",
        checkpoint_prefix=str(tmp_path),
        results_prefix=str(tmp_path),
    )

    try:
        trainer_run.load_run_checkpoint(str(checkpoint_path))
    except TrainingCheckpointError as exc:
        assert "Legacy optimizer checkpoints are not resumable" in str(exc)
    else:
        raise AssertionError("expected TrainingCheckpointError for legacy checkpoint payload")


def test_trainer_run_round_trips_typed_run_checkpoint(tmp_path):
    trainer_run = TrainerRun(
        config={"model": {"name": "demo"}, "phases": [{"name": "p0", "trainer": "Linear"}]},
        model_class=object,
        device=torch.device("cpu"),
        dtype=torch.float32,
        run_name="demo",
        checkpoint_prefix=str(tmp_path),
        results_prefix=str(tmp_path),
    )
    state = TrainerState(
        config={"model": {"name": "demo"}},
        device=torch.device("cpu"),
        epoch=3,
        best_loss={"valid_total": 1.23},
        phase_cursor=2,
        phase_records=[
            PhaseRecord(name="p0", kind="optimizer", started_epoch=0, completed_epoch=3)
        ],
    )
    artifacts = ArtifactRegistry()
    trainer_run.save_run_checkpoint(state, artifacts)

    loaded_state, loaded_artifacts = trainer_run.load_run_checkpoint()

    assert loaded_state.epoch == 3
    assert loaded_state.phase_cursor == 2
    assert list(loaded_artifacts.keys()) == []
