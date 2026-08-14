import json
import logging
import random
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
from scipy.signal import savgol_filter

import dymad.training.phases as phases_module
from dymad.core import GraphSeries, RegularSeries, RegularTrainerBatch
from dymad.numerics import denoise, generate_discrete_weak_weights
from dymad.training import driver
from dymad.training.execution_services import ExecutionServices
from dymad.training.helper import (
    CVResult,
    batch_pattern_search_points,
    bounded_nelder_mead_search_points,
    multi_start_bounded_nelder_mead_search_points,
    nelder_mead_like_search_indices,
    select_best_cv_result,
)
from dymad.training.ls_update import _comp_linear_eval_ct, _comp_linear_eval_dt, _ct_target
from dymad.training.phase_runtime import (
    ArtifactRegistry,
    ModelArtifact,
    OptimizerStateArtifact,
    PhaseContext,
    PhaseRecord,
    PhaseResult,
    TrainerState,
    TrainingCheckpointError,
    TrainingHistoryArtifact,
)
from dymad.training.phases import (
    AnalysisPhaseSpec,
    DataPhaseSpec,
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


def _build_regular_series(offset: float = 0.0) -> RegularSeries:
    time = torch.linspace(0.0, 0.8, 9)
    state = torch.stack((torch.sin(time + offset), torch.cos(time + offset)), dim=1)
    control = torch.stack((time, time**2), dim=1)
    return RegularSeries(time=time, state=state, control=control, meta={"series": float(offset)})


def _build_graph_series(offset: float = 0.0) -> GraphSeries:
    time = torch.linspace(0.0, 0.8, 9)
    node_state = torch.stack(
        (
            torch.stack((torch.sin(time + offset), torch.cos(time + offset)), dim=1),
            torch.stack((torch.sin(time + offset + 0.1), torch.cos(time + offset + 0.1)), dim=1),
        ),
        dim=1,
    )
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_weight = torch.tensor([1.0, 1.0])
    return GraphSeries(
        time=time,
        node_state=node_state,
        edge_index=edge_index,
        edge_weight=edge_weight,
        meta={"graph": True},
    )


def _build_data_phase(
    config: dict,
    spec: DataPhaseSpec,
):
    execution_services = ExecutionServices.from_config(config, default_device=torch.device("cpu"))
    return build_phase(
        spec,
        config=config,
        model_class=object,
        dtype=torch.float32,
        execution_services=execution_services,
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


def test_safe_plot_returns_false_when_plotting_fails(caplog):
    with caplog.at_level(logging.WARNING):
        wrote_plot = phases_module._safe_plot(
            logging.getLogger("test.plot"),
            label="history plot 'demo'",
            fn=lambda: (_ for _ in ()).throw(RuntimeError("backend unavailable")),
        )

    assert wrote_plot is False
    assert "Skipping history plot 'demo' due to plotting failure." in caplog.text


def test_prediction_diagnostic_sampling_uses_persisted_local_rng(monkeypatch):
    config = {
        "model": {"name": "demo"},
        "prediction_diagnostic": {"sample_seed": 12345},
    }
    execution_services = ExecutionServices.from_config(config, default_device=torch.device("cpu"))
    phase = build_phase(
        OptimizerPhaseSpec(name="node", trainer="NODE", config={}),
        config=config,
        model_class=object,
        dtype=torch.float32,
        execution_services=execution_services,
    )
    history = TrainingHistoryArtifact()
    train_set = list(range(30))
    valid_set = list(range(100, 110))
    phase_context = PhaseContext(train_set=train_set, valid_set=valid_set)

    monkeypatch.setattr(
        phase,
        "_evaluate_prediction_criterion_single",
        lambda _model, _optimizer_state, sample, **_kwargs: float(sample),
    )
    for epoch in range(2):
        random.seed(epoch)
        phase._update_prediction_history(
            object(),
            object(),
            phase_context,
            history,
            epoch=epoch,
            ode_method="rk4",
            ode_args={},
        )

    expected_rng = random.Random(12345)
    expected = [
        [0, expected_rng.choice(train_set), expected_rng.choice(valid_set)],
        [1, expected_rng.choice(train_set), expected_rng.choice(valid_set)],
    ]
    assert history.crit == expected


def test_fourth_order_continuous_target_is_exact_for_quartic_data():
    time = torch.linspace(-1.0, 1.0, 9, dtype=torch.float64)
    values = torch.stack((time**4, time**3), dim=-1).unsqueeze(0)

    derivative = _ct_target(values, float(time[1] - time[0]), order=4)

    expected = torch.stack((4.0 * time**3, 3.0 * time**2), dim=-1).unsqueeze(0)
    torch.testing.assert_close(derivative, expected, atol=1.0e-12, rtol=1.0e-12)


def test_linear_optimizer_uses_named_solver_kwargs(monkeypatch):
    captured = {}

    def fake_ls_updater(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(phases_module, "LSUpdater", fake_ls_updater)
    phase = build_phase(
        OptimizerPhaseSpec(
            name="linear",
            trainer="Linear",
            config={"method": "full", "linear_solver_kwargs": {"order": 4}},
        ),
        config={"model": {"name": "demo"}},
        model_class=object,
        dtype=torch.float64,
        execution_services=ExecutionServices.from_config(
            {"model": {"name": "demo"}}, default_device=torch.device("cpu")
        ),
    )

    result = phase._create_ls_updater(
        torch.nn.Linear(1, 1), PhaseContext(train_md={"dt_and_n_steps": [(0.1, 9)]})
    )

    assert result is not None
    assert captured["method"] == "full"
    assert captured["dt"] == pytest.approx(0.1)
    assert captured["order"] == 4


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


def test_normalize_phase_specs_accepts_smoothing_data_phase():
    specs = normalize_phase_specs(
        {
            "model": {"name": "demo"},
            "phases": [
                {
                    "type": "data",
                    "name": "smooth",
                    "operation": "smooth",
                    "method": "savgol",
                    "window_length": 5,
                    "polyorder": 2,
                }
            ],
        }
    )

    data_spec = specs[0]
    assert isinstance(data_spec, DataPhaseSpec)
    assert data_spec.operation == "smooth"
    assert data_spec.config["method"] == "savgol"


def test_normalize_phase_specs_accepts_kernel_smoothing_data_phase():
    specs = normalize_phase_specs(
        {
            "model": {"name": "demo"},
            "phases": [
                {
                    "type": "data",
                    "name": "smooth",
                    "operation": "smooth",
                    "method": "kernel_smoothing",
                    "kernel": "compact_polynomial",
                    "anchor_count": 5,
                    "bandwidth_multiplier": 2.0,
                    "degree": 4.0,
                }
            ],
        }
    )

    data_spec = specs[0]
    assert isinstance(data_spec, DataPhaseSpec)
    assert data_spec.config["method"] == "kernel_smoothing"
    assert data_spec.config["kernel"] == "compact_polynomial"


def test_normalize_phase_specs_accepts_optimizer_reset_flag():
    specs = normalize_phase_specs(
        {
            "model": {"name": "demo"},
            "phases": [
                {
                    "type": "optimizer",
                    "name": "warmup",
                    "trainer": "Weak",
                    "reset_optimizer": True,
                    "n_epochs": 5,
                },
                {
                    "trainer": "NODE",
                    "reset_optimizer": True,
                    "n_epochs": 7,
                },
            ],
        }
    )

    warmup = specs[0]
    refine = specs[1]
    assert isinstance(warmup, OptimizerPhaseSpec)
    assert isinstance(refine, OptimizerPhaseSpec)
    assert warmup.reset_optimizer is True
    assert refine.reset_optimizer is True
    assert "reset_optimizer" not in warmup.config
    assert "reset_optimizer" not in refine.config


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


def test_smoothing_data_phase_rewrites_regular_context_and_loader():
    train_series = _build_regular_series()
    valid_series = _build_regular_series(offset=0.2)
    original_context = PhaseContext(
        train_set=[train_series],
        valid_set=[valid_series],
        train_loader=object(),
        valid_loader=object(),
        train_md={"dt_and_n_steps": [(0.1, 9)]},
        valid_md={"dt_and_n_steps": [(0.1, 9)]},
    )
    config = {
        "model": {"name": "demo"},
        "dataloader": {"batch_size": 1, "shuffle": False},
        "phases": [],
    }
    phase = _build_data_phase(
        config,
        DataPhaseSpec(
            name="smooth",
            operation="smooth",
            config={"method": "savgol", "window_length": 5, "polyorder": 2},
        ),
    )

    result = phase.execute(
        trainer_state=TrainerState(config=config, device=torch.device("cpu")),
        phase_context=original_context,
        artifacts=ArtifactRegistry(),
        run_name="demo",
        logger=logging.getLogger("test.smooth.regular"),
    )

    assert result.phase_context.train_set is not original_context.train_set
    assert result.phase_context.valid_set is not original_context.valid_set
    assert isinstance(next(iter(result.phase_context.train_loader)), RegularTrainerBatch)
    np.testing.assert_allclose(
        result.phase_context.train_set[0].state.detach().cpu().numpy(),
        savgol_filter(train_series.state.detach().cpu().numpy(), 5, 2, axis=0),
    )
    train_delta = (
        result.phase_context.train_set[0].state.detach().cpu().numpy()
        - train_series.state.detach().cpu().numpy()
    )
    assert result.metrics["train_delta_rmse"] == pytest.approx(
        float(np.sqrt(np.mean(np.square(train_delta))))
    )
    assert "train_num_trajectories" not in result.metrics
    assert "train_num_elements" not in result.metrics
    assert "train_num_diff_elements" not in result.metrics
    assert result.metrics["train_delta_mae"] == pytest.approx(float(np.mean(np.abs(train_delta))))
    assert result.metrics["train_roughness_ratio"] == pytest.approx(
        result.metrics["train_roughness_after"] / result.metrics["train_roughness_before"]
    )
    assert result.record is not None
    assert result.record.metrics["train_delta_rmse"] == pytest.approx(
        result.metrics["train_delta_rmse"]
    )
    assert result.phase_context.train_md["data_phase_history"][-1]["method"] == "savgol"
    assert result.phase_context.train_md["data_phase_history"][-1]["metrics"][
        "train_delta_rmse"
    ] == pytest.approx(result.metrics["train_delta_rmse"])
    assert result.phase_context.valid_md["data_phase_history"][-1]["split"] == "valid"


def test_smoothing_data_phase_logs_train_valid_table(caplog):
    train_series = _build_regular_series()
    valid_series = _build_regular_series(offset=0.2)
    context = PhaseContext(
        train_set=[train_series],
        valid_set=[valid_series],
        train_loader=object(),
        valid_loader=object(),
        train_md={"dt_and_n_steps": [(0.1, 9)]},
        valid_md={"dt_and_n_steps": [(0.1, 9)]},
    )
    config = {
        "model": {"name": "demo"},
        "dataloader": {"batch_size": 1, "shuffle": False},
        "phases": [],
    }
    phase = _build_data_phase(
        config,
        DataPhaseSpec(
            name="smooth",
            operation="smooth",
            config={"method": "savgol", "window_length": 5, "polyorder": 2},
        ),
    )

    with caplog.at_level(logging.INFO):
        phase.execute(
            trainer_state=TrainerState(config=config, device=torch.device("cpu")),
            phase_context=context,
            artifacts=ArtifactRegistry(),
            run_name="demo",
            logger=logging.getLogger("test.smooth.table"),
        )

    assert "Data phase 'smooth' completed:" in caplog.text
    assert "metric" in caplog.text
    assert "train" in caplog.text
    assert "valid" in caplog.text
    assert "delta_rmse" in caplog.text
    assert "roughness_ratio" in caplog.text
    assert "num_trajectories" not in caplog.text
    assert "num_elements" not in caplog.text


def test_smoothing_data_phase_smooths_each_regular_series_independently():
    train_series = [_build_regular_series(), _build_regular_series(offset=0.4)]
    valid_series = [_build_regular_series(offset=0.2), _build_regular_series(offset=0.6)]
    context = PhaseContext(
        train_set=train_series,
        valid_set=valid_series,
        train_loader=object(),
        valid_loader=object(),
        train_md={"dt_and_n_steps": [(0.1, 9), (0.1, 9)]},
        valid_md={"dt_and_n_steps": [(0.1, 9), (0.1, 9)]},
    )
    config = {
        "model": {"name": "demo"},
        "dataloader": {"batch_size": 2, "shuffle": False},
        "phases": [],
    }
    phase = _build_data_phase(
        config,
        DataPhaseSpec(
            name="smooth",
            operation="smooth",
            config={"method": "savgol", "window_length": 5, "polyorder": 2},
        ),
    )

    result = phase.execute(
        trainer_state=TrainerState(config=config, device=torch.device("cpu")),
        phase_context=context,
        artifacts=ArtifactRegistry(),
        run_name="demo",
        logger=logging.getLogger("test.smooth.multiple"),
    )

    for actual_series, original_series in zip(
        result.phase_context.train_set,
        train_series,
        strict=False,
    ):
        np.testing.assert_allclose(
            actual_series.state.detach().cpu().numpy(),
            savgol_filter(original_series.state.detach().cpu().numpy(), 5, 2, axis=0),
        )

    for actual_series, original_series in zip(
        result.phase_context.valid_set,
        valid_series,
        strict=False,
    ):
        np.testing.assert_allclose(
            actual_series.state.detach().cpu().numpy(),
            savgol_filter(original_series.state.detach().cpu().numpy(), 5, 2, axis=0),
        )


def test_smoothing_data_phase_supports_kernel_smoothing_config():
    train_series = _build_regular_series()
    context = PhaseContext(
        train_set=[train_series],
        valid_set=None,
        train_loader=object(),
        valid_loader=None,
        train_md={"dt_and_n_steps": [(0.1, 9)]},
        valid_md=None,
    )
    config = {
        "model": {"name": "demo"},
        "dataloader": {"batch_size": 1, "shuffle": False},
        "phases": [],
    }
    denoise_config = {
        "method": "kernel_smoothing",
        "kernel": "gaussian",
        "anchor_count": 5,
        "bandwidth_multiplier": 2.0,
    }
    phase = _build_data_phase(
        config,
        DataPhaseSpec(
            name="smooth_kernel",
            operation="smooth",
            config={**denoise_config, "splits": ["train"]},
        ),
    )

    result = phase.execute(
        trainer_state=TrainerState(config=config, device=torch.device("cpu")),
        phase_context=context,
        artifacts=ArtifactRegistry(),
        run_name="demo",
        logger=logging.getLogger("test.smooth.kernel"),
    )

    expected = denoise(train_series.state, axis=0, **denoise_config)
    np.testing.assert_allclose(
        result.phase_context.train_set[0].state.detach().cpu().numpy(),
        expected.detach().cpu().numpy(),
    )
    history = result.phase_context.train_md["data_phase_history"][-1]
    assert history["method"] == "kernel_smoothing"
    assert history["config"] == {
        "kernel": "gaussian",
        "anchor_count": 5,
        "bandwidth_multiplier": 2.0,
    }


def test_smoothing_data_phase_uses_standalone_denoising_helpers(monkeypatch):
    train_series = _build_regular_series()
    context = PhaseContext(
        train_set=[train_series],
        valid_set=None,
        train_loader=object(),
        valid_loader=None,
        train_md={"dt_and_n_steps": [(0.1, 9)]},
        valid_md=None,
    )
    config = {
        "model": {"name": "demo"},
        "dataloader": {"batch_size": 1, "shuffle": False},
        "phases": [],
    }
    phase = _build_data_phase(
        config,
        DataPhaseSpec(
            name="smooth_train",
            operation="smooth",
            config={"splits": ["train"], "window_length": 5, "polyorder": 2},
        ),
    )
    calls: dict[str, object] = {}

    def fake_denoise(data, *, method, axis=0, **kwargs):
        calls["denoise"] = {
            "method": method,
            "axis": axis,
            "window_length": kwargs["window_length"],
            "polyorder": kwargs["polyorder"],
        }
        return data + 1.0

    def fake_metrics(*, original, denoised):
        calls["metrics"] = {
            "num_original": len(original),
            "num_denoised": len(denoised),
        }
        return {"delta_rmse": 2.5, "roughness_ratio": 0.4}

    monkeypatch.setattr(phases_module, "denoise", fake_denoise)
    monkeypatch.setattr(phases_module, "denoising_metrics", fake_metrics)

    result = phase.execute(
        trainer_state=TrainerState(config=config, device=torch.device("cpu")),
        phase_context=context,
        artifacts=ArtifactRegistry(),
        run_name="demo",
        logger=logging.getLogger("test.smooth.delegate"),
    )

    assert calls["denoise"] == {
        "method": "savgol",
        "axis": 0,
        "window_length": 5,
        "polyorder": 2,
    }
    assert calls["metrics"] == {"num_original": 1, "num_denoised": 1}
    np.testing.assert_allclose(
        result.phase_context.train_set[0].state.detach().cpu().numpy(),
        train_series.state.detach().cpu().numpy() + 1.0,
    )
    assert result.metrics["train_delta_rmse"] == pytest.approx(2.5)
    assert result.metrics["train_roughness_ratio"] == pytest.approx(0.4)


def test_smoothing_data_phase_supports_train_only_split():
    train_series = _build_regular_series()
    valid_series = _build_regular_series(offset=0.2)
    original_context = PhaseContext(
        train_set=[train_series],
        valid_set=[valid_series],
        train_loader=object(),
        valid_loader=object(),
        train_md={"dt_and_n_steps": [(0.1, 9)]},
        valid_md={"dt_and_n_steps": [(0.1, 9)]},
    )
    config = {
        "model": {"name": "demo"},
        "dataloader": {"batch_size": 1, "shuffle": False},
        "phases": [],
    }
    phase = _build_data_phase(
        config,
        DataPhaseSpec(
            name="smooth_train",
            operation="smooth",
            config={"splits": ["train"], "window_length": 5, "polyorder": 2},
        ),
    )

    result = phase.execute(
        trainer_state=TrainerState(config=config, device=torch.device("cpu")),
        phase_context=original_context,
        artifacts=ArtifactRegistry(),
        run_name="demo",
        logger=logging.getLogger("test.smooth.train"),
    )

    assert result.phase_context.valid_set is original_context.valid_set
    assert result.phase_context.valid_loader is original_context.valid_loader
    assert "data_phase_history" not in result.phase_context.valid_md
    assert result.phase_context.train_set is not original_context.train_set
    assert "valid_delta_rmse" not in result.metrics
    assert "train_delta_rmse" in result.metrics


def test_smoothing_data_phase_supports_graph_node_state_only():
    graph_series = _build_graph_series()
    original_context = PhaseContext(
        train_set=[graph_series],
        valid_set=[graph_series],
        train_loader=object(),
        valid_loader=object(),
        train_md={"dt_and_n_steps": [(0.1, 9)]},
        valid_md={"dt_and_n_steps": [(0.1, 9)]},
    )
    config = {
        "model": {"name": "demo"},
        "dataloader": {"batch_size": 1, "shuffle": False},
        "phases": [],
    }
    phase = _build_data_phase(
        config,
        DataPhaseSpec(
            name="smooth_graph",
            operation="smooth",
            config={"splits": ["valid"], "window_length": 5, "polyorder": 2},
        ),
    )

    result = phase.execute(
        trainer_state=TrainerState(config=config, device=torch.device("cpu")),
        phase_context=original_context,
        artifacts=ArtifactRegistry(),
        run_name="demo",
        logger=logging.getLogger("test.smooth.graph"),
    )

    smoothed = result.phase_context.valid_set[0]
    np.testing.assert_allclose(
        smoothed.node_state.detach().cpu().numpy(),
        savgol_filter(graph_series.node_state.detach().cpu().numpy(), 5, 2, axis=0),
    )
    assert torch.equal(smoothed.edge_index, graph_series.edge_index)
    assert torch.equal(smoothed.edge_weight, graph_series.edge_weight)
    assert result.phase_context.train_set is original_context.train_set


@pytest.mark.parametrize(
    "phase_config,match",
    [
        ({"window_length": 4, "polyorder": 2}, "odd, positive window_length"),
        ({"window_length": 5, "polyorder": 5}, "polyorder < window_length"),
        ({"window_length": 11, "polyorder": 2}, "at least 11 steps"),
        (
            {"method": "median", "window_length": 5, "polyorder": 2},
            "Unsupported data smoothing method",
        ),
        ({"axis": 3, "window_length": 5, "polyorder": 2}, "Invalid axis"),
        ({"splits": ["test"], "window_length": 5, "polyorder": 2}, "invalid splits"),
    ],
)
def test_smoothing_data_phase_rejects_invalid_configs(phase_config, match):
    config = {
        "model": {"name": "demo"},
        "dataloader": {"batch_size": 1, "shuffle": False},
        "phases": [],
    }
    phase = _build_data_phase(
        config,
        DataPhaseSpec(
            name="smooth_invalid",
            operation="smooth",
            config=phase_config,
        ),
    )
    context = PhaseContext(
        train_set=[_build_regular_series()],
        valid_set=[_build_regular_series(offset=0.2)],
        train_loader=object(),
        valid_loader=object(),
        train_md={"dt_and_n_steps": [(0.1, 9)]},
        valid_md={"dt_and_n_steps": [(0.1, 9)]},
    )

    with pytest.raises(PhaseSpecValidationError, match=match):
        phase.execute(
            trainer_state=TrainerState(config=config, device=torch.device("cpu")),
            phase_context=context,
            artifacts=ArtifactRegistry(),
            run_name="demo",
            logger=logging.getLogger("test.smooth.invalid"),
        )


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
  search:
    mode: nelder_mead_like
    bounds:
      training.weight_decay: [1.0e-4, 1.0e-2]
""".strip(),
        encoding="utf-8",
    )

    config = load_config(str(config_path))

    assert "training" not in config
    assert config["phases"][0]["ode_method"] == "rk4"
    assert config["phases"][0]["ode_args"] == {"step_size": 0.2}
    assert config["cv"]["param_grid"] == {"phases.0.learning_rate": [0.1, 0.2]}
    assert config["cv"]["search"]["bounds"] == {"phases.0.weight_decay": [1.0e-4, 1.0e-2]}

    execution_services = ExecutionServices.from_config(config, default_device=torch.device("cpu"))
    phase = build_phase(
        AnalysisPhaseSpec(name="analysis"),
        config=config,
        model_class=object,
        dtype=torch.float32,
        execution_services=execution_services,
    )

    assert phase._prediction_settings() == ("rk4", {"step_size": 0.2})


def test_load_config_preserves_explicit_phases_when_training_is_disabled(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model:
  name: demo
training:
  n_epochs: 1000
  save_interval: 10
  chop_mode: unfold
""".strip(),
        encoding="utf-8",
    )

    config = load_config(
        str(config_path),
        config_mod={
            "training": None,
            "phases": [
                {
                    "type": "optimizer",
                    "name": "warm",
                    "trainer": "OneStep",
                    "n_epochs": 300,
                    "save_interval": 20,
                },
                {
                    "type": "optimizer",
                    "name": "node",
                    "trainer": "NODE",
                    "n_epochs": 900,
                    "chop_mode": "initial",
                },
            ],
        },
    )

    assert config["training"] is None
    assert config["phases"][0]["n_epochs"] == 300
    assert config["phases"][0]["save_interval"] == 20
    assert "chop_mode" not in config["phases"][0]
    assert config["phases"][1]["n_epochs"] == 900
    assert config["phases"][1]["chop_mode"] == "initial"


def test_load_config_koopman_example_phase_overrides_survive_shared_config():
    config_path = Path(__file__).resolve().parents[1] / "examples" / "2d_koopman" / "kp_model.yaml"

    sequential = load_config(
        str(config_path),
        config_mod={
            "phases": [
                {
                    "type": "optimizer",
                    "trainer": "Weak",
                    "n_epochs": 500,
                    "save_interval": 20,
                    "load_checkpoint": False,
                    "learning_rate": 5e-3,
                    "decay_rate": 0.999,
                    "weak_form_params": {"N": 13, "dN": 2, "ordpol": 2, "ordint": 2},
                },
                {
                    "type": "optimizer",
                    "trainer": "NODE",
                    "n_epochs": 500,
                    "save_interval": 20,
                    "load_checkpoint": False,
                    "learning_rate": 5e-3,
                    "decay_rate": 0.999,
                    "sweep_lengths": [100, 200],
                    "sweep_epoch_step": 20,
                    "ode_method": "dopri5",
                    "ode_args": {"rtol": 1.0e-7, "atol": 1.0e-9},
                },
            ]
        },
    )

    assert "training" not in sequential
    assert sequential["phases"][0]["n_epochs"] == 500
    assert sequential["phases"][0]["learning_rate"] == pytest.approx(5e-3)
    assert sequential["phases"][1]["n_epochs"] == 500
    assert sequential["phases"][1]["learning_rate"] == pytest.approx(5e-3)

    accelerated = load_config(
        str(config_path),
        config_mod={
            "phases": [
                {"type": "linear_solve", "method": "full"},
                {
                    "repeat": {
                        "times": 3,
                        "phases": [
                            {
                                "type": "optimizer",
                                "trainer": "NODE",
                                "n_epochs": 100,
                                "save_interval": 20,
                                "load_checkpoint": False,
                                "learning_rate": 5e-3,
                                "decay_rate": 0.999,
                                "sweep_lengths": [100, 200],
                                "sweep_epoch_step": 20,
                                "ode_method": "dopri5",
                                "ode_args": {"rtol": 1.0e-7, "atol": 1.0e-9},
                            },
                            {"type": "linear_solve", "method": "full"},
                        ],
                    }
                },
                {
                    "type": "optimizer",
                    "trainer": "NODE",
                    "n_epochs": 1200,
                    "save_interval": 20,
                    "load_checkpoint": False,
                    "learning_rate": 5e-3,
                    "decay_rate": 0.999,
                    "sweep_lengths": [100, 200],
                    "sweep_epoch_step": 20,
                    "ode_method": "dopri5",
                    "ode_args": {"rtol": 1.0e-7, "atol": 1.0e-9},
                },
            ]
        },
    )

    assert "training" not in accelerated
    assert accelerated["phases"][2]["n_epochs"] == 1200
    assert accelerated["phases"][2]["learning_rate"] == pytest.approx(5e-3)


def test_seed_cv_trial_reproducibly_resets_numpy_and_torch() -> None:
    driver._seed_cv_trial(123)
    first_numpy = np.random.random(4)
    first_torch = torch.rand(4)

    driver._seed_cv_trial(123)
    second_numpy = np.random.random(4)
    second_torch = torch.rand(4)

    np.testing.assert_array_equal(first_numpy, second_numpy)
    torch.testing.assert_close(first_torch, second_torch, rtol=0.0, atol=0.0)


def test_cv_base_seed_uses_active_torch_seed_unless_configured() -> None:
    with torch.random.fork_rng():
        torch.manual_seed(12345)

        assert driver._resolve_cv_base_seed({}) == 12345
        assert driver._resolve_cv_base_seed({"seed": 17}) == 17


def test_cv_trial_seeds_depend_on_fold_not_combo_order() -> None:
    trainer = object.__new__(driver.DriverBase)
    trainer.base_seed = 101
    trainer.base_name = "demo"
    trainer.checkpoint_prefix = "checkpoints"
    trainer.results_prefix = "results"
    trainer.train_sets = []
    trainer.valid_sets = []
    trainer.model_class = torch.nn.Module
    trainer.device = torch.device("cpu")
    trainer.metric = "total"
    fold_specs = [(0, {}), (1, {})]

    first = trainer._trial_args_for_combo(combo_idx=0, combo={"value": 1}, fold_specs=fold_specs)
    reordered = trainer._trial_args_for_combo(
        combo_idx=9, combo={"value": 1}, fold_specs=fold_specs
    )

    assert [args["seed"] for args in first] == [101, 102]
    assert [args["seed"] for args in reordered] == [101, 102]


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


def test_one_step_optimizer_phase_uses_discrete_next_state_targets():
    class _DiscreteModel(torch.nn.Module):
        CONT = False

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def linear_eval(self, runtime):
            z = runtime.x
            return self.weight * z, z

    config = {"model": {"name": "demo"}, "phases": []}
    execution_services = ExecutionServices.from_config(config, default_device=torch.device("cpu"))
    phase = build_phase(
        OptimizerPhaseSpec(name="one_step", trainer="OneStep", config={}),
        config=config,
        model_class=_DiscreteModel,
        dtype=torch.float32,
        execution_services=execution_services,
    )
    model = _DiscreteModel()
    optimizer_state = OptimizerStateArtifact(
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        criteria=[torch.nn.MSELoss(), torch.nn.MSELoss()],
        criteria_weights=[1.0],
        criteria_names=["dynamics", "mse"],
        _one_step_dt=1.0,
    )
    series = RegularSeries(
        time=torch.tensor([0.0, 1.0, 2.0]),
        state=torch.tensor([[1.0], [2.0], [4.0]]),
        control=None,
        meta={},
    )
    batch = RegularTrainerBatch.collate_series([series])

    losses = phase._compute_losses(model, optimizer_state, batch, "dopri5", {})

    assert losses[0].item() == pytest.approx(2.5)


def test_optimizer_phase_reuses_prior_state_but_honors_new_phase_lr():
    class _SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

    config = {"model": {"name": "demo"}, "phases": []}
    execution_services = ExecutionServices.from_config(config, default_device=torch.device("cpu"))
    phase = build_phase(
        OptimizerPhaseSpec(
            name="weak",
            trainer="Weak",
            config={
                "learning_rate": 5.0e-3,
                "weak_form_params": {"N": 5, "dN": 1, "ordpol": 1, "ordint": 1},
            },
        ),
        config=config,
        model_class=_SimpleModel,
        dtype=torch.float32,
        execution_services=execution_services,
    )
    model = _SimpleModel()
    prior_optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    prior_optimizer.zero_grad(set_to_none=True)
    (model.weight.square()).backward()
    prior_optimizer.step()

    artifacts = ArtifactRegistry()
    artifacts.put(
        "optimizer_state",
        OptimizerStateArtifact(
            optimizer=prior_optimizer,
            criteria=[],
            criteria_weights=[],
            criteria_names=[],
        ),
    )

    optimizer_state = phase._build_optimizer_artifact(model, artifacts)

    assert optimizer_state.optimizer.state
    assert optimizer_state.optimizer.param_groups[0]["lr"] == pytest.approx(5.0e-3)


def test_optimizer_phase_reset_optimizer_starts_with_fresh_state():
    class _SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

    config = {"model": {"name": "demo"}, "phases": []}
    execution_services = ExecutionServices.from_config(config, default_device=torch.device("cpu"))
    phase = build_phase(
        OptimizerPhaseSpec(
            name="weak",
            trainer="Weak",
            config={
                "learning_rate": 5.0e-3,
                "weak_form_params": {"N": 5, "dN": 1, "ordpol": 1, "ordint": 1},
            },
            reset_optimizer=True,
        ),
        config=config,
        model_class=_SimpleModel,
        dtype=torch.float32,
        execution_services=execution_services,
    )
    model = _SimpleModel()
    prior_optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    prior_optimizer.zero_grad(set_to_none=True)
    (model.weight.square()).backward()
    prior_optimizer.step()

    artifacts = ArtifactRegistry()
    artifacts.put(
        "optimizer_state",
        OptimizerStateArtifact(
            optimizer=prior_optimizer,
            criteria=[],
            criteria_weights=[],
            criteria_names=[],
        ),
    )

    optimizer_state = phase._build_optimizer_artifact(model, artifacts)

    assert optimizer_state.optimizer.state == {}
    assert optimizer_state.optimizer.param_groups[0]["lr"] == pytest.approx(5.0e-3)


def test_one_step_optimizer_phase_uses_continuous_rate_targets():
    class _ContinuousModel(torch.nn.Module):
        CONT = True

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def linear_eval(self, runtime):
            z = runtime.x
            return self.weight * z, z

    config = {"model": {"name": "demo"}, "phases": []}
    execution_services = ExecutionServices.from_config(config, default_device=torch.device("cpu"))
    phase = build_phase(
        OptimizerPhaseSpec(name="one_step", trainer="OneStep", config={"order": 1}),
        config=config,
        model_class=_ContinuousModel,
        dtype=torch.float32,
        execution_services=execution_services,
    )
    model = _ContinuousModel()
    optimizer_state = OptimizerStateArtifact(
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        criteria=[torch.nn.MSELoss(), torch.nn.MSELoss()],
        criteria_weights=[1.0],
        criteria_names=["dynamics", "mse"],
        _one_step_dt=1.0,
        _one_step_kwargs={"order": 1},
    )
    series = RegularSeries(
        time=torch.tensor([0.0, 1.0, 2.0]),
        state=torch.tensor([[0.0], [1.0], [4.0]]),
        control=None,
        meta={},
    )
    batch = RegularTrainerBatch.collate_series([series])

    losses = phase._compute_losses(model, optimizer_state, batch, "dopri5", {})

    assert losses[0].item() == pytest.approx(2.0)


def test_one_step_optimizer_phase_supports_default_continuous_targets_with_trainable_encoder():
    class _ContinuousEncoderModel(torch.nn.Module):
        CONT = True

        def __init__(self):
            super().__init__()
            self.encoder_scale = torch.nn.Parameter(torch.tensor(2.0))
            self.prediction_scale = torch.nn.Parameter(torch.tensor(0.5))

        def encoder(self, runtime):
            return self.encoder_scale * runtime.x

        def linear_eval(self, runtime):
            z = self.encoder(runtime)
            return self.prediction_scale * z, z

    config = {"model": {"name": "demo"}, "phases": []}
    execution_services = ExecutionServices.from_config(config, default_device=torch.device("cpu"))
    phase = build_phase(
        OptimizerPhaseSpec(name="one_step", trainer="OneStep", config={}),
        config=config,
        model_class=_ContinuousEncoderModel,
        dtype=torch.float32,
        execution_services=execution_services,
    )
    model = _ContinuousEncoderModel()
    optimizer_state = OptimizerStateArtifact(
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        criteria=[torch.nn.MSELoss(), torch.nn.MSELoss()],
        criteria_weights=[1.0],
        criteria_names=["dynamics", "mse"],
        _one_step_dt=1.0,
    )
    series = RegularSeries(
        time=torch.tensor([0.0, 1.0, 2.0]),
        state=torch.tensor([[0.0], [1.0], [4.0]]),
        control=None,
        meta={},
    )
    batch = RegularTrainerBatch.collate_series([series])

    losses = phase._compute_losses(model, optimizer_state, batch, "dopri5", {})
    losses[0].backward()

    assert torch.isfinite(losses[0])
    assert model.encoder_scale.grad is not None
    assert model.encoder_scale.grad.item() != pytest.approx(0.0)


def test_one_step_optimizer_phase_detaches_discrete_targets_for_trainable_encoder():
    class _DiscreteEncoderModel(torch.nn.Module):
        CONT = False

        def __init__(self):
            super().__init__()
            self.encoder_scale = torch.nn.Parameter(torch.tensor(2.0))
            self.prediction_scale = torch.nn.Parameter(torch.tensor(0.5))

        def encoder(self, runtime):
            return self.encoder_scale * runtime.x

        def linear_eval(self, runtime):
            z = self.encoder(runtime)
            return self.prediction_scale * z, z

    config = {"model": {"name": "demo"}, "phases": []}
    execution_services = ExecutionServices.from_config(config, default_device=torch.device("cpu"))
    phase = build_phase(
        OptimizerPhaseSpec(name="one_step", trainer="OneStep", config={}),
        config=config,
        model_class=_DiscreteEncoderModel,
        dtype=torch.float32,
        execution_services=execution_services,
    )
    model = _DiscreteEncoderModel()
    criterion = torch.nn.MSELoss()
    optimizer_state = OptimizerStateArtifact(
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        criteria=[criterion, torch.nn.MSELoss()],
        criteria_weights=[1.0],
        criteria_names=["dynamics", "mse"],
        _one_step_dt=1.0,
    )
    series = RegularSeries(
        time=torch.tensor([0.0, 1.0, 2.0]),
        state=torch.tensor([[1.0], [2.0], [4.0]]),
        control=None,
        meta={},
    )
    batch = RegularTrainerBatch.collate_series([series])

    loss = phase._compute_losses(model, optimizer_state, batch, "dopri5", {})[0]
    actual_grad = torch.autograd.grad(loss, model.encoder_scale)[0]

    runtime_batch = batch.to(phase.device)
    predictions, targets = _comp_linear_eval_dt(model, runtime_batch, dt=1.0)
    expected_loss = criterion(predictions, targets.detach())
    expected_grad = torch.autograd.grad(expected_loss, model.encoder_scale)[0]

    assert targets.requires_grad is False
    assert actual_grad is not None
    assert actual_grad.item() == pytest.approx(expected_grad.item())


def test_one_step_optimizer_phase_detaches_first_order_continuous_targets_for_trainable_encoder():
    class _ContinuousEncoderModel(torch.nn.Module):
        CONT = True

        def __init__(self):
            super().__init__()
            self.encoder_scale = torch.nn.Parameter(torch.tensor(2.0))
            self.prediction_scale = torch.nn.Parameter(torch.tensor(0.5))

        def encoder(self, runtime):
            return self.encoder_scale * runtime.x

        def linear_eval(self, runtime):
            z = self.encoder(runtime)
            return self.prediction_scale * z, z

    config = {"model": {"name": "demo"}, "phases": []}
    execution_services = ExecutionServices.from_config(config, default_device=torch.device("cpu"))
    phase = build_phase(
        OptimizerPhaseSpec(name="one_step", trainer="OneStep", config={"order": 1}),
        config=config,
        model_class=_ContinuousEncoderModel,
        dtype=torch.float32,
        execution_services=execution_services,
    )
    model = _ContinuousEncoderModel()
    criterion = torch.nn.MSELoss()
    optimizer_state = OptimizerStateArtifact(
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        criteria=[criterion, torch.nn.MSELoss()],
        criteria_weights=[1.0],
        criteria_names=["dynamics", "mse"],
        _one_step_dt=1.0,
        _one_step_kwargs={"order": 1},
    )
    series = RegularSeries(
        time=torch.tensor([0.0, 1.0, 2.0]),
        state=torch.tensor([[0.0], [1.0], [4.0]]),
        control=None,
        meta={},
    )
    batch = RegularTrainerBatch.collate_series([series])

    loss = phase._compute_losses(model, optimizer_state, batch, "dopri5", {})[0]
    actual_grad = torch.autograd.grad(loss, model.encoder_scale)[0]

    runtime_batch = batch.to(phase.device)
    predictions, targets = _comp_linear_eval_ct(model, runtime_batch, dt=1.0, order=1)
    expected_loss = criterion(predictions, targets.detach())
    expected_grad = torch.autograd.grad(expected_loss, model.encoder_scale)[0]

    assert targets.requires_grad is False
    assert actual_grad is not None
    assert actual_grad.item() == pytest.approx(expected_grad.item())


def test_weak_form_optimizer_phase_uses_discrete_projected_residuals():
    class _DiscreteWeakModel(torch.nn.Module):
        CONT = False

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(2.0))

        def encoder(self, runtime):
            return runtime.x

        def dynamics(self, z, runtime):
            return self.weight * z

        def decoder(self, z, runtime):
            return z

    config = {"model": {"name": "demo"}, "phases": []}
    execution_services = ExecutionServices.from_config(config, default_device=torch.device("cpu"))
    phase = build_phase(
        OptimizerPhaseSpec(
            name="weak",
            trainer="Weak",
            config={"weak_form_params": {"N": 3, "dN": 1, "ordpol": 1, "ordint": 1}},
        ),
        config=config,
        model_class=_DiscreteWeakModel,
        dtype=torch.float32,
        execution_services=execution_services,
    )
    model = _DiscreteWeakModel()
    optimizer_state = OptimizerStateArtifact(
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        criteria=[torch.nn.MSELoss(), torch.nn.MSELoss()],
        criteria_weights=[1.0],
        criteria_names=["dynamics", "mse"],
    )
    phase._customize_optimizer_artifact(
        optimizer_state,
        model,
        PhaseContext(train_md={"dt_and_n_steps": [(1.0, 4)]}),
        logging.getLogger("test.weak.discrete"),
    )
    series = RegularSeries(
        time=torch.tensor([0.0, 1.0, 2.0, 3.0]),
        state=torch.tensor([[1.0], [2.0], [4.0], [8.0]]),
        control=None,
        meta={},
    )
    batch = RegularTrainerBatch.collate_series([series])

    losses = phase._compute_losses(model, optimizer_state, batch, "dopri5", {})

    assert losses[0].item() == pytest.approx(0.0)


def test_weak_form_optimizer_phase_matches_expected_discrete_projected_loss():
    class _DiscreteWeakModel(torch.nn.Module):
        CONT = False

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def encoder(self, runtime):
            return runtime.x

        def dynamics(self, z, runtime):
            return self.weight * z

        def decoder(self, z, runtime):
            return z

    config = {"model": {"name": "demo"}, "phases": []}
    execution_services = ExecutionServices.from_config(config, default_device=torch.device("cpu"))
    phase = build_phase(
        OptimizerPhaseSpec(
            name="weak",
            trainer="Weak",
            config={"weak_form_params": {"N": 3, "dN": 1, "ordpol": 1, "ordint": 1}},
        ),
        config=config,
        model_class=_DiscreteWeakModel,
        dtype=torch.float32,
        execution_services=execution_services,
    )
    model = _DiscreteWeakModel()
    criterion = torch.nn.MSELoss()
    optimizer_state = OptimizerStateArtifact(
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        criteria=[criterion, torch.nn.MSELoss()],
        criteria_weights=[1.0],
        criteria_names=["dynamics", "mse"],
    )
    phase._customize_optimizer_artifact(
        optimizer_state,
        model,
        PhaseContext(train_md={"dt_and_n_steps": [(1.0, 4)]}),
        logging.getLogger("test.weak.discrete"),
    )
    series = RegularSeries(
        time=torch.tensor([0.0, 1.0, 2.0, 3.0]),
        state=torch.tensor([[1.0], [2.0], [4.0], [8.0]]),
        control=None,
        meta={},
    )
    batch = RegularTrainerBatch.collate_series([series])

    loss = phase._compute_losses(model, optimizer_state, batch, "dopri5", {})[0]
    weights = torch.tensor(
        generate_discrete_weak_weights(1.0, 3, poly_order=1, int_rule_order=1),
        dtype=torch.float32,
    )
    projected_pred = weights @ torch.tensor([1.0, 2.0, 4.0], dtype=torch.float32)
    projected_true = weights @ torch.tensor([2.0, 4.0, 8.0], dtype=torch.float32)
    expected = criterion(projected_pred, projected_true)

    assert loss.item() == pytest.approx(expected.item())


def test_weak_form_optimizer_phase_detaches_discrete_targets_for_trainable_encoder():
    class _DiscreteEncoderWeakModel(torch.nn.Module):
        CONT = False

        def __init__(self):
            super().__init__()
            self.encoder_scale = torch.nn.Parameter(torch.tensor(2.0))
            self.prediction_scale = torch.nn.Parameter(torch.tensor(0.5))

        def encoder(self, runtime):
            return self.encoder_scale * runtime.x

        def dynamics(self, z, runtime):
            return self.prediction_scale * z

        def decoder(self, z, runtime):
            return z

    config = {"model": {"name": "demo"}, "phases": []}
    execution_services = ExecutionServices.from_config(config, default_device=torch.device("cpu"))
    phase = build_phase(
        OptimizerPhaseSpec(
            name="weak",
            trainer="Weak",
            config={"weak_form_params": {"N": 3, "dN": 1, "ordpol": 1, "ordint": 1}},
        ),
        config=config,
        model_class=_DiscreteEncoderWeakModel,
        dtype=torch.float32,
        execution_services=execution_services,
    )
    model = _DiscreteEncoderWeakModel()
    criterion = torch.nn.MSELoss()
    optimizer_state = OptimizerStateArtifact(
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        criteria=[criterion, torch.nn.MSELoss()],
        criteria_weights=[1.0],
        criteria_names=["dynamics", "mse"],
    )
    phase._customize_optimizer_artifact(
        optimizer_state,
        model,
        PhaseContext(train_md={"dt_and_n_steps": [(1.0, 5)]}),
        logging.getLogger("test.weak.discrete"),
    )
    series = RegularSeries(
        time=torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0]),
        state=torch.tensor([[1.0], [2.0], [4.0], [8.0], [16.0]]),
        control=None,
        meta={},
    )
    batch = RegularTrainerBatch.collate_series([series])

    loss = phase._compute_losses(model, optimizer_state, batch, "dopri5", {})[0]
    actual_grad = torch.autograd.grad(loss, model.encoder_scale)[0]

    runtime = phases_module.batch_to_runtime(batch.to(phase.device))
    latent = model.encoder(runtime)
    projected_pred = model.dynamics(latent, runtime)[:, :-1, :].unfold(1, 3, 1) @ cast(
        torch.Tensor, optimizer_state._weak_D
    )
    projected_target = latent[:, 1:, :].unfold(1, 3, 1) @ cast(
        torch.Tensor, optimizer_state._weak_D
    )
    expected_loss = criterion(projected_pred, projected_target.detach())
    expected_grad = torch.autograd.grad(expected_loss, model.encoder_scale)[0]

    assert projected_target.requires_grad is True
    assert actual_grad is not None
    assert actual_grad.item() == pytest.approx(expected_grad.item())


def test_weak_form_optimizer_phase_rejects_invalid_discrete_weight_normalization():
    config = {"model": {"name": "demo"}, "phases": []}
    execution_services = ExecutionServices.from_config(config, default_device=torch.device("cpu"))

    with pytest.raises(
        PhaseSpecValidationError,
        match="weak_form_params.discrete_weight_normalization must be one of 'l2', 'l1', or 'none'",
    ):
        build_phase(
            OptimizerPhaseSpec(
                name="weak",
                trainer="Weak",
                config={
                    "weak_form_params": {
                        "N": 2,
                        "dN": 1,
                        "ordpol": 1,
                        "ordint": 1,
                        "discrete_weight_normalization": "bad",
                    }
                },
            ),
            config=config,
            model_class=object,
            dtype=torch.float32,
            execution_services=execution_services,
        )


def test_weak_form_optimizer_phase_rejects_too_short_discrete_trajectories():
    class _DiscreteWeakModel(torch.nn.Module):
        CONT = False

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def encoder(self, runtime):
            return runtime.x

        def dynamics(self, z, runtime):
            return self.weight * z

        def decoder(self, z, runtime):
            return z

    config = {"model": {"name": "demo"}, "phases": []}
    execution_services = ExecutionServices.from_config(config, default_device=torch.device("cpu"))
    phase = build_phase(
        OptimizerPhaseSpec(
            name="weak",
            trainer="Weak",
            config={"weak_form_params": {"N": 3, "dN": 1, "ordpol": 1, "ordint": 1}},
        ),
        config=config,
        model_class=_DiscreteWeakModel,
        dtype=torch.float32,
        execution_services=execution_services,
    )
    model = _DiscreteWeakModel()
    optimizer_state = OptimizerStateArtifact(
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        criteria=[torch.nn.MSELoss(), torch.nn.MSELoss()],
        criteria_weights=[1.0],
        criteria_names=["dynamics", "mse"],
    )

    with pytest.raises(
        PhaseSpecValidationError,
        match="Discrete Weak phase requires at least 4 time samples when weak_form_params.N=3 residual weights are requested",
    ):
        phase._customize_optimizer_artifact(
            optimizer_state,
            model,
            PhaseContext(train_md={"dt_and_n_steps": [(1.0, 3)]}),
            logging.getLogger("test.weak.discrete"),
        )


def test_select_best_cv_result_uses_goal_and_tie_breakers() -> None:
    cv_results = [
        CVResult(params={"model.koopman_dimension": 6}, fold_metrics=[1.0, 1.0]),
        CVResult(params={"model.koopman_dimension": 4}, fold_metrics=[1.0, 1.2]),
        CVResult(params={"model.koopman_dimension": 8}, fold_metrics=[1.1, 1.1]),
    ]

    best_idx = select_best_cv_result(
        cv_results,
        goal="minimize",
        tie_breakers=("std_metric", "param_l1", "combo_index"),
    )

    assert best_idx == 0


def test_select_best_cv_result_supports_maximize_goal() -> None:
    cv_results = [
        CVResult(params={"model.koopman_dimension": 6}, fold_metrics=[1.0, 1.0]),
        CVResult(params={"model.koopman_dimension": 4}, fold_metrics=[2.0, 2.0]),
    ]

    best_idx = select_best_cv_result(
        cv_results,
        goal="maximize",
        tie_breakers=("std_metric", "combo_index"),
    )

    assert best_idx == 1


def test_select_best_cv_result_uses_explicit_combo_indices_for_tie_breaker() -> None:
    cv_results = [
        CVResult(params={"model.koopman_dimension": 8}, fold_metrics=[1.0]),
        CVResult(params={"model.koopman_dimension": 5}, fold_metrics=[1.0]),
        CVResult(params={"model.koopman_dimension": 3}, fold_metrics=[1.0]),
    ]

    best_idx = select_best_cv_result(
        cv_results,
        goal="minimize",
        tie_breakers=("std_metric", "combo_index"),
        combo_indices=(8, 5, 3),
    )

    assert best_idx == 2


def test_select_best_cv_result_rejects_unsupported_tie_breaker() -> None:
    cv_results = [CVResult(params={"model.koopman_dimension": 4}, fold_metrics=[1.0])]

    with pytest.raises(ValueError, match="unsupported tie breaker"):
        select_best_cv_result(cv_results, tie_breakers=("unknown_rule",))


def test_nelder_mead_like_search_indices_limits_evaluations() -> None:
    combos = [{"model.koopman_dimension": value} for value in range(9)]
    evaluation_calls: list[int] = []

    def _evaluate(index: int) -> float:
        evaluation_calls.append(index)
        value = float(combos[index]["model.koopman_dimension"])
        return (value - 5.0) ** 2

    evaluated = nelder_mead_like_search_indices(
        combos,
        evaluate_index=_evaluate,
        max_iterations=1,
    )

    assert evaluated == evaluation_calls
    assert evaluated[:2] == [0, 8]
    assert len(evaluated) == 4
    assert 6 in evaluated


def test_nelder_mead_like_search_indices_falls_back_to_grid_for_non_numeric_values() -> None:
    combos = [{"model.variant": name} for name in ("a", "b", "c")]
    evaluation_calls: list[int] = []

    def _evaluate(index: int) -> float:
        evaluation_calls.append(index)
        return float(index)

    evaluated = nelder_mead_like_search_indices(
        combos,
        evaluate_index=_evaluate,
        max_iterations=3,
    )

    assert evaluated == [0, 1, 2]
    assert evaluation_calls == [0, 1, 2]


def test_bounded_nelder_mead_search_points_respects_bounds() -> None:
    target = np.array([0.25, 0.75], dtype=float)

    def _evaluate(point: np.ndarray) -> float:
        return float(np.sum((point - target) ** 2))

    evaluated = bounded_nelder_mead_search_points(
        lower_bounds=[0.0, 0.0],
        upper_bounds=[1.0, 1.0],
        evaluate_point=_evaluate,
        max_iterations=6,
    )

    assert evaluated
    for point in evaluated:
        assert np.all(point >= 0.0)
        assert np.all(point <= 1.0)
    best_point = min(evaluated, key=_evaluate)
    assert np.linalg.norm(best_point - target) <= 0.25


def test_batch_pattern_search_points_respects_bounds_and_batches() -> None:
    target = np.array([0.75, 0.25], dtype=float)
    batch_lengths: list[int] = []

    def _evaluate(points: list[np.ndarray]) -> list[float]:
        batch_lengths.append(len(points))
        return [float(np.sum((point - target) ** 2)) for point in points]

    evaluated = batch_pattern_search_points(
        lower_bounds=[0.0, 0.0],
        upper_bounds=[1.0, 1.0],
        evaluate_points=_evaluate,
        max_evaluations=8,
        batch_size=3,
    )

    assert evaluated
    assert any(length > 1 for length in batch_lengths)
    for point in evaluated:
        assert np.all(point >= 0.0)
        assert np.all(point <= 1.0)
    best_point = min(evaluated, key=lambda point: float(np.sum((point - target) ** 2)))
    assert np.linalg.norm(best_point - target) <= 0.25


def test_multi_start_bounded_nelder_mead_search_points_respects_bounds() -> None:
    target = np.array([0.75, 0.25], dtype=float)

    def _evaluate(point: np.ndarray) -> float:
        return float(np.sum((point - target) ** 2))

    evaluated = multi_start_bounded_nelder_mead_search_points(
        lower_bounds=[0.0, 0.0],
        upper_bounds=[1.0, 1.0],
        evaluate_point=_evaluate,
        max_iterations=8,
        num_simplices=4,
        max_workers=2,
        seed=5,
    )

    assert evaluated
    for point in evaluated:
        assert np.all(point >= 0.0)
        assert np.all(point <= 1.0)
    best_point = min(evaluated, key=_evaluate)
    assert np.linalg.norm(best_point - target) <= 0.35


def test_single_split_driver_train_supports_nelder_mead_like_search(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model:
  name: demo
phases:
  - trainer: Linear
cv:
  param_grid:
    model.koopman_dimension: [0, 1, 2, 3, 4]
  search:
    mode: nelder_mead_like
    max_iterations: 1
""".strip(),
        encoding="utf-8",
    )

    class _FakeTrainSet:
        dtype = torch.float32

    def _fake_init_trajectory_managers(self):
        self.train_sets = [_FakeTrainSet()]
        self.valid_sets = [_FakeTrainSet()]

    monkeypatch.setattr(
        driver.SingleSplitDriver,
        "_init_trajectory_managers",
        _fake_init_trajectory_managers,
    )
    monkeypatch.setattr(driver.SingleSplitDriver, "_init_fold_split", lambda self: None)

    evaluated_dims: list[int] = []

    def _fake_run_cv_single(args):
        value = int(args["combo"]["model.koopman_dimension"])
        evaluated_dims.append(value)
        model_prefix = f"{args['checkpoint_prefix']}/fake_{args['combo_idx']}_{args['fold_idx']}"
        with open(f"{model_prefix}.pt", "wb") as handle:
            handle.write(b"pt")
        np.savez_compressed(f"{model_prefix}_summary.npz", koopman_dimension=np.array([value]))
        return {
            "combo_idx": args["combo_idx"],
            "fold_idx": args["fold_idx"],
            "combo": args["combo"],
            "metric_value": float((value - 2) ** 2),
            "model_prefix": model_prefix,
        }

    monkeypatch.setattr(driver, "run_cv_single", _fake_run_cv_single)
    monkeypatch.setattr(driver, "plot_cv_results", lambda *args, **kwargs: None)

    trainer = driver.SingleSplitDriver(
        config_path=str(config_path),
        model_class=torch.nn.Module,
        device=torch.device("cpu"),
    )
    _, best_result, all_results = trainer.train()

    assert len(all_results) == 4
    assert len(evaluated_dims) == 4
    assert best_result.params["model.koopman_dimension"] == 2


def test_single_split_driver_train_supports_explicit_grid_search(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model:
  name: demo
phases:
  - trainer: Linear
cv:
  param_grid:
    model.koopman_dimension: [0, 1, 2, 3, 4]
  search:
    mode: grid
""".strip(),
        encoding="utf-8",
    )

    class _FakeTrainSet:
        dtype = torch.float32

    def _fake_init_trajectory_managers(self):
        self.train_sets = [_FakeTrainSet()]
        self.valid_sets = [_FakeTrainSet()]

    monkeypatch.setattr(
        driver.SingleSplitDriver,
        "_init_trajectory_managers",
        _fake_init_trajectory_managers,
    )
    monkeypatch.setattr(driver.SingleSplitDriver, "_init_fold_split", lambda self: None)

    evaluated_dims: list[int] = []

    def _fake_run_cv_single(args):
        value = int(args["combo"]["model.koopman_dimension"])
        evaluated_dims.append(value)
        model_prefix = f"{args['checkpoint_prefix']}/fake_{args['combo_idx']}_{args['fold_idx']}"
        with open(f"{model_prefix}.pt", "wb") as handle:
            handle.write(b"pt")
        np.savez_compressed(f"{model_prefix}_summary.npz", koopman_dimension=np.array([value]))
        return {
            "combo_idx": args["combo_idx"],
            "fold_idx": args["fold_idx"],
            "combo": args["combo"],
            "metric_value": float((value - 2) ** 2),
            "model_prefix": model_prefix,
        }

    monkeypatch.setattr(driver, "run_cv_single", _fake_run_cv_single)
    monkeypatch.setattr(driver, "plot_cv_results", lambda *args, **kwargs: None)

    trainer = driver.SingleSplitDriver(
        config_path=str(config_path),
        model_class=torch.nn.Module,
        device=torch.device("cpu"),
    )
    _, best_result, all_results = trainer.train()

    assert evaluated_dims == [0, 1, 2, 3, 4]
    assert len(all_results) == 5
    assert best_result.params["model.koopman_dimension"] == 2
    tuning_dir = tmp_path / "demo" / "demo_tuning"
    assert (tuning_dir / "tuning_result.json").is_file()
    assert (tuning_dir / "tuning_evaluations.csv").is_file()
    payload = json.loads((tuning_dir / "tuning_result.json").read_text(encoding="utf-8"))
    assert payload["selected_params"]["model.koopman_dimension"] == 2


def test_single_split_driver_uses_configured_split_seed(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model:
  name: demo
data:
  split_seed: 7
split:
  train_frac: 0.6
phases:
  - trainer: Linear
""".strip(),
        encoding="utf-8",
    )

    class _FakeTrainSet:
        dtype = torch.float32
        metadata = {"n_samples": 10}

        def __init__(self):
            self.data_index = None

        def set_data_index(self, data_index):
            self.data_index = data_index.clone()

    def _fake_init_trajectory_managers(self):
        self.train_sets = [_FakeTrainSet()]
        self.valid_sets = [_FakeTrainSet()]

    monkeypatch.setattr(
        driver.SingleSplitDriver,
        "_init_trajectory_managers",
        _fake_init_trajectory_managers,
    )

    torch.manual_seed(123)
    trainer_a = driver.SingleSplitDriver(
        config_path=str(config_path),
        model_class=torch.nn.Module,
        device=torch.device("cpu"),
    )

    torch.manual_seed(999)
    trainer_b = driver.SingleSplitDriver(
        config_path=str(config_path),
        model_class=torch.nn.Module,
        device=torch.device("cpu"),
    )

    trainer_c = driver.SingleSplitDriver(
        config_path=str(config_path),
        model_class=torch.nn.Module,
        config_mod={"data": {"split_seed": 11}},
        device=torch.device("cpu"),
    )

    assert trainer_a.base_config["data"]["split_seed"] == 7
    assert torch.equal(trainer_a.train_set_index, trainer_b.train_set_index)
    assert torch.equal(trainer_a.valid_set_index, trainer_b.valid_set_index)
    assert not torch.equal(trainer_a.train_set_index, trainer_c.train_set_index)
    assert not torch.equal(trainer_a.valid_set_index, trainer_c.valid_set_index)


def test_single_split_driver_train_supports_bounded_nelder_mead_search(
    monkeypatch, tmp_path
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model:
  name: demo
  koopman_dimension: 1
training:
  weak_form_params:
    N: 13
phases:
  - trainer: Weak
cv:
  search:
    mode: nelder_mead_like
    bounds:
      model.koopman_dimension: [0, 4]
      training.weak_form_params.N:
        lower: 9
        upper: 17
        parity: odd
    max_iterations: 6
""".strip(),
        encoding="utf-8",
    )

    class _FakeTrainSet:
        dtype = torch.float32

    def _fake_init_trajectory_managers(self):
        self.train_sets = [_FakeTrainSet()]
        self.valid_sets = [_FakeTrainSet()]

    monkeypatch.setattr(
        driver.SingleSplitDriver,
        "_init_trajectory_managers",
        _fake_init_trajectory_managers,
    )
    monkeypatch.setattr(driver.SingleSplitDriver, "_init_fold_split", lambda self: None)

    evaluated_params: list[tuple[int, int]] = []

    def _fake_run_cv_single(args):
        koopman_dimension = int(args["combo"]["model.koopman_dimension"])
        weak_n = int(args["combo"]["phases.0.weak_form_params.N"])
        evaluated_params.append((koopman_dimension, weak_n))
        model_prefix = f"{args['checkpoint_prefix']}/fake_{args['combo_idx']}_{args['fold_idx']}"
        with open(f"{model_prefix}.pt", "wb") as handle:
            handle.write(b"pt")
        np.savez_compressed(
            f"{model_prefix}_summary.npz",
            koopman_dimension=np.array([koopman_dimension]),
            weak_n=np.array([weak_n]),
        )
        return {
            "combo_idx": args["combo_idx"],
            "fold_idx": args["fold_idx"],
            "combo": args["combo"],
            "metric_value": float((koopman_dimension - 2) ** 2 + 0.01 * (weak_n - 11) ** 2),
            "model_prefix": model_prefix,
        }

    monkeypatch.setattr(driver, "run_cv_single", _fake_run_cv_single)
    monkeypatch.setattr(driver, "plot_cv_results", lambda *args, **kwargs: None)

    trainer = driver.SingleSplitDriver(
        config_path=str(config_path),
        model_class=torch.nn.Module,
        device=torch.device("cpu"),
    )
    _, best_result, all_results = trainer.train()

    assert all_results
    assert evaluated_params
    assert all(0 <= koopman_dimension <= 4 for koopman_dimension, _ in evaluated_params)
    assert all(9 <= weak_n <= 17 and weak_n % 2 == 1 for _, weak_n in evaluated_params)
    assert best_result.params["model.koopman_dimension"] == 2
    assert best_result.params["phases.0.weak_form_params.N"] == 11


def test_single_split_driver_train_supports_bounded_batch_pattern_search(
    monkeypatch, tmp_path
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model:
  name: demo
  koopman_dimension: 1
training:
  weak_form_params:
    N: 13
phases:
  - trainer: Weak
cv:
  search:
    mode: batch_pattern_search
    bounds:
      model.koopman_dimension: [0, 4]
      training.weak_form_params.N:
        lower: 9
        upper: 17
        parity: odd
    max_iterations: 8
""".strip(),
        encoding="utf-8",
    )

    class _FakeTrainSet:
        dtype = torch.float32

    def _fake_init_trajectory_managers(self):
        self.train_sets = [_FakeTrainSet()]
        self.valid_sets = [_FakeTrainSet()]

    monkeypatch.setattr(
        driver.SingleSplitDriver,
        "_init_trajectory_managers",
        _fake_init_trajectory_managers,
    )
    monkeypatch.setattr(driver.SingleSplitDriver, "_init_fold_split", lambda self: None)

    evaluated_params: list[tuple[int, int]] = []

    def _fake_run_cv_single(args):
        koopman_dimension = int(args["combo"]["model.koopman_dimension"])
        weak_n = int(args["combo"]["phases.0.weak_form_params.N"])
        evaluated_params.append((koopman_dimension, weak_n))
        model_prefix = f"{args['checkpoint_prefix']}/fake_{args['combo_idx']}_{args['fold_idx']}"
        with open(f"{model_prefix}.pt", "wb") as handle:
            handle.write(b"pt")
        np.savez_compressed(
            f"{model_prefix}_summary.npz",
            koopman_dimension=np.array([koopman_dimension]),
            weak_n=np.array([weak_n]),
        )
        return {
            "combo_idx": args["combo_idx"],
            "fold_idx": args["fold_idx"],
            "combo": args["combo"],
            "metric_value": float((koopman_dimension - 2) ** 2 + 0.01 * (weak_n - 11) ** 2),
            "model_prefix": model_prefix,
        }

    monkeypatch.setattr(driver, "run_cv_single", _fake_run_cv_single)
    monkeypatch.setattr(driver, "plot_cv_results", lambda *args, **kwargs: None)

    with pytest.warns(RuntimeWarning, match="batch_pattern_search"):
        trainer = driver.SingleSplitDriver(
            config_path=str(config_path),
            model_class=torch.nn.Module,
            device=torch.device("cpu"),
        )
    _, best_result, all_results = trainer.train()

    assert all_results
    assert evaluated_params
    assert all(0 <= koopman_dimension <= 4 for koopman_dimension, _ in evaluated_params)
    assert all(9 <= weak_n <= 17 and weak_n % 2 == 1 for _, weak_n in evaluated_params)
    assert best_result.params["model.koopman_dimension"] == 2
    assert best_result.params["phases.0.weak_form_params.N"] == 11


def test_single_split_driver_train_supports_multi_start_nelder_mead_search(
    monkeypatch, tmp_path
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
seed: 3
model:
  name: demo
  koopman_dimension: 1
training:
  weak_form_params:
    N: 13
phases:
  - trainer: Weak
cv:
  search:
    mode: multi_start_nelder_mead
    bounds:
      model.koopman_dimension: [0, 4]
      training.weak_form_params.N:
        lower: 9
        upper: 17
        parity: odd
    max_iterations: 16
""".strip(),
        encoding="utf-8",
    )

    class _FakeTrainSet:
        dtype = torch.float32

    def _fake_init_trajectory_managers(self):
        self.train_sets = [_FakeTrainSet()]
        self.valid_sets = [_FakeTrainSet()]

    monkeypatch.setattr(
        driver.SingleSplitDriver,
        "_init_trajectory_managers",
        _fake_init_trajectory_managers,
    )
    monkeypatch.setattr(driver.SingleSplitDriver, "_init_fold_split", lambda self: None)

    evaluated_params: list[tuple[int, int]] = []

    def _fake_run_cv_single(args):
        koopman_dimension = int(args["combo"]["model.koopman_dimension"])
        weak_n = int(args["combo"]["phases.0.weak_form_params.N"])
        evaluated_params.append((koopman_dimension, weak_n))
        model_prefix = f"{args['checkpoint_prefix']}/fake_{args['combo_idx']}_{args['fold_idx']}"
        with open(f"{model_prefix}.pt", "wb") as handle:
            handle.write(b"pt")
        np.savez_compressed(
            f"{model_prefix}_summary.npz",
            koopman_dimension=np.array([koopman_dimension]),
            weak_n=np.array([weak_n]),
        )
        return {
            "combo_idx": args["combo_idx"],
            "fold_idx": args["fold_idx"],
            "combo": args["combo"],
            "metric_value": float((koopman_dimension - 2) ** 2 + 0.01 * (weak_n - 11) ** 2),
            "model_prefix": model_prefix,
        }

    monkeypatch.setattr(driver, "run_cv_single", _fake_run_cv_single)
    monkeypatch.setattr(driver, "plot_cv_results", lambda *args, **kwargs: None)

    trainer = driver.SingleSplitDriver(
        config_path=str(config_path),
        model_class=torch.nn.Module,
        device=torch.device("cpu"),
        max_workers=4,
    )
    _, best_result, all_results = trainer.train()

    assert all_results
    assert evaluated_params
    assert all(0 <= koopman_dimension <= 4 for koopman_dimension, _ in evaluated_params)
    assert all(9 <= weak_n <= 17 and weak_n % 2 == 1 for _, weak_n in evaluated_params)
    assert best_result.mean_metric == min(result.mean_metric for result in all_results)


def test_single_split_driver_warns_for_parallel_nelder_mead_like_search(
    monkeypatch, tmp_path
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model:
  name: demo
phases:
  - trainer: Linear
cv:
  param_grid:
    model.koopman_dimension: [0, 1, 2]
  search:
    mode: nelder_mead_like
    max_iterations: 1
""".strip(),
        encoding="utf-8",
    )

    class _FakeTrainSet:
        dtype = torch.float32

    def _fake_init_trajectory_managers(self):
        self.train_sets = [_FakeTrainSet()]
        self.valid_sets = [_FakeTrainSet()]

    monkeypatch.setattr(
        driver.SingleSplitDriver,
        "_init_trajectory_managers",
        _fake_init_trajectory_managers,
    )
    monkeypatch.setattr(driver.SingleSplitDriver, "_init_fold_split", lambda self: None)

    with pytest.warns(RuntimeWarning, match="nelder_mead_like"):
        driver.SingleSplitDriver(
            config_path=str(config_path),
            model_class=torch.nn.Module,
            device=torch.device("cpu"),
            max_workers=2,
        )


def test_single_split_driver_bounded_nelder_mead_evaluates_integer_upper_endpoint(
    monkeypatch, tmp_path
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model:
  name: demo
  koopman_dimension: 1
phases:
  - trainer: Linear
cv:
  search:
    mode: nelder_mead_like
    bounds:
      model.koopman_dimension: [0, 3]
    max_iterations: 4
""".strip(),
        encoding="utf-8",
    )

    class _FakeTrainSet:
        dtype = torch.float32

    def _fake_init_trajectory_managers(self):
        self.train_sets = [_FakeTrainSet()]
        self.valid_sets = [_FakeTrainSet()]

    monkeypatch.setattr(
        driver.SingleSplitDriver,
        "_init_trajectory_managers",
        _fake_init_trajectory_managers,
    )
    monkeypatch.setattr(driver.SingleSplitDriver, "_init_fold_split", lambda self: None)

    evaluated_dims: list[int] = []

    def _fake_run_cv_single(args):
        value = int(args["combo"]["model.koopman_dimension"])
        evaluated_dims.append(value)
        model_prefix = f"{args['checkpoint_prefix']}/fake_{args['combo_idx']}_{args['fold_idx']}"
        with open(f"{model_prefix}.pt", "wb") as handle:
            handle.write(b"pt")
        np.savez_compressed(f"{model_prefix}_summary.npz", koopman_dimension=np.array([value]))
        return {
            "combo_idx": args["combo_idx"],
            "fold_idx": args["fold_idx"],
            "combo": args["combo"],
            "metric_value": float((value - 3) ** 2),
            "model_prefix": model_prefix,
        }

    monkeypatch.setattr(driver, "run_cv_single", _fake_run_cv_single)
    monkeypatch.setattr(driver, "plot_cv_results", lambda *args, **kwargs: None)

    trainer = driver.SingleSplitDriver(
        config_path=str(config_path),
        model_class=torch.nn.Module,
        device=torch.device("cpu"),
    )
    _, best_result, _ = trainer.train()

    assert 3 in evaluated_dims
    assert best_result.params["model.koopman_dimension"] == 3


@pytest.mark.parametrize(
    ("config_text", "expected_error", "expected_match"),
    [
        (
            """
model:
  name: demo
phases:
  - trainer: Linear
    learning_rate: 0.1
cv:
  search:
    mode: nelder_mead_like
    bounds:
      phases.0.learning_rate:
        lower: 0.01
        upper: 0.2
        parity: odd
""".strip(),
            TypeError,
            "parity is only supported for integer-valued config fields",
        ),
        (
            """
model:
  name: demo
phases:
  - trainer: Linear
cv:
  search:
    mode: nelder_mead_like
    bounds:
      model.name: [0, 4]
""".strip(),
            TypeError,
            "must target an integer or floating-point config value",
        ),
        (
            """
model:
  name: demo
phases:
  - trainer: Linear
cv:
  search:
    mode: nelder_mead_like
    bounds:
      model.missing_dimension: [0, 4]
""".strip(),
            TypeError,
            "does not resolve in the config",
        ),
    ],
)
def test_single_split_driver_rejects_invalid_bounded_nelder_mead_yaml_configs(
    monkeypatch, tmp_path, config_text, expected_error, expected_match
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_text, encoding="utf-8")

    class _FakeTrainSet:
        dtype = torch.float32

    def _fake_init_trajectory_managers(self):
        self.train_sets = [_FakeTrainSet()]
        self.valid_sets = [_FakeTrainSet()]

    monkeypatch.setattr(
        driver.SingleSplitDriver,
        "_init_trajectory_managers",
        _fake_init_trajectory_managers,
    )
    monkeypatch.setattr(driver.SingleSplitDriver, "_init_fold_split", lambda self: None)

    with pytest.raises(expected_error, match=expected_match):
        driver.SingleSplitDriver(
            config_path=str(config_path),
            model_class=torch.nn.Module,
            device=torch.device("cpu"),
        )


def test_single_split_driver_train_uses_grid_combo_index_for_tie_breaker(
    monkeypatch, tmp_path
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model:
  name: demo
phases:
  - trainer: Linear
cv:
  param_grid:
    model.koopman_dimension: [0, 1, 2, 3, 4, 5, 6, 7, 8]
  search:
    mode: nelder_mead_like
    max_iterations: 2
""".strip(),
        encoding="utf-8",
    )

    class _FakeTrainSet:
        dtype = torch.float32

    def _fake_init_trajectory_managers(self):
        self.train_sets = [_FakeTrainSet()]
        self.valid_sets = [_FakeTrainSet()]

    monkeypatch.setattr(
        driver.SingleSplitDriver,
        "_init_trajectory_managers",
        _fake_init_trajectory_managers,
    )
    monkeypatch.setattr(driver.SingleSplitDriver, "_init_fold_split", lambda self: None)

    def _fake_nelder_mead_like_search_indices(combos, *, evaluate_index, **kwargs):
        evaluated = [0, 8, 2, 5, 7, 4, 1, 3]
        for index in evaluated:
            evaluate_index(index)
        return evaluated

    monkeypatch.setattr(
        driver,
        "nelder_mead_like_search_indices",
        _fake_nelder_mead_like_search_indices,
    )

    evaluated_dims: list[int] = []

    def _fake_run_cv_single(args):
        value = int(args["combo"]["model.koopman_dimension"])
        evaluated_dims.append(value)
        model_prefix = f"{args['checkpoint_prefix']}/fake_{args['combo_idx']}_{args['fold_idx']}"
        with open(f"{model_prefix}.pt", "wb") as handle:
            handle.write(b"pt")
        np.savez_compressed(f"{model_prefix}_summary.npz", koopman_dimension=np.array([value]))
        metric_value = 0.0 if value in {8, 5, 3} else 1.0
        return {
            "combo_idx": args["combo_idx"],
            "fold_idx": args["fold_idx"],
            "combo": args["combo"],
            "metric_value": metric_value,
            "model_prefix": model_prefix,
        }

    monkeypatch.setattr(driver, "run_cv_single", _fake_run_cv_single)
    monkeypatch.setattr(driver, "plot_cv_results", lambda *args, **kwargs: None)

    trainer = driver.SingleSplitDriver(
        config_path=str(config_path),
        model_class=torch.nn.Module,
        device=torch.device("cpu"),
    )
    _, best_result, all_results = trainer.train()

    assert evaluated_dims == [0, 8, 2, 5, 7, 4, 1, 3]
    assert len(all_results) == 8
    assert best_result.params["model.koopman_dimension"] == 3


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


def test_trainer_run_replays_completed_data_phases_on_resume(monkeypatch, tmp_path):
    config = {
        "model": {"name": "demo"},
        "dataloader": {"batch_size": 1, "shuffle": False},
        "phases": [
            {
                "type": "data",
                "name": "smooth",
                "operation": "smooth",
                "window_length": 5,
                "polyorder": 2,
            },
            {"name": "linear", "trainer": "Linear"},
        ],
    }
    trainer_run = TrainerRun(
        config=config,
        model_class=object,
        device=torch.device("cpu"),
        dtype=torch.float32,
        run_name="demo",
        checkpoint_prefix=str(tmp_path),
        results_prefix=str(tmp_path),
    )
    initial_context = PhaseContext(
        train_set=[_build_regular_series()],
        valid_set=[_build_regular_series(offset=0.2)],
        train_loader=object(),
        valid_loader=object(),
        train_md={"dt_and_n_steps": [(0.1, 9)]},
        valid_md={"dt_and_n_steps": [(0.1, 9)]},
    )
    resumed_state = TrainerState(
        config=config,
        device=torch.device("cpu"),
        phase_cursor=1,
    )
    monkeypatch.setattr(trainer_run, "_maybe_resume", lambda: (resumed_state, ArtifactRegistry()))

    captured = {}

    def fake_run(*, initial_context, initial_state, artifacts, run_name, checkpoint_callback):
        captured["initial_context"] = initial_context
        captured["initial_state"] = initial_state
        return []

    monkeypatch.setattr(trainer_run.pipeline, "run", fake_run)

    trainer_run.run(initial_context=initial_context)

    replayed_context = captured["initial_context"]
    np.testing.assert_allclose(
        replayed_context.train_set[0].state.detach().cpu().numpy(),
        savgol_filter(initial_context.train_set[0].state.detach().cpu().numpy(), 5, 2, axis=0),
    )
    assert replayed_context.train_md["data_phase_history"][-1]["phase"] == "smooth"
    assert captured["initial_state"] is resumed_state


def test_summary_export_persists_phase_metrics(tmp_path):
    config = {
        "model": {"name": "demo"},
        "plotting": {"prediction": False},
        "path": {"checkpoint_prefix": str(tmp_path), "results_prefix": str(tmp_path)},
        "phases": [],
    }
    execution_services = ExecutionServices.from_config(config, default_device=torch.device("cpu"))
    phase = build_phase(
        ExportPhaseSpec(name="export_summary", export_kind="summary"),
        config=config,
        model_class=torch.nn.Linear,
        dtype=torch.float32,
        execution_services=execution_services,
    )
    trainer_state = TrainerState(
        config=config,
        device=torch.device("cpu"),
        best_loss={"valid_total": 0.3},
        phase_records=[
            PhaseRecord(
                name="smooth",
                kind="data",
                started_epoch=0,
                completed_epoch=0,
                metrics={"train_delta_rmse": 0.1, "train_roughness_ratio": 0.8},
            )
        ],
    )
    artifacts = ArtifactRegistry()
    artifacts.put(
        "model",
        ModelArtifact(
            model=torch.nn.Linear(2, 2),
            config=config,
            train_md={},
            valid_md={},
            dtype=torch.float32,
        ),
    )
    artifacts.put(
        "history",
        TrainingHistoryArtifact(
            hist=[{"train_total": [0.2], "valid_total": [0.3]}],
            epoch_times=[1.0],
            best_loss={"valid_total": 0.3},
        ),
    )

    result = phase.execute(
        trainer_state=trainer_state,
        phase_context=PhaseContext(),
        artifacts=artifacts,
        run_name="demo",
        logger=logging.getLogger("test.export.summary"),
    )

    assert result.metrics["exports_written"] >= 1.0
    with np.load(tmp_path / "demo_summary.npz", allow_pickle=True) as npz:
        phase_metrics = npz["phase_metrics"].item()
        phase_records = npz["phase_records"].tolist()

    assert phase_metrics["smooth"]["train_delta_rmse"] == pytest.approx(0.1)
    assert phase_metrics["smooth"]["train_roughness_ratio"] == pytest.approx(0.8)
    assert len(phase_records) == 1
    assert phase_records[0].name == "smooth"
