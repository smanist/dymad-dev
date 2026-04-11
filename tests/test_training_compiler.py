from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dymad.agent.compiler import (
    TrainingCompileValidationError,
    TrainingRequest,
    compile_training_request,
)
from dymad.agent.exec.context import build_default_context


def _write_regular_dataset(path: Path) -> None:
    t = np.linspace(0.0, 1.0, 6)
    x = np.array(
        [
            [[0.0, 0.0], [0.2, 0.0], [0.4, 0.0], [0.6, 0.0], [0.8, 0.0], [1.0, 0.0]],
            [[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [0.3, 0.0], [0.4, 0.0], [0.5, 0.0]],
        ]
    )
    u = np.ones((2, 6, 1)) * 0.1
    np.savez_compressed(path, t=t, x=x, u=u)


def _write_graph_dataset(path: Path) -> None:
    t = np.linspace(0.0, 1.0, 5)
    x = np.array(
        [
            [
                [0.0, 0.0, 0.1, 0.1],
                [0.1, 0.0, 0.2, 0.1],
                [0.2, 0.0, 0.3, 0.1],
                [0.3, 0.0, 0.4, 0.1],
                [0.4, 0.0, 0.5, 0.1],
            ],
            [
                [0.0, 0.0, 0.2, 0.2],
                [0.2, 0.0, 0.3, 0.2],
                [0.4, 0.0, 0.4, 0.2],
                [0.6, 0.0, 0.5, 0.2],
                [0.8, 0.0, 0.6, 0.2],
            ],
        ]
    )
    adj = np.array([[0, 1], [1, 0]])
    np.savez_compressed(path, t=t, x=x, adj=adj)


def test_compile_training_request_resolves_regular_model_family_and_profile(tmp_path) -> None:
    context = build_default_context(artifact_root=tmp_path / "artifacts")
    dataset_path = tmp_path / "train.npz"
    _write_regular_dataset(dataset_path)
    train_handle = context.facade.register_dataset_file(path=str(dataset_path)).handle

    compiled = compile_training_request(
        facade=context.facade,
        request=TrainingRequest(
            train_dataset_handle=train_handle,
            model_key="kbf",
            run_name="compile_kbf",
            overrides={"model": {"koopman_dimension": 8}},
        ),
    )

    assert compiled.model.key == "kbf"
    assert compiled.model_ref == "dymad.models.collections:KBF"
    assert compiled.profile.key == "kbf-regular-default"
    assert compiled.trainer_kind == "weak_form"
    assert compiled.effective_run_name == "compile_kbf"
    assert compiled.effective_config["data"]["path"] == str(dataset_path.resolve())
    assert compiled.effective_config["model"]["koopman_dimension"] == 8


def test_compile_training_request_resolves_graph_model_family_and_profile(tmp_path) -> None:
    context = build_default_context(artifact_root=tmp_path / "artifacts")
    dataset_path = tmp_path / "train_graph.npz"
    _write_graph_dataset(dataset_path)
    train_handle = context.facade.register_dataset_file(path=str(dataset_path), kind="graph").handle

    compiled = compile_training_request(
        facade=context.facade,
        request=TrainingRequest(
            train_dataset_handle=train_handle,
            model_key="lti",
            run_name="compile_lti_graph",
        ),
    )

    assert compiled.train_dataset_kind == "graph"
    assert compiled.model_ref == "dymad.models.collections:GLTI"
    assert compiled.profile.key == "lti-graph-default"


def test_compile_training_request_rejects_runtime_owned_override_paths(tmp_path) -> None:
    context = build_default_context(artifact_root=tmp_path / "artifacts")
    dataset_path = tmp_path / "train.npz"
    _write_regular_dataset(dataset_path)
    train_handle = context.facade.register_dataset_file(path=str(dataset_path)).handle

    with pytest.raises(TrainingCompileValidationError) as exc_info:
        compile_training_request(
            facade=context.facade,
            request=TrainingRequest(
                train_dataset_handle=train_handle,
                model_key="kbf",
                overrides={"model": {"name": "not_allowed"}},
            ),
        )

    assert exc_info.value.field_path == ("overrides", "model", "name")


def test_compile_training_request_rejects_unsupported_override_paths(tmp_path) -> None:
    context = build_default_context(artifact_root=tmp_path / "artifacts")
    dataset_path = tmp_path / "train.npz"
    _write_regular_dataset(dataset_path)
    train_handle = context.facade.register_dataset_file(path=str(dataset_path)).handle

    with pytest.raises(TrainingCompileValidationError) as exc_info:
        compile_training_request(
            facade=context.facade,
            request=TrainingRequest(
                train_dataset_handle=train_handle,
                model_key="kbf",
                overrides={"runtime": {"device": "cpu"}},
            ),
        )

    assert exc_info.value.field_path == ("overrides", "runtime")


def test_compile_training_request_rejects_incompatible_explicit_profile(tmp_path) -> None:
    context = build_default_context(artifact_root=tmp_path / "artifacts")
    dataset_path = tmp_path / "train_graph.npz"
    _write_graph_dataset(dataset_path)
    train_handle = context.facade.register_dataset_file(path=str(dataset_path), kind="graph").handle

    with pytest.raises(TrainingCompileValidationError) as exc_info:
        compile_training_request(
            facade=context.facade,
            request=TrainingRequest(
                train_dataset_handle=train_handle,
                model_key="ldm",
                reference_profile="kbf-regular-default",
            ),
        )

    assert exc_info.value.field_path == ("reference_profile",)


def test_compile_training_request_infers_stacked_trainer_kind_from_phases(tmp_path) -> None:
    context = build_default_context(artifact_root=tmp_path / "artifacts")
    dataset_path = tmp_path / "train.npz"
    _write_regular_dataset(dataset_path)
    train_handle = context.facade.register_dataset_file(path=str(dataset_path)).handle

    compiled = compile_training_request(
        facade=context.facade,
        request=TrainingRequest(
            train_dataset_handle=train_handle,
            model_key="kbf",
            run_name="stacked_compile",
            overrides={
                "phases": [
                    {"type": "optimizer", "name": "Warmup", "trainer": "Weak", "n_epochs": 5},
                    {"type": "optimizer", "name": "Refine", "trainer": "NODE", "n_epochs": 7},
                ]
            },
        ),
    )

    assert compiled.trainer_kind == "stacked"
    assert [phase["name"] for phase in compiled.effective_config["phases"]] == [
        "Warmup",
        "Refine",
    ]
