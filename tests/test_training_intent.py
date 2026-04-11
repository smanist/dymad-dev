from __future__ import annotations

from pathlib import Path

import numpy as np

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
        ]
    )
    adj = np.array([[0, 1], [1, 0]])
    np.savez_compressed(path, t=t, x=x, adj=adj)


def test_executor_resolves_concise_lti_request_and_validates(tmp_path) -> None:
    dataset_path = tmp_path / "lti.npz"
    _write_regular_dataset(dataset_path)
    context = build_default_context(artifact_root=tmp_path / "artifacts")

    intent = context.executor.resolve_training_intent(
        request_text=(
            "Under this folder is a training dataset lti.npz. Let's learn a discrete-time "
            "LTI model of 2 states with trivial encoder and decoder. "
            "For training use a linear fit first, and then use NODE to refine."
        ),
        cwd=str(tmp_path),
    )

    assert intent.is_valid is True
    assert intent.selected_train_dataset_handle is not None
    assert intent.model_ref == "dymad.models.collections:DLTI"
    assert intent.reference_profile == "lti-regular-default"
    assert intent.config_overrides["model"]["koopman_dimension"] == 2
    assert intent.config_overrides["model"]["encoder_layers"] == 0
    assert intent.config_overrides["model"]["decoder_layers"] == 0
    assert intent.config_overrides["transform_x"]["type"] == "identity"
    assert intent.config_overrides["transform_u"]["type"] == "identity"
    assert intent.phases_override is not None
    assert [phase["trainer"] for phase in intent.phases_override] == ["Linear", "NODE"]

    validation = context.executor.validate_training_config(
        train_dataset_handle=intent.selected_train_dataset_handle,
        model_ref=intent.model_ref,
        reference_profile=intent.reference_profile,
        config=intent.structured_config(),
        run_name=intent.run_name,
    )
    assert validation.is_valid is True
    assert validation.reference_profile == "lti-regular-default"


def test_executor_resolves_graph_family_request(tmp_path) -> None:
    dataset_path = tmp_path / "graph_train.npz"
    _write_graph_dataset(dataset_path)
    context = build_default_context(artifact_root=tmp_path / "artifacts")

    intent = context.executor.resolve_training_intent(
        request_text="Train a discrete graph KBF on graph_train.npz with hidden dimension 16.",
        cwd=str(tmp_path),
    )

    assert intent.is_valid is True
    assert intent.model_ref == "dymad.models.collections:DGKBF"
    assert intent.reference_profile == "kbf-graph-default"
    assert intent.train_dataset_kind == "graph"
    assert intent.config_overrides["model"]["hidden_dimension"] == 16


def test_structured_overrides_beat_prose_for_training_intent(tmp_path) -> None:
    dataset_path = tmp_path / "override_train.npz"
    _write_regular_dataset(dataset_path)
    context = build_default_context(artifact_root=tmp_path / "artifacts")

    intent = context.executor.resolve_training_intent(
        request_text="Train override_train.npz as a discrete-time LTI model of 2 states.",
        cwd=str(tmp_path),
        overrides={
            "model.koopman_dimension": 5,
            "artifact_root": "./custom_artifacts",
            "run_name": "override_case",
        },
    )

    assert intent.is_valid is True
    assert intent.config_overrides["model"]["koopman_dimension"] == 5
    assert intent.artifact_root == "./custom_artifacts"
    assert intent.run_name == "override_case"


def test_training_intent_rejects_reserved_runtime_config_override(tmp_path) -> None:
    dataset_path = tmp_path / "reserved_train.npz"
    _write_regular_dataset(dataset_path)
    context = build_default_context(artifact_root=tmp_path / "artifacts")

    intent = context.executor.resolve_training_intent(
        request_text="Train reserved_train.npz as a discrete-time LTI model.",
        cwd=str(tmp_path),
        overrides={"data.path": "/tmp/forbidden.npz"},
    )

    assert intent.is_valid is False
    assert intent.rejection is not None
    assert intent.rejection.code == "reserved_runtime_path"


def test_training_intent_rejects_ambiguous_training_dataset(tmp_path) -> None:
    _write_regular_dataset(tmp_path / "first.npz")
    _write_regular_dataset(tmp_path / "second.npz")
    context = build_default_context(artifact_root=tmp_path / "artifacts")

    intent = context.executor.resolve_training_intent(
        request_text="Train a discrete-time LTI model with 2 states.",
        cwd=str(tmp_path),
    )

    assert intent.is_valid is False
    assert intent.rejection is not None
    assert intent.rejection.code == "train_dataset_ambiguous"
