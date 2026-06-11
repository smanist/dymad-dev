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


def test_compile_training_request_accepts_json_string_overrides(tmp_path) -> None:
    context = build_default_context(artifact_root=tmp_path / "artifacts")
    dataset_path = tmp_path / "train.npz"
    _write_regular_dataset(dataset_path)
    train_handle = context.facade.register_dataset_file(path=str(dataset_path)).handle

    compiled = compile_training_request(
        facade=context.facade,
        request=TrainingRequest(
            train_dataset_handle=train_handle,
            model_key="kbf",
            overrides='{"model": {"koopman_dimension": 8}}',
        ),
    )

    assert compiled.request.overrides == {"model": {"koopman_dimension": 8}}
    assert compiled.effective_config["model"]["koopman_dimension"] == 8


def test_compile_training_request_accepts_cv_sweep_overrides(tmp_path) -> None:
    context = build_default_context(artifact_root=tmp_path / "artifacts")
    dataset_path = tmp_path / "train.npz"
    _write_regular_dataset(dataset_path)
    train_handle = context.facade.register_dataset_file(path=str(dataset_path)).handle

    compiled = compile_training_request(
        facade=context.facade,
        request=TrainingRequest(
            train_dataset_handle=train_handle,
            model_key="kbf",
            overrides={"cv": {"param_grid": {"model.koopman_dimension": [4, 6]}}},
        ),
    )

    assert compiled.request.overrides == {"cv": {"param_grid": {"model.koopman_dimension": [4, 6]}}}
    assert compiled.effective_config["cv"] == {"param_grid": {"model.koopman_dimension": [4, 6]}}


def test_compile_training_request_accepts_json_string_cv_overrides(tmp_path) -> None:
    context = build_default_context(artifact_root=tmp_path / "artifacts")
    dataset_path = tmp_path / "train.npz"
    _write_regular_dataset(dataset_path)
    train_handle = context.facade.register_dataset_file(path=str(dataset_path)).handle

    compiled = compile_training_request(
        facade=context.facade,
        request=TrainingRequest(
            train_dataset_handle=train_handle,
            model_key="kbf",
            overrides='{"cv": {"param_grid": {"model.koopman_dimension": [4, 6]}}}',
        ),
    )

    assert compiled.request.overrides == {"cv": {"param_grid": {"model.koopman_dimension": [4, 6]}}}


def test_compile_training_request_accepts_nelder_mead_like_cv_metadata(tmp_path) -> None:
    context = build_default_context(artifact_root=tmp_path / "artifacts")
    dataset_path = tmp_path / "train.npz"
    _write_regular_dataset(dataset_path)
    train_handle = context.facade.register_dataset_file(path=str(dataset_path)).handle

    compiled = compile_training_request(
        facade=context.facade,
        request=TrainingRequest(
            train_dataset_handle=train_handle,
            model_key="kbf",
            overrides={
                "cv": {
                    "param_grid": {"model.koopman_dimension": [4, 6, 8]},
                    "search": {
                        "mode": "nelder_mead_like",
                        "max_iterations": 8,
                        "reflection": 1.0,
                        "expansion": 2.0,
                        "contraction": 0.5,
                        "shrink": 0.5,
                    },
                    "selection": {
                        "goal": "minimize",
                        "tie_breakers": ["std_metric", "combo_index"],
                    },
                }
            },
        ),
    )

    assert compiled.request.overrides == {
        "cv": {
            "param_grid": {"model.koopman_dimension": [4, 6, 8]},
            "search": {
                "mode": "nelder_mead_like",
                "max_iterations": 8,
                "reflection": 1.0,
                "expansion": 2.0,
                "contraction": 0.5,
                "shrink": 0.5,
            },
            "selection": {
                "goal": "minimize",
                "tie_breakers": ["std_metric", "combo_index"],
            },
        }
    }
    assert compiled.effective_config["cv"]["search"]["mode"] == "nelder_mead_like"
    assert compiled.effective_config["cv"]["selection"]["goal"] == "minimize"


def test_compile_training_request_accepts_explicit_grid_cv_metadata(tmp_path) -> None:
    context = build_default_context(artifact_root=tmp_path / "artifacts")
    dataset_path = tmp_path / "train.npz"
    _write_regular_dataset(dataset_path)
    train_handle = context.facade.register_dataset_file(path=str(dataset_path)).handle

    compiled = compile_training_request(
        facade=context.facade,
        request=TrainingRequest(
            train_dataset_handle=train_handle,
            model_key="kbf",
            overrides={
                "cv": {
                    "param_grid": {"model.koopman_dimension": [4, 6, 8]},
                    "search": {"mode": "grid"},
                }
            },
        ),
    )

    assert compiled.request.overrides == {
        "cv": {
            "param_grid": {"model.koopman_dimension": [4, 6, 8]},
            "search": {"mode": "grid"},
        }
    }
    assert compiled.effective_config["cv"]["param_grid"] == {"model.koopman_dimension": [4, 6, 8]}
    assert compiled.effective_config["cv"]["search"]["mode"] == "grid"


def test_compile_training_request_accepts_bounded_nelder_mead_cv_metadata(tmp_path) -> None:
    context = build_default_context(artifact_root=tmp_path / "artifacts")
    dataset_path = tmp_path / "train.npz"
    _write_regular_dataset(dataset_path)
    train_handle = context.facade.register_dataset_file(path=str(dataset_path)).handle

    compiled = compile_training_request(
        facade=context.facade,
        request=TrainingRequest(
            train_dataset_handle=train_handle,
            model_key="kbf",
            overrides={
                "cv": {
                    "search": {
                        "mode": "nelder_mead_like",
                        "bounds": {"model.koopman_dimension": [4, 8]},
                        "max_iterations": 8,
                    }
                }
            },
        ),
    )

    assert compiled.request.overrides == {
        "cv": {
            "search": {
                "mode": "nelder_mead_like",
                "bounds": {"model.koopman_dimension": [4, 8]},
                "max_iterations": 8,
            }
        }
    }
    assert compiled.effective_config["cv"]["search"]["mode"] == "nelder_mead_like"
    assert compiled.effective_config["cv"]["search"]["bounds"] == {
        "model.koopman_dimension": [4, 8]
    }


def test_compile_training_request_accepts_bounded_batch_pattern_search_cv_metadata(
    tmp_path,
) -> None:
    context = build_default_context(artifact_root=tmp_path / "artifacts")
    dataset_path = tmp_path / "train.npz"
    _write_regular_dataset(dataset_path)
    train_handle = context.facade.register_dataset_file(path=str(dataset_path)).handle

    compiled = compile_training_request(
        facade=context.facade,
        request=TrainingRequest(
            train_dataset_handle=train_handle,
            model_key="kbf",
            overrides={
                "cv": {
                    "search": {
                        "mode": "batch_pattern_search",
                        "bounds": {"model.koopman_dimension": [4, 8]},
                        "max_iterations": 8,
                    }
                }
            },
        ),
    )

    assert compiled.effective_config["cv"]["search"]["mode"] == "batch_pattern_search"
    assert compiled.effective_config["cv"]["search"]["bounds"] == {
        "model.koopman_dimension": [4, 8]
    }


def test_compile_training_request_accepts_bounded_multi_start_nelder_mead_cv_metadata(
    tmp_path,
) -> None:
    context = build_default_context(artifact_root=tmp_path / "artifacts")
    dataset_path = tmp_path / "train.npz"
    _write_regular_dataset(dataset_path)
    train_handle = context.facade.register_dataset_file(path=str(dataset_path)).handle

    compiled = compile_training_request(
        facade=context.facade,
        request=TrainingRequest(
            train_dataset_handle=train_handle,
            model_key="kbf",
            overrides={
                "cv": {
                    "search": {
                        "mode": "multi_start_nelder_mead",
                        "bounds": {"model.koopman_dimension": [4, 8]},
                        "max_iterations": 80,
                    }
                }
            },
            max_workers=4,
        ),
    )

    assert compiled.effective_config["cv"]["search"]["mode"] == "multi_start_nelder_mead"
    assert compiled.effective_config["cv"]["search"]["bounds"] == {
        "model.koopman_dimension": [4, 8]
    }
    assert compiled.request.max_workers == 4


def test_compile_training_request_accepts_parity_constrained_bounded_nelder_mead_cv_metadata(
    tmp_path,
) -> None:
    context = build_default_context(artifact_root=tmp_path / "artifacts")
    dataset_path = tmp_path / "train.npz"
    _write_regular_dataset(dataset_path)
    train_handle = context.facade.register_dataset_file(path=str(dataset_path)).handle

    compiled = compile_training_request(
        facade=context.facade,
        request=TrainingRequest(
            train_dataset_handle=train_handle,
            model_key="kbf",
            overrides={
                "cv": {
                    "search": {
                        "mode": "nelder_mead_like",
                        "bounds": {
                            "model.koopman_dimension": [4, 8],
                            "training.weak_form_params.N": {
                                "lower": 9,
                                "upper": 17,
                                "parity": "odd",
                            },
                        },
                        "max_iterations": 8,
                    }
                }
            },
        ),
    )

    assert compiled.effective_config["cv"]["search"]["bounds"] == {
        "model.koopman_dimension": [4, 8],
        "phases.0.weak_form_params.N": {"lower": 9, "upper": 17, "parity": "odd"},
    }


def test_compile_training_request_normalizes_json_string_cv_range_overrides(tmp_path) -> None:
    context = build_default_context(artifact_root=tmp_path / "artifacts")
    dataset_path = tmp_path / "train.npz"
    _write_regular_dataset(dataset_path)
    train_handle = context.facade.register_dataset_file(path=str(dataset_path)).handle

    compiled = compile_training_request(
        facade=context.facade,
        request=TrainingRequest(
            train_dataset_handle=train_handle,
            model_key="kbf",
            overrides=(
                '{"cv": {"param_grid": {"model.koopman_dimension": ["linspace", [4, 8, 3]]}}}'
            ),
        ),
    )

    assert compiled.request.overrides == {
        "cv": {"param_grid": {"model.koopman_dimension": ("linspace", (4, 8, 3))}}
    }
    assert compiled.effective_config["cv"] == {
        "param_grid": {"model.koopman_dimension": ("linspace", (4, 8, 3))}
    }


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
    assert "data.path" in str(exc_info.value)
    assert "model.name" in str(exc_info.value)


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


def test_compile_training_request_rewrites_legacy_training_param_grid_paths(tmp_path) -> None:
    context = build_default_context(artifact_root=tmp_path / "artifacts")
    dataset_path = tmp_path / "train.npz"
    _write_regular_dataset(dataset_path)
    train_handle = context.facade.register_dataset_file(path=str(dataset_path)).handle

    compiled = compile_training_request(
        facade=context.facade,
        request=TrainingRequest(
            train_dataset_handle=train_handle,
            model_key="kbf",
            overrides={"cv": {"param_grid": {"training.learning_rate": [0.1, 0.2]}}},
        ),
    )

    assert compiled.effective_config["cv"]["param_grid"] == {"phases.0.learning_rate": [0.1, 0.2]}


def test_compile_training_request_rewrites_legacy_training_search_bounds_paths(tmp_path) -> None:
    context = build_default_context(artifact_root=tmp_path / "artifacts")
    dataset_path = tmp_path / "train.npz"
    _write_regular_dataset(dataset_path)
    train_handle = context.facade.register_dataset_file(path=str(dataset_path)).handle

    compiled = compile_training_request(
        facade=context.facade,
        request=TrainingRequest(
            train_dataset_handle=train_handle,
            model_key="kbf",
            overrides={
                "cv": {
                    "search": {
                        "mode": "nelder_mead_like",
                        "bounds": {"training.learning_rate": [0.1, 0.2]},
                    }
                }
            },
        ),
    )

    assert compiled.effective_config["cv"]["search"]["bounds"] == {
        "phases.0.learning_rate": [0.1, 0.2]
    }


def test_compile_training_request_rewrites_legacy_training_search_bounds_paths_with_parity(
    tmp_path,
) -> None:
    context = build_default_context(artifact_root=tmp_path / "artifacts")
    dataset_path = tmp_path / "train.npz"
    _write_regular_dataset(dataset_path)
    train_handle = context.facade.register_dataset_file(path=str(dataset_path)).handle

    compiled = compile_training_request(
        facade=context.facade,
        request=TrainingRequest(
            train_dataset_handle=train_handle,
            model_key="kbf",
            overrides={
                "cv": {
                    "search": {
                        "mode": "nelder_mead_like",
                        "bounds": {
                            "training.weak_form_params.N": {
                                "lower": 9,
                                "upper": 17,
                                "parity": "odd",
                            }
                        },
                    }
                }
            },
        ),
    )

    assert compiled.effective_config["cv"]["search"]["bounds"] == {
        "phases.0.weak_form_params.N": {"lower": 9, "upper": 17, "parity": "odd"}
    }


@pytest.mark.parametrize(
    ("overrides", "expected_field_path"),
    [
        ({"cv": []}, ("overrides", "cv")),
        ({"cv": {"metric": "total"}}, ("overrides", "cv")),
        ({"cv": {"param_grid": {}}}, ("overrides", "cv", "param_grid")),
        (
            {"cv": {"param_grid": {"model.koopman_dimension": [4, 6]}, "metric": 1}},
            ("overrides", "cv", "metric"),
        ),
        (
            {"cv": {"param_grid": {"data.path": ["/tmp/other.npz"]}}},
            ("overrides", "data", "path"),
        ),
        ({"cv": {"param_grid": {"": [4, 6]}}}, ("overrides", "cv", "param_grid")),
        (
            {"cv": {"param_grid": {"model.koopman_dimension": []}}},
            ("overrides", "cv", "param_grid", "model.koopman_dimension"),
        ),
        (
            {"cv": {"param_grid": {"model.koopman_dimension": "bad"}}},
            ("overrides", "cv", "param_grid", "model.koopman_dimension"),
        ),
        (
            {"cv": {"param_grid": {"model.koopman_dimension": ("geomspace", [1, 2, 3])}}},
            ("overrides", "cv", "param_grid", "model.koopman_dimension"),
        ),
        (
            {"cv": {"param_grid": {"model.koopman_dimension": [4, 6]}, "search": []}},
            ("overrides", "cv", "search"),
        ),
        (
            {
                "cv": {
                    "search": {
                        "mode": "nelder_mead_like",
                        "bounds": {"model.koopman_dimension": [8, 4]},
                    }
                }
            },
            ("overrides", "cv", "search", "bounds", "model.koopman_dimension"),
        ),
        (
            {
                "cv": {
                    "search": {
                        "mode": "nelder_mead_like",
                        "bounds": {
                            "model.koopman_dimension": {
                                "lower": 4,
                                "upper": 8,
                                "parity": "prime",
                            }
                        },
                    }
                }
            },
            ("overrides", "cv", "search", "bounds", "model.koopman_dimension", "parity"),
        ),
        (
            {
                "cv": {
                    "param_grid": {"model.koopman_dimension": [4, 6]},
                    "search": {
                        "mode": "nelder_mead_like",
                        "bounds": {"model.koopman_dimension": [4, 8]},
                    },
                }
            },
            ("overrides", "cv"),
        ),
        (
            {"cv": {"search": {"mode": "multi_start_nelder_mead"}}},
            ("overrides", "cv", "search", "bounds"),
        ),
        (
            {
                "cv": {
                    "param_grid": {"model.koopman_dimension": [4, 6]},
                    "selection": {"goal": "downhill"},
                }
            },
            ("overrides", "cv", "selection", "goal"),
        ),
        (
            {
                "cv": {
                    "param_grid": {"model.koopman_dimension": [4, 6]},
                    "selection": {"tie_breakers": ["std_metric", "std_metric"]},
                }
            },
            ("overrides", "cv", "selection", "tie_breakers", "1"),
        ),
    ],
)
def test_compile_training_request_rejects_invalid_cv_overrides(
    tmp_path,
    overrides,
    expected_field_path,
) -> None:
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
                overrides=overrides,
            ),
        )

    assert exc_info.value.field_path == expected_field_path


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


def test_compile_training_request_accepts_minimal_staged_legacy_optimizer_shorthand(
    tmp_path,
) -> None:
    context = build_default_context(artifact_root=tmp_path / "artifacts")
    dataset_path = tmp_path / "train.npz"
    _write_regular_dataset(dataset_path)
    train_handle = context.facade.register_dataset_file(path=str(dataset_path)).handle

    compiled = compile_training_request(
        facade=context.facade,
        request=TrainingRequest(
            train_dataset_handle=train_handle,
            model_key="kbf",
            overrides={
                "phases": [
                    {"trainer": "Linear"},
                    {"trainer": "NODE"},
                ]
            },
        ),
    )

    assert compiled.trainer_kind == "stacked"
    assert [phase["trainer"] for phase in compiled.effective_config["phases"]] == [
        "Linear",
        "NODE",
    ]


@pytest.mark.parametrize(
    ("phase_sequence", "expected_trainers"),
    [
        ([{"trainer": "Linear"}, {"trainer": "Weak"}], ["Linear", "Weak"]),
        ([{"trainer": "OneStep"}, {"trainer": "NODE"}], ["OneStep", "NODE"]),
        ([{"trainer": "Weak"}, {"trainer": "NODE"}], ["Weak", "NODE"]),
    ],
)
def test_compile_training_request_accepts_other_supported_legacy_trainer_sequences(
    tmp_path,
    phase_sequence,
    expected_trainers,
) -> None:
    context = build_default_context(artifact_root=tmp_path / "artifacts")
    dataset_path = tmp_path / "train.npz"
    _write_regular_dataset(dataset_path)
    train_handle = context.facade.register_dataset_file(path=str(dataset_path)).handle

    compiled = compile_training_request(
        facade=context.facade,
        request=TrainingRequest(
            train_dataset_handle=train_handle,
            model_key="kbf",
            overrides={"phases": phase_sequence},
        ),
    )

    assert compiled.trainer_kind == "stacked"
    assert [phase["trainer"] for phase in compiled.effective_config["phases"]] == expected_trainers


def test_compile_training_request_preserves_profile_weak_phase_defaults_in_staged_schedule(
    tmp_path,
) -> None:
    context = build_default_context(artifact_root=tmp_path / "artifacts")
    dataset_path = tmp_path / "train.npz"
    _write_regular_dataset(dataset_path)
    train_handle = context.facade.register_dataset_file(path=str(dataset_path)).handle

    compiled = compile_training_request(
        facade=context.facade,
        request=TrainingRequest(
            train_dataset_handle=train_handle,
            model_key="lti",
            overrides={
                "phases": [
                    {"trainer": "Linear", "name": "initialization"},
                    {"trainer": "Weak", "name": "refinement"},
                ]
            },
        ),
    )

    weak_phase = compiled.effective_config["phases"][1]

    assert compiled.trainer_kind == "stacked"
    assert weak_phase["trainer"] == "Weak"
    assert weak_phase["name"] == "refinement"
    assert weak_phase["n_epochs"] == 25
    assert weak_phase["learning_rate"] == 5e-3
    assert weak_phase["weak_form_params"] == {"N": 13, "dN": 2, "ordpol": 2, "ordint": 2}


def test_compile_training_request_preserves_profile_weak_defaults_unless_overridden(
    tmp_path,
) -> None:
    context = build_default_context(artifact_root=tmp_path / "artifacts")
    dataset_path = tmp_path / "train.npz"
    _write_regular_dataset(dataset_path)
    train_handle = context.facade.register_dataset_file(path=str(dataset_path)).handle

    compiled = compile_training_request(
        facade=context.facade,
        request=TrainingRequest(
            train_dataset_handle=train_handle,
            model_key="lti",
            overrides={
                "phases": [
                    {
                        "type": "optimizer",
                        "name": "refinement",
                        "trainer": "Weak",
                        "n_epochs": 12,
                        "weak_form_params": {"N": 17},
                    }
                ]
            },
        ),
    )

    weak_phase = compiled.effective_config["phases"][0]

    assert compiled.trainer_kind == "weak_form"
    assert weak_phase["trainer"] == "Weak"
    assert weak_phase["name"] == "refinement"
    assert weak_phase["n_epochs"] == 12
    assert weak_phase["weak_form_params"] == {"N": 17, "dN": 2, "ordpol": 2, "ordint": 2}


def test_compile_training_request_rejects_zero_layer_identity_dimension_mismatch(tmp_path) -> None:
    context = build_default_context(artifact_root=tmp_path / "artifacts")
    dataset_path = tmp_path / "train.npz"
    _write_regular_dataset(dataset_path)
    train_handle = context.facade.register_dataset_file(path=str(dataset_path)).handle

    with pytest.raises(TrainingCompileValidationError) as exc_info:
        compile_training_request(
            facade=context.facade,
            request=TrainingRequest(
                train_dataset_handle=train_handle,
                model_key="lti",
                overrides={"model": {"encoder_layers": 0, "decoder_layers": 0}},
            ),
        )

    assert exc_info.value.field_path == ("overrides", "model", "koopman_dimension")
    assert "identity map" in str(exc_info.value)
