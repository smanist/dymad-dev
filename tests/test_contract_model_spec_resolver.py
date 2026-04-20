import pytest

import dymad.models.collections as collections_module
from dymad.models.model_spec import ModelSpecValidationError
from dymad.models.prediction import (
    predict_continuous_fenc,
    predict_continuous_np,
    predict_discrete_exp,
)
from dymad.models.recipes import resolve_recipe
from dymad.models.rollout_engine import select_rollout_engine


@pytest.mark.parametrize(
    ("typed_model", "model_config", "data_meta", "expected"),
    [
        (
            collections_module.LDM,
            {},
            {"n_total_state_features": 2, "n_total_control_features": 1, "delay": 0},
            {
                "encoder_key": "smpl_ctrl",
                "feature_key": "none",
                "decoder_key": "auto",
                "graph_mode": "none",
            },
        ),
        (
            collections_module.GLDM,
            {},
            {"n_total_state_features": 2, "n_total_control_features": 1, "delay": 1},
            {
                "encoder_key": "graph_ctrl",
                "feature_key": "none",
                "decoder_key": "graph",
                "graph_mode": "graph",
            },
        ),
        (
            collections_module.DSDMG,
            {"processor_layers": 1},
            {"n_total_state_features": 2, "n_total_control_features": 1, "delay": 1},
            {
                "encoder_key": "node_raw_ctrl",
                "feature_key": "none",
                "decoder_key": "node",
                "graph_mode": "node",
            },
        ),
        (
            collections_module.KBF,
            {"const_term": True, "koopman_dimension": 4},
            {"n_total_state_features": 2, "n_total_control_features": 1, "delay": 0},
            {
                "encoder_key": "smpl_auto",
                "feature_key": "blin_with_const",
                "decoder_key": "auto",
                "graph_mode": "none",
            },
        ),
        (
            collections_module.GLTI,
            {"const_term": True, "koopman_dimension": 4},
            {"n_total_state_features": 2, "n_total_control_features": 1, "delay": 0},
            {
                "encoder_key": "graph_auto",
                "feature_key": "graph_cat",
                "decoder_key": "graph",
                "graph_mode": "graph",
            },
        ),
    ],
)
def test_typed_recipe_resolution_covers_model_families(
    typed_model, model_config, data_meta, expected
) -> None:
    resolved = resolve_recipe(
        typed_model.typed_spec(),
        model_config,
        data_meta,
        dtype=None,
        device=None,
    )

    assert resolved.encoder_key == expected["encoder_key"]
    assert resolved.feature_key == expected["feature_key"]
    assert resolved.decoder_key == expected["decoder_key"]
    assert typed_model.typed_spec().graph_mode == expected["graph_mode"]


def test_rollout_engine_uses_kmm_fenc_path() -> None:
    spec = collections_module.KMM.typed_spec()
    resolved = resolve_recipe(
        spec,
        {
            "kernel_dimension": 4,
            "type": "share",
            "kernel": {
                "type": "sc_rbf",
                "input_dim": 2,
                "lengthscale_init": 1.0,
            },
        },
        {"n_total_state_features": 2, "n_total_control_features": 0, "delay": 0},
        dtype=None,
        device=None,
    )

    engine = select_rollout_engine(spec, {"predictor_type": "ode"}, resolved.dims)

    assert engine.predictor is predict_continuous_fenc


def test_rollout_engine_supports_np_override_for_continuous_models() -> None:
    spec = collections_module.LTI.typed_spec()
    resolved = resolve_recipe(
        spec,
        {"koopman_dimension": 4},
        {"n_total_state_features": 2, "n_total_control_features": 0, "delay": 0},
        dtype=None,
        device=None,
    )

    engine = select_rollout_engine(spec, {"predictor_type": "np"}, resolved.dims)

    assert engine.predictor is predict_continuous_np


def test_rollout_engine_maps_discrete_np_to_discrete_exp() -> None:
    spec = collections_module.DKBF.typed_spec()
    resolved = resolve_recipe(
        spec,
        {"koopman_dimension": 4},
        {"n_total_state_features": 2, "n_total_control_features": 0, "delay": 0},
        dtype=None,
        device=None,
    )

    engine = select_rollout_engine(spec, {"predictor_type": "np"}, resolved.dims)

    assert engine.predictor is predict_discrete_exp


def test_rollout_engine_rejects_exp_override_with_control_inputs() -> None:
    spec = collections_module.LTI.typed_spec()
    resolved = resolve_recipe(
        spec,
        {"koopman_dimension": 4},
        {"n_total_state_features": 2, "n_total_control_features": 1, "delay": 0},
        dtype=None,
        device=None,
    )

    with pytest.raises(ModelSpecValidationError):
        select_rollout_engine(spec, {"predictor_type": "exp"}, resolved.dims)
