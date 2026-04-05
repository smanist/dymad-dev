import pytest

import dymad.models.collections as collections_module
import dymad.models.helpers as helpers_module
from dymad.models.model_spec import ModelSpec, ModelSpecValidationError


def test_predefined_model_routes_via_typed_build_model(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def fake_build_model(model_spec, model_config, data_meta, dtype=None, device=None):
        calls["model_spec"] = model_spec
        calls["model_config"] = model_config
        calls["data_meta"] = data_meta
        return "stub-model"

    monkeypatch.setattr(collections_module, "build_model", fake_build_model)

    result = collections_module.LTI(
        {"name": "lti_model"},
        {"n_total_state_features": 2, "n_total_control_features": 1, "delay": 0},
    )

    assert result == "stub-model"
    assert isinstance(calls["model_spec"], ModelSpec)
    assert calls["model_spec"].recipe.kind == "lfm"
    assert calls["model_spec"].feature.family == "cat"
    assert calls["model_spec"].dynamics.family == "direct"
    assert calls["model_spec"].rollout.family == "lti"
    assert calls["model_spec"].rollout.default_predictor == "continuous"
    assert calls["model_spec"].rollout.supports_control_inputs is True
    assert calls["model_spec"].memory is not None
    assert calls["model_spec"].memory.family == "concat-latent-control"
    assert calls["model_spec"].memory.latent_state == "cat"
    assert calls["model_spec"].memory.requires_delay_window is True


def test_build_model_from_spec_is_typed_alias(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def fake_build_model(model_spec, model_config, data_meta, dtype=None, device=None):
        calls["model_spec"] = model_spec
        calls["model_config"] = model_config
        calls["data_meta"] = data_meta
        return "typed-stub"

    monkeypatch.setattr(helpers_module, "build_model", fake_build_model)

    spec = collections_module.LTI.typed_spec()
    result = helpers_module.build_model_from_spec(spec, {"name": "lti_model"}, {"delay": 0})

    assert result == "typed-stub"
    assert calls["model_spec"] is spec
    assert calls["model_config"] == {"name": "lti_model"}
    assert calls["data_meta"] == {"delay": 0}


def test_build_model_rejects_legacy_tuple_input() -> None:
    with pytest.raises(ModelSpecValidationError):
        helpers_module.build_model(
            [True, "smpl_auto", "cat", "direct", "auto", object],
            {"name": "legacy"},
            {"delay": 0},
        )
