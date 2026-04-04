import pytest

import dymad.models.collections as collections_module
import dymad.models.helpers as helpers_module
from dymad.models import LTI
from dymad.models.model_spec import LegacyPredefinedModelAdapter, ModelSpec
from dymad.models.prediction import predict_continuous, predict_discrete


def test_predefined_model_routes_via_typed_model_spec(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def fake_build_model_from_spec(model_spec, model_config, data_meta, dtype=None, device=None):
        calls["model_spec"] = model_spec
        calls["model_config"] = model_config
        calls["data_meta"] = data_meta
        return "stub-model"

    monkeypatch.setattr(collections_module, "build_model_from_spec", fake_build_model_from_spec)

    result = LTI({"name": "lti_model"}, {"n_total_state_features": 2, "n_total_control_features": 1, "delay": 0})

    assert result == "stub-model"
    assert isinstance(calls["model_spec"], ModelSpec)
    assert calls["model_spec"].feature.family == "cat"
    assert calls["model_spec"].dynamics.family == "direct"
    assert calls["model_spec"].rollout is not None
    assert calls["model_spec"].rollout.family == "lti"
    assert calls["model_spec"].rollout.predictor == "continuous"
    assert calls["model_spec"].rollout.supports_control_inputs is True
    assert calls["model_spec"].memory is not None
    assert calls["model_spec"].memory.family == "concat-latent-control"
    assert calls["model_spec"].memory.latent_state == "cat"
    assert calls["model_spec"].memory.requires_delay_window is True


def test_build_model_from_spec_adapts_to_legacy_builder(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def fake_build_model(model_spec, model_config, data_meta, dtype=None, device=None):
        calls["model_spec"] = model_spec
        return "legacy-stub"

    monkeypatch.setattr(helpers_module, "build_model", fake_build_model)

    spec = LegacyPredefinedModelAdapter.from_legacy_parts(
        continuous_time=True,
        encoder="smpl_auto",
        feature="cat",
        dynamics="direct",
        decoder="auto",
        model_cls=object,
        name="LTI",
    )
    result = helpers_module.build_model_from_spec(spec, {"name": "lti_model"}, {"delay": 0})

    assert result == "legacy-stub"
    assert calls["model_spec"] == [True, "smpl_auto", "cat", "direct", "auto", object]


def test_build_model_from_spec_uses_typed_dispatch_for_lti(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def fake_build_model(model_spec, model_config, data_meta, dtype=None, device=None):
        calls["model_spec"] = model_spec
        return "typed-dispatch-stub"

    def fail_to_legacy_tuple(_self):
        raise AssertionError("legacy tuple fallback should not run for LTI typed dispatch")

    monkeypatch.setattr(helpers_module, "build_model", fake_build_model)
    monkeypatch.setattr(ModelSpec, "to_legacy_tuple", fail_to_legacy_tuple)

    result = helpers_module.build_model_from_spec(
        collections_module.LTI.typed_spec(),
        {"name": "lti_model"},
        {"delay": 0},
    )

    assert result == "typed-dispatch-stub"
    assert calls["model_spec"] == [True, "smpl_auto", "cat", "direct", "auto", collections_module.CD_LFM]


@pytest.mark.parametrize(
    ("typed_model", "expected_predictor"),
    [
        (collections_module.LTI, predict_continuous),
        (collections_module.DLTI, predict_discrete),
    ],
)
def test_build_model_from_spec_selects_rollout_engine_from_typed_metadata(
    monkeypatch,
    typed_model,
    expected_predictor,
) -> None:
    class DummyModel:
        _predict = None

    def fake_build_model(model_spec, model_config, data_meta, dtype=None, device=None):
        return DummyModel()

    monkeypatch.setattr(helpers_module, "build_model", fake_build_model)

    model = helpers_module.build_model_from_spec(
        typed_model.typed_spec(),
        {"predictor_type": "exp"},
        {"delay": 0},
    )

    assert model._predict is expected_predictor
