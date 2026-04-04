import dymad.models.collections as collections_module
import dymad.models.helpers as helpers_module
from dymad.models import LTI
from dymad.models.model_spec import LegacyPredefinedModelAdapter, ModelSpec


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
