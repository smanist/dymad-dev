from __future__ import annotations

from pathlib import Path

from dymad.exec.context import build_default_context
from dymad.io.load_model_compat import load_model_compat


class DummyModel:
    pass


def test_load_model_compat_routes_via_boundary(monkeypatch, tmp_path: Path) -> None:
    context = build_default_context()
    checkpoint_path = tmp_path / "dummy_model.pt"
    checkpoint_path.write_text("placeholder", encoding="utf-8")

    captured: dict[str, object] = {}

    def fake_load_model(model_class, path):
        captured["model_class"] = model_class
        captured["checkpoint_path"] = path
        return "model-object", "predict-function"

    import dymad.io.checkpoint as checkpoint_module

    monkeypatch.setattr(checkpoint_module, "load_model", fake_load_model)

    model, predict_fn, trace = load_model_compat(
        DummyModel,
        checkpoint_path,
        context=context,
        horizon=9,
        has_control=True,
        return_trace=True,
    )

    assert model == "model-object"
    assert predict_fn == "predict-function"
    assert captured == {
        "model_class": DummyModel,
        "checkpoint_path": str(checkpoint_path),
    }
    assert trace.model_ref.endswith(":DummyModel")

    checkpoint_summary = context.facade.describe_object(trace.plan.checkpoint_handle)
    request = context.facade.get_prediction_request(trace.plan.prediction_handle)
    assert checkpoint_summary.kind == "checkpoint"
    assert request.horizon == 9
    assert request.has_control is True
    assert request.has_graph is False
