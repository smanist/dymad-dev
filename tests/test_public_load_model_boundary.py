from __future__ import annotations

from pathlib import Path

from dymad.exec.context import build_default_context
from dymad.io import load_model


class DummyModel:
    pass


def test_public_load_model_routes_via_boundary(monkeypatch, tmp_path: Path) -> None:
    context = build_default_context()
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_text("placeholder", encoding="utf-8")

    events: list[str] = []

    original_plan = context.executor.plan_checkpoint_prediction
    original_materialize = context.executor.materialize_checkpoint_prediction

    def traced_plan(**kwargs):
        events.append("exec.plan")
        return original_plan(**kwargs)

    def traced_materialize(**kwargs):
        events.append("exec.materialize")
        return original_materialize(**kwargs)

    monkeypatch.setattr(context.executor, "plan_checkpoint_prediction", traced_plan)
    monkeypatch.setattr(context.executor, "materialize_checkpoint_prediction", traced_materialize)

    import dymad.io.checkpoint as checkpoint_module

    def fake_load_model(model_class, path):
        events.append("legacy.load_model")
        assert model_class is DummyModel
        assert path == str(checkpoint_path)
        return "model-object", "predict-function"

    monkeypatch.setattr(checkpoint_module, "_load_model_legacy", fake_load_model)

    model, predict_fn, trace = load_model(
        DummyModel,
        checkpoint_path,
        context=context,
        return_trace=True,
    )

    assert model == "model-object"
    assert predict_fn == "predict-function"
    assert trace.plan.entrypoint == "dymad.io.checkpoint.load_model"
    assert events == [
        "exec.plan",
        "exec.materialize",
        "legacy.load_model",
    ]
