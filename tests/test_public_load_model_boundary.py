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

    def traced_plan(**kwargs):
        events.append("exec.plan")
        return original_plan(**kwargs)

    monkeypatch.setattr(context.executor, "plan_checkpoint_prediction", traced_plan)
    monkeypatch.setattr(
        context.executor,
        "materialize_checkpoint_prediction",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("executor materializer should not run")),
    )

    import dymad.io.checkpoint as checkpoint_module

    def fake_load_model(model_class, path):
        events.append("checkpoint.load_model")
        assert model_class is DummyModel
        assert path == str(checkpoint_path)
        return "model-object", "predict-function"

    monkeypatch.setattr(checkpoint_module, "_load_model_checkpoint", fake_load_model)

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
        "checkpoint.load_model",
    ]


def test_executor_checkpoint_materializer_is_placeholder(tmp_path: Path) -> None:
    context = build_default_context(artifact_root=tmp_path / "artifacts")
    plan = context.executor.plan_checkpoint_prediction(
        model_ref="dymad.models.collections:LDM",
        checkpoint_path="checkpoints/lti.pt",
        horizon=3,
    )

    try:
        context.executor.materialize_checkpoint_prediction(plan=plan, model_class=DummyModel)
    except NotImplementedError as exc:
        assert "dymad.io.load_model" in str(exc)
    else:
        raise AssertionError("CompatibilityExecutor.materialize_checkpoint_prediction should fail")
