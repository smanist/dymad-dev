from __future__ import annotations

from pathlib import Path

from dymad.agent.exec.context import build_default_context
from dymad.io import load_model


class DummyModel:
    pass


def test_checkpoint_e2e_path_routes_facade_store_exec(monkeypatch, tmp_path: Path) -> None:
    context = build_default_context()
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_text("placeholder", encoding="utf-8")

    events: list[str] = []
    captured: dict[str, object] = {}

    original_plan = context.executor.plan_checkpoint_prediction
    original_register_checkpoint = context.facade.register_checkpoint
    original_prepare_prediction = context.facade.prepare_prediction_request

    def traced_plan(**kwargs):
        events.append("exec.plan")
        return original_plan(**kwargs)

    def traced_register_checkpoint(*, model_ref: str, checkpoint_path: str, device: str = "cpu"):
        events.append("facade.register_checkpoint")
        return original_register_checkpoint(
            model_ref=model_ref,
            checkpoint_path=checkpoint_path,
            device=device,
        )

    def traced_prepare_prediction_request(
        *, checkpoint_handle: str, horizon: int, has_control: bool = False, has_graph: bool = False
    ):
        events.append("facade.prepare_prediction_request")
        return original_prepare_prediction(
            checkpoint_handle=checkpoint_handle,
            horizon=horizon,
            has_control=has_control,
            has_graph=has_graph,
        )

    monkeypatch.setattr(context.executor, "plan_checkpoint_prediction", traced_plan)
    monkeypatch.setattr(
        context.executor,
        "materialize_checkpoint_prediction",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("executor materializer should not run")),
    )
    monkeypatch.setattr(context.facade, "register_checkpoint", traced_register_checkpoint)
    monkeypatch.setattr(
        context.facade, "prepare_prediction_request", traced_prepare_prediction_request
    )

    def fake_load_model(model_class, path):
        events.append("checkpoint.load_model")
        captured["model_class"] = model_class
        captured["checkpoint_path"] = path
        return "model-object", "predict-function"

    import dymad.io.checkpoint as checkpoint_module

    monkeypatch.setattr(checkpoint_module, "_load_model_checkpoint", fake_load_model)

    model, predict_fn, trace = load_model(
        DummyModel,
        checkpoint_path,
        context=context,
        horizon=12,
        has_control=True,
        return_trace=True,
    )

    checkpoint_summary = context.facade.describe_object(trace.plan.checkpoint_handle)
    prediction_summary = context.facade.describe_object(trace.plan.prediction_handle)

    assert model == "model-object"
    assert predict_fn == "predict-function"
    assert trace.plan.entrypoint == "dymad.io.checkpoint.load_model"
    assert checkpoint_summary.kind == "checkpoint"
    assert prediction_summary.kind == "prediction_request"
    assert prediction_summary.derived_from == checkpoint_summary.handle
    assert captured == {
        "model_class": DummyModel,
        "checkpoint_path": str(checkpoint_path),
    }
    assert events == [
        "exec.plan",
        "facade.register_checkpoint",
        "facade.prepare_prediction_request",
        "checkpoint.load_model",
    ]
