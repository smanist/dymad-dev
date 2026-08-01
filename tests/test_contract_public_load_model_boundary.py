from __future__ import annotations

from pathlib import Path

import torch

from dymad.agent.exec.context import build_default_context
from dymad.io import load_model
from dymad.io.checkpoint import _move_model_to_device


class DummyModel:
    pass


class DeviceTrackingChild(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.device = None
        self.projection = torch.nn.Linear(2, 2)


class DeviceTrackingModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.device = None
        self.child = DeviceTrackingChild()


def test_move_model_to_device_synchronizes_explicit_device_metadata() -> None:
    model = DeviceTrackingModule()
    target = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    moved = _move_model_to_device(model, target)

    assert moved is model
    assert model.device == target
    assert model.child.device == target
    assert model.child.projection.weight.device == target


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
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("executor materializer should not run")
        ),
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
