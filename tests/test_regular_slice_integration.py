from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from dymad.core import GraphModelContext, RegularModelContext
from dymad.core.transform_module import SeriesTransformPipeline
from dymad.io import load_model
from dymad.io.trajectory_manager import TrajectoryManager
from dymad.transform import make_transform


class DummyPredictModel:
    def __init__(self, config, md, dtype=torch.float32):
        self.dtype = dtype
        self.config = config
        self.md = md
        self.state_dict_loaded = None
        self.predict_calls = []
        self.predict_times = []

    def load_state_dict(self, state_dict):
        self.state_dict_loaded = state_dict

    def predict(self, x0, data, t, **kwargs):
        self.predict_calls.append(kwargs)
        self.predict_times.append(t.detach().clone())
        steps = t.shape[-1]
        if x0.ndim == 1:
            return x0.unsqueeze(0).repeat(steps, 1)
        return x0.unsqueeze(1).repeat(1, steps, 1)


def _build_checkpoint_payload():
    x_fit = [
        np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]], dtype=float),
        np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], dtype=float),
    ]
    u_fit = [
        np.array([[0.0], [1.0], [2.0]], dtype=float),
        np.array([[1.0], [2.0], [3.0]], dtype=float),
    ]
    transform_x = make_transform({"type": "Scaler", "mode": "01"})
    transform_u = make_transform({"type": "Scaler", "mode": "-11"})
    transform_x.fit(x_fit)
    transform_u.fit(u_fit)
    return {
        "config": {
            "data": {"double_precision": False},
            "transform_x": {"type": "Scaler", "mode": "01"},
            "transform_u": {"type": "Scaler", "mode": "-11"},
        },
        "train_md": {
            "transform_x_state": transform_x.state_dict(),
            "transform_u_state": transform_u.state_dict(),
        },
        "model_state_dict": {"dummy": torch.tensor(1.0)},
    }


def test_regular_checkpoint_prediction_uses_typed_series(monkeypatch, tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "dummy.pt"
    checkpoint_path.write_text("placeholder", encoding="utf-8")

    import dymad.io.checkpoint as checkpoint_module

    payload = _build_checkpoint_payload()
    apply_events: list[int] = []
    original_forward = SeriesTransformPipeline.forward

    def traced_forward(self, batch):
        apply_events.append(len(batch))
        return original_forward(self, batch)

    monkeypatch.setattr(checkpoint_module.torch, "load", lambda *args, **kwargs: payload)
    monkeypatch.setattr(
        checkpoint_module.SeriesTransformPipeline,
        "forward",
        traced_forward,
    )

    _, predict_fn = load_model(DummyPredictModel, checkpoint_path)
    x0 = np.array([[1.0, 3.0], [2.0, 4.0]], dtype=float)
    u = np.array([[0.2], [0.6]], dtype=float)
    t = np.array([0.0, 1.0, 2.0], dtype=float)
    prediction = predict_fn(x0, t, u=u)

    assert apply_events == [1]
    assert prediction.shape == (3, 2)


def test_regular_checkpoint_prediction_routes_through_model_context(
    monkeypatch, tmp_path: Path
) -> None:
    checkpoint_path = tmp_path / "dummy.pt"
    checkpoint_path.write_text("placeholder", encoding="utf-8")

    import dymad.io.checkpoint as checkpoint_module

    payload = _build_checkpoint_payload()
    captured: dict[str, object] = {}
    original_build_model_context = checkpoint_module.build_model_context

    def traced_build_model_context(batch):
        context = original_build_model_context(batch)
        captured["context_type"] = type(context)
        captured["initial_state"] = context.initial_state_tensor(squeeze_single=True).clone()
        captured["runtime"] = context.to_runtime()
        return context

    monkeypatch.setattr(checkpoint_module.torch, "load", lambda *args, **kwargs: payload)
    monkeypatch.setattr(checkpoint_module, "build_model_context", traced_build_model_context)

    _, predict_fn = load_model(DummyPredictModel, checkpoint_path)
    x0 = np.array([[1.0, 3.0], [2.0, 4.0]], dtype=float)
    u = np.array([[0.2], [0.6]], dtype=float)
    t = np.array([0.0, 1.0, 2.0], dtype=float)
    prediction = predict_fn(x0, t, u=u)

    assert captured["context_type"] is RegularModelContext
    runtime = captured["runtime"]
    assert torch.equal(runtime.x[:, 0, :], captured["initial_state"].unsqueeze(0))
    assert runtime.batch_size == 1
    assert runtime.n_steps == 2
    assert prediction.shape == (3, 2)


def test_graph_checkpoint_prediction_routes_through_model_context(
    monkeypatch, tmp_path: Path
) -> None:
    checkpoint_path = tmp_path / "dummy.pt"
    checkpoint_path.write_text("placeholder", encoding="utf-8")

    import dymad.io.checkpoint as checkpoint_module

    payload = _build_checkpoint_payload()
    captured: dict[str, object] = {}
    original_build_model_context = checkpoint_module.build_model_context

    def traced_build_model_context(batch):
        context = original_build_model_context(batch)
        captured["context_type"] = type(context)
        captured["initial_state"] = context.initial_state_tensor(squeeze_single=True).clone()
        captured["runtime"] = context.to_runtime()
        return context

    monkeypatch.setattr(checkpoint_module.torch, "load", lambda *args, **kwargs: payload)
    monkeypatch.setattr(checkpoint_module, "build_model_context", traced_build_model_context)

    _, predict_fn = load_model(DummyPredictModel, checkpoint_path)
    x0 = np.array([[1.0, 3.0, 5.0, 7.0], [2.0, 4.0, 6.0, 8.0]], dtype=float)
    u = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=float)
    t = np.array([0.0, 1.0], dtype=float)
    edge_index = np.array([[0, 1], [1, 0]], dtype=int)
    prediction = predict_fn(x0, t, u=u, ei=edge_index)

    assert captured["context_type"] is GraphModelContext
    runtime = captured["runtime"]
    assert runtime.is_graph
    assert runtime.batch_size == 1
    assert runtime.n_nodes == 2
    assert torch.equal(runtime.x[:, 0, :], captured["initial_state"].unsqueeze(0))
    assert prediction.shape == (2, 4)


def test_regular_slice_integration_touches_typed_transform_seam(
    monkeypatch, tmp_path: Path
) -> None:
    data_path = tmp_path / "regular_slice.npz"
    t = np.stack(
        [
            np.linspace(0.0, 1.0, 6),
            np.linspace(0.0, 1.0, 6),
        ]
    )
    x = np.stack(
        [
            np.column_stack((np.linspace(0.0, 1.0, 6), np.linspace(1.0, 2.0, 6))),
            np.column_stack((np.linspace(2.0, 3.0, 6), np.linspace(3.0, 4.0, 6))),
        ]
    )
    u = np.stack(
        [
            np.linspace(0.0, 0.5, 6).reshape(-1, 1),
            np.linspace(0.5, 1.0, 6).reshape(-1, 1),
        ]
    )
    np.savez(data_path, t=t, x=x, u=u)

    checkpoint_path = tmp_path / "dummy.pt"
    checkpoint_path.write_text("placeholder", encoding="utf-8")

    import dymad.io.checkpoint as checkpoint_module

    payload = _build_checkpoint_payload()
    apply_events: list[int] = []
    original_forward = SeriesTransformPipeline.forward

    def traced_forward(self, batch):
        apply_events.append(len(batch))
        return original_forward(self, batch)

    monkeypatch.setattr(checkpoint_module.torch, "load", lambda *args, **kwargs: payload)
    monkeypatch.setattr(
        checkpoint_module.SeriesTransformPipeline,
        "forward",
        traced_forward,
    )

    manager = TrajectoryManager(
        metadata={
            "data_key": "data",
            "config": {
                "data": {"path": str(data_path)},
                "transform_x": [
                    {"type": "Scaler", "mode": "01"},
                    {"type": "delay", "delay": 2},
                ],
                "transform_u": {"type": "Scaler", "mode": "-11"},
            },
        }
    )
    manager.prepare_data()
    manager.set_data_index([0, 1])
    manager.apply_data_transformations()

    _, predict_fn = load_model(DummyPredictModel, checkpoint_path)
    prediction = predict_fn(x[0], t[0], u=u[0])

    assert len(manager.dataset) == 2
    assert apply_events == [2, 1]
    assert prediction.shape == (6, 2)


def test_checkpoint_prediction_uses_saved_ode_defaults(monkeypatch, tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "dummy.pt"
    checkpoint_path.write_text("placeholder", encoding="utf-8")

    import dymad.io.checkpoint as checkpoint_module

    payload = _build_checkpoint_payload()
    payload["config"]["phases"] = [
        {
            "name": "NODE",
            "trainer": "NODE",
            "ode_method": "rk4",
            "ode_args": {"step_size": 0.05},
        }
    ]

    monkeypatch.setattr(checkpoint_module.torch, "load", lambda *args, **kwargs: payload)

    model, predict_fn = load_model(DummyPredictModel, checkpoint_path)
    x0 = np.array([[1.0, 3.0], [2.0, 4.0]], dtype=float)
    u = np.array([[0.2], [0.6]], dtype=float)
    t = np.array([0.0, 1.0, 2.0], dtype=float)

    prediction = predict_fn(x0, t, u=u)

    assert prediction.shape == (3, 2)
    assert model.predict_calls[-1]["method"] == "rk4"
    assert model.predict_calls[-1]["step_size"] == 0.05


def test_checkpoint_prediction_explicit_kwargs_override_saved_defaults(
    monkeypatch, tmp_path: Path
) -> None:
    checkpoint_path = tmp_path / "dummy.pt"
    checkpoint_path.write_text("placeholder", encoding="utf-8")

    import dymad.io.checkpoint as checkpoint_module

    payload = _build_checkpoint_payload()
    payload["config"]["training"] = {
        "ode_method": "rk4",
        "ode_args": {"step_size": 0.05},
    }

    monkeypatch.setattr(checkpoint_module.torch, "load", lambda *args, **kwargs: payload)

    model, predict_fn = load_model(DummyPredictModel, checkpoint_path)
    x0 = np.array([[1.0, 3.0], [2.0, 4.0]], dtype=float)
    u = np.array([[0.2], [0.6]], dtype=float)
    t = np.array([0.0, 1.0, 2.0], dtype=float)

    prediction = predict_fn(x0, t, u=u, method="dopri5", step_size=0.1)

    assert prediction.shape == (3, 2)
    assert model.predict_calls[-1]["method"] == "dopri5"
    assert model.predict_calls[-1]["step_size"] == 0.1


def test_checkpoint_prediction_aligns_full_time_for_delayed_runtime(
    monkeypatch, tmp_path: Path
) -> None:
    checkpoint_path = tmp_path / "dummy.pt"
    checkpoint_path.write_text("placeholder", encoding="utf-8")

    import dymad.io.checkpoint as checkpoint_module

    payload = _build_checkpoint_payload()
    delayed = make_transform({"type": "delay", "delay": 1})
    delayed.fit(
        [
            np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]], dtype=float),
            np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], dtype=float),
        ]
    )
    payload["config"]["transform_x"] = {"type": "delay", "delay": 1}
    payload["train_md"]["transform_x_state"] = delayed.state_dict()

    monkeypatch.setattr(checkpoint_module.torch, "load", lambda *args, **kwargs: payload)

    model, predict_fn = load_model(DummyPredictModel, checkpoint_path)
    x0 = np.array([[1.0, 3.0], [2.0, 4.0], [3.0, 5.0]], dtype=float)
    u = np.array([[0.2], [0.6], [0.8]], dtype=float)
    t = np.array([0.0, 1.0, 2.0], dtype=float)

    prediction = predict_fn(x0, t, u=u)

    assert prediction.shape == (3, 2)
    assert model.predict_times[-1].shape[0] == 2
    assert torch.allclose(
        model.predict_times[-1],
        torch.tensor([1.0, 2.0], dtype=model.predict_times[-1].dtype),
    )


def test_graph_checkpoint_prediction_aligns_full_time_for_delayed_runtime(
    monkeypatch, tmp_path: Path
) -> None:
    checkpoint_path = tmp_path / "dummy.pt"
    checkpoint_path.write_text("placeholder", encoding="utf-8")

    import dymad.io.checkpoint as checkpoint_module

    payload = _build_checkpoint_payload()
    delayed = make_transform({"type": "delay", "delay": 1})
    delayed.fit(
        [
            np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]], dtype=float),
            np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], dtype=float),
        ]
    )
    payload["config"]["transform_x"] = {"type": "delay", "delay": 1}
    payload["train_md"]["transform_x_state"] = delayed.state_dict()

    monkeypatch.setattr(checkpoint_module.torch, "load", lambda *args, **kwargs: payload)

    model, predict_fn = load_model(DummyPredictModel, checkpoint_path)
    x0 = np.array([[1.0, 3.0, 5.0, 7.0], [2.0, 4.0, 6.0, 8.0], [3.0, 5.0, 7.0, 9.0]], dtype=float)
    u = np.array([[0.2, 0.4], [0.6, 0.8], [0.9, 1.1]], dtype=float)
    t = np.array([0.0, 1.0, 2.0], dtype=float)
    edge_index = (
        np.array([[0, 1], [1, 0]], dtype=int),
        np.array([[0, 1], [1, 0]], dtype=int),
        np.array([[0, 1], [1, 0]], dtype=int),
    )
    edge_weight = np.array(
        [
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=float,
    )

    prediction = predict_fn(x0, t, u=u, ei=edge_index, ew=edge_weight)

    assert prediction.shape == (3, 4)
    assert model.predict_times[-1].shape[0] == 2
    assert torch.allclose(
        model.predict_times[-1],
        torch.tensor([1.0, 2.0], dtype=model.predict_times[-1].dtype),
    )
