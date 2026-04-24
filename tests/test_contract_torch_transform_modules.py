from __future__ import annotations

import numpy as np
import pytest
import torch

from dymad.core import (
    AddOneTransform,
    ComposeTransform,
    DelayEmbeddingTransform,
    DenoisingTransform,
    FieldTransformModule,
    FixedGraphSeries,
    GraphSeriesBatch,
    IdentityTransform,
    LiftTransform,
    RegularSeries,
    RegularSeriesBatch,
    ScalerTransform,
    SeriesTransformPipeline,
    build_transform_module,
)
from dymad.numerics import denoise


def _as_torch_batch(arrays: list[np.ndarray]) -> list[torch.Tensor]:
    return [torch.as_tensor(item, dtype=torch.float32) for item in arrays]


def test_torch_compose_transform_matches_legacy_regular_payloads() -> None:
    payloads = [
        np.array([[0.0, 1.0], [1.0, 3.0], [2.0, 5.0], [3.0, 7.0]], dtype=float),
        np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]], dtype=float),
    ]
    reference = build_transform_module(
        [
            {"type": "Scaler", "mode": "01"},
            {"type": "delay", "delay": 1},
            {"type": "add_one"},
        ]
    )
    reference.fit(_as_torch_batch(payloads))
    legacy_out = reference.transform(payloads)

    transform = ComposeTransform(
        [
            ScalerTransform("01"),
            DelayEmbeddingTransform(delay=1),
            AddOneTransform(),
        ]
    )
    transform.fit(_as_torch_batch(payloads))
    torch_out = transform.transform_batch(_as_torch_batch(payloads))

    assert transform.delay == 1
    for source, expected, actual in zip(payloads, legacy_out, torch_out, strict=False):
        torch.testing.assert_close(actual, torch.as_tensor(expected, dtype=actual.dtype))
        recovered = transform.inverse(actual)
        torch.testing.assert_close(
            recovered,
            torch.as_tensor(source, dtype=actual.dtype),
        )


def test_series_transform_pipeline_aligns_regular_fields() -> None:
    batch = RegularSeriesBatch.collate(
        [
            RegularSeries(
                time=torch.arange(5, dtype=torch.float32),
                state=torch.arange(10, dtype=torch.float32).reshape(5, 2),
                control=torch.arange(5, dtype=torch.float32).reshape(5, 1),
            ),
            RegularSeries(
                time=torch.arange(5, dtype=torch.float32),
                state=torch.arange(10, 20, dtype=torch.float32).reshape(5, 2),
                control=torch.arange(5, 10, dtype=torch.float32).reshape(5, 1),
            ),
        ]
    )

    pipeline = SeriesTransformPipeline(
        [
            FieldTransformModule(
                "state",
                ComposeTransform([ScalerTransform("01"), DelayEmbeddingTransform(delay=2)]),
            ),
            FieldTransformModule("control", ScalerTransform("-11")),
        ]
    )
    pipeline.fit(batch)
    transformed = pipeline(batch)

    assert len(transformed) == 2
    assert transformed[0].time.shape[0] == 3
    assert transformed[0].state.shape == (3, 6)
    assert transformed[0].control.shape == (3, 1)
    assert transformed[0].meta["delay"] == 2
    assert transformed[0].meta["field_delays"]["state"] == 2
    assert transformed[0].meta["field_delays"]["control"] == 0


def test_series_transform_pipeline_handles_graph_node_state() -> None:
    batch = GraphSeriesBatch.collate(
        [
            FixedGraphSeries(
                time=torch.arange(4, dtype=torch.float32),
                node_state=torch.arange(24, dtype=torch.float32).reshape(4, 3, 2),
                edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
                edge_weight=torch.arange(8, dtype=torch.float32).reshape(4, 2),
            )
        ]
    )

    pipeline = SeriesTransformPipeline(
        [
            FieldTransformModule("node_state", DelayEmbeddingTransform(delay=1)),
            FieldTransformModule("edge_weight", IdentityTransform()),
        ]
    )
    pipeline.fit(batch)
    transformed = pipeline(batch)

    assert transformed[0].time.shape[0] == 3
    assert transformed[0].node_state.shape == (3, 3, 4)
    assert transformed[0].edge_weight.shape == (3, 2)
    assert transformed[0].meta["delay"] == 1


def test_lift_transform_poly_matches_legacy_behavior() -> None:
    payload = np.array(
        [
            [1.0, 2.0, -0.1],
            [1.1, 3.0, -0.2],
            [1.2, 4.0, -0.3],
            [1.3, 5.0, -0.4],
        ],
        dtype=float,
    )
    reference = build_transform_module({"type": "Lift", "fobs": "poly", "Ks": [3, 2, 4]})
    reference.fit([torch.as_tensor(payload, dtype=torch.float32)])
    expected = reference.transform([payload])[0]

    transform = LiftTransform(fobs="poly", Ks=[3, 2, 4])
    transform.fit([torch.as_tensor(payload, dtype=torch.float32)])
    actual = transform(torch.as_tensor(payload, dtype=torch.float32))

    torch.testing.assert_close(actual, torch.as_tensor(expected, dtype=actual.dtype))
    recovered = transform.inverse(actual)
    torch.testing.assert_close(recovered, torch.as_tensor(payload, dtype=actual.dtype))


def test_lift_transform_mixed_matches_legacy_behavior() -> None:
    payload = np.array(
        [
            [1.0, 0.4, -0.1, 2.0],
            [1.1, 0.3, -0.2, 2.1],
            [1.2, -0.2, -0.3, 2.2],
            [1.3, -0.1, -0.4, 2.3],
        ],
        dtype=float,
    )
    opts = [
        (0, "m", 5),
        (2, "f", 2),
        ([3, 1], "p", [4, 3]),
    ]
    reference = build_transform_module({"type": "Lift", "fobs": "mixed", "opts": opts})
    reference.fit([torch.as_tensor(payload, dtype=torch.float32)])
    expected = reference.transform([payload])[0]

    transform = LiftTransform(fobs="mixed", opts=opts)
    transform.fit([torch.as_tensor(payload, dtype=torch.float32)])
    actual = transform(torch.as_tensor(payload, dtype=torch.float32))

    torch.testing.assert_close(actual, torch.as_tensor(expected, dtype=actual.dtype))
    recovered = transform.inverse(actual)
    torch.testing.assert_close(recovered, torch.as_tensor(payload, dtype=actual.dtype))


def test_denoising_transform_matches_helper_and_inverse_is_identity() -> None:
    payload = torch.tensor(
        [
            [0.0, 0.2],
            [0.5, 0.1],
            [1.0, -0.1],
            [0.5, 0.0],
            [0.0, 0.2],
        ],
        dtype=torch.float64,
    )
    transform = DenoisingTransform(method="savgol", window_length=5, polyorder=2)
    transform.fit([payload])

    expected = denoise(payload, method="savgol", window_length=5, polyorder=2)
    actual = transform(payload)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(transform.inverse(actual), actual)
    assert transform.metadata.supports_gradients == "false"
    assert transform.input_dim == 2
    assert transform.output_dim == 2

    grad_payload = payload.detach().clone().requires_grad_(True)
    assert not transform(grad_payload).requires_grad
    assert not transform.inverse(grad_payload).requires_grad


def test_denoising_stage_disables_gradient_modes_for_compose() -> None:
    payload = torch.tensor(
        [
            [0.0, 1.0],
            [1.0, 3.0],
            [2.0, 5.0],
            [3.0, 7.0],
            [4.0, 9.0],
        ],
        dtype=torch.float32,
    )
    transform = ComposeTransform(
        [
            DenoisingTransform(method="savgol", window_length=5, polyorder=2),
            ScalerTransform("std"),
        ]
    )
    transform.fit([payload])

    assert transform.supports_gradients == "false"

    with pytest.raises(NotImplementedError, match="gradient-dependent operation"):
        transform.get_forward_modes(ref=payload)

    with pytest.raises(NotImplementedError, match="gradient-dependent operation"):
        transform.get_backward_modes(ref=transform(payload))


def test_range_scoped_modes_use_selected_differentiable_stage() -> None:
    payload = torch.tensor(
        [
            [0.0, 1.0],
            [1.0, 3.0],
            [2.0, 5.0],
            [3.0, 7.0],
            [4.0, 9.0],
        ],
        dtype=torch.float64,
    )
    transform = ComposeTransform(
        [
            ScalerTransform("std"),
            DenoisingTransform(method="savgol", window_length=5, polyorder=2),
            ScalerTransform("01"),
        ]
    )
    transform.fit([payload])

    differentiable_stage = transform.transforms[0]
    ref = payload[2]
    scaled_ref = differentiable_stage(ref)

    expected_forward = differentiable_stage.get_forward_modes(ref=ref)
    actual_forward = transform.get_forward_modes(ref=ref, rng=[0, 1])
    np.testing.assert_allclose(actual_forward, expected_forward)

    expected_backward = differentiable_stage.get_backward_modes(ref=scaled_ref)
    actual_backward = transform.get_backward_modes(ref=scaled_ref, rng=[0, 1])
    np.testing.assert_allclose(actual_backward, expected_backward)
