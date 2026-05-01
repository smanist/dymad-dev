import numpy as np
import pytest
import torch
from scipy.signal import savgol_filter

from dymad.numerics import denoise, denoising_metrics
from dymad.numerics.denoising import _build_kernel_design_matrix


def test_denoise_savgol_matches_scipy_for_numpy_arrays():
    data = np.stack(
        (
            np.linspace(0.0, 1.0, 9),
            np.sin(np.linspace(0.0, np.pi, 9)),
        ),
        axis=1,
    )

    result = denoise(data, method="savgol", window_length=5, polyorder=2)

    np.testing.assert_allclose(result, savgol_filter(data, 5, 2, axis=0))


def test_denoise_savgol_preserves_torch_dtype_and_device():
    data = torch.stack(
        (
            torch.linspace(0.0, 1.0, 9, dtype=torch.float32),
            torch.cos(torch.linspace(0.0, 1.0, 9, dtype=torch.float32)),
        ),
        dim=1,
    )

    result = denoise(data, method="savgol", window_length=5, polyorder=2)

    assert isinstance(result, torch.Tensor)
    assert result.dtype is data.dtype
    assert result.device == data.device
    np.testing.assert_allclose(
        result.detach().cpu().numpy(),
        savgol_filter(data.detach().cpu().numpy(), 5, 2, axis=0),
    )


def test_denoise_savgol_rejects_window_longer_than_signal():
    data = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 0.0],
            [3.0, 1.0],
        ]
    )

    with pytest.raises(ValueError, match="window_length"):
        denoise(data, method="savgol", window_length=5, polyorder=2)


def test_denoise_savgol_requires_window_length_and_polyorder():
    with pytest.raises(TypeError, match="window_length"):
        denoise(np.ones((9, 2)), method="savgol", polyorder=2)

    with pytest.raises(TypeError, match="polyorder"):
        denoise(np.ones((9, 2)), method="savgol", window_length=5)


@pytest.mark.parametrize(
    ("kernel", "degree", "expected"),
    [
        (
            "gaussian",
            4.0,
            np.exp(
                -0.5
                * np.square(
                    (np.array([0.0, 0.5, 1.0])[:, None] - np.array([0.0, 1.0])[None, :]) / 0.5
                )
            ),
        ),
        (
            "compact_polynomial",
            2.0,
            np.power(
                np.maximum(
                    1.0
                    - np.square(
                        (np.array([0.0, 0.5, 1.0])[:, None] - np.array([0.0, 1.0])[None, :]) / 0.75
                    ),
                    0.0,
                ),
                2.0,
            ),
        ),
    ],
)
def test_kernel_smoothing_design_matrix_matches_closed_forms(
    kernel: str,
    degree: float,
    expected: np.ndarray,
) -> None:
    time = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    anchors = np.array([0.0, 1.0], dtype=np.float64)
    bandwidth = 0.5 if kernel == "gaussian" else 0.75

    actual = _build_kernel_design_matrix(
        time,
        anchors,
        kernel=kernel,
        bandwidth=bandwidth,
        degree=degree,
    )

    np.testing.assert_allclose(actual, expected)


def test_denoise_kernel_smoothing_preserves_shape_and_handles_negative_axis() -> None:
    base = np.linspace(0.0, 1.0, 7)
    data = np.stack(
        [
            np.stack((base, np.sin(base * np.pi)), axis=1),
            np.stack((base**2, np.cos(base * np.pi)), axis=1),
            np.stack((np.sqrt(base + 1.0), np.sin(base * 2.0 * np.pi)), axis=1),
        ],
        axis=0,
    )

    actual = denoise(
        data,
        method="kernel_smoothing",
        axis=-2,
        kernel="gaussian",
        anchor_count=5,
        bandwidth_multiplier=1.5,
    )
    expected = np.stack(
        [
            denoise(
                trajectory,
                method="kernel_smoothing",
                axis=0,
                kernel="gaussian",
                anchor_count=5,
                bandwidth_multiplier=1.5,
            )
            for trajectory in data
        ],
        axis=0,
    )

    assert actual.shape == data.shape
    np.testing.assert_allclose(actual, expected)


def test_denoise_kernel_smoothing_preserves_torch_dtype_and_device() -> None:
    time = np.linspace(0.0, 1.0, 9, dtype=np.float64)
    data = torch.stack(
        (
            torch.linspace(0.0, 1.0, 9, dtype=torch.float32),
            torch.cos(torch.linspace(0.0, np.pi, 9, dtype=torch.float32)),
        ),
        dim=1,
    )

    result = denoise(
        data,
        method="kernel_smoothing",
        kernel="compact_polynomial",
        anchor_count=6,
        bandwidth_multiplier=2.0,
        degree=4,
        time=time,
    )

    assert isinstance(result, torch.Tensor)
    assert result.dtype is data.dtype
    assert result.device == data.device
    np.testing.assert_allclose(
        result.detach().cpu().numpy(),
        denoise(
            data.detach().cpu().numpy(),
            method="kernel_smoothing",
            kernel="compact_polynomial",
            anchor_count=6,
            bandwidth_multiplier=2.0,
            degree=4,
            time=time,
        ).astype(np.float32),
        atol=1e-6,
        rtol=1e-6,
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {"kernel": "triangle", "anchor_count": 3, "bandwidth_multiplier": 1.0},
            "Unsupported kernel",
        ),
        (
            {"axis": 2, "anchor_count": 3, "bandwidth_multiplier": 1.0},
            "Invalid axis",
        ),
        (
            {"anchor_count": 3, "bandwidth": 0.0},
            "bandwidth must be positive",
        ),
        (
            {"anchor_count": 3, "bandwidth_multiplier": 0.0},
            "bandwidth_multiplier must be positive",
        ),
        (
            {
                "kernel": "compact_polynomial",
                "anchor_count": 3,
                "bandwidth_multiplier": 1.0,
                "degree": 0.0,
            },
            "degree must be positive",
        ),
        (
            {"bandwidth_multiplier": 1.0},
            "anchor_count or anchor_density",
        ),
        (
            {"anchor_count": 3, "bandwidth_multiplier": 1.0, "time": np.ones((5, 1))},
            "time must be a 1D array",
        ),
        (
            {"anchor_count": 3, "bandwidth_multiplier": 1.0, "time": np.ones(4)},
            "same length",
        ),
        (
            {
                "anchor_count": 3,
                "bandwidth_multiplier": 1.0,
                "time": np.array([0.0, 0.2, 0.2, 0.7, 1.0]),
            },
            "strictly increasing",
        ),
    ],
)
def test_denoise_kernel_smoothing_rejects_invalid_configuration(
    kwargs: dict[str, object],
    match: str,
) -> None:
    data = np.ones((5, 2), dtype=np.float64)

    with pytest.raises(ValueError, match=match):
        denoise(data, method="kernel_smoothing", **kwargs)


def test_denoise_kernel_smoothing_rejects_short_and_nonfinite_inputs() -> None:
    with pytest.raises(ValueError, match="at least 2 samples"):
        denoise(
            np.ones((1, 2), dtype=np.float64),
            method="kernel_smoothing",
            anchor_count=1,
            bandwidth_multiplier=1.0,
        )

    with pytest.raises(ValueError, match="finite"):
        denoise(
            np.array([[0.0], [np.nan]], dtype=np.float64),
            method="kernel_smoothing",
            anchor_count=2,
            bandwidth_multiplier=1.0,
        )


def test_denoise_kernel_smoothing_improves_rmse_on_noisy_signal() -> None:
    rng = np.random.default_rng(12)
    time = np.linspace(0.0, 4.0 * np.pi, 128, dtype=np.float64)
    clean = np.stack((np.sin(time), np.cos(0.5 * time)), axis=1)
    noisy = clean + rng.normal(scale=0.35, size=clean.shape)

    smoothed = denoise(
        noisy,
        method="kernel_smoothing",
        kernel="gaussian",
        anchor_count=32,
        bandwidth_multiplier=2.0,
        time=time,
    )

    noisy_rmse = float(np.sqrt(np.mean(np.square(noisy - clean))))
    smoothed_rmse = float(np.sqrt(np.mean(np.square(smoothed - clean))))

    assert smoothed_rmse < noisy_rmse


def test_denoising_metrics_aggregate_multiple_signals():
    original = [
        np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 1.0]]),
        np.array([[1.0, -1.0], [2.0, 0.0], [3.0, 1.0]]),
    ]
    denoised = [
        np.array([[0.0, 0.0], [0.5, 1.5], [1.5, 1.0]]),
        np.array([[1.0, -0.5], [1.5, 0.5], [2.5, 1.5]]),
    ]

    metrics = denoising_metrics(original=original, denoised=denoised)

    deltas = [after - before for before, after in zip(original, denoised, strict=False)]
    n_elements = sum(delta.size for delta in deltas)
    sum_sq_delta = sum(float(np.square(delta).sum()) for delta in deltas)
    sum_abs_delta = sum(float(np.abs(delta).sum()) for delta in deltas)
    max_abs_delta = max(float(np.abs(delta).max(initial=0.0)) for delta in deltas)
    sum_sq_signal = sum(float(np.square(signal).sum()) for signal in original)
    diffs_before = [np.diff(signal, axis=0) for signal in original]
    diffs_after = [np.diff(signal, axis=0) for signal in denoised]
    n_diff_elements = sum(diff.size for diff in diffs_before)
    roughness_before = sum(float(np.square(diff).sum()) for diff in diffs_before) / n_diff_elements
    roughness_after = sum(float(np.square(diff).sum()) for diff in diffs_after) / n_diff_elements
    delta_rmse = float(np.sqrt(sum_sq_delta / n_elements))
    signal_rms = float(np.sqrt(sum_sq_signal / n_elements))

    assert metrics["delta_rmse"] == pytest.approx(delta_rmse)
    assert metrics["delta_mae"] == pytest.approx(sum_abs_delta / n_elements)
    assert metrics["delta_max_abs"] == pytest.approx(max_abs_delta)
    assert metrics["delta_rel_rmse"] == pytest.approx(delta_rmse / signal_rms)
    assert metrics["roughness_before"] == pytest.approx(roughness_before)
    assert metrics["roughness_after"] == pytest.approx(roughness_after)
    assert metrics["roughness_delta"] == pytest.approx(roughness_after - roughness_before)
    assert metrics["roughness_ratio"] == pytest.approx(roughness_after / roughness_before)


def test_denoise_rejects_unknown_method():
    with pytest.raises(ValueError, match="Unsupported denoising method"):
        denoise(np.ones((9, 2)), method="median")
