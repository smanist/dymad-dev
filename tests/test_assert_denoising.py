import numpy as np
import pytest
import torch
from scipy.signal import savgol_filter

from dymad.numerics import denoise, denoising_metrics
from dymad.numerics.kernel_smoothing import kernel_smoothing


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


def _kernel_basis(
    time: np.ndarray,
    *,
    kernel: str,
    anchor_count: int,
    bandwidth: float,
    degree: int,
) -> np.ndarray:
    anchors = np.linspace(time[0], time[-1], anchor_count, dtype=np.float64)
    scaled_distance = np.abs(time[:, None] - anchors[None, :]) / bandwidth
    if kernel == "gaussian":
        return np.exp(-0.5 * np.square(scaled_distance))

    support = scaled_distance <= 1.0
    if degree == 0:
        return support.astype(np.float64)

    basis = np.clip(1.0 - scaled_distance, a_min=0.0, a_max=None)
    return np.where(support, basis**degree, 0.0)


def test_kernel_smoothing_matches_global_lstsq_on_flattened_targets():
    time = np.array([0.0, 0.2, 0.45, 0.9, 1.4, 2.0], dtype=np.float64)
    base = np.stack(
        (
            np.sin(time),
            np.cos(time),
        ),
        axis=1,
    )
    data = np.stack((base, base + np.array([0.3, -0.1])), axis=0)

    result = kernel_smoothing(
        data,
        axis=1,
        time=time,
        kernel="gaussian",
        anchor_count=3,
        bandwidth=0.7,
        rcond=1e-10,
    )

    moved = np.moveaxis(data, 1, 0)
    design = _kernel_basis(time, kernel="gaussian", anchor_count=3, bandwidth=0.7, degree=3)
    expected_flat, _, _, _ = np.linalg.lstsq(design, moved.reshape(len(time), -1), rcond=1e-10)
    expected = (design @ expected_flat).reshape(moved.shape)
    expected = np.moveaxis(expected, 0, 1)

    assert result.shape == data.shape
    np.testing.assert_allclose(result, expected)


def test_kernel_smoothing_supports_compact_polynomial_kernel_and_negative_axis():
    time = np.array([0.0, 0.15, 0.5, 1.1, 1.8], dtype=np.float64)
    data = np.array(
        [
            [[0.0, 1.0, 0.5, -0.5, -1.0], [1.0, 0.0, 0.5, 1.0, 0.0]],
            [[0.2, 1.2, 0.7, -0.2, -0.8], [0.8, 0.1, 0.4, 1.1, 0.2]],
        ],
        dtype=np.float64,
    )

    result = kernel_smoothing(
        data,
        axis=-1,
        time=time,
        kernel="compact_polynomial",
        anchor_count=4,
        bandwidth=0.9,
        degree=2,
    )

    design = _kernel_basis(
        time,
        kernel="compact_polynomial",
        anchor_count=4,
        bandwidth=0.9,
        degree=2,
    )
    moved = np.moveaxis(data, -1, 0)
    expected_flat, _, _, _ = np.linalg.lstsq(design, moved.reshape(len(time), -1), rcond=None)
    expected = (design @ expected_flat).reshape(moved.shape)
    expected = np.moveaxis(expected, 0, -1)

    assert result.shape == data.shape
    np.testing.assert_allclose(result, expected)


def test_kernel_smoothing_preserves_compact_support_for_zero_degree():
    time = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    data = np.array([1.0, 999.0, 2.0], dtype=np.float64)

    result = kernel_smoothing(
        data,
        time=time,
        kernel="compact_polynomial",
        anchor_count=2,
        bandwidth=0.75,
        degree=0,
    )

    expected_design = np.array(
        [
            [1.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )
    expected_coefficients, _, _, _ = np.linalg.lstsq(
        expected_design,
        data[:, None],
        rcond=None,
    )
    expected = (expected_design @ expected_coefficients).ravel()

    np.testing.assert_allclose(result, expected)


def test_kernel_smoothing_default_bandwidth_scales_with_time_for_single_anchor():
    time = np.array([0.0, 0.25, 0.6, 1.0, 1.55, 2.1, 2.8, 3.7], dtype=np.float64)
    data = np.stack((np.sin(time), np.cos(time)), axis=1)

    baseline = kernel_smoothing(data, time=time, kernel="gaussian")
    scaled = kernel_smoothing(data, time=time * 100.0, kernel="gaussian")

    np.testing.assert_allclose(scaled, baseline)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"axis": -3}, "out of bounds"),
        ({"anchor_count": 2, "anchor_density": 0.5}, "Specify only one"),
        ({"bandwidth": 1.0, "bandwidth_multiplier": 2.0, "anchor_count": 2}, "Specify only one"),
        ({"kernel": "epanechnikov", "anchor_count": 2, "bandwidth": 1.0}, "Unsupported kernel"),
        ({"anchor_density": 0.0}, "anchor_density"),
        ({"anchor_count": 0}, "anchor_count"),
        ({"degree": -1}, "degree"),
        ({"ridge": -1.0}, "ridge"),
        ({"rcond": -1.0}, "rcond"),
    ],
)
def test_kernel_smoothing_rejects_invalid_hyperparameters(kwargs, match):
    with pytest.raises(ValueError, match=match):
        kernel_smoothing(np.ones((6, 2)), **kwargs)


@pytest.mark.parametrize(
    ("time", "match"),
    [
        (np.array([0.0, 0.5, 1.0]), "shape"),
        (np.array([0.0, 0.5, 0.5, 1.0, 1.5, 2.0]), "strictly increasing"),
        (np.array([0.0, 0.5, np.nan, 1.0, 1.5, 2.0]), "finite"),
    ],
)
def test_kernel_smoothing_rejects_invalid_time(time, match):
    with pytest.raises(ValueError, match=match):
        kernel_smoothing(np.ones((6, 2)), time=time)


def test_kernel_smoothing_rejects_non_finite_input_values():
    data = np.ones((6, 2))
    data[2, 1] = np.inf

    with pytest.raises(ValueError, match="finite input values"):
        kernel_smoothing(data)


def test_kernel_smoothing_rejects_length_one_axis():
    with pytest.raises(ValueError, match="at least two samples"):
        kernel_smoothing(np.ones((1, 3)))
