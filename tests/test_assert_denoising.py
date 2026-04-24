import numpy as np
import pytest
import torch
from scipy.signal import savgol_filter

from dymad.numerics import denoise, denoising_metrics


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
