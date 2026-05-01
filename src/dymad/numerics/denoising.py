from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, SupportsFloat, TypeAlias, overload

import numpy as np
import torch
from scipy.signal import savgol_filter

ArrayLike: TypeAlias = np.ndarray | torch.Tensor
DenoiseMethod = Literal["savgol", "kernel_smoothing"]
KernelName = Literal["gaussian", "compact_polynomial"]


def _denoise_savgol(data: ArrayLike, *, axis: int, **kwargs) -> ArrayLike:
    array = _to_numpy(data)
    smoothed = savgol_filter(array, axis=axis, **kwargs)
    return _from_numpy(np.asarray(smoothed), like=data)


def _normalize_axis(axis: int, ndim: int) -> int:
    if not -ndim <= axis < ndim:
        raise ValueError(f"Invalid axis {axis} for array with {ndim} dimensions.")
    return axis % ndim


def _from_numpy(array: np.ndarray, *, like: ArrayLike) -> ArrayLike:
    if isinstance(like, torch.Tensor):
        return torch.as_tensor(array, device=like.device, dtype=like.dtype)
    return np.asarray(array, dtype=like.dtype)


def _validate_finite(name: str, array: np.ndarray) -> None:
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")


def _coerce_positive_scalar(
    name: str,
    value: SupportsFloat,
    *,
    integer: bool = False,
    allow_zero: bool = False,
) -> int | float:
    scalar = float(value)
    lower_bound = 0.0 if allow_zero else 0.0
    if not np.isfinite(scalar) or scalar < lower_bound or (not allow_zero and scalar == 0.0):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}.")
    if integer:
        if not scalar.is_integer():
            raise ValueError(f"{name} must be an integer.")
        return int(scalar)
    return scalar


def _resolve_time(time: Sequence[float] | np.ndarray | None, *, n_samples: int) -> np.ndarray:
    if time is None:
        return np.linspace(0.0, 1.0, n_samples, dtype=np.float64)

    grid = np.asarray(time, dtype=np.float64)
    if grid.ndim != 1:
        raise ValueError("time must be a 1D array.")
    if grid.shape[0] != n_samples:
        raise ValueError("time must have the same length as the denoising axis.")
    _validate_finite("time", grid)
    if np.any(np.diff(grid) <= 0.0):
        raise ValueError("time must be strictly increasing.")
    return grid


def _resolve_anchor_count(
    time: np.ndarray,
    *,
    anchor_count: int | float | None,
    anchor_density: float | None,
) -> int:
    if anchor_count is not None:
        count = int(_coerce_positive_scalar("anchor_count", anchor_count, integer=True))
    elif anchor_density is not None:
        density = float(_coerce_positive_scalar("anchor_density", anchor_density))
        span = float(time[-1] - time[0])
        count = int(np.ceil(density * span))
    else:
        raise ValueError(
            "kernel_smoothing requires anchor_count or anchor_density to produce at least one anchor."
        )

    if count < 1:
        raise ValueError("kernel_smoothing anchor configuration must produce at least one anchor.")
    return count


def _resolve_bandwidth(
    time: np.ndarray,
    *,
    anchor_count: int,
    bandwidth: float | None,
    bandwidth_multiplier: float | None,
) -> float:
    if bandwidth is not None:
        return float(_coerce_positive_scalar("bandwidth", bandwidth))

    multiplier = 2.0 if bandwidth_multiplier is None else bandwidth_multiplier
    multiplier = float(_coerce_positive_scalar("bandwidth_multiplier", multiplier))
    span = float(time[-1] - time[0])
    spacing = span if anchor_count == 1 else span / float(anchor_count - 1)
    return multiplier * spacing


def _build_kernel_design_matrix(
    time: np.ndarray,
    anchors: np.ndarray,
    *,
    kernel: KernelName | str,
    bandwidth: float,
    degree: float = 4.0,
) -> np.ndarray:
    kernel_name = str(kernel)
    degree_value = float(_coerce_positive_scalar("degree", degree))
    bandwidth_value = float(_coerce_positive_scalar("bandwidth", bandwidth))
    scaled = (time[:, None] - anchors[None, :]) / bandwidth_value

    if kernel_name == "gaussian":
        return np.exp(-0.5 * np.square(scaled))
    if kernel_name == "compact_polynomial":
        return np.power(np.maximum(1.0 - np.square(scaled), 0.0), degree_value)
    raise ValueError(f"Unsupported kernel '{kernel_name}'.")


def _solve_least_squares(
    design: np.ndarray,
    response: np.ndarray,
    *,
    ridge: float = 0.0,
    rcond: float | None = None,
) -> np.ndarray:
    ridge_value = float(_coerce_positive_scalar("ridge", ridge, allow_zero=True))
    effective_rcond = None
    if rcond is not None:
        effective_rcond = float(_coerce_positive_scalar("rcond", rcond, allow_zero=True))

    if ridge_value > 0.0:
        regularizer = np.sqrt(ridge_value) * np.eye(design.shape[1], dtype=np.float64)
        system_matrix = np.vstack((design, regularizer))
        system_rhs = np.vstack((response, np.zeros((design.shape[1], response.shape[1]))))
    else:
        system_matrix = design
        system_rhs = response

    try:
        coefficients, _residuals, rank, _singular_values = np.linalg.lstsq(
            system_matrix,
            system_rhs,
            rcond=effective_rcond,
        )
        if rank == system_matrix.shape[1]:
            return coefficients
    except np.linalg.LinAlgError:
        pass

    pinv_rcond = 1e-15 if effective_rcond is None else effective_rcond
    return np.linalg.pinv(system_matrix, rcond=pinv_rcond) @ system_rhs


def _denoise_kernel_smoothing(
    data: ArrayLike,
    *,
    axis: int,
    kernel: KernelName | str = "gaussian",
    anchor_count: int | float | None = None,
    anchor_density: float | None = None,
    bandwidth: float | None = None,
    bandwidth_multiplier: float | None = None,
    degree: float = 4.0,
    ridge: float = 0.0,
    rcond: float | None = None,
    time: Sequence[float] | np.ndarray | None = None,
) -> ArrayLike:
    original = _to_numpy(data)
    normalized_axis = _normalize_axis(axis, original.ndim)
    leading = np.moveaxis(original, normalized_axis, 0)
    n_samples = int(leading.shape[0])
    if n_samples < 2:
        raise ValueError("kernel_smoothing requires at least 2 samples along the denoising axis.")

    flat = leading.reshape(n_samples, -1).astype(np.float64, copy=False)
    _validate_finite("data", flat)

    sample_time = _resolve_time(time, n_samples=n_samples)
    count = _resolve_anchor_count(
        sample_time,
        anchor_count=anchor_count,
        anchor_density=anchor_density,
    )
    anchors = np.linspace(sample_time[0], sample_time[-1], count, dtype=np.float64)
    kernel_bandwidth = _resolve_bandwidth(
        sample_time,
        anchor_count=count,
        bandwidth=bandwidth,
        bandwidth_multiplier=bandwidth_multiplier,
    )
    design = _build_kernel_design_matrix(
        sample_time,
        anchors,
        kernel=kernel,
        bandwidth=kernel_bandwidth,
        degree=degree,
    )
    coefficients = _solve_least_squares(design, flat, ridge=ridge, rcond=rcond)
    smoothed = (design @ coefficients).reshape(leading.shape)
    restored = np.moveaxis(smoothed, 0, normalized_axis)
    return _from_numpy(restored, like=data)


def _to_numpy(data: ArrayLike) -> np.ndarray:
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.asarray(data)


@overload
def denoise(data: np.ndarray, *, method: str = "savgol", axis: int = 0, **kwargs) -> np.ndarray: ...


@overload
def denoise(
    data: torch.Tensor, *, method: str = "savgol", axis: int = 0, **kwargs
) -> torch.Tensor: ...


def denoise(data: ArrayLike, *, method: str = "savgol", axis: int = 0, **kwargs) -> ArrayLike:
    """Apply a model-independent denoising method while preserving the input array type."""
    if method == "savgol":
        return _denoise_savgol(data, axis=axis, **kwargs)
    if method == "kernel_smoothing":
        return _denoise_kernel_smoothing(data, axis=axis, **kwargs)
    raise ValueError(f"Unsupported denoising method '{method}'.")


def denoising_metrics(
    *,
    original: Sequence[ArrayLike],
    denoised: Sequence[ArrayLike],
) -> dict[str, float]:
    """Aggregate denoising deltas and roughness statistics across one or more signals."""
    if len(original) != len(denoised):
        raise ValueError("Denoising metrics require matching dataset lengths.")

    sum_sq_delta = 0.0
    sum_abs_delta = 0.0
    max_abs_delta = 0.0
    sum_sq_signal = 0.0
    n_elements = 0
    sum_sq_diff_before = 0.0
    sum_sq_diff_after = 0.0
    n_diff_elements = 0

    for before_signal, after_signal in zip(original, denoised, strict=False):
        before = _to_numpy(before_signal)
        after = _to_numpy(after_signal)
        if before.shape != after.shape:
            raise ValueError("Denoising metrics require matching signal shapes.")

        delta = after - before
        sum_sq_delta += float(np.square(delta).sum())
        sum_abs_delta += float(np.abs(delta).sum())
        max_abs_delta = max(max_abs_delta, float(np.abs(delta).max(initial=0.0)))
        sum_sq_signal += float(np.square(before).sum())
        n_elements += int(delta.size)

        if before.shape[0] > 1:
            diff_before = np.diff(before, axis=0)
            diff_after = np.diff(after, axis=0)
            sum_sq_diff_before += float(np.square(diff_before).sum())
            sum_sq_diff_after += float(np.square(diff_after).sum())
            n_diff_elements += int(diff_before.size)

    delta_rmse = float(np.sqrt(sum_sq_delta / n_elements)) if n_elements > 0 else 0.0
    signal_rms = float(np.sqrt(sum_sq_signal / n_elements)) if n_elements > 0 else 0.0
    roughness_before = sum_sq_diff_before / n_diff_elements if n_diff_elements > 0 else 0.0
    roughness_after = sum_sq_diff_after / n_diff_elements if n_diff_elements > 0 else 0.0
    if roughness_before > 0.0:
        roughness_ratio = roughness_after / roughness_before
    elif roughness_after == 0.0:
        roughness_ratio = 1.0
    else:
        roughness_ratio = float("inf")

    return {
        "delta_rmse": delta_rmse,
        "delta_mae": sum_abs_delta / n_elements if n_elements > 0 else 0.0,
        "delta_max_abs": max_abs_delta,
        "delta_rel_rmse": (delta_rmse / signal_rms) if signal_rms > 0.0 else 0.0,
        "roughness_before": roughness_before,
        "roughness_after": roughness_after,
        "roughness_delta": roughness_after - roughness_before,
        "roughness_ratio": roughness_ratio,
    }
