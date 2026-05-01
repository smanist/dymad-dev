from __future__ import annotations

from numbers import Integral, Real
from typing import Literal

import numpy as np

KernelName = Literal["gaussian", "compact_polynomial"]


def _normalize_axis(axis: int, ndim: int) -> int:
    if axis < -ndim or axis >= ndim:
        raise ValueError(f"axis={axis} is out of bounds for an array with {ndim} dimensions.")
    return axis % ndim


def _validate_time(time: np.ndarray | None, *, n_samples: int) -> np.ndarray:
    if time is None:
        return np.arange(n_samples, dtype=np.float64)

    time_array = np.asarray(time, dtype=np.float64)
    if time_array.shape != (n_samples,):
        raise ValueError(f"time must have shape ({n_samples},), got {time_array.shape}.")
    if not np.isfinite(time_array).all():
        raise ValueError("time must contain only finite values.")
    if np.any(np.diff(time_array) <= 0):
        raise ValueError("time must be strictly increasing.")
    return time_array


def _resolve_anchor_count(
    *, n_samples: int, anchor_count: int | None, anchor_density: float | None
) -> int:
    if anchor_count is None and anchor_density is None:
        anchor_density = 0.125
    if anchor_count is not None and anchor_density is not None:
        raise ValueError("Specify only one of anchor_count or anchor_density.")

    if anchor_count is not None:
        if not isinstance(anchor_count, Integral) or isinstance(anchor_count, bool):
            raise ValueError("anchor_count must be a positive integer.")
        if anchor_count < 1:
            raise ValueError("anchor_count must be a positive integer.")
        return int(anchor_count)

    assert anchor_density is not None
    if not isinstance(anchor_density, Real) or isinstance(anchor_density, bool):
        raise ValueError("anchor_density must be a finite number in (0, 1].")
    if not np.isfinite(anchor_density) or anchor_density <= 0.0 or anchor_density > 1.0:
        raise ValueError("anchor_density must be a finite number in (0, 1].")
    return max(1, int(np.ceil(n_samples * anchor_density)))


def _resolve_bandwidth(
    *,
    time: np.ndarray,
    anchors: np.ndarray,
    bandwidth: float | None,
    bandwidth_multiplier: float | None,
) -> float:
    if bandwidth is None and bandwidth_multiplier is None:
        bandwidth_multiplier = 2.0
    if bandwidth is not None and bandwidth_multiplier is not None:
        raise ValueError("Specify only one of bandwidth or bandwidth_multiplier.")

    if bandwidth is not None:
        if not isinstance(bandwidth, Real) or isinstance(bandwidth, bool):
            raise ValueError("bandwidth must be a positive finite number.")
        if not np.isfinite(bandwidth) or bandwidth <= 0.0:
            raise ValueError("bandwidth must be a positive finite number.")
        return float(bandwidth)

    if bandwidth_multiplier is None:
        raise ValueError("Specify bandwidth or bandwidth_multiplier.")
    if not isinstance(bandwidth_multiplier, Real) or isinstance(bandwidth_multiplier, bool):
        raise ValueError("bandwidth_multiplier must be a positive finite number.")
    if not np.isfinite(bandwidth_multiplier) or bandwidth_multiplier <= 0.0:
        raise ValueError("bandwidth_multiplier must be a positive finite number.")

    if anchors.size == 1:
        spacing = float(np.mean(np.diff(time)))
    else:
        spacing = float(np.mean(np.diff(anchors)))
    return float(bandwidth_multiplier) * spacing


def _validate_degree(degree: int) -> int:
    if not isinstance(degree, Integral) or isinstance(degree, bool) or degree < 0:
        raise ValueError("degree must be a non-negative integer.")
    return int(degree)


def _validate_regularization(
    *, ridge: float | None, rcond: float | None
) -> tuple[float | None, float | None]:
    if ridge is not None:
        if not isinstance(ridge, Real) or isinstance(ridge, bool):
            raise ValueError("ridge must be a non-negative finite number.")
        if not np.isfinite(ridge) or ridge < 0.0:
            raise ValueError("ridge must be a non-negative finite number.")
        ridge = float(ridge)

    if rcond is not None:
        if not isinstance(rcond, Real) or isinstance(rcond, bool):
            raise ValueError("rcond must be a non-negative finite number.")
        if not np.isfinite(rcond) or rcond < 0.0:
            raise ValueError("rcond must be a non-negative finite number.")
        rcond = float(rcond)

    return ridge, rcond


def _build_design_matrix(
    *,
    time: np.ndarray,
    kernel: KernelName,
    anchor_count: int,
    bandwidth: float,
    degree: int,
) -> np.ndarray:
    if kernel not in {"gaussian", "compact_polynomial"}:
        raise ValueError(
            f"Unsupported kernel '{kernel}'. Expected 'gaussian' or 'compact_polynomial'."
        )

    anchors = np.linspace(time[0], time[-1], anchor_count, dtype=time.dtype)
    scaled_distance = np.abs(time[:, None] - anchors[None, :]) / bandwidth
    if kernel == "gaussian":
        return np.exp(-0.5 * np.square(scaled_distance))

    support = scaled_distance <= 1.0
    if degree == 0:
        return support.astype(np.float64)

    basis = np.clip(1.0 - scaled_distance, a_min=0.0, a_max=None)
    return np.where(support, basis**degree, 0.0)


def _solve_coefficients(
    design_matrix: np.ndarray,
    targets: np.ndarray,
    *,
    ridge: float | None,
    rcond: float | None,
) -> np.ndarray:
    if ridge is None or ridge == 0.0:
        coefficients, _, _, _ = np.linalg.lstsq(design_matrix, targets, rcond=rcond)
        return coefficients

    eye = np.eye(design_matrix.shape[1], dtype=design_matrix.dtype)
    augmented_design = np.concatenate((design_matrix, np.sqrt(ridge) * eye), axis=0)
    augmented_targets = np.concatenate(
        (targets, np.zeros((eye.shape[0], targets.shape[1]), dtype=targets.dtype)),
        axis=0,
    )
    coefficients, _, _, _ = np.linalg.lstsq(
        augmented_design,
        augmented_targets,
        rcond=rcond,
    )
    return coefficients


def kernel_smoothing(
    data: np.ndarray,
    *,
    kernel: KernelName = "gaussian",
    axis: int = 0,
    time: np.ndarray | None = None,
    anchor_count: int | None = None,
    anchor_density: float | None = None,
    bandwidth: float | None = None,
    bandwidth_multiplier: float | None = None,
    degree: int = 3,
    ridge: float | None = None,
    rcond: float | None = None,
) -> np.ndarray:
    """Smooth data along one axis with a global kernel-basis least-squares fit."""
    array = np.asarray(data)
    if array.ndim == 0:
        raise ValueError("kernel_smoothing requires an array with at least one dimension.")

    _normalize_axis(axis, array.ndim)
    moved = np.moveaxis(array, axis, 0)
    n_samples = moved.shape[0]
    if n_samples < 2:
        raise ValueError("kernel_smoothing requires at least two samples along the smoothing axis.")

    solve_dtype = np.result_type(moved.dtype, np.float64)
    values = np.asarray(moved, dtype=solve_dtype)
    if not np.isfinite(values).all():
        raise ValueError("kernel_smoothing requires finite input values.")

    degree = _validate_degree(degree)
    ridge, rcond = _validate_regularization(ridge=ridge, rcond=rcond)
    time_array = _validate_time(time, n_samples=n_samples)
    resolved_anchor_count = _resolve_anchor_count(
        n_samples=n_samples,
        anchor_count=anchor_count,
        anchor_density=anchor_density,
    )
    anchors = np.linspace(time_array[0], time_array[-1], resolved_anchor_count, dtype=np.float64)
    resolved_bandwidth = _resolve_bandwidth(
        time=time_array,
        anchors=anchors,
        bandwidth=bandwidth,
        bandwidth_multiplier=bandwidth_multiplier,
    )
    design_matrix = _build_design_matrix(
        time=time_array,
        kernel=kernel,
        anchor_count=resolved_anchor_count,
        bandwidth=resolved_bandwidth,
        degree=degree,
    )

    flattened = values.reshape(n_samples, -1)
    coefficients = _solve_coefficients(
        design_matrix,
        flattened,
        ridge=ridge,
        rcond=rcond,
    )
    smoothed = (design_matrix @ coefficients).reshape(values.shape)

    result = np.moveaxis(smoothed, 0, axis)
    if np.issubdtype(array.dtype, np.floating) or np.issubdtype(array.dtype, np.complexfloating):
        return np.asarray(result, dtype=array.dtype)
    return result
