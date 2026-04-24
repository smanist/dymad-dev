from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypeAlias, overload

import numpy as np
import torch
from scipy.signal import savgol_filter

ArrayLike: TypeAlias = np.ndarray | torch.Tensor
DenoiseMethod = Literal["savgol"]


def _denoise_savgol(data: ArrayLike, *, axis: int, **kwargs) -> ArrayLike:
    smoothed = savgol_filter(_to_numpy(data), axis=axis, **kwargs)
    if isinstance(data, torch.Tensor):
        return torch.as_tensor(smoothed, device=data.device, dtype=data.dtype)
    return np.asarray(smoothed, dtype=data.dtype)


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
