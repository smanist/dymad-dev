"""Typed spectral snapshot records for checkpoint-backed analysis seams."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class KoopmanWeightSnapshot:
    """Typed Koopman weight payload from checkpoint-backed models."""

    mode: str
    full_matrix: np.ndarray | None = None
    left_factor: np.ndarray | None = None
    right_factor: np.ndarray | None = None


@dataclass(frozen=True)
class SpectralSnapshot:
    """Typed spectral inputs extracted from one checkpoint-backed model state."""

    model_class: str
    checkpoint_path: str
    encoded_p0: np.ndarray
    encoded_p1: np.ndarray
    koopman_weights: KoopmanWeightSnapshot
    input_dim: int
    obs_dim: int
    sample_count: int
    metadata: dict[str, Any] = field(default_factory=dict)


def build_spectral_snapshot(
    *,
    model_class: str,
    checkpoint_path: str,
    encoded_p0: np.ndarray,
    encoded_p1: np.ndarray,
    weights: tuple[np.ndarray, ...],
    input_dim: int,
    obs_dim: int,
    metadata: Mapping[str, Any] | None = None,
) -> SpectralSnapshot:
    """Build a typed spectral snapshot from checkpoint-derived arrays."""
    if encoded_p0.shape != encoded_p1.shape:
        raise ValueError("encoded_p0 and encoded_p1 must have matching shapes.")

    if len(weights) == 1:
        koopman_weights = KoopmanWeightSnapshot(mode="full", full_matrix=weights[0])
    elif len(weights) == 2:
        koopman_weights = KoopmanWeightSnapshot(
            mode="low_rank",
            left_factor=weights[0],
            right_factor=weights[1],
        )
    else:
        raise ValueError("weights must contain either one full matrix or two low-rank factors.")

    return SpectralSnapshot(
        model_class=model_class,
        checkpoint_path=checkpoint_path,
        encoded_p0=encoded_p0,
        encoded_p1=encoded_p1,
        koopman_weights=koopman_weights,
        input_dim=input_dim,
        obs_dim=obs_dim,
        sample_count=encoded_p0.shape[0],
        metadata=dict(metadata or {}),
    )
