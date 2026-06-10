from __future__ import annotations

import numpy as np


def oscillatory_values(points: np.ndarray) -> np.ndarray:
    x = points[:, 0]
    y = points[:, 1]
    values = np.sin(18.0 * x + 11.0 * y) + 0.5 * np.cos(22.0 * x - 5.0 * y)
    return values.reshape(-1, 1)


def smooth_radial_values(points: np.ndarray) -> np.ndarray:
    radius_sq = np.sum(points * points, axis=1)
    values = np.exp(-2.5 * radius_sq) + 0.25 * points[:, 0] - 0.15 * points[:, 1]
    return values.reshape(-1, 1)


TARGETS = {
    "oscillatory": oscillatory_values,
    "smooth_radial": smooth_radial_values,
}
