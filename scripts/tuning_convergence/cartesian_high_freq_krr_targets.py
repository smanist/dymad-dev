from __future__ import annotations

import numpy as np
from scipy import special


def oscillatory_values(points: np.ndarray) -> np.ndarray:
    x = points[:, 0]
    y = points[:, 1]
    values = np.sin(18.0 * x + 11.0 * y) + 0.5 * np.cos(22.0 * x - 5.0 * y)
    return values.reshape(-1, 1)


def localized_bump_values(points: np.ndarray) -> np.ndarray:
    x = points[:, 0]
    y = points[:, 1]
    values = np.exp(-80.0 * ((x - 0.35) ** 2 + (y + 0.25) ** 2))
    return values.reshape(-1, 1)


def polar(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    radius = np.sqrt(np.sum(points * points, axis=1))
    theta = np.arctan2(points[:, 1], points[:, 0])
    return radius, theta


def laplace_neumann_m2_k2_values(points: np.ndarray) -> np.ndarray:
    radius, theta = polar(points)
    wavenumber = float(special.jnp_zeros(2, 3)[2])
    values = special.jv(2, wavenumber * radius) * np.cos(2.0 * theta)
    return values.reshape(-1, 1)


TARGETS = {
    "laplace_neumann_m2_k2": laplace_neumann_m2_k2_values,
    "localized_bump": localized_bump_values,
    "oscillatory": oscillatory_values,
}
