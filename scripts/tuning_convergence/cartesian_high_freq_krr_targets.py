from __future__ import annotations

import math
from functools import cache

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import special
from scipy.interpolate import CubicSpline
from scipy.linalg import eigh

RBF_EIGEN_SIGMA = 0.2
RBF_EIGEN_N_QUAD = 700
RBF_EIGEN_N_MODES = 3


def oscillatory_values(points: np.ndarray) -> np.ndarray:
    x = points[:, 0]
    y = points[:, 1]
    values = np.sin(18.0 * x + 11.0 * y) + 0.5 * np.cos(22.0 * x - 5.0 * y)
    return values.reshape(-1, 1)


def smooth_radial_values(points: np.ndarray) -> np.ndarray:
    radius_sq = np.sum(points * points, axis=1)
    values = np.exp(-2.5 * radius_sq) + 0.25 * points[:, 0] - 0.15 * points[:, 1]
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


def rbf_radial_kernel(order: int, sigma: float, r: np.ndarray, s: np.ndarray) -> np.ndarray:
    rr = np.asarray(r, dtype=float)[..., None]
    ss = np.asarray(s, dtype=float)[None, ...]
    return (
        2.0
        * math.pi
        * np.exp(-(rr * rr + ss * ss) / (2.0 * sigma * sigma))
        * special.iv(order, rr * ss / (sigma * sigma))
    )


def rbf_endpoint_values(
    order: int,
    sigma: float,
    eigenvalues: np.ndarray,
    r: np.ndarray,
    w: np.ndarray,
    radial_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    left = np.zeros(radial_values.shape[1], dtype=float)
    if order == 0:
        left_kernel = rbf_radial_kernel(order, sigma, np.array([0.0]), r)[0]
        left = (left_kernel[:, None] * radial_values * (w * r)[:, None]).sum(axis=0) / eigenvalues
    right_kernel = rbf_radial_kernel(order, sigma, np.array([1.0]), r)[0]
    right = (right_kernel[:, None] * radial_values * (w * r)[:, None]).sum(axis=0) / eigenvalues
    return left, right


@cache
def rbf_radial_eigen_interpolants(order: int, sigma: float) -> tuple[CubicSpline, ...]:
    x, w = leggauss(RBF_EIGEN_N_QUAD)
    r = 0.5 * (x + 1.0)
    w = 0.5 * w
    rr = r[:, None]
    ss = r[None, :]
    matrix = (
        np.sqrt(w)[:, None]
        * (
            2.0
            * math.pi
            * np.sqrt(rr * ss)
            * np.exp(-(rr * rr + ss * ss) / (2.0 * sigma * sigma))
            * special.iv(order, rr * ss / (sigma * sigma))
        )
        * np.sqrt(w)[None, :]
    )
    eigenvalues, eigenvectors = eigh(matrix)
    sort_order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sort_order][:RBF_EIGEN_N_MODES]
    eigenvectors = eigenvectors[:, sort_order][:, :RBF_EIGEN_N_MODES]

    radial = eigenvectors / (np.sqrt(w)[:, None] * np.sqrt(r)[:, None])
    for mode in range(RBF_EIGEN_N_MODES):
        norm = math.sqrt(float(np.sum(w * radial[:, mode] * radial[:, mode] * r)))
        radial[:, mode] /= norm
        if radial[0, mode] < 0.0:
            radial[:, mode] *= -1.0

    left, right = rbf_endpoint_values(order, sigma, eigenvalues, r, w, radial)
    knots = np.concatenate(([0.0], r, [1.0]))
    interpolants = []
    for mode in range(RBF_EIGEN_N_MODES):
        values = np.concatenate(([left[mode]], radial[:, mode], [right[mode]]))
        interpolants.append(CubicSpline(knots, values, bc_type="not-a-knot", extrapolate=False))
    return tuple(interpolants)


def rbf_eigen_m2_k2_values(points: np.ndarray) -> np.ndarray:
    radius, theta = polar(points)
    radial = rbf_radial_eigen_interpolants(2, RBF_EIGEN_SIGMA)[2](np.clip(radius, 0.0, 1.0))
    values = radial * np.cos(2.0 * theta)
    return values.reshape(-1, 1)


TARGETS = {
    "laplace_neumann_m2_k2": laplace_neumann_m2_k2_values,
    "localized_bump": localized_bump_values,
    "oscillatory": oscillatory_values,
    "rbf_eigen_m2_k2": rbf_eigen_m2_k2_values,
    "smooth_radial": smooth_radial_values,
}
