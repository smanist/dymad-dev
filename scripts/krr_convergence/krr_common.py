from __future__ import annotations

import argparse
import csv
import math
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import TwoSlopeNorm
from numpy.polynomial.legendre import leggauss
from scipy import special
from scipy.interpolate import CubicSpline
from scipy.linalg import eigh
from scipy.special import jn_zeros, jnp_zeros, jv

from dymad.io import Split
from dymad.modules import make_krr
from dymad.studies.convergence import (
    ArrayRegressionProblem,
    ArrayRegressionStudyConfig,
    CurveStyle,
    LevelSamplePlan,
    MedianPlotContext,
    NestedArraySamples,
    plot_convergence_summary,
    run_array_regression_study,
)
from dymad.tuning import ParameterSpec, TuningSpec

PLOT_FONT_SIZE = 14
matplotlib.rcParams.update(
    {
        "axes.labelsize": PLOT_FONT_SIZE,
        "axes.titlesize": PLOT_FONT_SIZE,
        "figure.titlesize": PLOT_FONT_SIZE,
        "legend.fontsize": PLOT_FONT_SIZE,
        "xtick.labelsize": PLOT_FONT_SIZE,
        "ytick.labelsize": PLOT_FONT_SIZE,
    }
)

RBF_METHOD = "rbf_krr"
DM_METHOD = "dm_krr"
METHODS = (RBF_METHOD, DM_METHOD)
METHOD_LABELS = {RBF_METHOD: "RBF KRR", DM_METHOD: "Diffusion maps KRR"}
METHOD_COLORS = {RBF_METHOD: "#d95f02", DM_METHOD: "#1b9e77"}

RBF_INTEGRAL_SIGMA = 0.2
RBF_INTEGRAL_N_QUAD = 700
RBF_INTEGRAL_MAX_MODE = 7


@dataclass(frozen=True)
class Case:
    name: str
    title: str
    target: Callable[[np.ndarray], np.ndarray]
    ambient_dim: int = 2
    embedding: str = "augmented"


@dataclass(frozen=True)
class Group:
    name: str
    slug: str
    cases: tuple[Case, ...]
    show_targets: bool = True


def comma_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in value.split(","))
    if not values or any(item < 2 for item in values):
        raise argparse.ArgumentTypeError("levels must be comma-separated integers at least 2")
    return values


def add_study_args(parser: argparse.ArgumentParser, groups: tuple[Group, ...]) -> None:
    parser.add_argument("--workdir", type=Path)
    parser.add_argument("--groups", nargs="+", choices=[group.slug for group in groups])
    parser.add_argument("--levels", type=comma_ints, default=(512, 1024, 2048, 4096))
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--n-val", type=int, default=1024)
    parser.add_argument("--n-test", type=int, default=4096)
    parser.add_argument("--initial-budget", type=int, default=9)
    parser.add_argument("--refinement-budget", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--validation-size", type=int, default=1024)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--no-prediction-plots", action="store_true")


def selected_groups(groups: tuple[Group, ...], selected: list[str] | None) -> tuple[Group, ...]:
    names = set(selected or (group.slug for group in groups))
    return tuple(group for group in groups if group.slug in names)


def output_root(workdir: Path | None, script_dir: Path) -> Path:
    return (workdir.resolve() if workdir is not None else script_dir) / "runs"


def report_root(workdir: Path | None, script_dir: Path) -> Path:
    return (workdir.resolve() if workdir is not None else script_dir) / "reports"


def unit_disk_sample(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    radius = np.sqrt(rng.random(n_samples))
    theta = 2.0 * math.pi * rng.random(n_samples)
    return np.column_stack((radius * np.cos(theta), radius * np.sin(theta)))


def _polar(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    radius = np.sqrt(np.sum(points[:, :2] * points[:, :2], axis=1))
    theta = np.arctan2(points[:, 1], points[:, 0])
    return radius, theta


def make_neumann_disk_target(
    angular_mode: int, radial_index: int
) -> Callable[[np.ndarray], np.ndarray]:
    alpha = float(jnp_zeros(angular_mode, radial_index)[radial_index - 1])

    def target(points: np.ndarray) -> np.ndarray:
        radius, theta = _polar(points)
        angular = 1.0 if angular_mode == 0 else np.cos(angular_mode * theta)
        return (jv(angular_mode, alpha * radius) * angular).reshape(-1, 1)

    return target


def make_dirichlet_disk_target(
    angular_mode: int, radial_index: int
) -> Callable[[np.ndarray], np.ndarray]:
    alpha = float(jn_zeros(angular_mode, radial_index)[radial_index - 1])

    def target(points: np.ndarray) -> np.ndarray:
        radius, theta = _polar(points)
        angular = 1.0 if angular_mode == 0 else np.cos(angular_mode * theta)
        return (jv(angular_mode, alpha * radius) * angular).reshape(-1, 1)

    return target


def rbf_radial_kernel(order: int, sigma: float, r: np.ndarray, s: np.ndarray) -> np.ndarray:
    rr = np.asarray(r, dtype=float)[..., None]
    ss = np.asarray(s, dtype=float)[None, ...]
    return (
        2.0
        * math.pi
        * np.exp(-(rr * rr + ss * ss) / (2.0 * sigma * sigma))
        * special.iv(order, rr * ss / (sigma * sigma))
    )


def _rbf_endpoint_values(
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
        left = (left_kernel[:, None] * radial_values * (w * r)[:, None]).sum(axis=0)
        left = left / eigenvalues
    right_kernel = rbf_radial_kernel(order, sigma, np.array([1.0]), r)[0]
    right = (right_kernel[:, None] * radial_values * (w * r)[:, None]).sum(axis=0)
    return left, right / eigenvalues


@cache
def rbf_radial_eigenpairs(order: int, sigma: float) -> tuple[np.ndarray, tuple[CubicSpline, ...]]:
    x, w = leggauss(RBF_INTEGRAL_N_QUAD)
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
    eigenvalues = eigenvalues[sort_order][:RBF_INTEGRAL_MAX_MODE]
    eigenvectors = eigenvectors[:, sort_order][:, :RBF_INTEGRAL_MAX_MODE]

    radial = eigenvectors / (np.sqrt(w)[:, None] * np.sqrt(r)[:, None])
    for mode in range(RBF_INTEGRAL_MAX_MODE):
        norm = math.sqrt(float(np.sum(w * radial[:, mode] * radial[:, mode] * r)))
        radial[:, mode] /= norm
        if radial[0, mode] < 0.0:
            radial[:, mode] *= -1.0

    left, right = _rbf_endpoint_values(order, sigma, eigenvalues, r, w, radial)
    knots = np.concatenate(([0.0], r, [1.0]))
    interpolants = []
    for mode in range(RBF_INTEGRAL_MAX_MODE):
        values = np.concatenate(([left[mode]], radial[:, mode], [right[mode]]))
        interpolants.append(CubicSpline(knots, values, bc_type="not-a-knot", extrapolate=False))
    return eigenvalues, tuple(interpolants)


def make_rbf_integral_disk_target(
    angular_mode: int, radial_mode: int
) -> Callable[[np.ndarray], np.ndarray]:
    if radial_mode < 1 or radial_mode > RBF_INTEGRAL_MAX_MODE:
        raise ValueError(f"radial_mode must be in [1, {RBF_INTEGRAL_MAX_MODE}]")

    def target(points: np.ndarray) -> np.ndarray:
        radius, theta = _polar(points)
        _, interpolants = rbf_radial_eigenpairs(angular_mode, RBF_INTEGRAL_SIGMA)
        radial = interpolants[radial_mode - 1](np.clip(radius, 0.0, 1.0))
        angular = 1.0 if angular_mode == 0 else np.cos(angular_mode * theta)
        return (radial * angular).reshape(-1, 1)

    return target


def make_ambient_periodic_disk_target(kx: int, ky: int) -> Callable[[np.ndarray], np.ndarray]:
    def target(points: np.ndarray) -> np.ndarray:
        u = 0.5 * (points[:, 0] + 1.0)
        v = 0.5 * (points[:, 1] + 1.0)
        values = np.cos(2.0 * math.pi * kx * u) * np.cos(2.0 * math.pi * ky * v)
        return values.reshape(-1, 1)

    return target


def wrap_scaled_target(
    target_raw: Callable[[np.ndarray], np.ndarray], *, coordinate_scale: float
) -> Callable[[np.ndarray], np.ndarray]:
    def target(points: np.ndarray) -> np.ndarray:
        return target_raw(points * coordinate_scale)

    return target


@dataclass(frozen=True)
class SemiTorusGeometry:
    major_radius: float = 2.0
    theta_grid_size: int = 512

    def points_from_angles(
        self,
        theta: np.ndarray | float,
        phi: np.ndarray | float,
        *,
        ambient_dim: int = 3,
        embedding: str = "augmented",
    ) -> np.ndarray:
        theta_values = np.asarray(theta, dtype=float)
        phi_values = np.asarray(phi, dtype=float)
        radius = self.major_radius + np.cos(theta_values)
        if ambient_dim < 3:
            raise ValueError("ambient_dim must be at least 3")
        flat_theta = theta_values.reshape(-1)
        flat_phi = phi_values.reshape(-1)
        flat_radius = radius.reshape(-1)
        if embedding == "harmonic":
            if ambient_dim % 2 == 0:
                raise ValueError("harmonic torus embedding requires an odd ambient_dim")
            harmonic_count = (ambient_dim - 1) // 2
            coordinates: list[np.ndarray] = []
            for order in range(1, harmonic_count + 1):
                amplitude = flat_radius / order
                coordinates.extend(
                    (amplitude * np.cos(order * flat_phi), amplitude * np.sin(order * flat_phi))
                )
            theta_normalizer = math.sqrt(
                sum(1.0 / (order * order) for order in range(1, harmonic_count + 1))
            )
            coordinates.append(np.sin(flat_theta) / theta_normalizer)
            return np.column_stack(coordinates)
        if embedding != "augmented":
            raise ValueError(f"unknown semi-torus embedding: {embedding}")
        base = np.column_stack(
            (
                flat_radius * np.cos(flat_phi),
                flat_radius * np.sin(flat_phi),
                np.sin(flat_theta),
            )
        )
        if ambient_dim == 3:
            return base

        extras: list[np.ndarray] = []
        extra_count = ambient_dim - 3
        for idx in range(extra_count):
            order = 1 + idx // 4
            selector = idx % 4
            if selector == 0:
                extras.append(0.2 * np.cos(order * flat_theta))
            elif selector == 1:
                extras.append(0.2 * np.sin(order * flat_theta))
            elif selector == 2:
                extras.append(0.2 * np.cos(order * flat_phi))
            else:
                extras.append(0.2 * np.sin(order * flat_phi))
        return np.column_stack((base, *extras))

    def angles_from_points(
        self, points: np.ndarray, *, embedding: str = "augmented"
    ) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(points, dtype=float)
        if values.shape[1] < 3:
            raise ValueError("semi-torus points must have at least three coordinates")
        if embedding == "harmonic":
            if values.shape[1] % 2 == 0:
                raise ValueError("harmonic torus embedding requires an odd ambient_dim")
            harmonic_count = (values.shape[1] - 1) // 2
            theta_normalizer = math.sqrt(
                sum(1.0 / (order * order) for order in range(1, harmonic_count + 1))
            )
            sin_theta = values[:, -1] * theta_normalizer
        elif embedding == "augmented":
            sin_theta = values[:, 2]
        else:
            raise ValueError(f"unknown semi-torus embedding: {embedding}")
        rho = np.sqrt(values[:, 0] * values[:, 0] + values[:, 1] * values[:, 1])
        theta = np.mod(np.arctan2(sin_theta, rho - self.major_radius), 2.0 * math.pi)
        phi = np.mod(np.arctan2(values[:, 1], values[:, 0]), 2.0 * math.pi)
        phi = np.where(phi > math.pi, 2.0 * math.pi - phi, phi)
        return theta, phi

    def sample_angles(
        self, n_samples: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        theta_chunks: list[np.ndarray] = []
        accepted = 0
        while accepted < n_samples:
            proposal_count = max(n_samples - accepted, 32)
            theta_proposal = 2.0 * math.pi * rng.random(proposal_count)
            accept_probability = (self.major_radius + np.cos(theta_proposal)) / (
                self.major_radius + 1.0
            )
            keep = rng.random(proposal_count) < accept_probability
            if np.any(keep):
                theta_chunks.append(theta_proposal[keep])
                accepted += int(np.count_nonzero(keep))
        theta = np.concatenate(theta_chunks)[:n_samples]
        phi = math.pi * rng.random(n_samples)
        return theta, phi

    def sample_points(
        self,
        n_samples: int,
        rng: np.random.Generator,
        *,
        ambient_dim: int = 3,
        embedding: str = "augmented",
    ) -> np.ndarray:
        theta, phi = self.sample_angles(n_samples, rng)
        return self.points_from_angles(theta, phi, ambient_dim=ambient_dim, embedding=embedding)


@dataclass(frozen=True)
class SemiTorusThetaMode:
    theta_nodes: np.ndarray
    values: np.ndarray
    eigenvalue: float
    global_mode: int

    def evaluate(self, theta: np.ndarray | float) -> np.ndarray:
        values = np.asarray(theta, dtype=float)
        period = 2.0 * math.pi
        knots = np.concatenate((self.theta_nodes, [period]))
        samples = np.concatenate((self.values, [self.values[0]]))
        return np.interp(np.mod(values, period), knots, samples)


@dataclass(frozen=True)
class SemiTorusFourierThetaMode:
    coefficients: np.ndarray
    eigenvalue: float
    global_mode: int
    fourier_order: int

    def evaluate(self, theta: np.ndarray | float) -> np.ndarray:
        values = np.asarray(theta, dtype=float)
        flat_theta = np.mod(values.reshape(-1), 2.0 * math.pi)
        result = np.full(flat_theta.shape, self.coefficients[0], dtype=float)
        for order in range(1, self.fourier_order + 1):
            cos_index = 2 * order - 1
            sin_index = 2 * order
            result += self.coefficients[cos_index] * np.cos(order * flat_theta)
            result += self.coefficients[sin_index] * np.sin(order * flat_theta)
        return result.reshape(values.shape)


@cache
def semi_torus_theta_modes(major_radius: float, m: int, grid_size: int) -> tuple[float, ...]:
    modes = _compute_semi_torus_theta_modes(major_radius, m, grid_size)
    payload: list[float] = []
    for mode in modes:
        payload.extend((mode.eigenvalue, float(mode.global_mode), *mode.values.tolist()))
    return tuple(payload)


@cache
def semi_torus_fourier_theta_modes(
    major_radius: float, m: int, fourier_order: int, quadrature_size: int
) -> tuple[float, ...]:
    modes = _compute_semi_torus_fourier_theta_modes(
        major_radius,
        m,
        fourier_order,
        quadrature_size,
    )
    payload: list[float] = []
    for mode in modes:
        payload.extend((mode.eigenvalue, float(mode.global_mode), *mode.coefficients.tolist()))
    return tuple(payload)


def _compute_semi_torus_theta_modes(
    major_radius: float, m: int, grid_size: int
) -> tuple[SemiTorusThetaMode, ...]:
    theta = np.linspace(0.0, 2.0 * math.pi, grid_size, endpoint=False)
    step = 2.0 * math.pi / grid_size
    weight = major_radius + np.cos(theta)
    right_weight = 0.5 * (weight + np.roll(weight, -1))
    left_weight = np.roll(right_weight, 1)

    stiffness = np.zeros((grid_size, grid_size), dtype=float)
    diag = (right_weight + left_weight) / (step * step) + (m * m) / weight
    np.fill_diagonal(stiffness, diag)
    indices = np.arange(grid_size)
    stiffness[indices, (indices + 1) % grid_size] = -right_weight / (step * step)
    stiffness[indices, (indices - 1) % grid_size] = -left_weight / (step * step)
    mass = np.diag(weight)
    eigenvalues, eigenvectors = eigh(stiffness, mass)
    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    modes: list[SemiTorusThetaMode] = []
    for global_mode, (eigenvalue, eigenvector) in enumerate(zip(eigenvalues, eigenvectors.T)):
        values = np.asarray(eigenvector, dtype=float)
        norm = math.sqrt(float(np.sum(values * values * weight) * step))
        values = values / norm
        if values[int(np.argmax(np.abs(values)))] < 0.0:
            values = -values
        modes.append(
            SemiTorusThetaMode(
                theta_nodes=theta,
                values=values,
                eigenvalue=float(eigenvalue),
                global_mode=global_mode,
            )
        )
    return tuple(modes)


def _semi_torus_fourier_basis(
    theta: np.ndarray, fourier_order: int
) -> tuple[np.ndarray, np.ndarray]:
    columns = [np.ones_like(theta)]
    derivative_columns = [np.zeros_like(theta)]
    for order in range(1, fourier_order + 1):
        columns.append(np.cos(order * theta))
        derivative_columns.append(-order * np.sin(order * theta))
        columns.append(np.sin(order * theta))
        derivative_columns.append(order * np.cos(order * theta))
    return np.column_stack(columns), np.column_stack(derivative_columns)


def _compute_semi_torus_fourier_theta_modes(
    major_radius: float,
    m: int,
    fourier_order: int,
    quadrature_size: int,
) -> tuple[SemiTorusFourierThetaMode, ...]:
    if fourier_order < 0:
        raise ValueError("fourier_order must be nonnegative")
    if quadrature_size < 2 * (2 * fourier_order + 1):
        raise ValueError("quadrature_size is too small for the requested Fourier order")

    theta = np.linspace(0.0, 2.0 * math.pi, quadrature_size, endpoint=False)
    step = 2.0 * math.pi / quadrature_size
    radius = major_radius + np.cos(theta)
    basis, basis_derivative = _semi_torus_fourier_basis(theta, fourier_order)
    mass = step * (basis.T @ (radius[:, None] * basis))
    stiffness = step * (
        basis_derivative.T @ (radius[:, None] * basis_derivative)
        + basis.T @ (((m * m) / radius)[:, None] * basis)
    )
    eigenvalues, eigenvectors = eigh(stiffness, mass)
    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    modes: list[SemiTorusFourierThetaMode] = []
    for global_mode, (eigenvalue, coefficients) in enumerate(zip(eigenvalues, eigenvectors.T)):
        coefficients = np.asarray(coefficients, dtype=float)
        values = basis @ coefficients
        norm = math.sqrt(float(np.sum(values * values * radius) * step))
        coefficients = coefficients / norm
        values = values / norm
        if values[int(np.argmax(np.abs(values)))] < 0.0:
            coefficients = -coefficients
        modes.append(
            SemiTorusFourierThetaMode(
                coefficients=coefficients,
                eigenvalue=float(eigenvalue),
                global_mode=global_mode,
                fourier_order=fourier_order,
            )
        )
    return tuple(modes)


def get_semi_torus_theta_mode(geometry: SemiTorusGeometry, *, m: int, j: int) -> SemiTorusThetaMode:
    payload = semi_torus_theta_modes(geometry.major_radius, m, geometry.theta_grid_size)
    record_size = geometry.theta_grid_size + 2
    start = j * record_size
    if start + record_size > len(payload):
        raise ValueError(
            f"theta mode j={j} is unavailable for grid size {geometry.theta_grid_size}"
        )
    theta = np.linspace(0.0, 2.0 * math.pi, geometry.theta_grid_size, endpoint=False)
    return SemiTorusThetaMode(
        theta_nodes=theta,
        values=np.asarray(payload[start + 2 : start + record_size], dtype=float),
        eigenvalue=float(payload[start]),
        global_mode=int(payload[start + 1]),
    )


def get_semi_torus_fourier_theta_mode(
    geometry: SemiTorusGeometry,
    *,
    m: int,
    j: int,
    fourier_order: int,
    quadrature_size: int = 4096,
) -> SemiTorusFourierThetaMode:
    payload = semi_torus_fourier_theta_modes(
        geometry.major_radius,
        m,
        fourier_order,
        quadrature_size,
    )
    coefficient_count = 2 * fourier_order + 1
    record_size = coefficient_count + 2
    start = j * record_size
    if start + record_size > len(payload):
        raise ValueError(f"theta mode j={j} is unavailable for Fourier order {fourier_order}")
    return SemiTorusFourierThetaMode(
        coefficients=np.asarray(payload[start + 2 : start + record_size], dtype=float),
        eigenvalue=float(payload[start]),
        global_mode=int(payload[start + 1]),
        fourier_order=fourier_order,
    )


def make_semi_torus_sample(
    geometry: SemiTorusGeometry,
    *,
    ambient_dim: int,
    coordinate_scale: float,
    embedding: str = "augmented",
) -> Callable[[int, np.random.Generator], np.ndarray]:
    def sample(n_samples: int, rng: np.random.Generator) -> np.ndarray:
        return (
            geometry.sample_points(n_samples, rng, ambient_dim=ambient_dim, embedding=embedding)
            / coordinate_scale
        )

    return sample


def make_semi_torus_target(
    geometry: SemiTorusGeometry,
    *,
    boundary: str,
    m: int,
    j: int,
    embedding: str = "augmented",
) -> Callable[[np.ndarray], np.ndarray]:
    mode = get_semi_torus_theta_mode(geometry, m=m, j=j)
    normalized_boundary = boundary.lower()
    if normalized_boundary not in {"dirichlet", "neumann"}:
        raise ValueError("boundary must be 'dirichlet' or 'neumann'")

    def target(points: np.ndarray) -> np.ndarray:
        theta, phi = geometry.angles_from_points(points, embedding=embedding)
        theta_values = mode.evaluate(theta)
        if normalized_boundary == "dirichlet":
            angular = np.sin(m * phi)
        else:
            angular = np.cos(m * phi)
        return (theta_values * angular).reshape(-1, 1)

    return target


def make_semi_torus_fourier_target(
    geometry: SemiTorusGeometry,
    *,
    boundary: str,
    m: int,
    j: int,
    fourier_order: int,
    quadrature_size: int = 4096,
    embedding: str = "augmented",
) -> Callable[[np.ndarray], np.ndarray]:
    mode = get_semi_torus_fourier_theta_mode(
        geometry,
        m=m,
        j=j,
        fourier_order=fourier_order,
        quadrature_size=quadrature_size,
    )
    normalized_boundary = boundary.lower()
    if normalized_boundary not in {"dirichlet", "neumann"}:
        raise ValueError("boundary must be 'dirichlet' or 'neumann'")

    def target(points: np.ndarray) -> np.ndarray:
        theta, phi = geometry.angles_from_points(points, embedding=embedding)
        theta_values = mode.evaluate(theta)
        if normalized_boundary == "dirichlet":
            angular = np.sin(m * phi)
        else:
            angular = np.cos(m * phi)
        return (theta_values * angular).reshape(-1, 1)

    return target


def make_semi_torus_ambient_fourier_target(
    geometry: SemiTorusGeometry,
    *,
    frequency: int,
) -> Callable[[np.ndarray], np.ndarray]:
    if frequency < 1:
        raise ValueError("frequency must be positive")

    def target(points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=float)
        if values.shape[1] < 2:
            raise ValueError("semi-torus ambient Fourier target requires at least two coordinates")
        scale = geometry.major_radius + 1.0
        u = 0.5 * (values[:, 0] / scale + 1.0)
        v = 0.5 * (values[:, 1] / scale + 1.0)
        cycles = 0.5 * frequency
        mode = np.cos(2.0 * math.pi * cycles * u) * np.cos(2.0 * math.pi * cycles * v)
        return mode.reshape(-1, 1)

    return target


def semi_torus_target_grid(
    geometry: SemiTorusGeometry,
    target: Callable[[np.ndarray], np.ndarray],
    *,
    ambient_dim: int = 3,
    embedding: str = "augmented",
    n_theta: int = 96,
    n_phi: int = 48,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, 2.0 * math.pi, n_theta, endpoint=False)
    phi = np.linspace(0.0, math.pi, n_phi)
    tt, pp = np.meshgrid(theta, phi)
    points = geometry.points_from_angles(
        tt.ravel(), pp.ravel(), ambient_dim=ambient_dim, embedding=embedding
    )
    values = target(points).reshape(n_phi, n_theta)
    x = points[:, 0].reshape(n_phi, n_theta)
    y = points[:, 1].reshape(n_phi, n_theta)
    z = points[:, 2].reshape(n_phi, n_theta)
    return x, y, z, values, points


def kernel_config(method: str, input_dim: int, bandwidth_init: float) -> dict[str, Any]:
    if method == RBF_METHOD:
        return {"type": "sc_rbf", "input_dim": input_dim, "lengthscale_init": bandwidth_init}
    if method == DM_METHOD:
        return {"type": "sc_dm", "input_dim": input_dim, "eps_init": bandwidth_init}
    raise ValueError(f"unknown method {method!r}")


def realized_kernel_value(model: Any, method: str) -> float:
    with torch.no_grad():
        if method == RBF_METHOD:
            return float(model.kernel.ell.detach().cpu())
        return float(model.kernel.eps.detach().cpu())


def rmse(truth: np.ndarray, pred: np.ndarray) -> float:
    if not np.isfinite(pred).all():
        return 1.0e12
    value = float(np.sqrt(np.mean((truth - pred) ** 2)))
    return value if math.isfinite(value) else 1.0e12


def max_abs_error(truth: np.ndarray, pred: np.ndarray) -> float:
    if not np.isfinite(pred).all():
        return 1.0e12
    value = float(np.max(np.abs(truth - pred)))
    return value if math.isfinite(value) else 1.0e12


def fit_model(method: str, split: Split, params: Mapping[str, Any]) -> Any:
    model = make_krr(
        type="share",
        kernel=kernel_config(method, split.x_train.shape[1], float(params["bandwidth_init"])),
        dtype=torch.float64,
        ridge_init=float(params["ridge_init"]),
        jitter=0.0,
    )
    model.set_train_data(split.x_train, split.y_train)
    model.fit()
    return model


def fit_and_score(
    method: str,
    split: Split,
    params: Mapping[str, Any],
    include_test: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    model = fit_model(method, split, params)
    with torch.no_grad():
        y_val_pred = model(torch.as_tensor(split.x_val, dtype=torch.float64)).cpu().numpy()
        y_test_pred = None
        if include_test:
            y_test_pred = model(torch.as_tensor(split.x_test, dtype=torch.float64)).cpu().numpy()
        train_residual = (
            float(model._residual.detach().cpu()) if model._residual is not None else math.nan
        )
    row: dict[str, Any] = {
        "validation_normalized_rmse": rmse(split.y_val, y_val_pred),
        "fit_seconds": time.perf_counter() - started,
        "realized_bandwidth": realized_kernel_value(model, method),
        "realized_ridge": float(model.ridge.detach().cpu()),
        "train_residual": train_residual,
    }
    if y_test_pred is not None:
        y_test_physical_pred = split.inverse_y(y_test_pred)
        row.update(
            {
                "error": rmse(split.y_test, y_test_pred),
                "test_physical_rmse": rmse(split.y_test_raw, y_test_physical_pred),
                "test_normalized_max_abs": max_abs_error(split.y_test, y_test_pred),
            }
        )
    return row


def fit_and_score_folds(
    method: str,
    samples: NestedArraySamples,
    plan: LevelSamplePlan,
    trial: int | str,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    fold_rows = [
        fit_and_score(method, samples.split_for_fold(trial, fold), params, include_test=False)
        for fold in plan.validation_folds
    ]
    values = np.asarray([float(row["validation_normalized_rmse"]) for row in fold_rows])
    return {
        "validation_normalized_rmse": float(np.mean(values)),
        "std_metric": float(np.std(values)),
        "fold_metrics": values.tolist(),
        "fit_seconds": float(sum(float(row["fit_seconds"]) for row in fold_rows)),
    }


def tuning_spec(
    metric_name: str,
    initial_budget: int | tuple[int, ...],
    refinement_budget: int,
    refinement_strategy: str | None,
) -> TuningSpec:
    strategy = (refinement_strategy or "nelder_mead_like") if refinement_budget > 0 else None
    return TuningSpec(
        parameters=(
            ParameterSpec("bandwidth_init", bounds=(1.0e-4, 1.0e2), scale="log"),
            ParameterSpec("ridge_init", bounds=(1.0e-16, 1.0e1), scale="log"),
        ),
        metric_name=metric_name,
        initial_budget=initial_budget,
        initial_strategy="grid",
        refinement_strategy=strategy,
        refinement_budget=refinement_budget,
        metadata={"study": "krr_dm_rbf_comparison"},
    )


def make_convergence_plot(result: Any, output_dir: Path, center: str, band: str) -> None:
    plot_convergence_summary(
        result,
        output_dir / "convergence.png",
        methods=METHODS,
        center=center,
        band=band,
        title="KRR kernel comparison",
        xlabel="N",
        ylabel="test normalized RMSE",
        styles={
            RBF_METHOD: CurveStyle(
                label=METHOD_LABELS[RBF_METHOD], color=METHOD_COLORS[RBF_METHOD]
            ),
            DM_METHOD: CurveStyle(label=METHOD_LABELS[DM_METHOD], color=METHOD_COLORS[DM_METHOD]),
        },
    )


def plot_truth_vs_prediction(context: MedianPlotContext, split: Split) -> None:
    model = fit_model(context.method, split, context.params)
    with torch.no_grad():
        y_pred_norm = model(torch.as_tensor(split.x_test, dtype=torch.float64)).cpu().numpy()
    truth = split.y_test_raw.reshape(-1)
    pred = split.inverse_y(y_pred_norm).reshape(-1)
    signed_error = pred - truth
    color_max = max(float(np.max(np.abs(truth))), float(np.max(np.abs(pred))), 1.0e-12)
    error_max = max(float(np.max(np.abs(signed_error))), 1.0e-12)
    error_norm = TwoSlopeNorm(vmin=-error_max, vcenter=0.0, vmax=error_max)

    x_coord = split.x_test_raw[:, 0]
    y_coord = split.x_test_raw[:, 1] if split.x_test_raw.shape[1] > 1 else np.zeros_like(x_coord)
    context.output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.3), constrained_layout=True)
    panels = (
        ("truth", truth, "coolwarm", None, -color_max, color_max),
        ("prediction", pred, "coolwarm", None, -color_max, color_max),
        ("signed error", signed_error, "seismic", error_norm, None, None),
    )
    for ax, (title, values, cmap, norm, vmin, vmax) in zip(axes, panels, strict=True):
        scatter = ax.scatter(
            x_coord,
            y_coord,
            c=values,
            s=5,
            cmap=cmap,
            norm=norm,
            vmin=vmin,
            vmax=vmax,
            linewidths=0,
        )
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(
        f"{context.method}, n_train={context.refinement}, "
        f"trial={context.trial}, {context.metric_name}={context.metric_value:.3g}",
        fontsize=PLOT_FONT_SIZE,
    )
    fig.savefig(context.output_path, dpi=160)
    plt.close(fig)


def make_problem(
    *,
    name: str,
    sample: Callable[[int, np.random.Generator], np.ndarray],
    target: Callable[[np.ndarray], np.ndarray],
    prediction_plots: bool,
    x_transform: str | None = None,
    y_transform: str | None = "std",
) -> ArrayRegressionProblem:
    return ArrayRegressionProblem(
        name=name,
        methods=METHODS,
        sample=sample,
        target=target,
        fit_and_score=fit_and_score,
        fit_and_score_folds=fit_and_score_folds,
        tuning_spec=tuning_spec,
        metrics=(
            "error",
            "test_physical_rmse",
            "test_normalized_max_abs",
            "fit_seconds",
            "realized_bandwidth",
            "realized_ridge",
            "train_residual",
        ),
        primary_metric="error",
        prediction_plotter=plot_truth_vs_prediction if prediction_plots else None,
        x_transform=x_transform,
        y_transform=y_transform,
    )


def run_group_cases(
    groups: tuple[Group, ...],
    *,
    root: Path,
    args: argparse.Namespace,
    sample_for_case: Callable[[Case], Callable[[int, np.random.Generator], np.ndarray]],
    x_transform: str | None = None,
    y_transform: str | None = "std",
) -> None:
    for group in groups:
        for case in group.cases:
            config = ArrayRegressionStudyConfig(
                output_dir=root / group.slug / case.name,
                levels=args.levels,
                trials=args.trials,
                n_val=args.n_val,
                n_test=args.n_test,
                initial_budget=(args.initial_budget, args.initial_budget),
                refinement_budget=args.refinement_budget,
                refinement_strategy=("multi_start_nelder_mead" if args.refinement_budget else None),
                tuning_policy="per_trial",
                seed=args.seed,
                max_workers=args.max_workers,
                resampling_mode="nested-fixed-test",
                validation_mode="train-valid-count",
                validation_size=args.validation_size,
                pool_multiplier=2,
                restart=True,
                plot=not args.no_plot,
                prediction_plots=not args.no_prediction_plots,
            )
            problem = make_problem(
                name=f"{group.slug}_{case.name}",
                sample=sample_for_case(case),
                target=case.target,
                prediction_plots=not args.no_prediction_plots,
                x_transform=x_transform,
                y_transform=y_transform,
            )
            result = run_array_regression_study(
                problem,
                config,
                make_plot=None if args.no_plot else make_convergence_plot,
            )
            print(f"Wrote convergence artifacts to {Path(config.output_dir).resolve()}")
            if result.diagnostics:
                print(f"Diagnostics: {len(result.diagnostics)} advisory item(s)")


def read_error_curves(case_dir: Path) -> dict[str, dict[str, np.ndarray]]:
    rows = []
    for name in ("trial_statistics.csv", "convergence_summary.csv"):
        path = case_dir / name
        if path.exists():
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            break
    by_method: dict[str, list[tuple[float, float, float, float]]] = {}
    for row in rows:
        if row.get("metric") != "error":
            continue
        try:
            point = (
                float(row["refinement"]),
                float(row["median"]),
                float(row["q25"]),
                float(row["q75"]),
            )
        except (KeyError, ValueError):
            continue
        by_method.setdefault(str(row.get("method", "")), []).append(point)

    curves: dict[str, dict[str, np.ndarray]] = {}
    for method, points in by_method.items():
        ordered = sorted(points, key=lambda item: item[0])
        curves[method] = {
            "x": np.asarray([item[0] for item in ordered], dtype=float),
            "median": np.asarray([item[1] for item in ordered], dtype=float),
            "q25": np.asarray([item[2] for item in ordered], dtype=float),
            "q75": np.asarray([item[3] for item in ordered], dtype=float),
        }
    return curves


def convergence_y_limits(all_curves: list[dict[str, dict[str, np.ndarray]]]) -> tuple[float, float]:
    values: list[float] = []
    for curves in all_curves:
        for curve in curves.values():
            values.extend(curve["q25"].tolist())
            values.extend(curve["q75"].tolist())
    finite = np.asarray([value for value in values if np.isfinite(value) and value > 0.0])
    if finite.size == 0:
        return 1.0e-12, 1.0
    lower = 10.0 ** math.floor(math.log10(float(finite.min())))
    upper = 10.0 ** math.ceil(math.log10(float(finite.max())))
    return lower, upper if upper > lower else lower * 10.0


def plot_error_curves(
    ax: plt.Axes, curves: dict[str, dict[str, np.ndarray]], y_limits: tuple[float, float]
) -> None:
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_ylim(*y_limits)
    ax.grid(True, which="major", color="#d9d9d9", linewidth=0.45)
    ax.grid(True, which="minor", color="#eeeeee", linewidth=0.25)
    for method in METHODS:
        curve = curves.get(method)
        if curve is None:
            continue
        ax.plot(
            curve["x"],
            curve["median"],
            marker="o",
            markersize=3.2,
            linewidth=1.25,
            color=METHOD_COLORS[method],
            label="RBF" if method == RBF_METHOD else "DM",
        )
        ax.fill_between(
            curve["x"],
            curve["q25"],
            curve["q75"],
            color=METHOD_COLORS[method],
            alpha=0.16,
            linewidth=0.0,
        )


def draw_missing(ax: plt.Axes, message: str = "missing") -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, color="#666666")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#d0d0d0")


def target_grid_on_disk(
    target: Callable[[np.ndarray], np.ndarray], n: int = 120
) -> tuple[np.ndarray, np.ndarray]:
    radius = np.linspace(0.0, 1.0, n)
    theta = np.linspace(0.0, 2.0 * math.pi, n)
    rr, tt = np.meshgrid(radius, theta)
    points = np.column_stack((rr.ravel() * np.cos(tt.ravel()), rr.ravel() * np.sin(tt.ravel())))
    return points, target(points).reshape(-1)
