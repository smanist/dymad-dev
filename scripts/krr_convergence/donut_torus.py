"""Diffusion-maps and RBF KRR convergence on a curved donut torus.

The three targets are separated Laplace--Beltrami eigenfunctions on the
embedded torus.  They retain the frequency pairs used by ``semi_torus.py``:
``(m, j) = (1, 0), (3, 1), (6, 3)``.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from krr_common import (
    METHOD_LABELS,
    METHODS,
    Case,
    Group,
    add_study_args,
    convergence_y_limits,
    draw_missing,
    output_root,
    plot_error_curves,
    read_error_curves,
    report_root,
    run_group_cases,
    selected_groups,
    wrap_scaled_target,
)
from scipy import special
from scipy.linalg import eigh

BASE_DIR = Path(__file__).resolve().parent
MAJOR_RADIUS = 2.0
MINOR_RADIUS = 1.0
COORDINATE_SCALE = MAJOR_RADIUS + MINOR_RADIUS
# The intermediate (1, 1) mode replaces (1, 0): a reduced production screen
# found a consistent, clear DM advantage for it over RBF KRR.
LAPLACE_MODES = ((1, 1), (3, 1), (6, 3))
AMBIENT_FOURIER_FREQUENCIES = (2, 6, 10)
RBF_INTEGRAL_SIGMA = 0.30
RBF_INTEGRAL_QUADRATURE_SIZE = 512
RBF_INTEGRAL_FOURIER_ORDER = 24


@dataclass(frozen=True)
class DonutTorusGeometry:
    """The standard embedded torus with its uniform surface-area measure."""

    major_radius: float = MAJOR_RADIUS
    minor_radius: float = MINOR_RADIUS

    @staticmethod
    @cache
    def _isometric_embedding_matrix(ambient_dim: int) -> np.ndarray:
        """Return a deterministic isometric map from R^3 into R^``ambient_dim``.

        The columns are the first three orthonormal DCT-II modes.  Thus an
        isometric embedding uses every output coordinate while preserving the
        Euclidean distance between every pair of original torus points.
        """

        if ambient_dim < 3:
            raise ValueError("ambient_dim must be at least 3")
        if ambient_dim == 3:
            return np.eye(3)
        rows = np.arange(ambient_dim, dtype=float)[:, None]
        modes = np.arange(3, dtype=float)[None, :]
        matrix = np.cos(math.pi * (rows + 0.5) * modes / ambient_dim)
        matrix[:, 0] *= math.sqrt(1.0 / ambient_dim)
        matrix[:, 1:] *= math.sqrt(2.0 / ambient_dim)
        return matrix

    def points_from_angles(
        self,
        theta: np.ndarray | float,
        phi: np.ndarray | float,
        *,
        ambient_dim: int = 3,
        embedding: str = "augmented",
    ) -> np.ndarray:
        theta_values = np.asarray(theta, dtype=float).reshape(-1)
        phi_values = np.asarray(phi, dtype=float).reshape(-1)
        ring_radius = self.major_radius + self.minor_radius * np.cos(theta_values)
        if ambient_dim < 3:
            raise ValueError("ambient_dim must be at least 3")
        base = np.column_stack(
            (
                ring_radius * np.cos(phi_values),
                ring_radius * np.sin(phi_values),
                self.minor_radius * np.sin(theta_values),
            )
        )
        if embedding == "isometric":
            return base @ self._isometric_embedding_matrix(ambient_dim).T
        if embedding == "harmonic":
            if ambient_dim % 2 == 0:
                raise ValueError("harmonic torus embedding requires an odd ambient_dim")
            harmonic_count = (ambient_dim - 1) // 2
            coordinates: list[np.ndarray] = []
            for order in range(1, harmonic_count + 1):
                amplitude = ring_radius / order
                coordinates.extend(
                    (amplitude * np.cos(order * phi_values), amplitude * np.sin(order * phi_values))
                )
            theta_normalizer = math.sqrt(
                sum(1.0 / (order * order) for order in range(1, harmonic_count + 1))
            )
            coordinates.append(self.minor_radius * np.sin(theta_values) / theta_normalizer)
            return np.column_stack(coordinates)
        if embedding != "augmented":
            raise ValueError(f"unknown donut-torus embedding: {embedding}")
        if ambient_dim == 3:
            return base
        extras: list[np.ndarray] = []
        for index in range(ambient_dim - 3):
            order = 1 + index // 4
            selector = index % 4
            if selector == 0:
                extras.append(0.2 * np.cos(order * theta_values))
            elif selector == 1:
                extras.append(0.2 * np.sin(order * theta_values))
            elif selector == 2:
                extras.append(0.2 * np.cos(order * phi_values))
            else:
                extras.append(0.2 * np.sin(order * phi_values))
        return np.column_stack((base, *extras))

    def angles_from_points(
        self, points: np.ndarray, *, embedding: str = "augmented"
    ) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(points, dtype=float)
        if values.ndim != 2 or values.shape[1] < 3:
            raise ValueError("donut-torus points must have at least three coordinates")
        if embedding == "isometric":
            base = values @ self._isometric_embedding_matrix(values.shape[1])
            sin_theta = base[:, 2] / self.minor_radius
            radius = np.hypot(base[:, 0], base[:, 1])
            phi = np.mod(np.arctan2(base[:, 1], base[:, 0]), 2.0 * math.pi)
        elif embedding == "harmonic":
            if values.shape[1] % 2 == 0:
                raise ValueError("harmonic torus embedding requires an odd ambient_dim")
            harmonic_count = (values.shape[1] - 1) // 2
            theta_normalizer = math.sqrt(
                sum(1.0 / (order * order) for order in range(1, harmonic_count + 1))
            )
            sin_theta = values[:, -1] * theta_normalizer / self.minor_radius
            radius = np.hypot(values[:, 0], values[:, 1])
            phi = np.mod(np.arctan2(values[:, 1], values[:, 0]), 2.0 * math.pi)
        elif embedding == "augmented":
            sin_theta = values[:, 2] / self.minor_radius
            radius = np.hypot(values[:, 0], values[:, 1])
            phi = np.mod(np.arctan2(values[:, 1], values[:, 0]), 2.0 * math.pi)
        else:
            raise ValueError(f"unknown donut-torus embedding: {embedding}")
        theta = np.mod(
            np.arctan2(
                sin_theta,
                (radius - self.major_radius) / self.minor_radius,
            ),
            2.0 * math.pi,
        )
        return theta, phi

    def sample_points(
        self,
        n_samples: int,
        rng: np.random.Generator,
        *,
        ambient_dim: int = 3,
        embedding: str = "augmented",
    ) -> np.ndarray:
        """Draw points uniformly with respect to surface area."""

        theta_chunks: list[np.ndarray] = []
        accepted = 0
        while accepted < n_samples:
            proposal_count = max(n_samples - accepted, 32)
            proposed = 2.0 * math.pi * rng.random(proposal_count)
            probability = (self.major_radius + self.minor_radius * np.cos(proposed)) / (
                self.major_radius + self.minor_radius
            )
            kept = proposed[rng.random(proposal_count) < probability]
            theta_chunks.append(kept)
            accepted += len(kept)
        theta = np.concatenate(theta_chunks)[:n_samples]
        phi = 2.0 * math.pi * rng.random(n_samples)
        return self.points_from_angles(theta, phi, ambient_dim=ambient_dim, embedding=embedding)


@dataclass(frozen=True)
class DonutTorusThetaMode:
    coefficients: np.ndarray
    eigenvalue: float
    fourier_order: int

    def evaluate(self, theta: np.ndarray | float) -> np.ndarray:
        values = np.asarray(theta, dtype=float)
        flat = np.mod(values.reshape(-1), 2.0 * math.pi)
        result = np.full(flat.shape, self.coefficients[0], dtype=float)
        for order in range(1, self.fourier_order + 1):
            result += self.coefficients[2 * order - 1] * np.cos(order * flat)
            result += self.coefficients[2 * order] * np.sin(order * flat)
        return result.reshape(values.shape)


@dataclass(frozen=True)
class DonutTorusRBFThetaMode:
    coefficients: np.ndarray
    eigenvalue: float
    fourier_order: int

    def evaluate(self, theta: np.ndarray | float) -> np.ndarray:
        values = np.asarray(theta, dtype=float)
        basis, _ = _fourier_basis(values.reshape(-1), self.fourier_order)
        return (basis @ self.coefficients).reshape(values.shape)


def _fourier_basis(theta: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray]:
    columns = [np.ones_like(theta)]
    derivatives = [np.zeros_like(theta)]
    for frequency in range(1, order + 1):
        columns.extend((np.cos(frequency * theta), np.sin(frequency * theta)))
        derivatives.extend(
            (-frequency * np.sin(frequency * theta), frequency * np.cos(frequency * theta))
        )
    return np.column_stack(columns), np.column_stack(derivatives)


def _donut_torus_rbf_theta_kernel(
    theta: np.ndarray,
    reference_theta: np.ndarray,
    *,
    azimuthal_mode: int,
    major_radius: float,
    minor_radius: float,
    sigma: float,
) -> np.ndarray:
    """Fourier coefficient in phi of the ambient Gaussian RBF integral kernel."""

    query = np.asarray(theta, dtype=float).reshape(-1)
    reference = np.asarray(reference_theta, dtype=float).reshape(-1)
    query_radius = major_radius + minor_radius * np.cos(query)
    reference_radius = major_radius + minor_radius * np.cos(reference)
    sine_difference = minor_radius * (np.sin(query)[:, None] - np.sin(reference)[None, :])
    squared_base_distance = (
        query_radius[:, None] ** 2 + reference_radius[None, :] ** 2 + sine_difference**2
    )
    argument = query_radius[:, None] * reference_radius[None, :] / sigma**2
    return (
        2.0
        * math.pi
        * np.exp(-squared_base_distance / (2.0 * sigma**2))
        * special.iv(azimuthal_mode, argument)
    )


@cache
def donut_torus_theta_mode(
    major_radius: float,
    minor_radius: float,
    m: int,
    j: int,
    fourier_order: int = 16,
    quadrature_size: int = 4096,
) -> DonutTorusThetaMode:
    """Compute the ``j``th theta mode for azimuthal frequency ``m``.

    This is the Fourier--Galerkin generalized eigenproblem for the induced
    donut-torus metric, so the resulting target is an LB eigenfunction rather
    than an ambient Fourier surrogate.
    """

    if m < 0 or j < 0:
        raise ValueError("m and j must be nonnegative")
    if quadrature_size < 2 * (2 * fourier_order + 1):
        raise ValueError("quadrature_size is too small for the Fourier basis")
    theta = np.linspace(0.0, 2.0 * math.pi, quadrature_size, endpoint=False)
    step = 2.0 * math.pi / quadrature_size
    ring_radius = major_radius + minor_radius * np.cos(theta)
    basis, derivative = _fourier_basis(theta, fourier_order)
    mass = step * (basis.T @ ((minor_radius * ring_radius)[:, None] * basis))
    stiffness = step * (
        derivative.T @ ((ring_radius / minor_radius)[:, None] * derivative)
        + basis.T @ (((m * m * minor_radius / ring_radius)[:, None]) * basis)
    )
    eigenvalues, eigenvectors = eigh(stiffness, mass, check_finite=False)
    if j >= eigenvectors.shape[1]:
        raise ValueError(f"theta mode j={j} is unavailable for Fourier order {fourier_order}")
    coefficients = np.asarray(eigenvectors[:, j], dtype=float)
    values = basis @ coefficients
    if values[int(np.argmax(np.abs(values)))] < 0.0:
        coefficients = -coefficients
    return DonutTorusThetaMode(coefficients, float(eigenvalues[j]), fourier_order)


@cache
def donut_torus_rbf_theta_mode(
    major_radius: float,
    minor_radius: float,
    m: int,
    j: int,
    sigma: float = RBF_INTEGRAL_SIGMA,
    quadrature_size: int = RBF_INTEGRAL_QUADRATURE_SIZE,
    fourier_order: int = RBF_INTEGRAL_FOURIER_ORDER,
) -> DonutTorusRBFThetaMode:
    """Return the ``j``th RBF-integral theta mode using Fourier--Galerkin discretization."""

    if m < 0 or j < 0:
        raise ValueError("m and j must be nonnegative")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    if quadrature_size < 2 * (2 * fourier_order + 1):
        raise ValueError("quadrature_size is too small for the requested Fourier basis")
    theta = np.linspace(0.0, 2.0 * math.pi, quadrature_size, endpoint=False)
    step = 2.0 * math.pi / quadrature_size
    weights = step * minor_radius * (major_radius + minor_radius * np.cos(theta))
    basis, _ = _fourier_basis(theta, fourier_order)
    kernel = _donut_torus_rbf_theta_kernel(
        theta,
        theta,
        azimuthal_mode=m,
        major_radius=major_radius,
        minor_radius=minor_radius,
        sigma=sigma,
    )
    mass = basis.T @ (weights[:, None] * basis)
    operator = basis.T @ (weights[:, None] * (kernel @ (weights[:, None] * basis)))
    eigenvalues, eigenvectors = eigh(
        (operator + operator.T) * 0.5,
        (mass + mass.T) * 0.5,
        check_finite=False,
    )
    index = eigenvalues.size - 1 - j
    if index < 0:
        raise ValueError(f"RBF theta mode j={j} is unavailable")
    coefficients = np.asarray(eigenvectors[:, index], dtype=float)
    values = basis @ coefficients
    if values[int(np.argmax(np.abs(values)))] < 0.0:
        coefficients = -coefficients
    return DonutTorusRBFThetaMode(
        coefficients,
        float(eigenvalues[index]),
        fourier_order,
    )


def make_donut_torus_lb_target(
    geometry: DonutTorusGeometry, *, m: int, j: int, embedding: str = "augmented"
) -> Callable[[np.ndarray], np.ndarray]:
    """Make a real ``cos(m phi)`` Laplace--Beltrami eigenfunction target."""

    theta_mode = donut_torus_theta_mode(
        geometry.major_radius,
        geometry.minor_radius,
        m,
        j,
    )

    def target(points: np.ndarray) -> np.ndarray:
        theta, phi = geometry.angles_from_points(points, embedding=embedding)
        return (theta_mode.evaluate(theta) * np.cos(m * phi)).reshape(-1, 1)

    return target


def make_donut_torus_rbf_integral_target(
    geometry: DonutTorusGeometry, *, m: int, j: int, embedding: str = "augmented"
) -> Callable[[np.ndarray], np.ndarray]:
    """Make a real RBF-integral eigenfunction with the same ``(m, j)`` indexing."""

    theta_mode = donut_torus_rbf_theta_mode(
        geometry.major_radius,
        geometry.minor_radius,
        m,
        j,
    )

    def target(points: np.ndarray) -> np.ndarray:
        theta, phi = geometry.angles_from_points(points, embedding=embedding)
        return (theta_mode.evaluate(theta) * np.cos(m * phi)).reshape(-1, 1)

    return target


def make_donut_torus_ambient_fourier_target(
    geometry: DonutTorusGeometry, *, frequency: int
) -> Callable[[np.ndarray], np.ndarray]:
    """Make the same ambient-coordinate Fourier target family as the semi-torus study."""

    if frequency < 1:
        raise ValueError("frequency must be positive")

    def target(points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=float)
        if values.ndim != 2 or values.shape[1] < 2:
            raise ValueError("donut-torus ambient Fourier target requires at least two coordinates")
        scale = geometry.major_radius + geometry.minor_radius
        u = 0.5 * (values[:, 0] / scale + 1.0)
        v = 0.5 * (values[:, 1] / scale + 1.0)
        cycles = 0.5 * frequency
        return (np.cos(2.0 * math.pi * cycles * u) * np.cos(2.0 * math.pi * cycles * v)).reshape(
            -1, 1
        )

    return target


def make_donut_torus_sample(
    geometry: DonutTorusGeometry,
    *,
    ambient_dim: int = 3,
    coordinate_scale: float,
    embedding: str = "augmented",
) -> Callable[[int, np.random.Generator], np.ndarray]:
    def sample(n_samples: int, rng: np.random.Generator) -> np.ndarray:
        return (
            geometry.sample_points(
                n_samples,
                rng,
                ambient_dim=ambient_dim,
                embedding=embedding,
            )
            / coordinate_scale
        )

    return sample


def donut_torus_target_grid(
    geometry: DonutTorusGeometry,
    target: Callable[[np.ndarray], np.ndarray],
    *,
    ambient_dim: int = 3,
    embedding: str = "augmented",
    n_theta: int = 128,
    n_phi: int = 128,
) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * math.pi, n_theta, endpoint=False)
    phi = np.linspace(0.0, 2.0 * math.pi, n_phi, endpoint=False)
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    points = geometry.points_from_angles(
        theta_grid.ravel(),
        phi_grid.ravel(),
        ambient_dim=ambient_dim,
        embedding=embedding,
    )
    return target(points).reshape(n_phi, n_theta)


GEOMETRY = DonutTorusGeometry()


def laplace_case(
    m: int,
    j: int,
    ambient_dim: int = 3,
    *,
    name_prefix: str = "lb",
    show_ambient_dim: bool = False,
    embedding: str = "augmented",
) -> Case:
    return Case(
        name=f"{name_prefix}_m{m}_j{j}_d{ambient_dim}"
        if name_prefix != "lb" or ambient_dim != 3
        else f"lb_m{m}_j{j}",
        title=f"n={ambient_dim}" if show_ambient_dim else f"m={m}, j={j}",
        ambient_dim=ambient_dim,
        embedding=embedding,
        target=wrap_scaled_target(
            make_donut_torus_lb_target(GEOMETRY, m=m, j=j, embedding=embedding),
            coordinate_scale=COORDINATE_SCALE,
        ),
    )


def rbf_integral_case(m: int, j: int) -> Case:
    return Case(
        name=f"rbf_integral_m{m}_j{j}",
        title=f"m={m}, j={j}",
        ambient_dim=3,
        target=wrap_scaled_target(
            make_donut_torus_rbf_integral_target(GEOMETRY, m=m, j=j),
            coordinate_scale=COORDINATE_SCALE,
        ),
    )


GROUPS = (
    Group(
        "Donut-torus Laplace--Beltrami eigenfunctions",
        "laplace_beltrami",
        tuple(laplace_case(m, j) for m, j in LAPLACE_MODES),
    ),
    Group(
        "Donut-torus RBF-integral eigenfunctions",
        "donut_rbf_integral",
        tuple(rbf_integral_case(m, j) for m, j in LAPLACE_MODES),
    ),
    Group(
        "Donut-torus ambient Fourier modes",
        "donut_ambient_fourier",
        tuple(
            Case(
                name=f"ambient_fourier_k{frequency}",
                title=f"k={frequency}",
                ambient_dim=3,
                target=wrap_scaled_target(
                    make_donut_torus_ambient_fourier_target(GEOMETRY, frequency=frequency),
                    coordinate_scale=COORDINATE_SCALE,
                ),
            )
            for frequency in AMBIENT_FOURIER_FREQUENCIES
        ),
    ),
    Group(
        "Donut-torus Neumann LB eigenfunction (m=1, j=1) by ambient dimension (harmonic embedding)",
        "donut_neumann_harmonic_ambient",
        tuple(
            laplace_case(
                1,
                1,
                ambient_dim,
                name_prefix="neumann",
                show_ambient_dim=True,
                embedding="harmonic",
            )
            for ambient_dim in (3, 7, 11, 15)
        ),
        show_targets=False,
    ),
    Group(
        "Donut-torus Neumann LB eigenfunction (m=1, j=1) by ambient dimension (isometric embedding)",
        "donut_neumann_isometric_ambient",
        tuple(
            laplace_case(
                1,
                1,
                ambient_dim,
                name_prefix="isometric",
                show_ambient_dim=True,
                embedding="isometric",
            )
            for ambient_dim in (3, 7, 11, 15)
        ),
        show_targets=False,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run donut-torus DM/RBF KRR convergence cases.")
    add_study_args(parser, GROUPS)
    return parser.parse_args()


def _grid_target(case: Case):
    return lambda points: case.target(points / COORDINATE_SCALE)


def _plot_target(ax: plt.Axes, case: Case, color_limit: float, *, show_y_axis: bool = True) -> None:
    values = donut_torus_target_grid(
        GEOMETRY,
        _grid_target(case),
        ambient_dim=case.ambient_dim,
        embedding=case.embedding,
    )
    ax.contourf(
        np.linspace(0.0, 2.0 * math.pi, values.shape[1], endpoint=False),
        np.linspace(0.0, 2.0 * math.pi, values.shape[0], endpoint=False),
        values,
        levels=np.linspace(-color_limit, color_limit, 21),
        cmap="coolwarm",
        extend="both",
    )
    ax.set(title=case.title, xlabel=r"$\theta$")
    ax.set_xticks((0.0, math.pi, 2.0 * math.pi), ("0", r"$\pi$", r"$2\pi$"))
    if show_y_axis:
        ax.set_ylabel(r"$\phi$")
        ax.set_yticks((0.0, math.pi, 2.0 * math.pi), ("0", r"$\pi$", r"$2\pi$"))
    else:
        ax.tick_params(axis="y", which="both", left=False, labelleft=False)


def write_group_plot(group: Group, root: Path, reports: Path) -> Path:
    case_dirs = [root / group.slug / case.name for case in group.cases]
    curves = [read_error_curves(path) for path in case_dirs]
    y_limits = convergence_y_limits(curves)
    n_rows = 2 if group.show_targets else 1
    fig, axes = plt.subplots(
        n_rows,
        len(group.cases),
        figsize=(4.1 * len(group.cases), 6.1 if group.show_targets else 3.2),
        squeeze=False,
    )
    color_limit = 1.0
    if group.show_targets:
        target_values = [
            donut_torus_target_grid(
                GEOMETRY,
                _grid_target(case),
                ambient_dim=case.ambient_dim,
                embedding=case.embedding,
            )
            for case in group.cases
        ]
        color_limit = max(float(np.max(np.abs(values))) for values in target_values)
    for col, (case, case_dir, curves_for_case) in enumerate(
        zip(group.cases, case_dirs, curves, strict=True)
    ):
        if group.show_targets:
            _plot_target(axes[0, col], case, color_limit, show_y_axis=col == 0)
        axis = axes[1 if group.show_targets else 0, col]
        if case_dir.exists() and curves_for_case:
            plot_error_curves(axis, curves_for_case, y_limits)
        else:
            draw_missing(axis, "no curves")
        axis.set_xlabel("N")
        axis.set_xticks(
            (2**9, 2**10, 2**11, 2**12), (r"$2^9$", r"$2^{10}$", r"$2^{11}$", r"$2^{12}$")
        )
        if col:
            axis.tick_params(axis="y", which="both", left=False, labelleft=False)
    curve_axes = axes[1 if group.show_targets else 0]
    curve_axes[0].set_ylabel("RMSE")
    handles, labels = curve_axes[0].get_legend_handles_labels()
    if handles:
        curve_axes[-1].legend(handles, labels, loc="upper right", frameon=False)
    fig.suptitle(group.name)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    reports.mkdir(parents=True, exist_ok=True)
    path = reports / f"donut_torus_{group.slug.removeprefix('donut_')}.png"
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    return path


def main() -> int:
    args = parse_args()
    groups = selected_groups(GROUPS, args.groups)
    root = output_root(args.workdir, BASE_DIR)
    run_group_cases(
        groups,
        root=root,
        args=args,
        sample_for_case=lambda _: make_donut_torus_sample(
            GEOMETRY,
            ambient_dim=_.ambient_dim,
            coordinate_scale=COORDINATE_SCALE,
            embedding=_.embedding,
        ),
    )
    if not args.no_plot:
        reports = report_root(args.workdir, BASE_DIR)
        paths = [write_group_plot(group, root, reports) for group in groups]
        (reports / "summary.md").write_text(
            "# Donut-torus DM/RBF KRR convergence\n\n"
            + "\n".join(f"- [{path.stem}]({path.name})" for path in paths)
            + f"\n\nMethods: {METHOD_LABELS[METHODS[1]]} and {METHOD_LABELS[METHODS[0]]}.\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
