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
from common import (
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
from scipy.linalg import eigh

BASE_DIR = Path(__file__).resolve().parent
MAJOR_RADIUS = 2.0
MINOR_RADIUS = 1.0
COORDINATE_SCALE = MAJOR_RADIUS + MINOR_RADIUS
LAPLACE_MODES = ((1, 0), (3, 1), (6, 3))


@dataclass(frozen=True)
class DonutTorusGeometry:
    """The standard embedded torus with its uniform surface-area measure."""

    major_radius: float = MAJOR_RADIUS
    minor_radius: float = MINOR_RADIUS

    def points_from_angles(self, theta: np.ndarray | float, phi: np.ndarray | float) -> np.ndarray:
        theta_values = np.asarray(theta, dtype=float).reshape(-1)
        phi_values = np.asarray(phi, dtype=float).reshape(-1)
        ring_radius = self.major_radius + self.minor_radius * np.cos(theta_values)
        return np.column_stack(
            (
                ring_radius * np.cos(phi_values),
                ring_radius * np.sin(phi_values),
                self.minor_radius * np.sin(theta_values),
            )
        )

    def angles_from_points(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(points, dtype=float)
        if values.ndim != 2 or values.shape[1] < 3:
            raise ValueError("donut-torus points must have at least three coordinates")
        radius = np.hypot(values[:, 0], values[:, 1])
        theta = np.mod(
            np.arctan2(
                values[:, 2] / self.minor_radius,
                (radius - self.major_radius) / self.minor_radius,
            ),
            2.0 * math.pi,
        )
        phi = np.mod(np.arctan2(values[:, 1], values[:, 0]), 2.0 * math.pi)
        return theta, phi

    def sample_points(self, n_samples: int, rng: np.random.Generator) -> np.ndarray:
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
        return self.points_from_angles(theta, phi)


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


def _fourier_basis(theta: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray]:
    columns = [np.ones_like(theta)]
    derivatives = [np.zeros_like(theta)]
    for frequency in range(1, order + 1):
        columns.extend((np.cos(frequency * theta), np.sin(frequency * theta)))
        derivatives.extend(
            (-frequency * np.sin(frequency * theta), frequency * np.cos(frequency * theta))
        )
    return np.column_stack(columns), np.column_stack(derivatives)


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


def make_donut_torus_lb_target(
    geometry: DonutTorusGeometry, *, m: int, j: int
) -> Callable[[np.ndarray], np.ndarray]:
    """Make a real ``cos(m phi)`` Laplace--Beltrami eigenfunction target."""

    theta_mode = donut_torus_theta_mode(
        geometry.major_radius,
        geometry.minor_radius,
        m,
        j,
    )

    def target(points: np.ndarray) -> np.ndarray:
        theta, phi = geometry.angles_from_points(points)
        return (theta_mode.evaluate(theta) * np.cos(m * phi)).reshape(-1, 1)

    return target


def make_donut_torus_sample(
    geometry: DonutTorusGeometry, *, coordinate_scale: float
) -> Callable[[int, np.random.Generator], np.ndarray]:
    def sample(n_samples: int, rng: np.random.Generator) -> np.ndarray:
        return geometry.sample_points(n_samples, rng) / coordinate_scale

    return sample


def donut_torus_target_grid(
    geometry: DonutTorusGeometry,
    target: Callable[[np.ndarray], np.ndarray],
    *,
    n_theta: int = 128,
    n_phi: int = 128,
) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * math.pi, n_theta, endpoint=False)
    phi = np.linspace(0.0, 2.0 * math.pi, n_phi, endpoint=False)
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    points = geometry.points_from_angles(theta_grid.ravel(), phi_grid.ravel())
    return target(points).reshape(n_phi, n_theta)


GEOMETRY = DonutTorusGeometry()


def laplace_case(m: int, j: int) -> Case:
    return Case(
        name=f"lb_m{m}_j{j}",
        title=f"m={m}, j={j}",
        ambient_dim=3,
        target=wrap_scaled_target(
            make_donut_torus_lb_target(GEOMETRY, m=m, j=j),
            coordinate_scale=COORDINATE_SCALE,
        ),
    )


GROUPS = (
    Group(
        "Donut-torus Laplace--Beltrami eigenfunctions",
        "laplace_beltrami",
        tuple(laplace_case(m, j) for m, j in LAPLACE_MODES),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run donut-torus DM/RBF KRR convergence cases.")
    add_study_args(parser, GROUPS)
    return parser.parse_args()


def _grid_target(case: Case):
    return lambda points: case.target(points / COORDINATE_SCALE)


def write_group_plot(group: Group, root: Path, reports: Path) -> Path:
    case_dirs = [root / group.slug / case.name for case in group.cases]
    curves = [read_error_curves(path) for path in case_dirs]
    y_limits = convergence_y_limits(curves)
    targets = [donut_torus_target_grid(GEOMETRY, _grid_target(case)) for case in group.cases]
    color_limit = max(float(np.max(np.abs(values))) for values in targets)
    fig, axes = plt.subplots(2, len(group.cases), figsize=(4.1 * len(group.cases), 6.1))
    for col, (case, case_dir, curves_for_case, values) in enumerate(
        zip(group.cases, case_dirs, curves, targets, strict=True)
    ):
        target_axis = axes[0, col]
        target_axis.contourf(
            np.linspace(0.0, 2.0 * math.pi, values.shape[1], endpoint=False),
            np.linspace(0.0, 2.0 * math.pi, values.shape[0], endpoint=False),
            values,
            levels=np.linspace(-color_limit, color_limit, 21),
            cmap="coolwarm",
            extend="both",
        )
        target_axis.set(title=case.title, xlabel=r"$\theta$", ylabel=r"$\phi$")
        target_axis.set_xticks((0.0, math.pi, 2.0 * math.pi), ("0", r"$\pi$", r"$2\pi$"))
        target_axis.set_yticks((0.0, math.pi, 2.0 * math.pi), ("0", r"$\pi$", r"$2\pi$"))

        axis = axes[1, col]
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
    axes[1, 0].set_ylabel("RMSE")
    handles, labels = axes[1, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="center right", bbox_to_anchor=(0.995, 0.27), frameon=False)
    fig.suptitle(group.name)
    fig.tight_layout(rect=(0.0, 0.0, 0.94, 0.95))
    reports.mkdir(parents=True, exist_ok=True)
    path = reports / "donut_torus_laplace_beltrami.png"
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
            GEOMETRY, coordinate_scale=COORDINATE_SCALE
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
