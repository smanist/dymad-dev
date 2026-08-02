"""Semi-torus KRR convergence with its own LB mode in each harmonic embedding.

This study is deliberately separate from ``semi_torus.py``'s harmonic-ambient
group.  Each ambient dimension changes the induced metric, so this script
solves the requested Neumann ``(m, j) = (1, 1)`` mode against that particular
metric.  The regression problem applies no input or output transforms: points
are sampled uniformly in each harmonic embedding's induced area measure,
then divided only by ``R + r`` before fitting.  Every target is scaled to unit
maximum magnitude before sampling.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
from krr_common import (
    Case,
    Group,
    add_study_args,
    output_root,
    report_root,
    run_group_cases,
    selected_groups,
    wrap_scaled_target,
)
from scipy.linalg import eigh
from semi_torus import GEOMETRY, write_group_plot

BASE_DIR = Path(__file__).resolve().parent
AMBIENT_DIMS = (3, 7, 11, 15)
AZIMUTHAL_MODE = 1
THETA_MODE = 1
FOURIER_ORDER = 36
QUADRATURE_SIZE = 8_192
NORMALIZATION_GRID_SIZE = 8_192
ALIGNMENT_GRID_SIZE = 8_192
INPUT_SCALE = GEOMETRY.major_radius + 1.0


@dataclass(frozen=True)
class MultiharmonicThetaMode:
    """Theta factor of an LB eigenfunction for one harmonic embedding."""

    coefficients: np.ndarray
    eigenvalue: float
    fourier_order: int

    def evaluate(self, theta: np.ndarray | float) -> np.ndarray:
        values = np.asarray(theta, dtype=float)
        basis, _ = _fourier_basis(np.mod(values.reshape(-1), 2.0 * math.pi), self.fourier_order)
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


def multiharmonic_metric_terms(
    theta: np.ndarray,
    *,
    major_radius: float,
    ambient_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(g_theta, g_phi)`` for the semi-torus harmonic embedding."""

    if ambient_dim < 3 or ambient_dim % 2 == 0:
        raise ValueError("multiharmonic ambient_dim must be an odd integer at least 3")
    harmonic_count = (ambient_dim - 1) // 2
    harmonic_sum = sum(1.0 / (order * order) for order in range(1, harmonic_count + 1))
    ring_radius = major_radius + np.cos(theta)
    g_theta = harmonic_sum * np.sin(theta) ** 2 + np.cos(theta) ** 2 / harmonic_sum
    g_phi = harmonic_count * ring_radius * ring_radius
    return g_theta, g_phi


def multiharmonic_area_density(
    theta: np.ndarray,
    *,
    major_radius: float,
    ambient_dim: int,
) -> np.ndarray:
    """Return the harmonic embedding's area density in ``(theta, phi)`` coordinates."""

    g_theta, g_phi = multiharmonic_metric_terms(
        np.asarray(theta, dtype=float),
        major_radius=major_radius,
        ambient_dim=ambient_dim,
    )
    return np.sqrt(g_theta * g_phi)


def make_multiharmonic_area_sample(
    ambient_dim: int,
) -> Callable[[int, np.random.Generator], np.ndarray]:
    """Sample uniformly with respect to one harmonic embedding's surface area."""

    harmonic_count = (ambient_dim - 1) // 2
    harmonic_sum = sum(1.0 / (order * order) for order in range(1, harmonic_count + 1))
    # ``rho`` and ``sqrt(g_theta)`` achieve their maxima at different angles
    # once more than one harmonic is present.  The product of their separate
    # maxima is a simple global rejection envelope.
    density_bound = (GEOMETRY.major_radius + 1.0) * math.sqrt(harmonic_count * harmonic_sum)

    def sample(n_samples: int, rng: np.random.Generator) -> np.ndarray:
        theta_chunks: list[np.ndarray] = []
        accepted = 0
        while accepted < n_samples:
            proposal_count = max(n_samples - accepted, 32)
            theta_proposal = 2.0 * math.pi * rng.random(proposal_count)
            density = multiharmonic_area_density(
                theta_proposal,
                major_radius=GEOMETRY.major_radius,
                ambient_dim=ambient_dim,
            )
            keep = rng.random(proposal_count) < density / density_bound
            if np.any(keep):
                theta_chunks.append(theta_proposal[keep])
                accepted += int(np.count_nonzero(keep))
        theta = np.concatenate(theta_chunks)[:n_samples]
        phi = math.pi * rng.random(n_samples)
        points = GEOMETRY.points_from_angles(
            theta,
            phi,
            ambient_dim=ambient_dim,
            embedding="harmonic",
        )
        return points / INPUT_SCALE

    return sample


@cache
def multiharmonic_theta_mode(
    major_radius: float,
    m: int,
    j: int,
    ambient_dim: int,
    fourier_order: int = FOURIER_ORDER,
    quadrature_size: int = QUADRATURE_SIZE,
) -> MultiharmonicThetaMode:
    """Solve theta mode ``j`` in Neumann azimuthal sector ``m`` for one embedding."""

    if m < 0 or j < 0:
        raise ValueError("m and j must be nonnegative")
    if quadrature_size < 2 * (2 * fourier_order + 1):
        raise ValueError("quadrature_size is too small for the requested Fourier basis")
    theta = np.linspace(0.0, 2.0 * math.pi, quadrature_size, endpoint=False)
    step = 2.0 * math.pi / quadrature_size
    basis, derivative = _fourier_basis(theta, fourier_order)
    g_theta, g_phi = multiharmonic_metric_terms(
        theta,
        major_radius=major_radius,
        ambient_dim=ambient_dim,
    )
    area = np.sqrt(g_theta * g_phi)
    mass = step * (basis.T @ (area[:, None] * basis))
    stiffness = step * (
        derivative.T @ ((area / g_theta)[:, None] * derivative)
        + basis.T @ ((m * m * area / g_phi)[:, None] * basis)
    )
    eigenvalues, eigenvectors = eigh(stiffness, mass, check_finite=False)
    if j >= eigenvalues.size:
        raise ValueError(f"theta mode j={j} is unavailable for Fourier order {fourier_order}")
    coefficients = np.asarray(eigenvectors[:, j], dtype=float)
    values = basis @ coefficients
    if values[int(np.argmax(np.abs(values)))] < 0.0:
        coefficients = -coefficients
    return MultiharmonicThetaMode(coefficients, float(eigenvalues[j]), fourier_order)


@cache
def aligned_multiharmonic_theta_mode(
    major_radius: float,
    m: int,
    j: int,
    ambient_dim: int,
    reference_ambient_dim: int = 3,
) -> MultiharmonicThetaMode:
    """Orient a theta mode to have nonnegative correlation with a reference mode."""

    mode = multiharmonic_theta_mode(major_radius, m, j, ambient_dim)
    if ambient_dim == reference_ambient_dim:
        return mode
    reference = multiharmonic_theta_mode(major_radius, m, j, reference_ambient_dim)
    theta = np.linspace(0.0, 2.0 * math.pi, ALIGNMENT_GRID_SIZE, endpoint=False)
    correlation = float(np.dot(mode.evaluate(theta), reference.evaluate(theta)))
    if correlation >= 0.0:
        return mode
    return MultiharmonicThetaMode(-mode.coefficients, mode.eigenvalue, mode.fourier_order)


@cache
def multiharmonic_target_scale(
    major_radius: float,
    m: int,
    j: int,
    ambient_dim: int,
) -> float:
    """Return the deterministic scale that makes the target's max magnitude one."""

    mode = multiharmonic_theta_mode(major_radius, m, j, ambient_dim)
    theta = np.linspace(0.0, 2.0 * math.pi, NORMALIZATION_GRID_SIZE, endpoint=False)
    amplitude = float(np.max(np.abs(mode.evaluate(theta))))
    if not math.isfinite(amplitude) or amplitude <= 0.0:
        raise RuntimeError("cannot normalize a zero or non-finite eigenfunction")
    return 1.0 / amplitude


def make_multiharmonic_neumann_lb_target(
    *,
    m: int,
    j: int,
    ambient_dim: int,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return the max-normalized real Neumann LB mode for one embedding."""

    mode = multiharmonic_theta_mode(GEOMETRY.major_radius, m, j, ambient_dim)
    scale = multiharmonic_target_scale(GEOMETRY.major_radius, m, j, ambient_dim)

    def target(points: np.ndarray) -> np.ndarray:
        theta, phi = GEOMETRY.angles_from_points(points, embedding="harmonic")
        return (scale * mode.evaluate(theta) * np.cos(m * phi)).reshape(-1, 1)

    return target


def make_multiharmonic_dirichlet_lb_target(
    *,
    m: int,
    j: int,
    source_ambient_dim: int,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return an aligned, max-normalized real Dirichlet LB target.

    ``source_ambient_dim`` controls the metric used to solve the theta mode;
    points may be represented in any harmonic ambient dimension.
    """

    mode = aligned_multiharmonic_theta_mode(
        GEOMETRY.major_radius,
        m,
        j,
        source_ambient_dim,
    )
    scale = multiharmonic_target_scale(GEOMETRY.major_radius, m, j, source_ambient_dim)

    def target(points: np.ndarray) -> np.ndarray:
        theta, phi = GEOMETRY.angles_from_points(points, embedding="harmonic")
        return (scale * mode.evaluate(theta) * np.sin(m * phi)).reshape(-1, 1)

    return target


def multiharmonic_case(ambient_dim: int) -> Case:
    mode = multiharmonic_theta_mode(
        GEOMETRY.major_radius,
        AZIMUTHAL_MODE,
        THETA_MODE,
        ambient_dim,
    )
    return Case(
        name=f"neumann_m{AZIMUTHAL_MODE}_j{THETA_MODE}_d{ambient_dim}",
        title=f"d={ambient_dim}, λ={mode.eigenvalue:.4f}",
        ambient_dim=ambient_dim,
        embedding="harmonic",
        target=wrap_scaled_target(
            make_multiharmonic_neumann_lb_target(
                m=AZIMUTHAL_MODE,
                j=THETA_MODE,
                ambient_dim=ambient_dim,
            ),
            coordinate_scale=INPUT_SCALE,
        ),
    )


GROUPS = (
    Group(
        "Semi-torus harmonic embedding: own Neumann m=1, j=1 LB mode by ambient dimension",
        "semi_torus_multiharmonic_neumann_lb",
        tuple(multiharmonic_case(ambient_dim) for ambient_dim in AMBIENT_DIMS),
        show_targets=True,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run semi-torus KRR on dimension-specific harmonic Neumann LB modes."
    )
    add_study_args(parser, GROUPS)
    parser.add_argument(
        "--ambient-dims",
        type=int,
        nargs="+",
        choices=AMBIENT_DIMS,
        help="run only the requested harmonic embedding dimensions",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    groups = selected_groups(GROUPS, args.groups)
    if args.ambient_dims is not None:
        selected_dims = set(args.ambient_dims)
        groups = tuple(
            Group(
                group.name,
                group.slug,
                tuple(case for case in group.cases if case.ambient_dim in selected_dims),
                show_targets=group.show_targets,
            )
            for group in groups
        )
    root = output_root(args.workdir, BASE_DIR)
    run_group_cases(
        groups,
        root=root,
        args=args,
        sample_for_case=lambda case: make_multiharmonic_area_sample(case.ambient_dim),
        x_transform=None,
        y_transform=None,
    )
    if not args.no_plot:
        reports = report_root(args.workdir, BASE_DIR)
        for group in groups:
            path = write_group_plot(group, root, reports)
            print(f"Wrote report to {path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
