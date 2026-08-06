"""Exploratory DM/RBF KRR study with dimension-specific multiharmonic LB targets.

Unlike ``donut_torus.py``'s harmonic-ambient group, this script solves the
``(m, j) = (1, 1)`` Laplace--Beltrami mode for each multiharmonic embedding's
own induced metric.  It deliberately retains the standard torus sampler so
the experiment isolates the effect of changing both the representation and
the target geometry while preserving the existing intrinsic sampling method.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
from donut_torus import (
    COORDINATE_SCALE,
    GEOMETRY,
    DonutTorusGeometry,
    make_donut_torus_sample,
    write_group_plot,
)
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

BASE_DIR = Path(__file__).resolve().parent
AMBIENT_DIMS = (3, 7, 11, 15)
AZIMUTHAL_MODE = 1
THETA_MODE = 1
FOURIER_ORDER = 36
QUADRATURE_SIZE = 8_192


@dataclass(frozen=True)
class MultiharmonicThetaMode:
    """A theta profile normalized in the multiharmonic induced area measure."""

    coefficients: np.ndarray
    eigenvalue: float
    fourier_order: int

    def evaluate(self, theta: np.ndarray | float) -> np.ndarray:
        values = np.asarray(theta, dtype=float)
        flat = np.mod(values.reshape(-1), 2.0 * math.pi)
        basis, _ = _fourier_basis(flat, self.fourier_order)
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
    minor_radius: float,
    ambient_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the diagonal induced metric coefficients for the harmonic embedding."""

    if ambient_dim < 3 or ambient_dim % 2 == 0:
        raise ValueError("multiharmonic ambient_dim must be an odd integer at least 3")
    harmonic_count = (ambient_dim - 1) // 2
    harmonic_sum = sum(1.0 / (order * order) for order in range(1, harmonic_count + 1))
    ring_radius = major_radius + minor_radius * np.cos(theta)
    g_theta = (
        minor_radius
        * minor_radius
        * (harmonic_sum * np.sin(theta) ** 2 + np.cos(theta) ** 2 / harmonic_sum)
    )
    g_phi = harmonic_count * ring_radius * ring_radius
    return g_theta, g_phi


@cache
def multiharmonic_theta_mode(
    major_radius: float,
    minor_radius: float,
    m: int,
    j: int,
    ambient_dim: int,
    fourier_order: int = FOURIER_ORDER,
    quadrature_size: int = QUADRATURE_SIZE,
) -> MultiharmonicThetaMode:
    """Solve the ``j``th theta mode in azimuthal sector ``m`` for one embedding."""

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
        minor_radius=minor_radius,
        ambient_dim=ambient_dim,
    )
    area = np.sqrt(g_theta * g_phi)
    mass = step * (basis.T @ (area[:, None] * basis))
    stiffness = step * (
        derivative.T @ ((area / g_theta)[:, None] * derivative)
        + basis.T @ (((m * m * area / g_phi)[:, None]) * basis)
    )
    eigenvalues, eigenvectors = eigh(stiffness, mass, check_finite=False)
    if j >= eigenvalues.size:
        raise ValueError(f"theta mode j={j} is unavailable for Fourier order {fourier_order}")
    coefficients = np.asarray(eigenvectors[:, j], dtype=float)
    values = basis @ coefficients
    if values[int(np.argmax(np.abs(values)))] < 0.0:
        coefficients = -coefficients
    return MultiharmonicThetaMode(coefficients, float(eigenvalues[j]), fourier_order)


def make_multiharmonic_lb_target(
    geometry: DonutTorusGeometry,
    *,
    m: int,
    j: int,
    ambient_dim: int,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return the real ``cos(m phi)`` LB mode for one multiharmonic geometry."""

    theta_mode = multiharmonic_theta_mode(
        geometry.major_radius,
        geometry.minor_radius,
        m,
        j,
        ambient_dim,
    )

    def target(points: np.ndarray) -> np.ndarray:
        theta, phi = geometry.angles_from_points(points, embedding="harmonic")
        return (theta_mode.evaluate(theta) * np.cos(m * phi)).reshape(-1, 1)

    return target


def multiharmonic_case(ambient_dim: int) -> Case:
    mode = multiharmonic_theta_mode(
        GEOMETRY.major_radius,
        GEOMETRY.minor_radius,
        AZIMUTHAL_MODE,
        THETA_MODE,
        ambient_dim,
    )
    return Case(
        name=f"multiharmonic_lb_m{AZIMUTHAL_MODE}_j{THETA_MODE}_d{ambient_dim}",
        title=f"d={ambient_dim}, λ={mode.eigenvalue:.4f}",
        ambient_dim=ambient_dim,
        embedding="harmonic",
        target=wrap_scaled_target(
            make_multiharmonic_lb_target(
                GEOMETRY,
                m=AZIMUTHAL_MODE,
                j=THETA_MODE,
                ambient_dim=ambient_dim,
            ),
            coordinate_scale=COORDINATE_SCALE,
        ),
    )


GROUPS = (
    Group(
        "Exploratory multiharmonic torus: own m=1, j=1 LB mode by ambient dimension",
        "donut_multiharmonic_lb_ambient",
        tuple(multiharmonic_case(ambient_dim) for ambient_dim in AMBIENT_DIMS),
        show_targets=True,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run exploratory DM/RBF KRR on dimension-specific multiharmonic LB modes."
    )
    add_study_args(parser, GROUPS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    groups = selected_groups(GROUPS, args.groups)
    root = output_root(args.workdir, BASE_DIR)
    run_group_cases(
        groups,
        root=root,
        args=args,
        sample_for_case=lambda case: make_donut_torus_sample(
            GEOMETRY,
            ambient_dim=case.ambient_dim,
            coordinate_scale=COORDINATE_SCALE,
            embedding="harmonic",
        ),
    )
    if not args.no_plot:
        reports = report_root(args.workdir, BASE_DIR)
        for group in groups:
            path = write_group_plot(group, root, reports)
            print(f"Wrote exploratory report to {path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
