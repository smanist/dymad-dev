from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np


def test_dimension_specific_semi_torus_neumann_targets_are_finite_and_max_normalized() -> None:
    sys.path.insert(0, str(Path.cwd() / "scripts/krr_convergence"))
    from semi_torus import GEOMETRY  # noqa: PLC0415
    from semi_torus_multiharmonic_lb import (  # noqa: PLC0415
        AMBIENT_DIMS,
        AZIMUTHAL_MODE,
        THETA_MODE,
        make_multiharmonic_neumann_lb_target,
        multiharmonic_theta_mode,
    )

    theta = np.linspace(0.0, 2.0 * np.pi, 257, endpoint=False)
    phi = np.linspace(0.0, np.pi, 129)
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    eigenvalues = []
    for ambient_dim in AMBIENT_DIMS:
        mode = multiharmonic_theta_mode(
            GEOMETRY.major_radius,
            AZIMUTHAL_MODE,
            THETA_MODE,
            ambient_dim,
        )
        eigenvalues.append(mode.eigenvalue)
        points = GEOMETRY.points_from_angles(
            theta_grid.ravel(),
            phi_grid.ravel(),
            ambient_dim=ambient_dim,
            embedding="harmonic",
        )
        target = make_multiharmonic_neumann_lb_target(
            m=AZIMUTHAL_MODE,
            j=THETA_MODE,
            ambient_dim=ambient_dim,
        )
        values = target(points)
        wrapped = target(
            GEOMETRY.points_from_angles(
                theta_grid.ravel() + 2.0 * np.pi,
                phi_grid.ravel() + 2.0 * np.pi,
                ambient_dim=ambient_dim,
                embedding="harmonic",
            )
        )
        assert np.isfinite(values).all()
        assert float(np.max(np.abs(values))) <= 1.0 + 1.0e-12
        np.testing.assert_allclose(wrapped, values, atol=1.0e-10)
    assert np.ptp(eigenvalues) > 0.1


def test_dimension_specific_semi_torus_problem_uses_no_data_transforms() -> None:
    sys.path.insert(0, str(Path.cwd() / "scripts/krr_convergence"))
    from krr_common import make_problem  # noqa: PLC0415

    problem = make_problem(
        name="semi_torus_multiharmonic_neumann_lb",
        sample=lambda n_samples, rng: rng.random((n_samples, 3)),
        target=lambda points: points[:, :1],
        prediction_plots=False,
        x_transform=None,
        y_transform=None,
    )

    assert problem.x_transform is None
    assert problem.y_transform is None


def test_harmonic_area_sampler_matches_the_induced_area_density() -> None:
    sys.path.insert(0, str(Path.cwd() / "scripts/krr_convergence"))
    from semi_torus import GEOMETRY  # noqa: PLC0415
    from semi_torus_multiharmonic_lb import (  # noqa: PLC0415
        INPUT_SCALE,
        make_multiharmonic_area_sample,
        multiharmonic_area_density,
    )

    ambient_dim = 11
    points = make_multiharmonic_area_sample(ambient_dim)(12_000, np.random.default_rng(23))
    theta, phi = GEOMETRY.angles_from_points(points * INPUT_SCALE, embedding="harmonic")
    grid = np.linspace(0.0, 2.0 * np.pi, 16_384, endpoint=False)
    density = multiharmonic_area_density(
        grid,
        major_radius=GEOMETRY.major_radius,
        ambient_dim=ambient_dim,
    )
    expected_cosine = float(np.sum(np.cos(grid) * density) / np.sum(density))

    assert abs(float(np.mean(np.cos(theta))) - expected_cosine) < 0.03
    assert abs(float(np.mean(phi)) - math.pi / 2.0) < 0.03
