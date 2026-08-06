from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def test_dirichlet_fixed_and_own_targets_are_aligned() -> None:
    sys.path.insert(0, str(Path.cwd() / "scripts/krr_convergence"))
    from semi_torus_multiharmonic_dirichlet_lb import (  # noqa: PLC0415
        AMBIENT_DIMS,
        AZIMUTHAL_MODE,
        GEOMETRY,
        REFERENCE_AMBIENT_DIM,
        THETA_MODE,
    )
    from semi_torus_multiharmonic_lb import (  # noqa: PLC0415
        aligned_multiharmonic_theta_mode,
        make_multiharmonic_dirichlet_lb_target,
    )

    theta = np.linspace(0.0, 2.0 * np.pi, 257, endpoint=False)
    phi = np.full_like(theta, np.pi / (2.0 * AZIMUTHAL_MODE))
    reference_mode = aligned_multiharmonic_theta_mode(
        GEOMETRY.major_radius,
        AZIMUTHAL_MODE,
        THETA_MODE,
        REFERENCE_AMBIENT_DIM,
    )
    reference_values = reference_mode.evaluate(theta)
    fixed_values = []
    for ambient_dim in AMBIENT_DIMS:
        points = GEOMETRY.points_from_angles(
            theta,
            phi,
            ambient_dim=ambient_dim,
            embedding="harmonic",
        )
        fixed = make_multiharmonic_dirichlet_lb_target(
            m=AZIMUTHAL_MODE,
            j=THETA_MODE,
            source_ambient_dim=REFERENCE_AMBIENT_DIM,
        )(points)
        own = make_multiharmonic_dirichlet_lb_target(
            m=AZIMUTHAL_MODE,
            j=THETA_MODE,
            source_ambient_dim=ambient_dim,
        )(points)
        fixed_values.append(fixed)
        own_mode = aligned_multiharmonic_theta_mode(
            GEOMETRY.major_radius,
            AZIMUTHAL_MODE,
            THETA_MODE,
            ambient_dim,
        )
        assert np.isfinite(own).all()
        assert float(np.dot(reference_values, own_mode.evaluate(theta))) > 0.0

        boundary_points = GEOMETRY.points_from_angles(
            theta,
            np.zeros_like(theta),
            ambient_dim=ambient_dim,
            embedding="harmonic",
        )
        np.testing.assert_allclose(
            make_multiharmonic_dirichlet_lb_target(
                m=AZIMUTHAL_MODE,
                j=THETA_MODE,
                source_ambient_dim=ambient_dim,
            )(boundary_points),
            0.0,
            atol=1.0e-12,
        )
    for values in fixed_values[1:]:
        np.testing.assert_allclose(values, fixed_values[0], atol=1.0e-12)
