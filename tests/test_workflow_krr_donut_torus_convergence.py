import csv
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


def test_donut_torus_lb_targets_are_periodic_and_finite() -> None:
    sys.path.insert(0, str(Path.cwd() / "scripts/krr_convergence"))
    from donut_torus import (  # noqa: PLC0415
        AMBIENT_FOURIER_FREQUENCIES,
        GROUPS,
        LAPLACE_MODES,
        RBF_INTEGRAL_FOURIER_ORDER,
        DonutTorusGeometry,
        donut_torus_rbf_theta_mode,
        make_donut_torus_ambient_fourier_target,
        make_donut_torus_lb_target,
        make_donut_torus_rbf_integral_target,
    )

    geometry = DonutTorusGeometry()
    assert LAPLACE_MODES == ((1, 1), (3, 1), (6, 3))
    theta = np.asarray([0.1, 1.3, 4.2])
    phi = np.asarray([0.2, 2.1, 5.7])
    points = geometry.points_from_angles(theta, phi)
    recovered_theta, recovered_phi = geometry.angles_from_points(points)
    np.testing.assert_allclose(recovered_theta, theta)
    np.testing.assert_allclose(recovered_phi, phi)

    harmonic_points = geometry.points_from_angles(
        theta,
        phi,
        ambient_dim=15,
        embedding="harmonic",
    )
    for target_factory in (make_donut_torus_lb_target, make_donut_torus_rbf_integral_target):
        for m, j in LAPLACE_MODES:
            target = target_factory(geometry, m=m, j=j)
            values = target(points)
            wrapped = target(geometry.points_from_angles(theta + 2.0 * np.pi, phi + 2.0 * np.pi))
            assert values.shape == (3, 1)
            assert np.isfinite(values).all()
            np.testing.assert_allclose(wrapped, values, atol=1e-10)
    rbf_mode = donut_torus_rbf_theta_mode(geometry.major_radius, geometry.minor_radius, 3, 1)
    assert rbf_mode.coefficients.shape == (2 * RBF_INTEGRAL_FOURIER_ORDER + 1,)
    assert np.isfinite(rbf_mode.eigenvalue)
    harmonic_target = make_donut_torus_lb_target(
        geometry,
        m=1,
        j=1,
        embedding="harmonic",
    )
    assert np.isfinite(harmonic_target(harmonic_points)).all()
    isometric_points = geometry.points_from_angles(
        theta,
        phi,
        ambient_dim=15,
        embedding="isometric",
    )
    np.testing.assert_allclose(
        np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1),
        np.linalg.norm(isometric_points[:, None, :] - isometric_points[None, :, :], axis=-1),
        atol=1e-12,
    )
    recovered_theta, recovered_phi = geometry.angles_from_points(
        isometric_points,
        embedding="isometric",
    )
    np.testing.assert_allclose(recovered_theta, theta, atol=1e-12)
    np.testing.assert_allclose(recovered_phi, phi, atol=1e-12)
    isometric_target = make_donut_torus_lb_target(
        geometry,
        m=1,
        j=1,
        embedding="isometric",
    )
    np.testing.assert_allclose(isometric_target(isometric_points), harmonic_target(harmonic_points))
    for frequency in AMBIENT_FOURIER_FREQUENCIES:
        assert np.isfinite(
            make_donut_torus_ambient_fourier_target(geometry, frequency=frequency)(points)
        ).all()
    assert [group.slug for group in GROUPS] == [
        "laplace_beltrami",
        "donut_rbf_integral",
        "donut_ambient_fourier",
        "donut_neumann_harmonic_ambient",
        "donut_neumann_isometric_ambient",
    ]


def test_donut_torus_small_run_writes_rbf_comparison_artifacts(tmp_path: Path) -> None:
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(tmp_path / "mpl"))
    result = subprocess.run(
        [
            sys.executable,
            str(Path.cwd() / "scripts/krr_convergence/donut_torus.py"),
            "--workdir",
            str(tmp_path),
            "--groups",
            "laplace_beltrami",
            "--levels",
            "8,10",
            "--trials",
            "1",
            "--n-val",
            "8",
            "--n-test",
            "16",
            "--initial-budget",
            "2",
            "--refinement-budget",
            "0",
            "--max-workers",
            "1",
            "--validation-size",
            "4",
            "--no-prediction-plots",
        ],
        cwd=Path.cwd(),
        env=env,
        text=True,
        capture_output=True,
        check=True,
        timeout=180,
    )
    assert "Wrote convergence artifacts" in result.stdout
    case_dir = tmp_path / "runs/laplace_beltrami/lb_m1_j1"
    for name in ("raw_results.csv", "convergence_summary.csv", "convergence_rates.json"):
        assert (case_dir / name).is_file()
    with (case_dir / "raw_results.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert {row["method"] for row in rows} == {"dm_krr", "rbf_krr"}
    assert len(list((case_dir / "tuning").glob("*/tuning_result.json"))) == 4
    assert (tmp_path / "reports/donut_torus_laplace_beltrami.png").is_file()


def test_donut_torus_extended_groups_run_with_their_target_families(tmp_path: Path) -> None:
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(tmp_path / "mpl"))
    result = subprocess.run(
        [
            sys.executable,
            str(Path.cwd() / "scripts/krr_convergence/donut_torus.py"),
            "--workdir",
            str(tmp_path),
            "--groups",
            "donut_rbf_integral",
            "donut_ambient_fourier",
            "donut_neumann_harmonic_ambient",
            "--levels",
            "8,10",
            "--trials",
            "1",
            "--n-val",
            "8",
            "--n-test",
            "16",
            "--initial-budget",
            "2",
            "--refinement-budget",
            "0",
            "--max-workers",
            "1",
            "--validation-size",
            "4",
            "--no-prediction-plots",
        ],
        cwd=Path.cwd(),
        env=env,
        text=True,
        capture_output=True,
        check=True,
        timeout=180,
    )
    assert result.stdout.count("Wrote convergence artifacts") == 10
    case_dirs = (
        tmp_path / "runs/donut_rbf_integral/rbf_integral_m1_j1",
        tmp_path / "runs/donut_ambient_fourier/ambient_fourier_k2",
        tmp_path / "runs/donut_neumann_harmonic_ambient/neumann_m1_j1_d15",
    )
    for case_dir in case_dirs:
        assert (case_dir / "raw_results.csv").is_file()
        assert len(list((case_dir / "tuning").glob("*/tuning_result.json"))) == 4
    for slug in ("rbf_integral", "ambient_fourier", "neumann_harmonic_ambient"):
        assert (tmp_path / f"reports/donut_torus_{slug}.png").is_file()
