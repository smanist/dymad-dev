import csv
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


def test_multiharmonic_modes_are_finite_periodic_and_dimension_specific() -> None:
    sys.path.insert(0, str(Path.cwd() / "scripts/krr_convergence"))
    from donut_torus import DonutTorusGeometry  # noqa: PLC0415
    from donut_torus_multiharmonic_lb import (  # noqa: PLC0415
        AMBIENT_DIMS,
        AZIMUTHAL_MODE,
        THETA_MODE,
        make_multiharmonic_lb_target,
        multiharmonic_theta_mode,
    )

    geometry = DonutTorusGeometry()
    theta = np.asarray([0.2, 1.1, 4.7])
    phi = np.asarray([0.4, 2.2, 5.9])
    eigenvalues = []
    for ambient_dim in AMBIENT_DIMS:
        mode = multiharmonic_theta_mode(
            geometry.major_radius,
            geometry.minor_radius,
            AZIMUTHAL_MODE,
            THETA_MODE,
            ambient_dim,
        )
        eigenvalues.append(mode.eigenvalue)
        points = geometry.points_from_angles(
            theta,
            phi,
            ambient_dim=ambient_dim,
            embedding="harmonic",
        )
        target = make_multiharmonic_lb_target(
            geometry,
            m=AZIMUTHAL_MODE,
            j=THETA_MODE,
            ambient_dim=ambient_dim,
        )
        values = target(points)
        wrapped = target(
            geometry.points_from_angles(
                theta + 2.0 * np.pi,
                phi + 2.0 * np.pi,
                ambient_dim=ambient_dim,
                embedding="harmonic",
            )
        )
        assert np.isfinite(values).all()
        np.testing.assert_allclose(wrapped, values, atol=1.0e-10)
    assert eigenvalues == sorted(eigenvalues, reverse=True)
    assert eigenvalues[0] - eigenvalues[-1] > 0.2


def test_multiharmonic_exploratory_run_writes_two_row_report(tmp_path: Path) -> None:
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(tmp_path / "mpl"))
    result = subprocess.run(
        [
            sys.executable,
            str(Path.cwd() / "scripts/krr_convergence/donut_torus_multiharmonic_lb.py"),
            "--workdir",
            str(tmp_path),
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
    assert result.stdout.count("Wrote convergence artifacts") == 4
    assert "Wrote exploratory report" in result.stdout
    for ambient_dim in (3, 7, 11, 15):
        case_dir = (
            tmp_path
            / "runs/donut_multiharmonic_lb_ambient"
            / f"multiharmonic_lb_m1_j1_d{ambient_dim}"
        )
        with (case_dir / "raw_results.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 4
        assert {row["method"] for row in rows} == {"dm_krr", "rbf_krr"}
        assert len(list((case_dir / "tuning").glob("*/tuning_result.json"))) == 4
    assert (tmp_path / "reports/donut_torus_multiharmonic_lb_ambient.png").is_file()
