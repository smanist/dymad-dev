import csv
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


def test_donut_torus_lb_targets_are_periodic_and_finite() -> None:
    sys.path.insert(0, str(Path.cwd() / "scripts/krr_convergence"))
    from donut_torus import (  # noqa: PLC0415
        LAPLACE_MODES,
        DonutTorusGeometry,
        make_donut_torus_lb_target,
    )

    geometry = DonutTorusGeometry()
    theta = np.asarray([0.1, 1.3, 4.2])
    phi = np.asarray([0.2, 2.1, 5.7])
    points = geometry.points_from_angles(theta, phi)
    recovered_theta, recovered_phi = geometry.angles_from_points(points)
    np.testing.assert_allclose(recovered_theta, theta)
    np.testing.assert_allclose(recovered_phi, phi)

    for m, j in LAPLACE_MODES:
        target = make_donut_torus_lb_target(geometry, m=m, j=j)
        values = target(points)
        wrapped = target(geometry.points_from_angles(theta + 2.0 * np.pi, phi + 2.0 * np.pi))
        assert values.shape == (3, 1)
        assert np.isfinite(values).all()
        np.testing.assert_allclose(wrapped, values, atol=1e-12)


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
    case_dir = tmp_path / "runs/laplace_beltrami/lb_m1_j0"
    for name in ("raw_results.csv", "convergence_summary.csv", "convergence_rates.json"):
        assert (case_dir / name).is_file()
    with (case_dir / "raw_results.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert {row["method"] for row in rows} == {"dm_krr", "rbf_krr"}
    assert len(list((case_dir / "tuning").glob("*/tuning_result.json"))) == 4
    assert (tmp_path / "reports/donut_torus_laplace_beltrami.png").is_file()
