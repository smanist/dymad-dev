import csv
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _run_small_study(script: str, workdir: Path, group: str) -> str:
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(workdir / "mpl"))
    result = subprocess.run(
        [
            sys.executable,
            str(Path.cwd() / "scripts/krr_convergence" / script),
            "--workdir",
            str(workdir),
            "--groups",
            group,
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
    return result.stdout


def _assert_study_artifacts(case_dir: Path) -> None:
    for name in ("raw_results.csv", "convergence_summary.csv", "convergence_rates.json"):
        assert (case_dir / name).is_file()
    with (case_dir / "raw_results.csv").open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 4
    assert len(list((case_dir / "tuning").glob("*/tuning_result.json"))) == 4


def test_manifold_targets_are_finite() -> None:
    sys.path.insert(0, str(Path.cwd() / "scripts/krr_convergence"))
    from krr_common import (  # noqa: PLC0415
        Case,  # noqa: PLC0415
        SemiTorusGeometry,
        make_ambient_periodic_disk_target,
        make_neumann_disk_target,
        make_semi_torus_ambient_fourier_target,
        make_semi_torus_fourier_target,
    )

    disk_points = np.asarray([[0.0, 0.0], [0.25, 0.5], [-0.4, 0.3]])
    torus = SemiTorusGeometry()
    torus_points = torus.points_from_angles(
        np.asarray([0.0, np.pi / 2.0, np.pi]),
        np.asarray([0.0, np.pi / 3.0, np.pi]),
        ambient_dim=15,
    )
    harmonic_torus_points = torus.points_from_angles(
        np.asarray([0.0, np.pi / 2.0, np.pi]),
        np.asarray([0.0, np.pi / 3.0, np.pi]),
        ambient_dim=15,
        embedding="harmonic",
    )
    targets = (
        (make_neumann_disk_target(3, 1), disk_points),
        (make_ambient_periodic_disk_target(2, 2), disk_points),
        (
            make_semi_torus_fourier_target(torus, boundary="dirichlet", m=3, j=1, fourier_order=16),
            torus_points,
        ),
        (
            make_semi_torus_fourier_target(
                torus,
                boundary="neumann",
                m=3,
                j=1,
                fourier_order=16,
                embedding="harmonic",
            ),
            harmonic_torus_points,
        ),
        (make_semi_torus_ambient_fourier_target(torus, frequency=2), torus_points),
    )
    for target, points in targets:
        values = target(points)
        assert values.shape == (len(points), 1)
        assert np.isfinite(values).all()
    assert Case("x", "x", targets[0][0]).ambient_dim == 2


def test_krr_studies_keep_their_script_level_coordinate_scale() -> None:
    sys.path.insert(0, str(Path.cwd() / "scripts/krr_convergence"))
    from krr_common import make_problem  # noqa: PLC0415

    problem = make_problem(
        name="identity_input_transform",
        sample=lambda n_samples, rng: rng.random((n_samples, 2)),
        target=lambda points: points[:, :1],
        prediction_plots=False,
    )

    assert problem.x_transform is None
    assert problem.y_transform == "std"


def test_semi_torus_targets_plot_in_intrinsic_coordinates() -> None:
    sys.path.insert(0, str(Path.cwd() / "scripts/krr_convergence"))
    from semi_torus import GROUPS, _plot_target  # noqa: PLC0415

    ambient_group = next(group for group in GROUPS if group.slug == "dirichlet_ambient")
    assert [case.title for case in ambient_group.cases] == ["n=3", "n=7", "n=11", "n=15"]

    figure, (axis, shared_axis) = plt.subplots(1, 2, sharey=True)
    try:
        _plot_target(axis, GROUPS[0].cases[0], color_limit=1.0)
        _plot_target(
            shared_axis,
            GROUPS[0].cases[1],
            color_limit=1.0,
            show_y_axis=False,
        )
        assert axis.name == "rectilinear"
        assert axis.get_xlabel() == r"$\theta$"
        assert axis.get_ylabel() == r"$\phi$"
        assert axis.get_xlim() == (0.0, 2.0 * np.pi)
        assert axis.get_ylim() == (0.0, np.pi)
        assert axis.collections
        assert axis.get_shared_y_axes().joined(axis, shared_axis)
        assert shared_axis.get_ylabel() == ""
        assert all(not label.get_visible() for label in shared_axis.get_yticklabels())
    finally:
        plt.close(figure)


def test_disk_and_semi_torus_small_runs_write_manifold_reports(tmp_path: Path) -> None:
    disk_stdout = _run_small_study("disk.py", tmp_path / "disk", "neumann")
    assert "Wrote convergence artifacts" in disk_stdout
    _assert_study_artifacts(tmp_path / "disk/runs/neumann/neumann_m3_r1")
    assert (tmp_path / "disk/reports/disk_neumann.png").is_file()

    torus_stdout = _run_small_study(
        "semi_torus.py", tmp_path / "semi_torus", "neumann_harmonic_ambient"
    )
    assert "Wrote convergence artifacts" in torus_stdout
    _assert_study_artifacts(tmp_path / "semi_torus/runs/neumann_harmonic_ambient/neumann_m3_j1_d3")
    assert (tmp_path / "semi_torus/reports/semi_torus_neumann_harmonic_ambient.png").is_file()
