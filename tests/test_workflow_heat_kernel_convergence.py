import csv
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pykeops")

sys.path.insert(0, str(Path(os.getcwd()) / "scripts/ker_heat"))

import backend_cmp  # noqa: E402
import circle  # noqa: E402
import disk  # noqa: E402
import torus  # noqa: E402


def test_circle_helpers_write_small_convergence_outputs(tmp_path: Path) -> None:
    ids, src = circle.source_angles()
    pts = circle.test_angles(32)
    truth = circle.reference_kernel(src, pts)
    rows = []
    for steps in (1, 2):
        ref = circle.sample_angles(16, seed=7 + steps)
        pred, _volume_hat = circle.dymad_section(ref, src, pts, steps)
        rows.extend(circle.metric_rows(ids, pred, truth, 16, 0, steps))
    raw_csv = tmp_path / "circle_raw_results.csv"
    fig_path = tmp_path / "convergence_circle.png"
    circle.write_rows(raw_csv, rows)
    circle.plot_convergence(circle.read_rows(raw_csv), fig_path)

    assert len(rows) == 8
    assert raw_csv.is_file()
    assert fig_path.is_file()
    with raw_csv.open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert {row["steps"] for row in csv_rows} == {"1", "2"}
    assert all(np.isfinite(float(row["relative_l2_error"])) for row in csv_rows)


def test_circle_helpers_write_small_subset_convergence_outputs(tmp_path: Path) -> None:
    ids, src = circle.source_angles()
    pts = circle.test_angles(16)
    truth = circle.reference_kernel(src[:2], pts)
    ref = circle.sample_angles(16, seed=17)
    pred, volume = circle.dymad_section(ref, src[:2], pts, steps=1)
    rows = circle.metric_rows(ids[:2], pred, truth, 16, 0, 1)
    raw_csv = tmp_path / "circle_subset_raw_results.csv"
    fig_path = tmp_path / "convergence_circle_subset.png"
    circle.write_rows(raw_csv, rows)
    circle.plot_convergence(circle.read_rows(raw_csv), fig_path)

    assert pred.shape == truth.shape
    assert np.isfinite(volume)
    assert raw_csv.is_file()
    assert fig_path.is_file()


def test_circle_reference_is_periodic() -> None:
    src = np.asarray([[0.2 * circle.TWO_PI]])
    pts = np.asarray([[0.1], [0.1 + circle.TWO_PI]])
    values = circle.reference_kernel(src, pts)

    assert np.allclose(values[:, 0], values[:, 1])


def test_torus_reference_is_product_of_circle_sections() -> None:
    pts = np.asarray([[0.1, 0.2], [0.4, 0.6], [1.2, 2.0]])
    src = np.asarray([[0.3, 0.7], [1.0, 1.5]])
    expected = torus.circle_kernel(src[:, 0], pts[:, 0]) * torus.circle_kernel(src[:, 1], pts[:, 1])

    assert np.allclose(torus.reference(src, pts), expected)


def test_disk_reference_is_symmetric() -> None:
    pts = np.asarray([[0.0, 0.0], [0.35, 0.0], [0.2, 0.4]])
    forward = disk.reference(pts[[0]], pts[[1]])[0, 0]
    backward = disk.reference(pts[[1]], pts[[0]])[0, 0]

    assert abs(forward - backward) < 1e-12


def test_disk_keops_section_rows_include_source_groups() -> None:
    ids, src, groups = disk.sources()
    pts = disk.locations(32)
    truth = disk.reference(src[:2], pts)
    pred, volume = disk.dymad_section(disk.sample(32, seed=11), src[:2], pts, steps=1)
    rows = disk.metric_rows(ids[:2], groups[:2], pred, truth, 32, 0, 1)

    assert pred.shape == truth.shape
    assert np.isfinite(volume)
    assert volume > 0.0
    assert {row["source_group"] for row in rows} == {"interior"}
    assert all(np.isfinite(float(row["relative_l2_error"])) for row in rows)


def test_torus_keops_section_uses_estimated_volume() -> None:
    ids, src, groups = torus.sources()
    pts = torus.locations(16)
    truth = torus.reference(src[:2], pts)
    pred, volume = torus.dymad_section(torus.sample(64, seed=13), src[:2], pts, steps=1)
    rows = torus.metric_rows(ids[:2], groups[:2], pred, truth, 64, 0, 1)

    assert pred.shape == truth.shape
    assert np.isfinite(volume)
    assert volume > 0.0
    assert {row["source_group"] for row in rows} == {"all"}
    assert all(np.isfinite(float(row["max_abs_error"])) for row in rows)


def test_dymad_section_uses_estimated_volume_normalization() -> None:
    ids, src = circle.source_angles()
    del ids
    pts = circle.test_angles(32)
    values, volume_hat = circle.dymad_section(
        circle.sample_angles(16, seed=7), src[:2], pts, steps=1
    )

    assert values.shape == (2, 32)
    assert np.max(np.abs(values.mean(axis=1) * volume_hat - 1.0)) < 1e-12


def test_circle_volume_estimate_is_finite() -> None:
    _values, volume_hat = circle.dymad_section(
        circle.sample_angles(512, seed=7),
        circle.source_angles()[1][:1],
        circle.test_angles(32),
        steps=8,
    )

    assert np.isfinite(volume_hat)
    assert 2.0 < volume_hat < 10.0


def test_circle_worker_runs_under_thread_pool() -> None:
    ids, src = circle.source_angles()
    pts = circle.test_angles(32)
    truth = circle.reference_kernel(src, pts)
    tasks = [(1, 16, 0, ids, src, pts, truth), (2, 16, 0, ids, src, pts, truth)]

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(circle.run_one, tasks))

    assert [item[0] for item in results] == [1, 2]
    assert all(len(item[3]) == len(ids) for item in results)


def test_circle_section_plot_writes_truth_prediction_error(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(circle, "SECTION_TEST_COUNT", 32)
    path = tmp_path / "section_circle.png"

    circle.plot_sections(16, 1, 0, (0, 1), path)

    assert path.is_file()


def test_keops_dense_verify_writes_small_visual_check(tmp_path: Path) -> None:
    rows, gram_diffs, section_diffs = backend_cmp.compare_case("circle", n_ref=24, n_loc=16)
    backend_cmp.write_rows(rows, tmp_path / "keops_dense_errors.csv")
    backend_cmp.plot_case("circle", rows, gram_diffs, section_diffs, tmp_path)

    assert len(rows) == 1
    assert all(row["case"] == "circle" for row in rows)
    assert all(np.isfinite(float(row["gram_max_abs"])) for row in rows)
    assert all(np.isfinite(float(row["section_max_abs"])) for row in rows)
    assert (tmp_path / "keops_dense_errors.csv").is_file()
    assert (tmp_path / "circle_keops_dense_verify.png").is_file()
