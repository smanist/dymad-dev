from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(os.getcwd()) / "scripts/ker_heat"))

import backend_cmp  # noqa: E402
import circle  # noqa: E402
import disk  # noqa: E402
import torus  # noqa: E402
import torus_donut  # noqa: E402
from common import (  # noqa: E402
    HeatCase,
    HeatSectionSpec,
    epsilon_curve_at_largest_n,
    evaluate_heat_section,
    fit_loglog_rate,
    plot_circle_torus_convergence,
    plot_convergence,
    plot_error_vs_epsilon_at_largest_n,
    run_study,
    section_plot_steps,
    section_plot_title,
    study_artifact_paths,
)


def test_shared_heat_section_evaluator_supports_dense_backend() -> None:
    points = np.linspace(-1.0, 1.0, 8, dtype=float)[:, None]
    spec = HeatSectionSpec(
        ambient_dim=1,
        encode=lambda values: values,
        mode="uniform",
        mass_normalization="none",
    )

    section = evaluate_heat_section(
        spec,
        points,
        points[[1, 5]],
        points,
        epsilon=0.1,
        steps=1,
        backend="torch",
    )

    assert section.shape == (2, 8)
    assert np.all(np.isfinite(section))


def test_parallel_study_retries_serially_when_workers_are_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    case = HeatCase(
        study="fallback",
        title="Fallback",
        target_time=0.1,
        steps=(1,),
        sample_counts=(1,),
        section_n=1,
        section_steps=1,
        section_source_indices=(0,),
        parallel=True,
        fit_point_count=1,
    )

    def unavailable(*args, **kwargs):
        del args, kwargs
        raise PermissionError("worker processes unavailable")

    monkeypatch.setattr("common.run_parallel", unavailable)

    def run_one(_task):
        return (
            1,
            1,
            0,
            [
                {
                    "case": case.study,
                    "steps": 1,
                    "epsilon": 0.1,
                    "n_samples": 1,
                    "trial": 0,
                    "source_id": "source",
                    "relative_l2_error": 0.0,
                    "max_abs_error": 0.0,
                }
            ],
        )

    raw_csv = run_study(tmp_path, case, [None], run_one)

    assert raw_csv.exists()


def test_convergence_plot_writes_all_n_values(tmp_path: Path) -> None:
    rows: list[dict[str, str]] = []
    for step in (1, 2):
        for n_samples in (32, 64, 128):
            for trial in (0, 1):
                rows.append(
                    {
                        "case": "toy",
                        "steps": str(step),
                        "epsilon": str(0.1 / step),
                        "n_samples": str(n_samples),
                        "trial": str(trial),
                        "source_id": "src",
                        "relative_l2_error": "1.0",
                        "max_abs_error": str(1.0 / n_samples + 0.1 * step),
                    }
                )
    path = tmp_path / "heat_conv_eN_toy.png"
    plot_convergence(
        rows,
        path=path,
        steps_values=(1, 2),
        target_time=0.1,
        title="Toy convergence",
    )
    assert path.stat().st_size > 0


def test_epsilon_convergence_plot_uses_largest_n_per_epsilon(tmp_path: Path) -> None:
    rows: list[dict[str, str]] = []
    for eps, step in ((0.1, 1), (0.05, 2)):
        for n_samples in (32, 64):
            for trial in (0, 1):
                rows.append(
                    {
                        "case": "toy",
                        "steps": str(step),
                        "epsilon": str(eps),
                        "n_samples": str(n_samples),
                        "trial": str(trial),
                        "source_id": "src",
                        "relative_l2_error": "1.0",
                        "max_abs_error": str(eps + 1.0 / n_samples + 0.01 * trial),
                    }
                )

    epsilons, errors, n_values = epsilon_curve_at_largest_n(rows, metric="max_abs_error")

    assert epsilons.tolist() == [0.05, 0.1]
    assert n_values.tolist() == [64.0, 64.0]
    assert errors.tolist() == pytest.approx([0.05 + 1.0 / 64.0 + 0.005, 0.1 + 1.0 / 64.0 + 0.005])
    assert fit_loglog_rate(epsilons, 3.0 * epsilons**2.0) == pytest.approx((2.0, 3.0))

    path = tmp_path / "epsilon.png"
    plot_error_vs_epsilon_at_largest_n(
        rows,
        path=path,
        title="Toy convergence at largest N",
        metric="max_abs_error",
        fit_point_count=2,
    )
    assert path.stat().st_size > 0


def test_compact_circle_torus_plot_writes_requested_curves(tmp_path: Path) -> None:
    def make_rows(
        *, target_time: float, steps_values: tuple[int, ...], sample_counts: tuple[int, ...]
    ) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for steps in steps_values:
            epsilon = target_time / steps
            for n_samples in sample_counts:
                rows.append(
                    {
                        "case": "toy",
                        "steps": str(steps),
                        "epsilon": str(epsilon),
                        "n_samples": str(n_samples),
                        "trial": "0",
                        "source_id": "src",
                        "relative_l2_error": str(epsilon + 1.0 / n_samples),
                        "max_abs_error": str(epsilon + 1.0 / n_samples),
                    }
                )
        return rows

    circle_case = HeatCase(
        study="circle",
        title="Circle",
        target_time=0.01,
        steps=(8, 16, 32),
        sample_counts=(32, 64, 128),
        section_n=128,
        section_steps=32,
        section_source_indices=(0,),
        parallel=False,
        fit_point_count=3,
    )
    torus_case = HeatCase(
        study="torus",
        title="Torus",
        target_time=0.04,
        steps=(2, 4, 8),
        sample_counts=(128, 256, 512),
        section_n=512,
        section_steps=8,
        section_source_indices=(0,),
        parallel=False,
        fit_point_count=3,
    )
    path = tmp_path / "heat_circle_torus_convergence.png"

    plot_circle_torus_convergence(
        circle_rows=make_rows(
            target_time=circle_case.target_time,
            steps_values=circle_case.steps,
            sample_counts=circle_case.sample_counts,
        ),
        torus_rows=make_rows(
            target_time=torus_case.target_time,
            steps_values=torus_case.steps,
            sample_counts=torus_case.sample_counts,
        ),
        circle_case=circle_case,
        torus_case=torus_case,
        circle_epsilon=0.000625,
        torus_epsilon=0.01,
        path=path,
    )

    assert path.stat().st_size > 0


@pytest.mark.parametrize(
    ("module", "expected_studies"),
    [
        (circle, {"circle_mass", "circle_no_mass", "circle_nonuniform"}),
        (torus, {"torus_mass", "torus_no_mass", "torus_nonuniform"}),
        (disk, {"disk_full_mass", "disk_interior_no_mass", "disk_full_nonuniform"}),
    ],
)
def test_each_manifold_exposes_all_case_artifacts(module, expected_studies: set[str]) -> None:
    assert len(module.ACTIVE_CASES) == len(expected_studies)
    studies = {module.CASES[case].study for case in module.ACTIVE_CASES}
    assert studies == expected_studies
    for case in module.ACTIVE_CASES:
        config = module.CASES[case]
        raw_csv, conv_en, conv, section = study_artifact_paths(module.BASE_DIR, config)
        study = config.study
        assert raw_csv == module.BASE_DIR / "runs" / study / "raw_results.csv"
        assert conv_en.name == f"heat_conv_eN_{study}.png"
        assert conv.name == f"heat_conv_{study}.png"
        assert section.name == f"heat_section_{study}.png"


def test_disk_interior_no_mass_configuration_and_sources() -> None:
    config = disk.CASES["interior_no_mass"]
    _ids, points, groups = disk.case_sources("interior_no_mass")

    assert set(groups) == {"interior"}
    assert points.shape == (5, 2)
    assert max(np.linalg.norm(points, axis=1)) <= 0.6 + 1e-10
    assert config.target_time == 0.04
    assert config.steps == (1, 2, 4, 8, 16, 32)
    assert config.sample_counts == (512, 1024, 2048, 4096, 8192, 16384, 32768)
    assert config.section_source_indices == (4,)


@pytest.mark.parametrize(
    ("config", "expected_steps"),
    [
        (circle.CASES["no_mass"], 32),
        (disk.CASES["interior_no_mass"], 32),
        (torus.CASES["no_mass"], 8),
    ],
)
def test_no_mass_section_plots_use_smallest_epsilon_and_compact_titles(
    config: HeatCase, expected_steps: int
) -> None:
    section_steps = section_plot_steps(config, smallest_epsilon=True)

    assert section_steps == expected_steps
    assert section_plot_title(config, section_steps, include_study=False) == (
        f"N={config.section_n}, eps={config.target_time / section_steps:g}, steps={section_steps}"
    )


def test_nonuniform_reference_samples_remain_on_manifold() -> None:
    circle_points = circle.nonuniform_angles(16, seed=123)
    torus_points = torus.nonuniform_sample(16, seed=123)
    disk_points = disk.nonuniform_sample(16, seed=123)
    donut_points = torus_donut.reference_sample("nonuniform", 16, seed=123)

    assert circle_points.shape == (16, 1)
    assert np.all(circle_points >= 0.0)
    assert np.all(circle_points < 2.0 * np.pi)
    assert torus_points.shape == (16, 2)
    assert np.all(torus_points >= 0.0)
    assert np.all(torus_points < 2.0 * np.pi)
    assert disk_points.shape == (16, 2)
    assert np.max(np.linalg.norm(disk_points, axis=1)) <= 1.0 + 1e-12
    assert donut_points.shape == (16, 2)
    assert np.all(donut_points >= 0.0)
    assert np.all(donut_points < 2.0 * np.pi)


def test_donut_uniform_surface_sampling_has_area_measure() -> None:
    points = torus_donut.uniform_surface_sample(4096, seed=123)

    assert np.mean(np.cos(points[:, 1])) == pytest.approx(
        torus_donut.DONUT_MINOR_RADIUS / (2.0 * torus_donut.DONUT_MAJOR_RADIUS), abs=2e-3
    )


def test_donut_spectral_reference_preserves_surface_mass() -> None:
    source = np.asarray([[0.25, 1.15]])
    points = torus_donut.locations(4096)
    spectrum = torus_donut.donut_spectrum()
    values = torus_donut.donut_reference(source, points)
    weights = torus_donut.donut_location_weights(points)

    assert spectrum.eigenvalues[0][0] == pytest.approx(0.0, abs=1e-10)
    assert np.all(spectrum.eigenvalues[0] >= -1e-10)
    assert np.all(np.isfinite(values))
    assert np.sum(weights * values[0]) == pytest.approx(1.0, abs=1e-8)


def test_donut_study_artifacts_are_isolated() -> None:
    expected_studies = {
        "torus_donut_mass",
        "torus_donut_no_mass",
        "torus_donut_nonuniform",
    }

    assert {torus_donut.CASES[case].study for case in torus_donut.ACTIVE_CASES} == expected_studies
    for case in torus_donut.ACTIVE_CASES:
        config = torus_donut.CASES[case]
        raw_csv, conv_en, conv, section = study_artifact_paths(torus_donut.BASE_DIR, config)
        assert raw_csv == torus_donut.BASE_DIR / "runs" / config.study / "raw_results.csv"
        assert conv_en.name == f"heat_conv_eN_{config.study}.png"
        assert conv.name == f"heat_conv_{config.study}.png"
        assert section.name == f"heat_section_{config.study}.png"


def test_backend_comparison_plot_uses_direct_difference_arrays(tmp_path: Path) -> None:
    gram_diff = np.asarray([[0.0, 1e-6], [-1e-6, 0.0]])
    section_diff = np.asarray([[2e-6, -2e-6], [0.0, 1e-6]])

    backend_cmp.plot_case("toy", gram_diff, section_diff, output_dir=tmp_path)

    assert (tmp_path / "toy_keops_dense_verify.png").stat().st_size > 0


def test_backend_comparison_gram_materializes_the_composed_kernel() -> None:
    reference = backend_cmp.circle_points(12)
    model = backend_cmp.kernel(reference, backend="torch")

    actual = backend_cmp.gram(model, reference)
    tensor = model.kernel.reference_points
    expected = model.kernel.materialize(tensor, tensor).detach().cpu().numpy()

    assert actual.shape == (reference.shape[0], reference.shape[0])
    assert np.allclose(actual, expected)
