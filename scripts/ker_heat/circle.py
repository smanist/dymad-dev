"""Circle heat-kernel convergence studies."""

from __future__ import annotations

# ruff: noqa: E402, I001

import math
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from runtime_env import configure_script_runtime  # noqa: E402

configure_script_runtime(__file__, matplotlib=True)

import matplotlib
import numpy as np
from scipy.stats import qmc

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import (  # noqa: E402
    HeatCase,
    HeatSectionSpec,
    evaluate_heat_section,
    metric_rows as build_metric_rows,
    periodic_heat_kernel,
    plot_study,
    run_study,
    study_artifact_paths,
    trial_seed as build_trial_seed,
)

BASE_DIR = Path(__file__).resolve().parent
TWO_PI = 2.0 * math.pi
CIRCLE_VOLUME = TWO_PI
TARGET_TIME = 0.01
STEPS = (1, 2, 4, 8, 16, 32)
SAMPLE_COUNTS = (512, 1024, 2048, 4096, 8192, 16384)
TRIALS = 8
TEST_COUNT = 4096
SEED = 2026061801
SOURCE_FRACTIONS = (0.125, 0.375, 0.625, 0.875)

RUN_TRIALS = TRIALS
RUN_TEST_COUNT = TEST_COUNT
SECTION_TRIAL = 0
SECTION_TEST_COUNT = TEST_COUNT
RUN_WORKERS = 4
RUN_PARALLEL = False


def circle_case(study: str, title: str, parallel: bool) -> HeatCase:
    return HeatCase(
        study=study,
        title=title,
        target_time=TARGET_TIME,
        steps=STEPS,
        sample_counts=SAMPLE_COUNTS,
        section_n=16384,
        section_steps=32,
        section_source_indices=(0,),
        parallel=parallel,
        fit_point_count=4,
    )


CASES: dict[str, HeatCase] = {
    "mass": circle_case("circle_mass", "Circle mass-normalized convergence", parallel=False),
    "no_mass": circle_case("circle_no_mass", "Circle no-mass convergence", parallel=RUN_PARALLEL),
    "nonuniform": circle_case(
        "circle_nonuniform", "Circle nonuniform-sample convergence", parallel=RUN_PARALLEL
    ),
}
ACTIVE_CASES = ("mass", "no_mass", "nonuniform")

ifrun = 1
ifplt = 1
ifsec = 1


def trial_seed(n_samples: int, trial: int) -> int:
    return build_trial_seed(SEED, n_samples, trial)


def sample_angles(n_samples: int, seed: int) -> np.ndarray:
    sampler = qmc.Sobol(d=1, scramble=True, seed=seed)
    return TWO_PI * sampler.random_base2(math.ceil(math.log2(n_samples)))[:n_samples]


def nonuniform_angles(n_samples: int, seed: int) -> np.ndarray:
    theta = sample_angles(n_samples, seed)
    return np.mod(theta[:, 0] + 0.55 * np.sin(theta[:, 0]), TWO_PI)[:, None]


def reference_sample(case: str, n_samples: int, seed: int) -> np.ndarray:
    if case == "nonuniform":
        return nonuniform_angles(n_samples, seed)
    return sample_angles(n_samples, seed)


def test_angles(n_points: int) -> np.ndarray:
    return (TWO_PI * (np.arange(n_points, dtype=float) + 0.5) / n_points)[:, None]


def source_angles() -> tuple[list[str], np.ndarray]:
    ids = [f"s1_q{i}" for i in range(len(SOURCE_FRACTIONS))]
    return ids, TWO_PI * np.asarray(SOURCE_FRACTIONS, dtype=float)[:, None]


def embed(theta: np.ndarray) -> np.ndarray:
    return np.column_stack((np.cos(theta[:, 0]), np.sin(theta[:, 0])))


def reference_kernel(sources: np.ndarray, points: np.ndarray) -> np.ndarray:
    return periodic_heat_kernel(sources[:, 0], points[:, 0], TARGET_TIME, TWO_PI)


def circle_location_weights(points: np.ndarray) -> np.ndarray:
    return np.full(points.shape[0], CIRCLE_VOLUME / points.shape[0], dtype=float)


SECTION_SPECS = {
    "mass": HeatSectionSpec(
        ambient_dim=2,
        encode=embed,
        mode="uniform",
        volume_normalization="estimate_volume",
        volume_dim=1,
    ),
    "no_mass": HeatSectionSpec(
        ambient_dim=2,
        encode=embed,
        mode="uniform",
        mass_normalization="none",
    ),
    "nonuniform": HeatSectionSpec(
        ambient_dim=2,
        encode=embed,
        mode="density",
        alpha=1.0,
        location_weights=circle_location_weights,
    ),
}


def dymad_section(
    case: str, ref_angles: np.ndarray, sources: np.ndarray, points: np.ndarray, steps: int
) -> np.ndarray:
    try:
        spec = SECTION_SPECS[case]
    except KeyError as error:
        raise ValueError(f"Unknown case: {case}") from error
    return evaluate_heat_section(
        spec,
        ref_angles,
        sources,
        points,
        epsilon=TARGET_TIME / steps,
        steps=steps,
    )


def case_truth(case: str, sources: np.ndarray, points: np.ndarray) -> np.ndarray:
    truth = reference_kernel(sources, points)
    return CIRCLE_VOLUME * truth if case == "no_mass" else truth


def run_one(task):
    case, step_count, n_samples, trial, source_ids, source_pts, test_pts, truth = task
    ref_pts = reference_sample(case, n_samples, trial_seed(n_samples, trial))
    pred = dymad_section(case, ref_pts, source_pts, test_pts, step_count)
    return (
        step_count,
        n_samples,
        trial,
        build_metric_rows(
            case=CASES[case].study,
            target_time=TARGET_TIME,
            ids=source_ids,
            estimate=pred,
            truth=truth,
            n_samples=n_samples,
            trial=trial,
            steps=step_count,
        ),
    )


def warm_keops_backend() -> None:
    _source_ids, source_pts = source_angles()
    points = test_angles(4)
    for case in ACTIVE_CASES:
        dymad_section(
            case,
            reference_sample(case, 8, trial_seed(8, 0)),
            source_pts,
            points,
            CASES[case].section_steps,
        )


def plot_sections(case: str, path: Path) -> None:
    config = CASES[case]
    _source_ids, source_pts_all = source_angles()
    source_idx = list(config.section_source_indices)
    source_pts = source_pts_all[source_idx]
    points = test_angles(SECTION_TEST_COUNT)
    truth = case_truth(case, source_pts, points)
    pred = dymad_section(
        case,
        reference_sample(case, config.section_n, trial_seed(config.section_n, SECTION_TRIAL)),
        source_pts,
        points,
        config.section_steps,
    )
    fig, axes = plt.subplots(
        len(source_idx),
        3,
        figsize=(12.0, 3.0 * len(source_idx)),
        squeeze=False,
        constrained_layout=True,
    )
    for row, idx in enumerate(source_idx):
        error = pred[row] - truth[row]
        panels = (("Truth", truth[row]), ("Prediction", pred[row]), ("Error", error))
        for column, (ax, (title, values)) in enumerate(zip(axes[row], panels, strict=True)):
            ax.plot(points[:, 0], values)
            if column == 0:
                ax.scatter(
                    source_pts_all[:, 0],
                    np.zeros(len(source_pts_all)),
                    facecolors="none",
                    edgecolors="black",
                    s=28,
                )
                ax.scatter(source_pts_all[idx, 0], 0.0, c="black", s=30)
            ax.set_title(title, fontsize=14)
            ax.set_xticks([0.0, math.pi, TWO_PI], [r"$0$", r"$\pi$", r"$2\pi$"])
            ax.set_xlabel(r"$\theta$")
            ax.grid(True, alpha=0.25)
            ax.tick_params(axis="both", labelsize=14)
    epsilon = config.target_time / config.section_steps
    fig.suptitle(
        f"{config.study}: N={config.section_n}, eps={epsilon:g}, steps={config.section_steps}",
        y=1.08,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_case(case: str) -> None:
    config = CASES[case]
    source_ids, source_pts = source_angles()
    test_pts = test_angles(RUN_TEST_COUNT)
    truth = case_truth(case, source_pts, test_pts)
    tasks = [
        (case, step_count, n_samples, trial, source_ids, source_pts, test_pts, truth)
        for step_count in config.steps
        for n_samples in config.sample_counts
        for trial in range(RUN_TRIALS)
    ]
    run_study(BASE_DIR, config, tasks, run_one, max_workers=RUN_WORKERS)


if __name__ == "__main__" and ifrun:
    warm_keops_backend()
    for active_case in ACTIVE_CASES:
        run_case(active_case)

if __name__ == "__main__" and ifplt:
    for active_case in ACTIVE_CASES:
        outputs = plot_study(BASE_DIR, CASES[active_case])
        if outputs is None:
            raw_csv, _conv_en_path, _conv_path, _section_path = study_artifact_paths(
                BASE_DIR, CASES[active_case]
            )
            print(f"Missing {raw_csv}; set ifrun = 1 or copy the CSV into place.")
        else:
            for output in outputs:
                print(f"Wrote {output}")

if __name__ == "__main__" and ifsec:
    for active_case in ACTIVE_CASES:
        _raw_csv, _conv_en_path, _conv_path, section_path = study_artifact_paths(
            BASE_DIR, CASES[active_case]
        )
        plot_sections(active_case, section_path)
        print(f"Wrote {section_path}")
