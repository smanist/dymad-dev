"""Flat-torus heat-kernel convergence studies."""

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
    SECTION_FONT_SIZE,
    evaluate_heat_section,
    metric_rows as build_metric_rows,
    periodic_heat_kernel,
    plot_study,
    run_study,
    section_plot_steps,
    section_plot_title,
    study_artifact_paths,
    trial_seed as build_trial_seed,
)

BASE_DIR = Path(__file__).resolve().parent
TWO_PI = 2.0 * math.pi
TORUS_AREA = TWO_PI * TWO_PI
TARGET_TIME = 0.04
STEPS = (1, 2, 4, 8)
SAMPLE_COUNTS = (4096, 8192, 16384, 32768, 65536)
TRIALS = 8
TEST_COUNT = 4096
SEED = 2026061801
CASE_INDEX = 4

RUN_TRIALS = TRIALS
RUN_TEST_COUNT = TEST_COUNT
SECTION_TRIAL = 0
SECTION_TEST_COUNT = TEST_COUNT
RUN_WORKERS = 4
RUN_PARALLEL = False


def torus_case(study: str, title: str, parallel: bool) -> HeatCase:
    return HeatCase(
        study=study,
        title=title,
        target_time=TARGET_TIME,
        steps=STEPS,
        sample_counts=SAMPLE_COUNTS,
        section_n=65536,
        section_steps=8,
        section_source_indices=(0,),
        parallel=parallel,
        fit_point_count=3,
    )


CASES: dict[str, HeatCase] = {
    "mass": torus_case("torus_mass", "Torus mass-normalized convergence", parallel=False),
    "no_mass": torus_case("torus_no_mass", "Torus no-mass convergence", parallel=RUN_PARALLEL),
    "nonuniform": torus_case(
        "torus_nonuniform", "Torus nonuniform-sample convergence", parallel=RUN_PARALLEL
    ),
}
ACTIVE_CASES = ("mass", "no_mass", "nonuniform")

ifrun = 1
ifplt = 1
ifsec = 1


def trial_seed(n_samples: int, trial: int) -> int:
    return build_trial_seed(SEED, n_samples, trial, CASE_INDEX)


def sample(n_samples: int, seed: int) -> np.ndarray:
    sampler = qmc.Sobol(d=2, scramble=True, seed=seed)
    return TWO_PI * sampler.random_base2(math.ceil(math.log2(n_samples)))[:n_samples]


def nonuniform_sample(n_samples: int, seed: int) -> np.ndarray:
    points = sample(n_samples, seed)
    amplitudes = np.asarray([0.55, -0.40], dtype=float)
    return np.mod(points + amplitudes[None, :] * np.sin(points), TWO_PI)


def reference_sample(case: str, n_samples: int, seed: int) -> np.ndarray:
    if case == "nonuniform":
        return nonuniform_sample(n_samples, seed)
    return sample(n_samples, seed)


def locations(n_points: int) -> np.ndarray:
    side = int(round(math.sqrt(n_points)))
    if side * side != n_points:
        raise ValueError("n_points must be a perfect square for the 2D torus")
    axis = TWO_PI * (np.arange(side, dtype=float) + 0.5) / side
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    return np.column_stack((xx.ravel(), yy.ravel()))


def sources() -> tuple[list[str], np.ndarray, list[str]]:
    ids = ["t2_q0", "t2_q1", "t2_q2", "t2_q3", "t2_q4"]
    coords = [[0.25, 0.25], [0.50, 0.50], [0.75, 0.25], [0.25, 0.75], [0.90, 0.10]]
    return ids, TWO_PI * np.asarray(coords, dtype=float), ["all"] * len(ids)


def embed(points: np.ndarray) -> np.ndarray:
    return np.column_stack(
        (
            np.cos(points[:, 0]),
            np.sin(points[:, 0]),
            np.cos(points[:, 1]),
            np.sin(points[:, 1]),
        )
    )


def reference(src: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return periodic_heat_kernel(src[:, 0], pts[:, 0], TARGET_TIME, TWO_PI) * periodic_heat_kernel(
        src[:, 1], pts[:, 1], TARGET_TIME, TWO_PI
    )


def torus_location_weights(points: np.ndarray) -> np.ndarray:
    return np.full(points.shape[0], TORUS_AREA / points.shape[0], dtype=float)


SECTION_SPECS = {
    "mass": HeatSectionSpec(
        ambient_dim=4,
        encode=embed,
        mode="uniform",
        volume_normalization="estimate_volume",
        volume_dim=2,
    ),
    "no_mass": HeatSectionSpec(
        ambient_dim=4,
        encode=embed,
        mode="uniform",
        mass_normalization="none",
    ),
    "nonuniform": HeatSectionSpec(
        ambient_dim=4,
        encode=embed,
        mode="density",
        alpha=1.0,
        location_weights=torus_location_weights,
    ),
}


def dymad_section(
    case: str, ref_angles: np.ndarray, src: np.ndarray, pts: np.ndarray, steps: int
) -> np.ndarray:
    try:
        spec = SECTION_SPECS[case]
    except KeyError as error:
        raise ValueError(f"Unknown case: {case}") from error
    return evaluate_heat_section(
        spec,
        ref_angles,
        src,
        pts,
        epsilon=TARGET_TIME / steps,
        steps=steps,
    )


def case_truth(case: str, src: np.ndarray, pts: np.ndarray) -> np.ndarray:
    truth = reference(src, pts)
    return TORUS_AREA * truth if case == "no_mass" else truth


def run_one(task):
    case, step_count, n_samples, trial, source_ids, source_pts, source_groups, test_pts, truth = (
        task
    )
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
            groups=source_groups,
            estimate=pred,
            truth=truth,
            n_samples=n_samples,
            trial=trial,
            steps=step_count,
            weights=torus_location_weights(test_pts),
        ),
    )


def warm_keops_backend() -> None:
    _source_ids, source_pts, _source_groups = sources()
    points = locations(4)
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
    section_steps = section_plot_steps(config, smallest_epsilon=case == "no_mass")
    _source_ids, source_pts_all, _source_groups = sources()
    source_idx = list(config.section_source_indices)
    source_pts = source_pts_all[source_idx]
    pts = locations(SECTION_TEST_COUNT)
    truth = case_truth(case, source_pts, pts)
    pred = dymad_section(
        case,
        reference_sample(case, config.section_n, trial_seed(config.section_n, SECTION_TRIAL)),
        source_pts,
        pts,
        section_steps,
    )
    side = int(round(math.sqrt(SECTION_TEST_COUNT)))
    xx = pts[:, 0].reshape(side, side)
    yy = pts[:, 1].reshape(side, side)
    fig, axes = plt.subplots(
        len(source_idx),
        3,
        figsize=(12.0, 3.6 * len(source_idx)),
        squeeze=False,
        constrained_layout=True,
    )
    tick_values = [0.0, math.pi, TWO_PI]
    tick_labels = [r"$0$", r"$\pi$", r"$2\pi$"]
    for row, idx in enumerate(source_idx):
        error = pred[row] - truth[row]
        vmin = min(float(np.min(truth[row])), float(np.min(pred[row])))
        vmax = max(float(np.max(truth[row])), float(np.max(pred[row])))
        err_max = max(float(np.max(np.abs(error))), np.finfo(float).tiny)
        panels = (("Truth", truth[row]), ("Prediction", pred[row]))
        for column, (ax, (title, values)) in enumerate(zip(axes[row, :2], panels, strict=True)):
            image = ax.pcolormesh(
                xx,
                yy,
                values.reshape(side, side),
                shading="auto",
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
            )
            if column == 0:
                ax.scatter(
                    source_pts_all[:, 0],
                    source_pts_all[:, 1],
                    facecolors="none",
                    edgecolors="black",
                    s=28,
                )
                ax.scatter(source_pts_all[idx, 0], source_pts_all[idx, 1], c="black", s=30)
            ax.set_title(title, fontsize=SECTION_FONT_SIZE)
            ax.set_aspect("equal")
            ax.set_xticks(tick_values, tick_labels)
            ax.set_yticks(tick_values, tick_labels)
            ax.set_xlabel(r"$\theta_1$", fontsize=SECTION_FONT_SIZE)
            ax.tick_params(axis="both", labelsize=SECTION_FONT_SIZE)
        colorbar = fig.colorbar(image, ax=axes[row, :2], shrink=0.78)
        colorbar.ax.tick_params(labelsize=SECTION_FONT_SIZE)
        error_image = axes[row, 2].pcolormesh(
            xx,
            yy,
            error.reshape(side, side),
            shading="auto",
            cmap="coolwarm",
            vmin=-err_max,
            vmax=err_max,
        )
        axes[row, 2].set_title("Error", fontsize=SECTION_FONT_SIZE)
        axes[row, 2].set_aspect("equal")
        axes[row, 2].set_xticks(tick_values, tick_labels)
        axes[row, 2].set_yticks(tick_values, tick_labels)
        axes[row, 2].set_xlabel(r"$\theta_1$", fontsize=SECTION_FONT_SIZE)
        axes[row, 2].tick_params(axis="both", labelsize=SECTION_FONT_SIZE)
        colorbar = fig.colorbar(error_image, ax=axes[row, 2], shrink=0.78)
        colorbar.ax.tick_params(labelsize=SECTION_FONT_SIZE)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\theta_2$", fontsize=SECTION_FONT_SIZE)
    fig.suptitle(
        section_plot_title(config, section_steps, include_study=case != "no_mass"),
        fontsize=SECTION_FONT_SIZE,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_case(case: str) -> None:
    config = CASES[case]
    source_ids, source_pts, source_groups = sources()
    test_pts = locations(RUN_TEST_COUNT)
    truth = case_truth(case, source_pts, test_pts)
    tasks = [
        (case, step_count, n_samples, trial, source_ids, source_pts, source_groups, test_pts, truth)
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
