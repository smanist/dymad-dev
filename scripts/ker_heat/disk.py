"""Neumann disk heat-kernel convergence studies."""

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
from scipy import special
from scipy.stats import qmc

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import (  # noqa: E402
    HeatCase,
    HeatSectionSpec,
    SECTION_FONT_SIZE,
    evaluate_heat_section,
    metric_rows as build_metric_rows,
    plot_study,
    run_study,
    section_plot_steps,
    section_plot_title,
    study_artifact_paths,
    trial_seed as build_trial_seed,
)

BASE_DIR = Path(__file__).resolve().parent
DISK_AREA = math.pi
SEED = 2026061801
CASE_INDEX = 8
REFERENCE_TERM_TOL = 1e-13
REFERENCE_ANGULAR_ORDER = 42
REFERENCE_RADIAL_ROOTS = 42
RUN_PARALLEL = False


CASES: dict[str, HeatCase] = {
    "mass": HeatCase(
        study="disk_full_mass",
        title="Disk full mass-normalized convergence",
        target_time=0.08,
        steps=(1, 2, 4, 8),
        sample_counts=(32, 64, 128, 256, 512, 1024, 2048, 4096),
        section_n=4096,
        section_steps=8,
        section_source_indices=(0,),
        parallel=False,
        fit_point_count=3,
    ),
    "interior_no_mass": HeatCase(
        study="disk_interior_no_mass",
        title="Disk interior no-mass convergence",
        target_time=0.04,
        steps=(1, 2, 4, 8, 16, 32),
        sample_counts=(512, 1024, 2048, 4096, 8192, 16384, 32768),
        section_n=4096,
        section_steps=1,
        section_source_indices=(4,),
        parallel=RUN_PARALLEL,
        fit_point_count=4,
    ),
    "nonuniform": HeatCase(
        study="disk_full_nonuniform",
        title="Disk full nonuniform-sample convergence",
        target_time=0.08,
        steps=(1, 2, 4, 8),
        sample_counts=(32, 64, 128, 256, 512, 1024, 2048, 4096),
        section_n=4096,
        section_steps=8,
        section_source_indices=(0,),
        parallel=RUN_PARALLEL,
        fit_point_count=3,
    ),
}
ACTIVE_CASES = ("mass", "interior_no_mass", "nonuniform")
SOURCE_SETS = {"mass": "full", "interior_no_mass": "interior", "nonuniform": "full"}
RUN_TRIALS = 8
RUN_TEST_COUNT = 4096
SECTION_TEST_COUNT = 4096
SECTION_TRIAL = 0
RUN_WORKERS = 4

ifrun = 1
ifplt = 1
ifsec = 1


def trial_seed(n_samples: int, trial: int) -> int:
    return build_trial_seed(SEED, n_samples, trial, CASE_INDEX)


def sample(n_samples: int, seed: int) -> np.ndarray:
    sampler = qmc.Sobol(d=2, scramble=True, seed=seed)
    unit = sampler.random_base2(math.ceil(math.log2(n_samples)))[:n_samples]
    radius = np.sqrt(unit[:, 0])
    theta = 2.0 * math.pi * unit[:, 1]
    return np.column_stack((radius * np.cos(theta), radius * np.sin(theta)))


def nonuniform_sample(n_samples: int, seed: int) -> np.ndarray:
    points = sample(n_samples, seed)
    radii = np.linalg.norm(points, axis=1)
    scale = np.divide(radii**0.65, radii, out=np.ones_like(radii), where=radii > 0.0)
    return points * scale[:, None]


def reference_sample(case: str, n_samples: int, seed: int) -> np.ndarray:
    if case == "nonuniform":
        return nonuniform_sample(n_samples, seed)
    return sample(n_samples, seed)


def locations(n_points: int) -> np.ndarray:
    return sample(n_points, seed=2026062101)


def full_sources() -> tuple[list[str], np.ndarray, list[str]]:
    data = [
        ("d_center", (0.0, 0.0), "interior"),
        ("d_int_x35", (0.35, 0.0), "interior"),
        ("d_int_diag60", (0.4242640687, 0.4242640687), "interior"),
        ("d_nb_x90", (0.90, 0.0), "near_boundary"),
        ("d_nb_diag90", (0.6363961031, 0.6363961031), "near_boundary"),
        ("d_nb_y95", (0.0, 0.95), "near_boundary"),
    ]
    return (
        [item[0] for item in data],
        np.asarray([item[1] for item in data], dtype=float),
        [item[2] for item in data],
    )


def interior_sources() -> tuple[list[str], np.ndarray, list[str]]:
    data = [
        ("d_center", (0.0, 0.0), "interior"),
        ("d_x25", (0.25, 0.0), "interior"),
        ("d_diag40", (0.2828427125, 0.2828427125), "interior"),
        ("d_y55", (0.0, 0.55), "interior"),
        ("d_diag60", (0.4242640687, 0.4242640687), "interior"),
    ]
    return (
        [item[0] for item in data],
        np.asarray([item[1] for item in data], dtype=float),
        [item[2] for item in data],
    )


def case_sources(case: str) -> tuple[list[str], np.ndarray, list[str]]:
    if SOURCE_SETS[case] == "interior":
        return interior_sources()
    return full_sources()


def mode_table(
    order_count: int = REFERENCE_ANGULAR_ORDER,
    root_count: int = REFERENCE_RADIAL_ROOTS,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    roots, norms = [], []
    for order in range(order_count + 1):
        rts = special.jnp_zeros(order, root_count).astype(float)
        radial = 0.5 * (
            special.jv(order, rts) ** 2 - special.jv(order - 1, rts) * special.jv(order + 1, rts)
        )
        angular = 2.0 * math.pi if order == 0 else math.pi
        roots.append(rts)
        norms.append(1.0 / (angular * radial))
    return roots, norms


def reference(src: np.ndarray, pts: np.ndarray, t: float) -> np.ndarray:
    roots, norms = mode_table()
    radii = np.linalg.norm(pts, axis=1)
    theta = np.arctan2(pts[:, 1], pts[:, 0])
    source_radii = np.linalg.norm(src, axis=1)
    source_theta = np.arctan2(src[:, 1], src[:, 0])
    values = np.full((src.shape[0], pts.shape[0]), 1.0 / math.pi, dtype=float)
    for order, rts in enumerate(roots):
        decay = np.exp(-(rts * rts) * t)
        keep = decay >= REFERENCE_TERM_TOL
        if not np.any(keep):
            continue
        rts_kept = rts[keep]
        decay_kept = decay[keep]
        norm_kept = norms[order][keep]
        target_radial = special.jv(order, np.outer(rts_kept, radii))
        source_radial = special.jv(order, np.outer(rts_kept, source_radii))
        contribution = ((decay_kept * norm_kept)[:, None] * source_radial).T @ target_radial
        if order > 0:
            contribution *= np.cos(order * (theta[None, :] - source_theta[:, None]))
        values += contribution
    return values


def disk_location_weights(points: np.ndarray) -> np.ndarray:
    return np.full(points.shape[0], DISK_AREA / points.shape[0], dtype=float)


def identity(points: np.ndarray) -> np.ndarray:
    return points


SECTION_SPECS = {
    "mass": HeatSectionSpec(
        ambient_dim=2,
        encode=identity,
        mode="uniform",
        volume_normalization="estimate_volume",
        volume_dim=2,
    ),
    "interior_no_mass": HeatSectionSpec(
        ambient_dim=2,
        encode=identity,
        mode="uniform",
        mass_normalization="none",
    ),
    "nonuniform": HeatSectionSpec(
        ambient_dim=2,
        encode=identity,
        mode="density",
        alpha=1.0,
        location_weights=disk_location_weights,
    ),
}


def dymad_section(
    case: str, ref_pts: np.ndarray, src: np.ndarray, pts: np.ndarray, steps: int
) -> np.ndarray:
    config = CASES[case]
    try:
        spec = SECTION_SPECS[case]
    except KeyError as error:
        raise ValueError(f"Unknown case: {case}") from error
    return evaluate_heat_section(
        spec,
        ref_pts,
        src,
        pts,
        epsilon=config.target_time / steps,
        steps=steps,
    )


def case_truth(case: str, src: np.ndarray, pts: np.ndarray) -> np.ndarray:
    truth = reference(src, pts, CASES[case].target_time)
    return DISK_AREA * truth if case == "interior_no_mass" else truth


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
            target_time=CASES[case].target_time,
            ids=source_ids,
            groups=source_groups,
            estimate=pred,
            truth=truth,
            n_samples=n_samples,
            trial=trial,
            steps=step_count,
            weights=disk_location_weights(test_pts),
        ),
    )


def warm_keops_backend() -> None:
    _source_ids, source_pts, _source_groups = full_sources()
    points = locations(4)
    for case in ACTIVE_CASES:
        dymad_section(
            case,
            reference_sample(case, 8, trial_seed(8, 0)),
            source_pts,
            points,
            CASES[case].section_steps,
        )


def case_source_groups(case: str) -> list[str]:
    return list(dict.fromkeys(case_sources(case)[2]))


def plot_sections(case: str, path: Path) -> None:
    config = CASES[case]
    section_steps = section_plot_steps(config, smallest_epsilon=case == "interior_no_mass")
    _source_ids, source_pts_all, _source_groups = case_sources(case)
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
    fig, axes = plt.subplots(
        len(source_idx),
        3,
        figsize=(12.0, 3.6 * len(source_idx)),
        squeeze=False,
        constrained_layout=True,
    )
    tick_values = [-1.0, 0.0, 1.0]
    for row, idx in enumerate(source_idx):
        error = pred[row] - truth[row]
        vmin = min(float(np.min(truth[row])), float(np.min(pred[row])))
        vmax = max(float(np.max(truth[row])), float(np.max(pred[row])))
        err_max = max(float(np.max(np.abs(error))), np.finfo(float).tiny)
        panels = (("Truth", truth[row]), ("Prediction", pred[row]))
        for column, (ax, (title, values)) in enumerate(zip(axes[row, :2], panels, strict=True)):
            image = ax.tricontourf(
                pts[:, 0], pts[:, 1], values, levels=40, cmap="viridis", vmin=vmin, vmax=vmax
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
            ax.set_xticks(tick_values)
            ax.set_yticks(tick_values)
            ax.set_xlabel("x", fontsize=SECTION_FONT_SIZE)
            ax.tick_params(axis="both", labelsize=SECTION_FONT_SIZE)
        colorbar = fig.colorbar(image, ax=axes[row, :2], shrink=0.78)
        colorbar.ax.tick_params(labelsize=SECTION_FONT_SIZE)
        error_image = axes[row, 2].tricontourf(
            pts[:, 0],
            pts[:, 1],
            error,
            levels=40,
            cmap="coolwarm",
            vmin=-err_max,
            vmax=err_max,
        )
        axes[row, 2].set_title("Error", fontsize=SECTION_FONT_SIZE)
        axes[row, 2].set_aspect("equal")
        axes[row, 2].set_xticks(tick_values)
        axes[row, 2].set_yticks(tick_values)
        axes[row, 2].set_xlabel("x", fontsize=SECTION_FONT_SIZE)
        axes[row, 2].tick_params(axis="both", labelsize=SECTION_FONT_SIZE)
        colorbar = fig.colorbar(error_image, ax=axes[row, 2], shrink=0.78)
        colorbar.ax.tick_params(labelsize=SECTION_FONT_SIZE)
    for ax in axes[:, 0]:
        ax.set_ylabel("y", fontsize=SECTION_FONT_SIZE)
    fig.suptitle(
        section_plot_title(config, section_steps, include_study=case != "interior_no_mass"),
        fontsize=SECTION_FONT_SIZE,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_case(case: str) -> None:
    config = CASES[case]
    source_ids, source_pts, groups = case_sources(case)
    test_pts = locations(RUN_TEST_COUNT)
    truth = case_truth(case, source_pts, test_pts)
    tasks = [
        (case, step_count, n_samples, trial, source_ids, source_pts, groups, test_pts, truth)
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
        outputs = plot_study(
            BASE_DIR, CASES[active_case], source_groups=case_source_groups(active_case)
        )
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
