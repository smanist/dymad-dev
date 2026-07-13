"""Curved 3D donut-torus heat-kernel convergence study."""

from __future__ import annotations

# ruff: noqa: E402, I001

import math
import sys
from dataclasses import dataclass
from functools import cache
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from runtime_env import configure_script_runtime  # noqa: E402

configure_script_runtime(__file__, matplotlib=True)

import matplotlib
import numpy as np
from scipy import linalg

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import (  # noqa: E402
    HeatCase,
    HeatSectionSpec,
    evaluate_heat_section,
    metric_rows as build_metric_rows,
    plot_study,
    run_study,
    study_artifact_paths,
)
from torus import (  # noqa: E402
    BASE_DIR,
    RUN_TEST_COUNT,
    RUN_TRIALS,
    RUN_WORKERS,
    SAMPLE_COUNTS,
    SECTION_TEST_COUNT,
    SECTION_TRIAL,
    STEPS,
    TARGET_TIME,
    TWO_PI,
    locations,
    sample,
    sources,
    trial_seed,
)

DONUT_MAJOR_RADIUS = 2.0
DONUT_MINOR_RADIUS = 0.75
DONUT_AREA = 4.0 * math.pi * math.pi * DONUT_MAJOR_RADIUS * DONUT_MINOR_RADIUS
DONUT_THETA_POINTS = 95
DONUT_ANGULAR_MODES = 32


def donut_case(study: str, title: str, parallel: bool) -> HeatCase:
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
    "mass": donut_case("torus_donut_mass", "Donut torus mass-normalized convergence", False),
    "no_mass": donut_case("torus_donut_no_mass", "Donut torus no-mass convergence", False),
    "nonuniform": donut_case(
        "torus_donut_nonuniform", "Donut torus nonuniform-sample convergence", False
    ),
}
ACTIVE_CASES = ("mass", "no_mass", "nonuniform")

ifrun = 1
ifplt = 1
ifsec = 1


def donut_embed(points: np.ndarray) -> np.ndarray:
    phi, theta = points[:, 0], points[:, 1]
    ring_radius = DONUT_MAJOR_RADIUS + DONUT_MINOR_RADIUS * np.cos(theta)
    return np.column_stack(
        (
            ring_radius * np.cos(phi),
            ring_radius * np.sin(phi),
            DONUT_MINOR_RADIUS * np.sin(theta),
        )
    )


def uniform_surface_sample(n_samples: int, seed: int) -> np.ndarray:
    """Map parameter-uniform Sobol points to the donut's uniform area measure."""

    points = sample(n_samples, seed)
    target_theta = points[:, 1]
    theta = target_theta.copy()
    ratio = DONUT_MINOR_RADIUS / DONUT_MAJOR_RADIUS
    for _ in range(8):
        theta -= (theta + ratio * np.sin(theta) - target_theta) / (1.0 + ratio * np.cos(theta))
    return np.column_stack((points[:, 0], np.mod(theta, TWO_PI)))


def reference_sample(case: str, n_samples: int, seed: int) -> np.ndarray:
    if case == "nonuniform":
        return sample(n_samples, seed)
    return uniform_surface_sample(n_samples, seed)


@dataclass(frozen=True)
class DonutSpectrum:
    eigenvalues: tuple[np.ndarray, ...]
    eigenvectors: tuple[np.ndarray, ...]


def periodic_derivative_matrix(n_points: int) -> np.ndarray:
    modes = np.fft.fftfreq(n_points, d=1.0 / n_points)
    identity = np.eye(n_points)
    return np.fft.ifft(1j * modes[:, None] * np.fft.fft(identity, axis=0), axis=0).real


@cache
def donut_spectrum(
    theta_points: int = DONUT_THETA_POINTS,
    angular_modes: int = DONUT_ANGULAR_MODES,
) -> DonutSpectrum:
    if theta_points % 2 == 0:
        raise ValueError("theta_points must be odd for the Fourier differentiation grid.")
    theta = TWO_PI * np.arange(theta_points, dtype=float) / theta_points
    derivative = periodic_derivative_matrix(theta_points)
    area_factor = DONUT_MAJOR_RADIUS + DONUT_MINOR_RADIUS * np.cos(theta)
    theta_step = TWO_PI / theta_points
    mass = theta_step * np.diag(DONUT_MINOR_RADIUS * area_factor)
    stiffness_base = (
        theta_step * derivative.T @ ((area_factor / DONUT_MINOR_RADIUS)[:, None] * derivative)
    )
    eigenvalues, eigenvectors = [], []
    for mode in range(angular_modes + 1):
        stiffness = stiffness_base + theta_step * np.diag(
            mode * mode * DONUT_MINOR_RADIUS / area_factor
        )
        values, vectors = linalg.eigh(stiffness, mass, check_finite=False)
        eigenvalues.append(values)
        eigenvectors.append(vectors)
    return DonutSpectrum(tuple(eigenvalues), tuple(eigenvectors))


def evaluate_theta_modes(theta: np.ndarray, eigenvectors: np.ndarray) -> np.ndarray:
    n_points = eigenvectors.shape[0]
    modes = np.fft.fftfreq(n_points, d=1.0 / n_points)
    coefficients = np.fft.fft(eigenvectors, axis=0) / n_points
    basis = np.exp(1j * np.outer(np.mod(theta, TWO_PI), modes))
    return np.real(basis @ coefficients)


def donut_reference(
    src: np.ndarray,
    pts: np.ndarray,
    t: float = TARGET_TIME,
    *,
    theta_points: int = DONUT_THETA_POINTS,
    angular_modes: int = DONUT_ANGULAR_MODES,
) -> np.ndarray:
    spectrum = donut_spectrum(theta_points, angular_modes)
    values = np.zeros((src.shape[0], pts.shape[0]), dtype=float)
    phi_difference = src[:, None, 0] - pts[None, :, 0]
    for mode, (eigenvalues, eigenvectors) in enumerate(
        zip(spectrum.eigenvalues, spectrum.eigenvectors, strict=True)
    ):
        keep = eigenvalues * t <= 36.0
        source_modes = evaluate_theta_modes(src[:, 1], eigenvectors[:, keep])
        point_modes = evaluate_theta_modes(pts[:, 1], eigenvectors[:, keep])
        theta_sum = (source_modes * np.exp(-eigenvalues[keep] * t)[None, :]) @ point_modes.T
        coefficient = 1.0 / TWO_PI if mode == 0 else 1.0 / math.pi
        values += coefficient * np.cos(mode * phi_difference) * theta_sum
    return values


def donut_location_weights(points: np.ndarray) -> np.ndarray:
    side = int(round(math.sqrt(points.shape[0])))
    if side * side != points.shape[0]:
        raise ValueError("Donut quadrature points must form a square parameter grid.")
    parameter_area = (TWO_PI / side) ** 2
    return (
        parameter_area
        * DONUT_MINOR_RADIUS
        * (DONUT_MAJOR_RADIUS + DONUT_MINOR_RADIUS * np.cos(points[:, 1]))
    )


SECTION_SPECS = {
    "mass": HeatSectionSpec(
        ambient_dim=3,
        encode=donut_embed,
        mode="uniform",
        volume_normalization="estimate_volume",
        volume_dim=2,
    ),
    "no_mass": HeatSectionSpec(
        ambient_dim=3,
        encode=donut_embed,
        mode="uniform",
        mass_normalization="none",
    ),
    "nonuniform": HeatSectionSpec(
        ambient_dim=3,
        encode=donut_embed,
        mode="density",
        alpha=1.0,
        location_weights=donut_location_weights,
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


def warm_keops_backend() -> None:
    _source_ids, source_pts, _source_groups = sources()
    for case in ACTIVE_CASES:
        dymad_section(
            case,
            reference_sample(case, 8, trial_seed(8, 0)),
            source_pts,
            locations(4),
            CASES[case].section_steps,
        )


def case_truth(case: str, src: np.ndarray, pts: np.ndarray) -> np.ndarray:
    truth = donut_reference(src, pts, CASES[case].target_time)
    return DONUT_AREA * truth if case == "no_mass" else truth


def run_one(task):
    case, step_count, n_samples, trial, source_ids, source_pts, source_groups, test_pts, truth = (
        task
    )
    pred = dymad_section(
        case,
        reference_sample(case, n_samples, trial_seed(n_samples, trial)),
        source_pts,
        test_pts,
        step_count,
    )
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
            weights=donut_location_weights(test_pts),
        ),
    )


def plot_sections(case: str, path: Path) -> None:
    config = CASES[case]
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
        config.section_steps,
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
            ax.set_title(title, fontsize=14)
            ax.set_aspect("equal")
            ax.set_xticks(tick_values, tick_labels)
            ax.set_yticks(tick_values, tick_labels)
            ax.set_xlabel(r"$\theta_1$")
        fig.colorbar(image, ax=axes[row, :2], shrink=0.78)
        error_image = axes[row, 2].pcolormesh(
            xx,
            yy,
            error.reshape(side, side),
            shading="auto",
            cmap="coolwarm",
            vmin=-err_max,
            vmax=err_max,
        )
        axes[row, 2].set_title("Error", fontsize=14)
        axes[row, 2].set_aspect("equal")
        axes[row, 2].set_xticks(tick_values, tick_labels)
        axes[row, 2].set_yticks(tick_values, tick_labels)
        axes[row, 2].set_xlabel(r"$\theta_1$")
        fig.colorbar(error_image, ax=axes[row, 2], shrink=0.78)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\theta_2$")
    epsilon = config.target_time / config.section_steps
    fig.suptitle(
        f"{config.study}: N={config.section_n}, eps={epsilon:g}, steps={config.section_steps}"
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
