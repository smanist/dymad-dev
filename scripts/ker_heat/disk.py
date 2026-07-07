"""Disk heat-kernel section convergence with DyMAD uniform mode."""

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
import torch
from scipy import special
from scipy.stats import qmc

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dymad.modules import KernelScDMHeat  # noqa: E402
from common import (  # noqa: E402
    metric_rows as build_metric_rows,
    plot_convergence as plot_convergence_curves,
    plot_max_abs_convergence as plot_max_abs_convergence_curves,
    read_rows,
    run_serial,
    trial_seed as build_trial_seed,
    write_rows,
)

BASE_DIR = Path(__file__).resolve().parent
OUT = BASE_DIR / "runs" / "disk_redo"
RAW_CSV = OUT / "disk_raw_results.csv"
FIG_PATH = OUT / "convergence_disk.png"
MAX_ABS_FIG_PATH = OUT / "convergence_disk_max_abs.png"
SECTION_FIG_PATH = OUT / "section_disk.png"

TARGET_TIME = 0.08
STEPS = (1, 2, 4, 8)
EPSILONS = tuple(TARGET_TIME / step for step in STEPS)
SAMPLE_COUNTS = (32, 64, 128, 256, 512, 1024, 2048, 4096)
TRIALS = 8
TEST_COUNT = 4096
SEED = 2026061801
CASE_INDEX = 8
DISK_AREA = math.pi
REFERENCE_TERM_TOL = 1e-13
REFERENCE_ANGULAR_ORDER = 42
REFERENCE_RADIAL_ROOTS = 42

RUN_STEPS = STEPS
RUN_SAMPLE_COUNTS = SAMPLE_COUNTS
RUN_TRIALS = TRIALS
RUN_TEST_COUNT = TEST_COUNT
SECTION_N = 4096
SECTION_STEPS = 8
SECTION_TRIAL = 0
SECTION_SOURCE_INDICES = (0, 3)
SECTION_TEST_COUNT = TEST_COUNT

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


def locations(n_points: int) -> np.ndarray:
    return sample(n_points, seed=2026062101)


def sources() -> tuple[list[str], np.ndarray, list[str]]:
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


def reference(src: np.ndarray, pts: np.ndarray, t: float = TARGET_TIME) -> np.ndarray:
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


def dymad_section(
    ref_pts: np.ndarray, src: np.ndarray, pts: np.ndarray, steps: int
) -> tuple[np.ndarray, float]:
    epsilon = TARGET_TIME / steps
    kernel = KernelScDMHeat(
        in_dim=2,
        eps_init=epsilon,
        alpha_init=1.0,
        dtype=torch.float64,
        backend="keops",
    )
    kernel.set_reference_data(torch.as_tensor(ref_pts, dtype=torch.float64))
    values = kernel.heat_kernel(
        torch.as_tensor(pts, dtype=torch.float64),
        torch.as_tensor(src, dtype=torch.float64),
        mode="uniform",
        steps=steps,
        volume_normalization="estimate_volume",
        volume_dim=2,
        volume_estimate_warnings=False,
        return_diagnostics=True,
    )
    section, diagnostics = values
    return section.detach().cpu().numpy().T, float(diagnostics["volume"])


def metric_rows(
    ids: list[str],
    groups: list[str],
    estimate: np.ndarray,
    truth: np.ndarray,
    n_samples: int,
    trial: int,
    steps: int,
) -> list[dict[str, object]]:
    weights = np.full(truth.shape[1], DISK_AREA / truth.shape[1], dtype=float)
    return build_metric_rows(
        case="disk",
        target_time=TARGET_TIME,
        ids=ids,
        groups=groups,
        estimate=estimate,
        truth=truth,
        n_samples=n_samples,
        trial=trial,
        steps=steps,
        weights=weights,
    )


def run_one(task):
    step_count, n, tr, source_ids, source_pts, source_groups, test_pts, truth = task
    ref_pts = sample(n, trial_seed(n, tr))
    pred, _volume = dymad_section(ref_pts, source_pts, test_pts, step_count)
    return step_count, n, tr, metric_rows(source_ids, source_groups, pred, truth, n, tr, step_count)


def plot_convergence(rows: list[dict[str, str]], path: Path) -> None:
    groups = [
        group
        for group in ("interior", "near_boundary")
        if any(row["source_group"] == group for row in rows)
    ]
    plot_convergence_curves(
        rows,
        path=path,
        steps_values=STEPS,
        target_time=TARGET_TIME,
        title="Disk uniform-mode convergence",
        source_groups=groups,
    )


def plot_max_abs_convergence(rows: list[dict[str, str]], path: Path) -> None:
    plot_max_abs_convergence_curves(
        rows,
        path=path,
        steps_values=STEPS,
        target_time=TARGET_TIME,
        title="Disk max-abs convergence",
    )


def plot_sections(n_samples: int, steps: int, trial: int, source_indices, path: Path) -> None:
    _source_ids, source_pts_all, _groups = sources()
    source_idx = list(source_indices)
    source_pts = source_pts_all[source_idx]
    pts = locations(SECTION_TEST_COUNT)
    truth = reference(source_pts, pts)
    pred, volume = dymad_section(
        sample(n_samples, trial_seed(n_samples, trial)), source_pts, pts, steps
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
        shared_image = None
        panels = (("Truth", truth[row]), ("Prediction", pred[row]))
        for ax, (title, values) in zip(axes[row, :2], panels, strict=True):
            image = ax.tricontourf(
                pts[:, 0], pts[:, 1], values, levels=40, cmap="viridis", vmin=vmin, vmax=vmax
            )
            shared_image = image
            ax.scatter(source_pts_all[idx, 0], source_pts_all[idx, 1], c="black", s=16)
            ax.set_title(title)
            ax.set_aspect("equal")
            ax.set_xticks(tick_values)
            ax.set_yticks(tick_values)
            ax.set_xlabel("x")
        if shared_image is not None:
            fig.colorbar(shared_image, ax=axes[row, :2], shrink=0.78)

        error_image = axes[row, 2].tricontourf(
            pts[:, 0],
            pts[:, 1],
            error,
            levels=40,
            cmap="coolwarm",
            vmin=-err_max,
            vmax=err_max,
        )
        axes[row, 2].scatter(source_pts_all[idx, 0], source_pts_all[idx, 1], c="black", s=16)
        axes[row, 2].set_title("Error")
        axes[row, 2].set_aspect("equal")
        axes[row, 2].set_xticks(tick_values)
        axes[row, 2].set_yticks(tick_values)
        axes[row, 2].set_xlabel("x")
        fig.colorbar(error_image, ax=axes[row, 2], shrink=0.78)
    for ax in axes[:, 0]:
        ax.set_ylabel("y")
    eps = TARGET_TIME / steps
    fig.suptitle(
        f"disk sections: N={n_samples}, eps={eps:g}, steps={steps}, trial={trial}, volume={volume:.6g}"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__" and ifrun:
    source_ids, source_pts, source_groups = sources()
    test_pts = locations(RUN_TEST_COUNT)
    truth = reference(source_pts, test_pts)
    tasks = [
        (step_count, n, tr, source_ids, source_pts, source_groups, test_pts, truth)
        for step_count in RUN_STEPS
        for n in RUN_SAMPLE_COUNTS
        for tr in range(RUN_TRIALS)
    ]
    write_rows(RAW_CSV, run_serial(tasks, run_one, case="disk", target_time=TARGET_TIME))

if __name__ == "__main__" and ifplt:
    if RAW_CSV.exists():
        raw_rows = read_rows(RAW_CSV)
        plot_convergence(raw_rows, FIG_PATH)
        print(f"Wrote {FIG_PATH}")
        plot_max_abs_convergence(raw_rows, MAX_ABS_FIG_PATH)
        print(f"Wrote {MAX_ABS_FIG_PATH}")
    else:
        print(f"Missing {RAW_CSV}; set ifrun = 1 or copy the disk CSV into place.")

if __name__ == "__main__" and ifsec:
    plot_sections(SECTION_N, SECTION_STEPS, SECTION_TRIAL, SECTION_SOURCE_INDICES, SECTION_FIG_PATH)
    print(f"Wrote {SECTION_FIG_PATH}")
