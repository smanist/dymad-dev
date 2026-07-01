"""Disk heat-kernel section convergence with DyMAD uniform mode."""

from __future__ import annotations

# ruff: noqa: E402, I001

import csv
import math
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_MPL_CONFIG_DIR = Path(tempfile.gettempdir()) / "dymad_matplotlib"
_MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))

import matplotlib
import numpy as np
import torch
from scipy import special
from scipy.stats import qmc

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dymad.modules import KernelSparseScDM  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent
OUT = BASE_DIR / "runs" / "disk_redo"
RAW_CSV = OUT / "disk_raw_results.csv"
FIG_PATH = OUT / "convergence_disk.png"
SECTION_FIG_PATH = OUT / "section_disk.png"

TARGET_TIME = 0.00125
STEPS = (1, 2, 4, 8)
EPSILONS = tuple(TARGET_TIME / step for step in STEPS)
SAMPLE_COUNTS = (2048, 4096, 8192, 16384, 32768, 65536)
TRIALS = 8
TEST_COUNT = 4096
SEED = 2026061801
CASE_INDEX = 8
DISK_AREA = math.pi
KERNEL_TOL = 1e-8
REFERENCE_TERM_TOL = 1e-13
REFERENCE_ANGULAR_ORDER = 42
REFERENCE_RADIAL_ROOTS = 42

# Full sparse sections at the largest counts are memory-sensitive.  If a run
# crashes, raise KERNEL_TOL slightly, keeping it as low as the machine allows.
RUN_STEPS = STEPS
RUN_SAMPLE_COUNTS = SAMPLE_COUNTS
RUN_TRIALS = TRIALS
RUN_TEST_COUNT = TEST_COUNT
MAX_WORKERS = 4
SECTION_N = 65536
SECTION_STEPS = 8
SECTION_TRIAL = 0
SECTION_SOURCE_INDICES = (0, 3)
SECTION_TEST_COUNT = TEST_COUNT

ifrun = 1
ifplt = 1
ifsec = 0


def trial_seed(n_samples: int, trial: int) -> int:
    return int(SEED + 1_000_003 * CASE_INDEX + 10_007 * trial + n_samples)


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
    kernel = KernelSparseScDM(
        in_dim=2,
        eps_init=epsilon,
        t_init=1.0,
        dtype=torch.float64,
        kernel_tol=KERNEL_TOL,
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
    diff = estimate - truth
    weights = np.full(truth.shape[1], DISK_AREA / truth.shape[1], dtype=float)
    l2_error = np.sqrt(np.sum(weights[None, :] * diff * diff, axis=1))
    reference_l2 = np.sqrt(np.sum(weights[None, :] * truth * truth, axis=1))
    rel_l2 = l2_error / np.maximum(reference_l2, 1e-300)
    max_abs = np.max(np.abs(diff), axis=1)
    return [
        {
            "case": "disk",
            "steps": steps,
            "epsilon": TARGET_TIME / steps,
            "n_samples": n_samples,
            "trial": trial,
            "source_id": ids[i],
            "source_group": groups[i],
            "relative_l2_error": float(rel_l2[i]),
            "max_abs_error": float(max_abs[i]),
        }
        for i in range(len(ids))
    ]


def run_one(task):
    step_count, n, tr, source_ids, source_pts, source_groups, test_pts, truth = task
    ref_pts = sample(n, trial_seed(n, tr))
    pred, _volume = dymad_section(ref_pts, source_pts, test_pts, step_count)
    return step_count, n, tr, metric_rows(source_ids, source_groups, pred, truth, n, tr, step_count)


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "case",
        "steps",
        "epsilon",
        "n_samples",
        "trial",
        "source_id",
        "source_group",
        "relative_l2_error",
        "max_abs_error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def median_curve(
    rows: list[dict[str, str]], steps: int, metric: str, source_group: str
) -> tuple[list[int], list[float]]:
    selected_ns = sorted(
        {
            int(row["n_samples"])
            for row in rows
            if int(row["steps"]) == steps and row["source_group"] == source_group
        }
    )
    medians = [
        float(
            np.median(
                [
                    float(row[metric])
                    for row in rows
                    if int(row["steps"]) == steps
                    and int(row["n_samples"]) == n
                    and row["source_group"] == source_group
                ]
            )
        )
        for n in selected_ns
    ]
    return selected_ns, medians


def plot_convergence(rows: list[dict[str, str]], path: Path) -> None:
    groups = [
        group
        for group in ("interior", "near_boundary")
        if any(row["source_group"] == group for row in rows)
    ]
    fig, axes = plt.subplots(
        len(groups), 2, figsize=(12.5, 4.4 * len(groups)), squeeze=False, constrained_layout=True
    )
    colors = dict(zip(STEPS, plt.cm.tab10(np.linspace(0.0, 1.0, len(STEPS)))))
    for row_idx, group in enumerate(groups):
        for steps in STEPS:
            eps = TARGET_TIME / steps
            ns, rel = median_curve(rows, steps, "relative_l2_error", group)
            _, mx = median_curve(rows, steps, "max_abs_error", group)
            label = f"eps={eps:g}, p={steps}"
            axes[row_idx, 0].loglog(ns, rel, marker="o", color=colors[steps], label=label)
            axes[row_idx, 1].loglog(ns, mx, marker="o", color=colors[steps], label=label)
        axes[row_idx, 0].set_title(f"{group}: relative L2")
        axes[row_idx, 1].set_title(f"{group}: max abs")
    for ax in axes.flat:
        ax.set_xlabel("sample count N")
        ax.set_ylabel("median error")
        ax.grid(True, which="both", alpha=0.28)
        ax.legend(fontsize=8)
    fig.suptitle("Disk DyMAD uniform-mode convergence")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_sections(n_samples: int, steps: int, trial: int, source_indices, path: Path) -> None:
    source_ids, source_pts_all, _groups = sources()
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
    for row, idx in enumerate(source_idx):
        error = pred[row] - truth[row]
        vmin = min(float(np.min(truth[row])), float(np.min(pred[row])))
        vmax = max(float(np.max(truth[row])), float(np.max(pred[row])))
        err_max = max(float(np.max(np.abs(error))), np.finfo(float).tiny)
        panels = (
            ("truth", truth[row], "viridis", vmin, vmax),
            ("DyMAD", pred[row], "viridis", vmin, vmax),
            ("error", error, "coolwarm", -err_max, err_max),
        )
        for ax, (title, values, cmap, lo, hi) in zip(axes[row], panels, strict=True):
            image = ax.tricontourf(
                pts[:, 0], pts[:, 1], values, levels=40, cmap=cmap, vmin=lo, vmax=hi
            )
            ax.scatter(source_pts_all[idx, 0], source_pts_all[idx, 1], c="black", s=16)
            ax.set_title(f"{source_ids[idx]} {title}")
            ax.set_aspect("equal")
            fig.colorbar(image, ax=ax, shrink=0.78)
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
    raw_rows: list[dict[str, object]] = []
    if MAX_WORKERS > 1:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = [pool.submit(run_one, task) for task in tasks]
            results = []
            for future in as_completed(futures):
                step_count, n, tr, rows = future.result()
                results.append((step_count, n, tr, rows))
                print(
                    f"done disk steps={step_count} eps={TARGET_TIME / step_count:g} n={n} trial={tr}",
                    flush=True,
                )
    else:
        results = [run_one(task) for task in tasks]
    for step_count, n, tr, rows in results:
        raw_rows.extend(rows)
        if MAX_WORKERS <= 1:
            print(f"done disk steps={step_count} eps={TARGET_TIME / step_count:g} n={n} trial={tr}")
    write_rows(RAW_CSV, raw_rows)

if __name__ == "__main__" and ifplt:
    if RAW_CSV.exists():
        plot_convergence(read_rows(RAW_CSV), FIG_PATH)
        print(f"Wrote {FIG_PATH}")
    else:
        print(f"Missing {RAW_CSV}; set ifrun = 1 or copy the disk CSV into place.")

if __name__ == "__main__" and ifsec:
    plot_sections(SECTION_N, SECTION_STEPS, SECTION_TRIAL, SECTION_SOURCE_INDICES, SECTION_FIG_PATH)
    print(f"Wrote {SECTION_FIG_PATH}")
