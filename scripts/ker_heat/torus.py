"""Flat 2-torus heat-kernel section convergence with DyMAD uniform mode."""

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
from scipy.stats import qmc

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dymad.modules import KernelSparseScDM  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent
OUT = BASE_DIR / "runs" / "torus_redo"
RAW_CSV = OUT / "torus_raw_results.csv"
FIG_PATH = OUT / "convergence_torus.png"
SECTION_FIG_PATH = OUT / "section_torus.png"

TWO_PI = 2.0 * math.pi
TORUS_AREA = TWO_PI * TWO_PI
TARGET_TIME = 0.01
STEPS = (1, 2, 4, 8)
EPSILONS = tuple(TARGET_TIME / step for step in STEPS)
SAMPLE_COUNTS = (8192, 16384, 32768, 65536)
TRIALS = 8
TEST_COUNT = 4096
SEED = 2026061801
CASE_INDEX = 4
KERNEL_TOL = 1e-10

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
SECTION_SOURCE_INDICES = (0,)
SECTION_TEST_COUNT = TEST_COUNT

ifrun = 0
ifplt = 0
ifsec = 1


def trial_seed(n_samples: int, trial: int) -> int:
    return int(SEED + 1_000_003 * CASE_INDEX + 10_007 * trial + n_samples)


def sample(n_samples: int, seed: int) -> np.ndarray:
    sampler = qmc.Sobol(d=2, scramble=True, seed=seed)
    return TWO_PI * sampler.random_base2(math.ceil(math.log2(n_samples)))[:n_samples]


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


def circle_kernel(src: np.ndarray, pts: np.ndarray, t: float = TARGET_TIME) -> np.ndarray:
    images = max(4, int(math.ceil(6.0 * math.sqrt(t) / TWO_PI)) + 4)
    diff = src[:, None] - pts[None, :]
    out = np.zeros_like(diff)
    scale = math.sqrt(4.0 * math.pi * t)
    for image in range(-images, images + 1):
        out += np.exp(-((diff + image * TWO_PI) ** 2) / (4.0 * t)) / scale
    return out


def reference(src: np.ndarray, pts: np.ndarray, t: float = TARGET_TIME) -> np.ndarray:
    return circle_kernel(src[:, 0], pts[:, 0], t) * circle_kernel(src[:, 1], pts[:, 1], t)


def dymad_section(
    ref_angles: np.ndarray, src: np.ndarray, pts: np.ndarray, steps: int
) -> tuple[np.ndarray, float]:
    epsilon = TARGET_TIME / steps
    kernel = KernelSparseScDM(
        in_dim=4,
        eps_init=epsilon,
        t_init=1.0,
        dtype=torch.float64,
        kernel_tol=KERNEL_TOL,
    )
    kernel.set_reference_data(torch.as_tensor(embed(ref_angles), dtype=torch.float64))
    values = kernel.heat_kernel(
        torch.as_tensor(embed(pts), dtype=torch.float64),
        torch.as_tensor(embed(src), dtype=torch.float64),
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
    weights = np.full(truth.shape[1], TORUS_AREA / truth.shape[1], dtype=float)
    l2_error = np.sqrt(np.sum(weights[None, :] * diff * diff, axis=1))
    reference_l2 = np.sqrt(np.sum(weights[None, :] * truth * truth, axis=1))
    rel_l2 = l2_error / np.maximum(reference_l2, 1e-300)
    max_abs = np.max(np.abs(diff), axis=1)
    return [
        {
            "case": "torus",
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
    rows: list[dict[str, str]], steps: int, metric: str
) -> tuple[list[int], list[float]]:
    selected_ns = sorted({int(row["n_samples"]) for row in rows if int(row["steps"]) == steps})
    medians = [
        float(
            np.median(
                [
                    float(row[metric])
                    for row in rows
                    if int(row["steps"]) == steps and int(row["n_samples"]) == n
                ]
            )
        )
        for n in selected_ns
    ]
    return selected_ns, medians


def plot_convergence(rows: list[dict[str, str]], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), constrained_layout=True)
    colors = dict(zip(STEPS, plt.cm.tab10(np.linspace(0.0, 1.0, len(STEPS)))))
    for steps in STEPS:
        eps = TARGET_TIME / steps
        ns, rel = median_curve(rows, steps, "relative_l2_error")
        _, mx = median_curve(rows, steps, "max_abs_error")
        label = f"eps={eps:g}, p={steps}"
        axes[0].loglog(ns, rel, marker="o", color=colors[steps], label=label)
        axes[1].loglog(ns, mx, marker="o", color=colors[steps], label=label)
    axes[0].set_title("relative L2")
    axes[1].set_title("max abs")
    for ax in axes:
        ax.set_xlabel("sample count N")
        ax.set_ylabel("median error")
        ax.grid(True, which="both", alpha=0.28)
        ax.legend(fontsize=8)
    fig.suptitle("Torus DyMAD uniform-mode convergence")
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
            image = ax.pcolormesh(
                xx, yy, values.reshape(side, side), shading="auto", cmap=cmap, vmin=lo, vmax=hi
            )
            ax.scatter(source_pts_all[idx, 0], source_pts_all[idx, 1], c="black", s=16)
            ax.set_title(f"{source_ids[idx]} {title}")
            ax.set_aspect("equal")
            fig.colorbar(image, ax=ax, shrink=0.78)
    eps = TARGET_TIME / steps
    fig.suptitle(
        f"torus sections: N={n_samples}, eps={eps:g}, steps={steps}, trial={trial}, volume={volume:.6g}"
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
                    f"done torus steps={step_count} eps={TARGET_TIME / step_count:g} n={n} trial={tr}",
                    flush=True,
                )
    else:
        results = [run_one(task) for task in tasks]
    for step_count, n, tr, rows in results:
        raw_rows.extend(rows)
        if MAX_WORKERS <= 1:
            print(
                f"done torus steps={step_count} eps={TARGET_TIME / step_count:g} n={n} trial={tr}"
            )
    write_rows(RAW_CSV, raw_rows)

if __name__ == "__main__" and ifplt:
    if RAW_CSV.exists():
        plot_convergence(read_rows(RAW_CSV), FIG_PATH)
        print(f"Wrote {FIG_PATH}")
    else:
        print(f"Missing {RAW_CSV}; set ifrun = 1 or copy the torus CSV into place.")

if __name__ == "__main__" and ifsec:
    plot_sections(SECTION_N, SECTION_STEPS, SECTION_TRIAL, SECTION_SOURCE_INDICES, SECTION_FIG_PATH)
    print(f"Wrote {SECTION_FIG_PATH}")
