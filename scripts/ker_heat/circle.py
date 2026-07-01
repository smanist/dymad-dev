"""Circle heat-kernel section convergence with DyMAD uniform mode."""

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

from dymad.modules import KernelScDM  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent
OUT = BASE_DIR / "runs" / "circle_redo"
RAW_CSV = OUT / "circle_raw_results.csv"
FIG_PATH = OUT / "convergence_circle.png"

TWO_PI = 2.0 * math.pi
TARGET_TIME = 0.01
# STEPS = (1, 2, 4, 8, 16, 32)
STEPS = (8,)
EPSILONS = tuple(TARGET_TIME / step for step in STEPS)
# SAMPLE_COUNTS = (512, 1024, 2048, 4096, 8192, 16384)
SAMPLE_COUNTS = (512, 1024, 2048, 4096, 8192)
TRIALS = 8
TEST_COUNT = 4096
SEED = 2026061801
SOURCE_FRACTIONS = (0.125, 0.375, 0.625, 0.875)

# Full DyMAD dense sections are expensive at the largest counts.  For quick
# smoke runs, override these near the if-block below.
RUN_STEPS = STEPS
RUN_SAMPLE_COUNTS = SAMPLE_COUNTS
RUN_TRIALS = TRIALS
RUN_TEST_COUNT = TEST_COUNT
MAX_WORKERS = 4

ifrun = 1
ifplt = 1


def trial_seed(n_samples: int, trial: int) -> int:
    return SEED + n_samples + 10_007 * trial


def sample_angles(n_samples: int, seed: int) -> np.ndarray:
    sampler = qmc.Sobol(d=1, scramble=True, seed=seed)
    return TWO_PI * sampler.random_base2(math.ceil(math.log2(n_samples)))[:n_samples]


def test_angles(n_points: int) -> np.ndarray:
    return (TWO_PI * (np.arange(n_points, dtype=float) + 0.5) / n_points)[:, None]


def source_angles() -> tuple[list[str], np.ndarray]:
    ids = [f"s1_q{i}" for i in range(len(SOURCE_FRACTIONS))]
    return ids, TWO_PI * np.asarray(SOURCE_FRACTIONS, dtype=float)[:, None]


def embed(theta: np.ndarray) -> np.ndarray:
    return np.column_stack((np.cos(theta[:, 0]), np.sin(theta[:, 0])))


def reference_kernel(sources: np.ndarray, points: np.ndarray) -> np.ndarray:
    images = max(4, int(math.ceil(6.0 * math.sqrt(TARGET_TIME) / TWO_PI)) + 4)
    diff = sources[:, 0, None] - points[:, 0][None, :]
    values = np.zeros_like(diff)
    scale = math.sqrt(4.0 * math.pi * TARGET_TIME)
    for image in range(-images, images + 1):
        values += np.exp(-((diff + image * TWO_PI) ** 2) / (4.0 * TARGET_TIME)) / scale
    return values


def dymad_section(
    ref_angles: np.ndarray, sources: np.ndarray, points: np.ndarray, steps: int
) -> tuple[np.ndarray, float]:
    epsilon = TARGET_TIME / steps
    kernel = KernelScDM(in_dim=2, eps_init=epsilon, t_init=1.0, dtype=torch.float64)
    kernel.set_reference_data(torch.as_tensor(embed(ref_angles), dtype=torch.float64))
    values = kernel.heat_kernel(
        torch.as_tensor(embed(points), dtype=torch.float64),
        torch.as_tensor(embed(sources), dtype=torch.float64),
        mode="uniform",
        steps=steps,
        volume_normalization="estimate_volume",
        volume_dim=1,
        volume_estimate_warnings=False,
        return_diagnostics=True,
    )
    section, diagnostics = values
    return section.detach().cpu().numpy().T, float(diagnostics["volume"])


def metric_rows(ids, estimate, truth, n_samples: int, trial: int, steps: int):
    diff = estimate - truth
    rel_l2 = np.sqrt(np.mean(diff * diff, axis=1)) / np.sqrt(np.mean(truth * truth, axis=1))
    max_abs = np.max(np.abs(diff), axis=1)
    return [
        {
            "case": "circle",
            "steps": steps,
            "epsilon": TARGET_TIME / steps,
            "n_samples": n_samples,
            "trial": trial,
            "source_id": ids[i],
            "relative_l2_error": float(rel_l2[i]),
            "max_abs_error": float(max_abs[i]),
        }
        for i in range(len(ids))
    ]


def run_one(task):
    step_count, n, tr, source_ids, source_pts, test_pts, truth = task
    ref_pts = sample_angles(n, trial_seed(n, tr))
    pred, _volume_hat = dymad_section(ref_pts, source_pts, test_pts, step_count)
    return step_count, n, tr, metric_rows(source_ids, pred, truth, n, tr, step_count)


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "case",
        "steps",
        "epsilon",
        "n_samples",
        "trial",
        "source_id",
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
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0), constrained_layout=True)
    for steps in STEPS:
        eps = TARGET_TIME / steps
        ns, rel = median_curve(rows, steps, "relative_l2_error")
        _, mx = median_curve(rows, steps, "max_abs_error")
        label = f"eps={eps:g}, p={steps}"
        axes[0].loglog(ns, rel, marker="o", label=label)
        axes[1].loglog(ns, mx, marker="o", label=label)
    axes[0].set_title("relative L2")
    axes[1].set_title("max abs")
    for ax in axes:
        ax.set_xlabel("sample count N")
        ax.grid(True, which="both", alpha=0.28)
        ax.legend(fontsize=8)
    fig.suptitle("Circle DyMAD uniform-mode convergence")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__" and ifrun:
    source_ids, source_pts = source_angles()
    test_pts = test_angles(RUN_TEST_COUNT)
    truth = reference_kernel(source_pts, test_pts)
    tasks = [
        (step_count, n, tr, source_ids, source_pts, test_pts, truth)
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
                    f"done circle steps={step_count} eps={TARGET_TIME / step_count:g} "
                    f"n={n} trial={tr}",
                    flush=True,
                )
    else:
        results = [run_one(task) for task in tasks]
    for step_count, n, tr, rows in results:
        raw_rows.extend(rows)
        if MAX_WORKERS <= 1:
            print(
                f"done circle steps={step_count} eps={TARGET_TIME / step_count:g} n={n} trial={tr}"
            )
    write_rows(RAW_CSV, raw_rows)

if __name__ == "__main__" and ifplt:
    if RAW_CSV.exists():
        plot_convergence(read_rows(RAW_CSV), FIG_PATH)
        print(f"Wrote {FIG_PATH}")
    else:
        print(f"Missing {RAW_CSV}; set ifrun = 1 or copy the circle CSV into place.")
