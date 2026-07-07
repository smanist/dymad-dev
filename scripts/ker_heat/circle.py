"""Circle heat-kernel section convergence with DyMAD uniform mode."""

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
from scipy.stats import qmc

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dymad.modules import KernelScDMHeat  # noqa: E402
from common import (  # noqa: E402
    metric_rows as build_metric_rows,
    periodic_heat_kernel,
    plot_convergence as plot_convergence_curves,
    plot_max_abs_convergence as plot_max_abs_convergence_curves,
    read_rows,
    run_serial,
    trial_seed as build_trial_seed,
    write_rows,
)

BASE_DIR = Path(__file__).resolve().parent
OUT = BASE_DIR / "runs" / "circle_redo"
RAW_CSV = OUT / "circle_raw_results.csv"
FIG_PATH = OUT / "convergence_circle.png"
MAX_ABS_FIG_PATH = OUT / "convergence_circle_max_abs.png"
SECTION_FIG_PATH = OUT / "section_circle.png"

TWO_PI = 2.0 * math.pi
TARGET_TIME = 0.01
STEPS = (1, 2, 4, 8, 16, 32)
EPSILONS = tuple(TARGET_TIME / step for step in STEPS)
SAMPLE_COUNTS = (512, 1024, 2048, 4096, 8192, 16384)
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
SECTION_N = 16384
SECTION_STEPS = 32
SECTION_TRIAL = 0
SECTION_SOURCE_INDICES = (0,)
SECTION_TEST_COUNT = TEST_COUNT

ifrun = 1
ifplt = 1
ifsec = 1


def trial_seed(n_samples: int, trial: int) -> int:
    return build_trial_seed(SEED, n_samples, trial)


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
    return periodic_heat_kernel(sources[:, 0], points[:, 0], TARGET_TIME, TWO_PI)


def dymad_section(
    ref_angles: np.ndarray, sources: np.ndarray, points: np.ndarray, steps: int
) -> tuple[np.ndarray, float]:
    epsilon = TARGET_TIME / steps
    kernel = KernelScDMHeat(
        in_dim=2,
        eps_init=epsilon,
        alpha_init=1.0,
        dtype=torch.float64,
        backend="keops",
    )
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
    return build_metric_rows(
        case="circle",
        target_time=TARGET_TIME,
        ids=ids,
        estimate=estimate,
        truth=truth,
        n_samples=n_samples,
        trial=trial,
        steps=steps,
    )


def run_one(task):
    step_count, n, tr, source_ids, source_pts, test_pts, truth = task
    ref_pts = sample_angles(n, trial_seed(n, tr))
    pred, _volume_hat = dymad_section(ref_pts, source_pts, test_pts, step_count)
    return step_count, n, tr, metric_rows(source_ids, pred, truth, n, tr, step_count)


def plot_convergence(rows: list[dict[str, str]], path: Path) -> None:
    plot_convergence_curves(
        rows,
        path=path,
        steps_values=STEPS,
        target_time=TARGET_TIME,
        title="Circle uniform-mode convergence",
    )


def plot_max_abs_convergence(rows: list[dict[str, str]], path: Path) -> None:
    plot_max_abs_convergence_curves(
        rows,
        path=path,
        steps_values=STEPS,
        target_time=TARGET_TIME,
        title="Circle max-abs convergence",
    )


def plot_sections(n_samples: int, steps: int, trial: int, source_indices, path: Path) -> None:
    _source_ids, sources = source_angles()
    source_idx = list(source_indices)
    source_pts = sources[source_idx]
    points = test_angles(SECTION_TEST_COUNT)
    truth = reference_kernel(source_pts, points)
    pred, volume_hat = dymad_section(
        sample_angles(n_samples, trial_seed(n_samples, trial)), source_pts, points, steps
    )
    fig, axes = plt.subplots(
        len(source_idx),
        3,
        figsize=(12.0, 3.0 * len(source_idx)),
        squeeze=False,
        constrained_layout=True,
    )
    for row, _idx in enumerate(source_idx):
        error = pred[row] - truth[row]
        panels = (("Truth", truth[row]), ("Prediction", pred[row]), ("Error", error))
        for ax, (title, values) in zip(axes[row], panels, strict=True):
            ax.plot(points[:, 0], values)
            ax.set_title(title)
            ax.set_xticks([0.0, math.pi, TWO_PI], [r"$0$", r"$\pi$", r"$2\pi$"])
            ax.set_xlabel(r"$\theta$")
            ax.grid(True, alpha=0.25)
    eps = TARGET_TIME / steps
    fig.suptitle(
        f"circle sections: N={n_samples}, eps={eps:g}, steps={steps}, trial={trial}, volume={volume_hat:.6g}",
        y=1.08,
    )
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
    write_rows(RAW_CSV, run_serial(tasks, run_one, case="circle", target_time=TARGET_TIME))

if __name__ == "__main__" and ifplt:
    if RAW_CSV.exists():
        raw_rows = read_rows(RAW_CSV)
        plot_convergence(raw_rows, FIG_PATH)
        print(f"Wrote {FIG_PATH}")
        plot_max_abs_convergence(raw_rows, MAX_ABS_FIG_PATH)
        print(f"Wrote {MAX_ABS_FIG_PATH}")
    else:
        print(f"Missing {RAW_CSV}; set ifrun = 1 or copy the circle CSV into place.")

if __name__ == "__main__" and ifsec:
    plot_sections(SECTION_N, SECTION_STEPS, SECTION_TRIAL, SECTION_SOURCE_INDICES, SECTION_FIG_PATH)
    print(f"Wrote {SECTION_FIG_PATH}")
