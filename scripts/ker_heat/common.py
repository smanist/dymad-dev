"""Shared helpers for heat-kernel convergence scripts."""

from __future__ import annotations

import csv
import math
import multiprocessing as mp
import os
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import torch

from dymad.kernel_analysis import DiffusionHeatSections
from dymad.modules.kernel import KernelBackend

BASE_FIELDS = [
    "case",
    "steps",
    "epsilon",
    "n_samples",
    "trial",
    "source_id",
    "relative_l2_error",
    "max_abs_error",
]

CONVERGENCE_FONT_SIZE = 14
SECTION_FONT_SIZE = 14


@dataclass(frozen=True)
class HeatCase:
    study: str
    title: str
    target_time: float
    steps: tuple[int, ...]
    sample_counts: tuple[int, ...]
    section_n: int
    section_steps: int
    section_source_indices: tuple[int, ...]
    parallel: bool
    fit_point_count: int


def section_plot_title(case: HeatCase, steps: int, *, include_study: bool = True) -> str:
    """Format a heat-section figure title for a selected step count."""

    epsilon = case.target_time / steps
    details = f"N={case.section_n}, eps={epsilon:g}, steps={steps}"
    return f"{case.study}: {details}" if include_study else details


def section_plot_steps(case: HeatCase, *, smallest_epsilon: bool = False) -> int:
    """Choose the configured section steps or the value with the smallest epsilon."""

    return max(case.steps) if smallest_epsilon else case.section_steps


@dataclass(frozen=True)
class HeatSectionSpec:
    """Geometry and normalization inputs shared by a heat-section evaluation."""

    ambient_dim: int
    encode: Callable[[np.ndarray], np.ndarray]
    mode: Literal["density", "uniform"]
    alpha: float | None = None
    mass_normalization: Literal["source", "median", "none"] = "source"
    volume_normalization: Literal["none", "estimate_volume"] = "none"
    volume_dim: int | None = None
    location_weights: Callable[[np.ndarray], np.ndarray] | None = None

    def __post_init__(self) -> None:
        if self.ambient_dim <= 0:
            raise ValueError("ambient_dim must be positive.")
        if self.alpha is not None and self.mode != "density":
            raise ValueError("alpha is only supported for density heat kernels.")
        if self.location_weights is not None and self.volume_normalization != "none":
            raise ValueError("location_weights cannot be combined with volume_normalization.")
        if self.volume_normalization == "estimate_volume" and self.volume_dim is None:
            raise ValueError("volume_dim is required when estimating volume.")
        if self.volume_dim is not None and self.volume_dim <= 0:
            raise ValueError("volume_dim must be positive.")


def evaluate_heat_section(
    spec: HeatSectionSpec,
    reference: np.ndarray,
    sources: np.ndarray,
    locations: np.ndarray,
    *,
    epsilon: float,
    steps: int,
    backend: KernelBackend = "keops",
) -> np.ndarray:
    """Evaluate one source-by-location heat-kernel section from a shared spec."""

    kernel = DiffusionHeatSections(
        in_dim=spec.ambient_dim,
        eps_init=epsilon,
        alpha_init=1.0,
        dtype=torch.float64,
        backend=backend,
    )
    kernel.set_reference_data(torch.as_tensor(spec.encode(reference), dtype=torch.float64))
    kwargs: dict[str, Any] = {
        "mode": spec.mode,
        "steps": steps,
        "mass_normalization": spec.mass_normalization,
        "volume_normalization": spec.volume_normalization,
    }
    if spec.alpha is not None:
        kwargs["alpha"] = spec.alpha
    if spec.volume_dim is not None:
        kwargs["volume_dim"] = spec.volume_dim
        kwargs["volume_estimate_warnings"] = False
    if spec.location_weights is not None:
        kwargs["location_weights"] = torch.as_tensor(
            spec.location_weights(locations), dtype=torch.float64
        )
    section = kernel.heat_kernel(
        torch.as_tensor(spec.encode(locations), dtype=torch.float64),
        torch.as_tensor(spec.encode(sources), dtype=torch.float64),
        **kwargs,
    )
    if not isinstance(section, torch.Tensor):
        raise RuntimeError("Expected a single heat-kernel section.")
    return section.detach().cpu().numpy().T


def trial_seed(seed: int, n_samples: int, trial: int, case_index: int = 0) -> int:
    return int(seed + 1_000_003 * case_index + 10_007 * trial + n_samples)


def periodic_heat_kernel(
    sources: np.ndarray, points: np.ndarray, t: float, period: float = 2.0 * math.pi
) -> np.ndarray:
    images = max(4, int(math.ceil(6.0 * math.sqrt(t) / period)) + 4)
    diff = sources[:, None] - points[None, :]
    values = np.zeros_like(diff)
    scale = math.sqrt(4.0 * math.pi * t)
    for image in range(-images, images + 1):
        values += np.exp(-((diff + image * period) ** 2) / (4.0 * t)) / scale
    return values


def metric_rows(
    *,
    case: str,
    target_time: float,
    ids: Sequence[str],
    estimate: np.ndarray,
    truth: np.ndarray,
    n_samples: int,
    trial: int,
    steps: int,
    weights: np.ndarray | None = None,
    groups: Sequence[str] | None = None,
) -> list[dict[str, object]]:
    diff = estimate - truth
    if weights is None:
        l2_error = np.sqrt(np.mean(diff * diff, axis=1))
        reference_l2 = np.sqrt(np.mean(truth * truth, axis=1))
    else:
        l2_error = np.sqrt(np.sum(weights[None, :] * diff * diff, axis=1))
        reference_l2 = np.sqrt(np.sum(weights[None, :] * truth * truth, axis=1))
    rel_l2 = l2_error / np.maximum(reference_l2, 1e-300)
    max_abs = np.max(np.abs(diff), axis=1)
    rows: list[dict[str, object]] = []
    for i, source_id in enumerate(ids):
        row: dict[str, object] = {
            "case": case,
            "steps": steps,
            "epsilon": target_time / steps,
            "n_samples": n_samples,
            "trial": trial,
            "source_id": source_id,
            "relative_l2_error": float(rel_l2[i]),
            "max_abs_error": float(max_abs[i]),
        }
        if groups is not None:
            row["source_group"] = groups[i]
        rows.append(row)
    return rows


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fields = BASE_FIELDS.copy()
    if any("source_group" in row for row in rows):
        fields.insert(fields.index("relative_l2_error"), "source_group")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def study_artifact_paths(base_dir: Path, case: HeatCase) -> tuple[Path, Path, Path, Path]:
    out = base_dir / "runs" / case.study
    return (
        out / "raw_results.csv",
        out / f"heat_conv_eN_{case.study}.png",
        out / f"heat_conv_{case.study}.png",
        out / f"heat_section_{case.study}.png",
    )


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def run_serial(
    tasks: Iterable[Any],
    run_one: Callable[[Any], tuple[int, int, int, list[dict[str, object]]]],
    *,
    case: str,
    target_time: float,
) -> list[dict[str, object]]:
    raw_rows: list[dict[str, object]] = []
    for task in tasks:
        step_count, n_samples, trial, rows = run_one(task)
        raw_rows.extend(rows)
        print(
            f"done {case} steps={step_count} eps={target_time / step_count:g} "
            f"n={n_samples} trial={trial}",
            flush=True,
        )
    return raw_rows


def run_parallel(
    tasks: Iterable[Any],
    run_one: Callable[[Any], tuple[int, int, int, list[dict[str, object]]]],
    *,
    case: str,
    target_time: float,
    max_workers: int | None = None,
) -> list[dict[str, object]]:
    worker_count = max_workers if max_workers is not None else max(1, (os.cpu_count() or 1) // 2)
    raw_rows: list[dict[str, object]] = []
    pool_kwargs: dict[str, Any] = {"max_workers": worker_count}
    if "fork" in mp.get_all_start_methods():
        pool_kwargs["mp_context"] = mp.get_context("fork")
    with ProcessPoolExecutor(**pool_kwargs) as pool:
        for step_count, n_samples, trial, rows in pool.map(run_one, tasks):
            raw_rows.extend(rows)
            print(
                f"done {case} steps={step_count} eps={target_time / step_count:g} "
                f"n={n_samples} trial={trial}",
                flush=True,
            )
    return raw_rows


def run_study(
    base_dir: Path,
    case: HeatCase,
    tasks: Iterable[Any],
    run_one: Callable[[Any], tuple[int, int, int, list[dict[str, object]]]],
    *,
    max_workers: int | None = None,
) -> Path:
    raw_csv, _conv_en_path, _conv_path, _section_path = study_artifact_paths(base_dir, case)
    task_values = list(tasks)
    if case.parallel:
        try:
            rows = run_parallel(
                task_values,
                run_one,
                case=case.study,
                target_time=case.target_time,
                max_workers=max_workers,
            )
        except PermissionError as error:
            print(f"parallel workers unavailable for {case.study}; retrying serially: {error}")
            rows = run_serial(task_values, run_one, case=case.study, target_time=case.target_time)
        except FileNotFoundError as error:
            if "brew_prefix" not in str(error):
                raise
            print(f"KeOps worker setup failed for {case.study}; retrying serially: {error}")
            rows = run_serial(task_values, run_one, case=case.study, target_time=case.target_time)
    else:
        rows = run_serial(task_values, run_one, case=case.study, target_time=case.target_time)
    write_rows(raw_csv, rows)
    return raw_csv


def median_curve(
    rows: list[dict[str, str]],
    *,
    steps: int,
    metric: str,
    source_group: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    selected_ns = sorted(
        {
            int(row["n_samples"])
            for row in rows
            if int(row["steps"]) == steps
            and (source_group is None or row.get("source_group") == source_group)
        }
    )
    centers = []
    tiny = np.finfo(float).tiny
    for n_samples in selected_ns:
        values = np.asarray(
            [
                float(row[metric])
                for row in rows
                if int(row["steps"]) == steps
                and int(row["n_samples"]) == n_samples
                and (source_group is None or row.get("source_group") == source_group)
            ],
            dtype=float,
        )
        centers.append(max(float(np.median(values)), tiny))
    return np.asarray(selected_ns, dtype=float), np.asarray(centers, dtype=float)


def epsilon_curve_at_largest_n(
    rows: list[dict[str, str]],
    *,
    metric: str,
    source_group: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    epsilons = sorted(
        {
            float(row["epsilon"])
            for row in rows
            if source_group is None or row.get("source_group") == source_group
        }
    )
    centers = []
    n_values = []
    tiny = np.finfo(float).tiny
    for epsilon in epsilons:
        matching = [
            row
            for row in rows
            if float(row["epsilon"]) == epsilon
            and (source_group is None or row.get("source_group") == source_group)
        ]
        largest_n = max(int(row["n_samples"]) for row in matching)
        values = np.asarray(
            [float(row[metric]) for row in matching if int(row["n_samples"]) == largest_n],
            dtype=float,
        )
        centers.append(max(float(np.median(values)), tiny))
        n_values.append(largest_n)
    return (
        np.asarray(epsilons, dtype=float),
        np.asarray(centers, dtype=float),
        np.asarray(n_values, dtype=float),
    )


def fit_loglog_rate(x_values: np.ndarray, y_values: np.ndarray) -> tuple[float, float] | None:
    mask = np.isfinite(x_values) & np.isfinite(y_values) & (x_values > 0.0) & (y_values > 0.0)
    if int(np.count_nonzero(mask)) < 2:
        return None
    slope, intercept = np.polyfit(np.log(x_values[mask]), np.log(y_values[mask]), deg=1)
    return float(slope), float(math.exp(intercept))


def line_colors(values: Sequence[int]) -> dict[int, tuple[float, float, float]]:
    base = np.asarray([0.04, 0.20, 0.55], dtype=float)
    white = np.ones(3, dtype=float)
    blends = np.linspace(0.0, 0.62, len(values))
    return {
        value: tuple(float(channel) for channel in (1.0 - blend) * base + blend * white)
        for value, blend in zip(values, blends, strict=True)
    }


def plot_convergence(
    rows: list[dict[str, str]],
    *,
    path: Path,
    steps_values: Sequence[int],
    target_time: float,
    title: str,
    source_groups: Sequence[str] | None = None,
    metrics: Sequence[tuple[str, str]] = (
        ("relative_l2_error", "relative L2"),
        ("max_abs_error", "MAE"),
    ),
) -> None:
    groups = [None] if source_groups is None else list(source_groups)
    fig, axes = plt.subplots(
        len(groups),
        len(metrics),
        figsize=(6.2 * len(metrics), 4.8 * len(groups)),
        squeeze=False,
        constrained_layout=True,
    )
    colors = line_colors(steps_values)
    for row_idx, group in enumerate(groups):
        for step_count in steps_values:
            eps = target_time / step_count
            label = f"eps={eps:g}, p={step_count}"
            for column, (metric, _label) in enumerate(metrics):
                ns, values = median_curve(rows, steps=step_count, metric=metric, source_group=group)
                axes[row_idx, column].loglog(
                    ns, values, marker="o", color=colors[step_count], label=label
                )
        if len(metrics) > 1:
            prefix = "" if group is None else f"{group}: "
            for column, (_metric, label) in enumerate(metrics):
                axes[row_idx, column].set_title(f"{prefix}{label}", fontsize=CONVERGENCE_FONT_SIZE)
    for ax in axes.flat:
        ax.set_xlabel("N", fontsize=CONVERGENCE_FONT_SIZE)
        ax.set_ylabel(
            "MAE" if len(metrics) == 1 else "median error", fontsize=CONVERGENCE_FONT_SIZE
        )
        ax.tick_params(axis="both", which="both", labelsize=CONVERGENCE_FONT_SIZE)
        ax.grid(True, which="both", alpha=0.28)
        ax.legend(fontsize=10 if len(steps_values) > 4 else CONVERGENCE_FONT_SIZE)
        if len(metrics) == 1:
            ns = sorted({int(row["n_samples"]) for row in rows})
            ax.set_xticks(ns, [rf"$2^{{{int(math.log2(n))}}}$" for n in ns])
            ax.set_yticks([])
    if len(metrics) > 1:
        fig.suptitle(title, fontsize=CONVERGENCE_FONT_SIZE)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_error_vs_epsilon_at_largest_n(
    rows: list[dict[str, str]],
    *,
    path: Path,
    title: str,
    metric: str = "max_abs_error",
    ylabel: str = "MAE",
    fit_point_count: int | None = None,
    source_groups: Sequence[str] | None = None,
) -> None:
    groups = [None] if source_groups is None else list(source_groups)
    fig, axes = plt.subplots(
        len(groups),
        1,
        figsize=(7.2, 4.8 * len(groups)),
        squeeze=False,
        constrained_layout=True,
    )
    color = "#093a6f"
    for row_idx, group in enumerate(groups):
        ax = axes[row_idx, 0]
        epsilons, errors, _n_values = epsilon_curve_at_largest_n(
            rows, metric=metric, source_group=group
        )
        if len(epsilons) == 0:
            continue
        ax.loglog(
            epsilons,
            errors,
            marker="o",
            linewidth=2.0,
            color=color,
            label="DM",
        )
        fit_epsilons = epsilons if fit_point_count is None else epsilons[:fit_point_count]
        fit_errors = errors if fit_point_count is None else errors[:fit_point_count]
        rate = fit_loglog_rate(fit_epsilons, fit_errors)
        if rate is not None:
            slope, coefficient = rate
            ax.loglog(
                fit_epsilons,
                coefficient * fit_epsilons**slope,
                linestyle="--",
                linewidth=1.8,
                color="#202020",
                label=rf"fit: $O(\epsilon^{{{slope:.2f}}})$",
            )
        ax.set_xlabel(r"$\epsilon$", fontsize=CONVERGENCE_FONT_SIZE)
        ax.set_ylabel(ylabel, fontsize=CONVERGENCE_FONT_SIZE)
        ax.tick_params(axis="both", which="both", labelsize=CONVERGENCE_FONT_SIZE)
        ax.grid(True, which="both", alpha=0.28)
        ax.legend(fontsize=CONVERGENCE_FONT_SIZE)
    fig.suptitle(title, fontsize=CONVERGENCE_FONT_SIZE)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_study(
    base_dir: Path,
    case: HeatCase,
    *,
    source_groups: Sequence[str] | None = None,
) -> tuple[Path, Path] | None:
    raw_csv, conv_en_path, conv_path, _section_path = study_artifact_paths(base_dir, case)
    if not raw_csv.exists():
        return None
    rows = read_rows(raw_csv)
    plot_convergence(
        rows,
        path=conv_en_path,
        steps_values=case.steps,
        target_time=case.target_time,
        title=case.title,
        source_groups=source_groups,
        metrics=(("max_abs_error", "MAE"),)
        if "no_mass" in case.study
        else (("relative_l2_error", "relative L2"), ("max_abs_error", "MAE")),
    )
    plot_error_vs_epsilon_at_largest_n(
        rows,
        path=conv_path,
        title=(f"{case.title.replace(' no-mass', '')} at N={max(case.sample_counts)}"),
        metric="max_abs_error",
        ylabel="MAE",
        fit_point_count=case.fit_point_count,
        source_groups=source_groups,
    )
    return conv_en_path, conv_path


def plot_max_abs_convergence(
    rows: list[dict[str, str]],
    *,
    path: Path,
    steps_values: Sequence[int],
    target_time: float,
    title: str,
    source_groups: Sequence[str] | None = None,
) -> None:
    del title
    groups = [None] if source_groups is None else list(source_groups)
    fig, axes = plt.subplots(
        len(groups),
        1,
        figsize=(7.2, 4.8 * len(groups)),
        squeeze=False,
        constrained_layout=True,
    )
    colors = line_colors(steps_values)
    for row_idx, group in enumerate(groups):
        ax = axes[row_idx, 0]
        for step_count in steps_values:
            eps = target_time / step_count
            ns, max_abs = median_curve(
                rows, steps=step_count, metric="max_abs_error", source_group=group
            )
            label = f"eps={eps:g}, p={step_count}"
            ax.loglog(ns, max_abs, marker="o", color=colors[step_count], label=label)
        ax.set_xlabel("N", fontsize=CONVERGENCE_FONT_SIZE)
        ax.set_ylabel("max abs error", fontsize=CONVERGENCE_FONT_SIZE)
        ax.tick_params(axis="both", which="both", labelsize=CONVERGENCE_FONT_SIZE)
        ax.grid(True, which="both", alpha=0.28)
        ax.legend(fontsize=CONVERGENCE_FONT_SIZE)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
