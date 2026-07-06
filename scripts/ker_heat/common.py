"""Shared helpers for heat-kernel convergence scripts."""

from __future__ import annotations

import csv
import math
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

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
) -> None:
    groups = [None] if source_groups is None else list(source_groups)
    fig, axes = plt.subplots(
        len(groups),
        2,
        figsize=(12.5, 4.4 * len(groups)),
        squeeze=False,
        constrained_layout=True,
    )
    colors = line_colors(steps_values)
    for row_idx, group in enumerate(groups):
        for step_count in steps_values:
            eps = target_time / step_count
            ns, rel = median_curve(
                rows, steps=step_count, metric="relative_l2_error", source_group=group
            )
            _, mx = median_curve(rows, steps=step_count, metric="max_abs_error", source_group=group)
            label = f"eps={eps:g}, p={step_count}"
            axes[row_idx, 0].loglog(ns, rel, marker="o", color=colors[step_count], label=label)
            axes[row_idx, 1].loglog(ns, mx, marker="o", color=colors[step_count], label=label)
        prefix = "" if group is None else f"{group}: "
        axes[row_idx, 0].set_title(f"{prefix}relative L2")
        axes[row_idx, 1].set_title(f"{prefix}max abs")
    for ax in axes.flat:
        ax.set_xlabel("sample count N")
        ax.set_ylabel("median error")
        ax.grid(True, which="both", alpha=0.28)
        ax.legend(fontsize=8)
    fig.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_max_abs_convergence(
    rows: list[dict[str, str]],
    *,
    path: Path,
    steps_values: Sequence[int],
    target_time: float,
    title: str,
    source_groups: Sequence[str] | None = None,
) -> None:
    groups = [None] if source_groups is None else list(source_groups)
    fig, axes = plt.subplots(
        len(groups),
        1,
        figsize=(6.6, 4.2 * len(groups)),
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
        prefix = "" if group is None else f"{group}: "
        ax.set_title(f"{prefix}max abs")
        ax.set_xlabel("sample count N")
        ax.set_ylabel("median max abs error")
        ax.grid(True, which="both", alpha=0.28)
        ax.legend(fontsize=8)
    fig.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
