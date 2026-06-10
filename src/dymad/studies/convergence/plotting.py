from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from dymad.studies.convergence.core import ConvergenceStudyResult


@dataclass(frozen=True)
class CurveStyle:
    label: str | None = None
    color: str | None = None
    marker: str = "o"
    linestyle: str = "-"
    alpha: float = 0.18


def plot_convergence_summary(
    result: ConvergenceStudyResult,
    output_path: str | Path,
    *,
    methods: Sequence[str] | None = None,
    center: str = "mean",
    band: str | None = "std",
    x_column: str = "refinement",
    title: str | None = None,
    xlabel: str = "refinement",
    ylabel: str = "error",
    xscale: str = "log",
    yscale: str = "log",
    xbase: float | None = 2,
    styles: Mapping[str, CurveStyle] | None = None,
    figsize: tuple[float, float] = (6.5, 4.2),
    dpi: int = 160,
) -> None:
    rows_by_method = _rows_by_method(result.convergence_summary)
    methods_to_plot = tuple(methods) if methods is not None else tuple(rows_by_method)
    fig, ax = plt.subplots(figsize=figsize)
    for method in methods_to_plot:
        rows = sorted(rows_by_method.get(method, []), key=lambda item: float(item[x_column]))
        if not rows:
            continue
        style = styles.get(method, CurveStyle()) if styles is not None else CurveStyle()
        x_values = np.array([float(row[x_column]) for row in rows])
        y_values = np.array([float(row[center]) for row in rows])
        label = style.label or method
        if band in {"iqr", "q05_q95"}:
            lower_column, upper_column = _band_columns(band)
            lower = np.array([float(row[lower_column]) for row in rows])
            upper = np.array([float(row[upper_column]) for row in rows])
            ax.plot(
                x_values,
                y_values,
                marker=style.marker,
                linestyle=style.linestyle,
                color=style.color,
                label=label,
            )
            ax.fill_between(x_values, lower, upper, color=style.color, alpha=style.alpha)
        elif band is None:
            ax.plot(
                x_values,
                y_values,
                marker=style.marker,
                linestyle=style.linestyle,
                color=style.color,
                label=label,
            )
        else:
            yerr = np.array([float(row[band]) for row in rows])
            ax.errorbar(
                x_values,
                y_values,
                yerr=yerr,
                marker=style.marker,
                linestyle=style.linestyle,
                color=style.color,
                capsize=3,
                label=label,
            )
    if xscale == "log" and xbase is not None:
        ax.set_xscale(xscale, base=xbase)
    else:
        ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _rows_by_method(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    by_method: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_method.setdefault(str(row["method"]), []).append(row)
    return by_method


def _band_columns(band: str) -> tuple[str, str]:
    if band == "iqr":
        return "q25", "q75"
    if band == "q05_q95":
        return "q05", "q95"
    raise ValueError(f"unsupported filled band {band!r}")
