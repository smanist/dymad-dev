"""The five visualizations included in the ambient-circle PDF report."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METHOD_COLORS = {"dm_krr": "#0072B2", "rbf_krr": "#D55E00"}


def _plot_target_ensembles(
    path: Path,
    semi: Any,
    full: Any,
    *,
    representative_family: int,
    representative_s: float,
    representative_values: np.ndarray,
) -> None:
    """Plot the actual endpoint targets and one representative family.

    The target construction is the object of the comparison, so this figure
    deliberately shows targets rather than continuum kernel eigenfunctions.
    """

    figure, axes = plt.subplots(2, 2, figsize=(10.5, 6.7), constrained_layout=True)

    def draw_endpoints(axis: plt.Axes, data: Any, start: int, title: str) -> None:
        order = np.argsort(data.theta_test)
        count = data.endpoint_dimension // 2 if data.modes.name == "semi_circle" else data.endpoint_dimension
        for index in range(count):
            axis.plot(
                data.theta_test[order],
                data.endpoint_test[order, start + index],
                linewidth=0.95,
                label=str(index + 1),
            )
        axis.set(title=title, xlabel=r"$\theta$", ylabel="target value")
        axis.grid(alpha=0.25)
        axis.legend(title="endpoint", fontsize="xx-small", ncol=3, loc="best")

    draw_endpoints(axes[0, 0], semi, 0, "Semicircle: 12 LB endpoints")
    draw_endpoints(
        axes[0, 1],
        semi,
        semi.endpoint_dimension // 2,
        "Semicircle: 12 RBF endpoints",
    )

    family_index = representative_family - 1
    # The crossover itself is very close to one and makes the mixed target
    # visually indistinguishable from v_i.  Show a genuinely mixed interior
    # example instead.
    order = np.argsort(semi.theta_test)
    axis = axes[1, 0]
    axis.plot(
        semi.theta_test[order],
        semi.endpoint_test[order, family_index],
        linewidth=1.1,
        label=rf"$u_{{{representative_family}}}$ (LB)",
    )
    axis.plot(
        semi.theta_test[order],
        semi.endpoint_test[order, family_index + semi.endpoint_dimension // 2],
        linewidth=1.1,
        label=rf"$v_{{{representative_family}}}$ (RBF)",
    )
    axis.plot(
        semi.theta_test[order],
        representative_values[order],
        color="0.12",
        linewidth=1.5,
        label=rf"$f_{{{representative_family}}}(s={representative_s:.1f})$",
    )
    axis.set(title="One representative semicircle family", xlabel=r"$\theta$", ylabel="target value")
    axis.grid(alpha=0.25)
    axis.legend(fontsize="x-small", loc="best")

    draw_endpoints(axes[1, 1], full, 0, "Full circle: 12 common-LB endpoints")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_kernel_mode_comparison(
    path: Path, semi: dict[str, Any], full: dict[str, Any]
) -> None:
    """Plot target modes against closest-aligned finite-sample kernel modes."""

    panels = list(semi["panels"]) + list(full["panels"])
    figure, axes = plt.subplots(3, 2, figsize=(10.5, 9.1), constrained_layout=True)
    for panel_index, (axis, panel) in enumerate(zip(axes.flat, panels, strict=True)):
        theta = np.asarray(semi["theta"] if panel_index < 4 else full["theta"])
        target = np.asarray(panel["target"])
        kernel = np.asarray(panel["kernel"])
        count = min(8, target.shape[1])
        amplitude = max(
            float(np.max(np.abs(target[:, :count]))),
            float(np.max(np.abs(kernel[:, :count]))),
        )
        spacing = 2.6 * amplitude
        offsets = spacing * np.arange(count, dtype=float)
        method_color = (
            METHOD_COLORS["dm_krr"] if panel["title"].endswith("DM") else METHOD_COLORS["rbf_krr"]
        )
        angle = float(panel["maximum_angle_degrees"])
        angle_text = f"{angle:.3g}"
        if "e" in angle_text:
            mantissa, exponent = angle_text.split("e")
            angle_text = rf"{mantissa}\times10^{{{int(exponent)}}}"
        for mode in range(count):
            axis.plot(
                theta,
                target[:, mode] + offsets[mode],
                color="0.28",
                linestyle="--",
                linewidth=0.9,
                label="target-space mode" if mode == 0 else None,
            )
            axis.plot(
                theta,
                kernel[:, mode] + offsets[mode],
                color=method_color,
                linewidth=1.1,
                label="aligned kernel mode" if mode == 0 else None,
            )
        axis.set(
            title=(
                f"{panel['title']}; "
                rf"$\theta_{{\max}}={{{angle_text}}}^\circ$"
            ),
            xlabel=r"$\theta$",
            yticks=offsets,
            yticklabels=[str(value) for value in range(1, count + 1)],
            ylabel="mode (vertically offset)",
        )
        axis.grid(alpha=0.18)
        if panel_index == 0:
            axis.legend(fontsize="x-small", loc="best")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _endpoint_number(row: dict[str, Any]) -> int:
    return int(str(row["target_label"]).rsplit(" ", maxsplit=1)[-1])


def _plot_endpoint_error_and_share(
    error_axis: plt.Axes,
    share_axis: plt.Axes,
    dm: list[dict[str, Any]],
    rbf: list[dict[str, Any]],
    *,
    title: str,
) -> None:
    """Plot total errors above squared leakage shares without curve ambiguity."""

    index = np.arange(1, len(dm) + 1, dtype=float)
    all_errors: list[float] = []
    for rows, method, color, marker in (
        (dm, "DM", METHOD_COLORS["dm_krr"], "o"),
        (rbf, "RBF", METHOD_COLORS["rbf_krr"], "s"),
    ):
        total = np.asarray([float(row["population_error"]) for row in rows])
        leakage_share = np.asarray([float(row["leakage_share"]) for row in rows])
        error_axis.semilogy(
            index,
            total,
            color=color,
            linewidth=1.45,
            marker=marker,
            markersize=3.0,
            label=f"{method} total $E$",
        )
        share_axis.plot(
            index,
            leakage_share,
            color=color,
            linewidth=1.35,
            marker=marker,
            markersize=3.0,
            label=rf"{method} $L^2/E^2$",
        )
        all_errors.extend(total)
    error_axis.set(
        title=title,
        ylabel=r"test $L^2$ error norm",
        xticks=index,
        xticklabels=[],
        yscale="log",
    )
    error_axis.set_ylim(float(np.min(all_errors)) * 0.55, float(np.max(all_errors)) * 1.8)
    share_axis.set(
        xlabel="endpoint",
        ylabel=r"leakage energy $L^2/E^2$",
        xticks=index,
        xticklabels=[str(value) for value in index.astype(int)],
        ylim=(-0.04, 1.04),
    )
    for axis in (error_axis, share_axis):
        axis.grid(alpha=0.25)
        axis.set_axisbelow(True)


def _plot_semicircle_endpoints(path: Path, rows: list[dict[str, Any]]) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 6.2), constrained_layout=True)
    for column, (kind, title) in enumerate(
        zip(("lb", "rbf"), ("LB endpoints", "RBF endpoints"), strict=True)
    ):
        dm = sorted(
            (row for row in rows if row["target_kind"] == kind and row["method"] == "dm_krr"),
            key=_endpoint_number,
        )
        rbf = sorted(
            (row for row in rows if row["target_kind"] == kind and row["method"] == "rbf_krr"),
            key=_endpoint_number,
        )
        _plot_endpoint_error_and_share(axes[0, column], axes[1, column], dm, rbf, title=title)
    axes[0, 0].legend(fontsize="x-small", loc="best")
    axes[1, 0].legend(fontsize="x-small", loc="best")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _family_lookup(rows: list[dict[str, Any]], family: int, method: str) -> list[dict[str, Any]]:
    return sorted(
        (row for row in rows if row["family_index"] == family and row["method"] == method),
        key=lambda row: float(row["s"]),
    )


def _plot_semicircle_family_focus(
    path: Path, rows: list[dict[str, Any]], diagnostics: list[dict[str, Any]]
) -> None:
    """Show one family in detail and condense all 12 in a heat map."""

    # ``_family_summary`` records the same representative in summary.json.  It
    # is recomputed here to keep this figure function independent of I/O.
    crossings = [row for row in diagnostics if row["exact_crossing"] is not None]
    if crossings:
        median_log_collapse = float(
            np.median([math.log(row["leakage_collapse"]) for row in crossings])
        )
        representative = min(
            crossings,
            key=lambda row: (
                abs(math.log(row["leakage_collapse"]) - median_log_collapse),
                float("inf")
                if row["floor_crossing_shift"] is None
                else row["floor_crossing_shift"],
                row["family_index"],
            ),
        )
    else:
        representative = diagnostics[0]

    family_index = int(representative["family_index"])
    dm = _family_lookup(rows, family_index, "dm_krr")
    rbf = _family_lookup(rows, family_index, "rbf_krr")
    s = np.asarray([float(row["s"]) for row in dm])
    dm_error = np.asarray([float(row["population_error"]) for row in dm])
    rbf_error = np.asarray([float(row["population_error"]) for row in rbf])
    dm_in = np.asarray([float(row["in_class_error"]) for row in dm])
    dm_leakage = np.asarray([float(row["leakage"]) for row in dm])
    rbf_in = np.asarray([float(row["in_class_error"]) for row in rbf])
    rbf_leakage = np.asarray([float(row["leakage"]) for row in rbf])

    figure = plt.figure(figsize=(10.5, 6.8), constrained_layout=True)
    grid = figure.add_gridspec(2, 2)
    full_axis = figure.add_subplot(grid[0, 0])
    zoom_axis = figure.add_subplot(grid[0, 1])
    heat_axis = figure.add_subplot(grid[1, :])

    def draw_layers(axis: plt.Axes, *, zoom: bool) -> None:
        selected = s >= 0.95 if zoom else np.ones_like(s, dtype=bool)
        axis.semilogy(
            s[selected],
            dm_in[selected],
            color=METHOD_COLORS["dm_krr"],
            linestyle="--",
            linewidth=1.0,
            label="DM in-class",
        )
        axis.semilogy(
            s[selected],
            dm_leakage[selected],
            color=METHOD_COLORS["dm_krr"],
            linestyle=":",
            linewidth=1.45,
            label="DM leakage",
        )
        axis.semilogy(
            s[selected],
            rbf_in[selected],
            color=METHOD_COLORS["rbf_krr"],
            linestyle="--",
            linewidth=1.0,
            label="RBF in-class",
        )
        axis.semilogy(
            s[selected],
            rbf_leakage[selected],
            color=METHOD_COLORS["rbf_krr"],
            linestyle=":",
            linewidth=1.45,
            label="RBF leakage",
        )
        # Draw total errors last and above the components.  Since
        # E^2=B^2+L^2, this makes the invariant E>=B,L visually explicit even
        # when E and L are indistinguishable at the plot resolution.
        axis.semilogy(
            s[selected],
            dm_error[selected],
            color=METHOD_COLORS["dm_krr"],
            linewidth=1.7,
            zorder=4,
            label="DM total",
        )
        axis.semilogy(
            s[selected],
            rbf_error[selected],
            color=METHOD_COLORS["rbf_krr"],
            linewidth=1.7,
            zorder=4,
            label="RBF total",
        )
        if representative["exact_crossing"] is not None:
            crossing = float(representative["exact_crossing"])
            crossing_error = float(np.interp(crossing, s, dm_error))
            axis.axvline(
                crossing,
                color="0.2",
                linestyle="--",
                linewidth=0.8,
                label="exact crossover",
            )
            axis.plot(crossing, crossing_error, "ko", markersize=3.5)
        values = np.concatenate(
            (
                dm_error[selected],
                rbf_error[selected],
                dm_in[selected],
                dm_leakage[selected],
                rbf_in[selected],
                rbf_leakage[selected],
            )
        )
        axis.set(
            xlabel=r"$s$",
            ylabel=r"test $L^2$ error norm",
            xlim=(0.95, 1.0) if zoom else (0.0, 1.0),
            ylim=(max(float(np.min(values)) * 0.45, 1.0e-16), float(np.max(values)) * 2.2),
        )
        axis.grid(alpha=0.24)

    draw_layers(full_axis, zoom=False)
    full_axis.set_title(f"Representative family {family_index}: full path")
    full_axis.legend(fontsize="xx-small", loc="best", ncol=2)
    draw_layers(zoom_axis, zoom=True)
    crossover_title = r"Same family: endpoint zoom ($s\geq0.95$)"
    if representative["exact_crossing"] is not None:
        crossover_title += rf", $s_\star={float(representative['exact_crossing']):.6f}$"
    zoom_axis.set_title(crossover_title)

    all_s = np.asarray([float(row["s"]) for row in _family_lookup(rows, 1, "dm_krr")])
    log_ratio = np.asarray(
        [
            np.log10(
                np.asarray(
                    [
                        float(row["population_error"])
                        for row in _family_lookup(rows, family, "rbf_krr")
                    ]
                )
                / np.asarray(
                    [
                        float(row["population_error"])
                        for row in _family_lookup(rows, family, "dm_krr")
                    ]
                )
            )
            for family in range(1, len(diagnostics) + 1)
        ]
    )
    scale = max(0.15, float(np.max(np.abs(log_ratio))))
    image = heat_axis.imshow(
        log_ratio,
        aspect="auto",
        interpolation="nearest",
        extent=(float(all_s[0]), float(all_s[-1]), len(diagnostics) + 0.5, 0.5),
        cmap="RdBu_r",
        vmin=-scale,
        vmax=scale,
    )
    heat_axis.contour(
        all_s,
        np.arange(1, len(diagnostics) + 1),
        log_ratio,
        levels=[0.0],
        colors="k",
        linewidths=0.7,
    )
    heat_axis.axhline(family_index, color="0.1", linewidth=1.0)
    heat_axis.set(
        title=r"All families: $\log_{10}(E_{\rm RBF}/E_{\rm DM})$",
        xlabel=r"$s$",
        ylabel="family",
        yticks=np.arange(1, len(diagnostics) + 1),
    )
    colorbar = figure.colorbar(image, ax=heat_axis, shrink=0.88)
    colorbar.set_label("RBF better  ←       →  DM better", fontsize="x-small")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_fullcircle_endpoints(path: Path, rows: list[dict[str, Any]]) -> None:
    figure, axis = plt.subplots(figsize=(8.1, 4.5), constrained_layout=True)
    dm = sorted((row for row in rows if row["method"] == "dm_krr"), key=_endpoint_number)
    rbf = sorted((row for row in rows if row["method"] == "rbf_krr"), key=_endpoint_number)
    index = np.arange(1, len(dm) + 1, dtype=float)
    values: list[float] = []
    method_rows = (
        (dm, "DM", METHOD_COLORS["dm_krr"], "o"),
        (rbf, "RBF", METHOD_COLORS["rbf_krr"], "s"),
    )
    for selected, label, color, marker in method_rows:
        in_class = np.asarray([float(row["in_class_error"]) for row in selected])
        leakage = np.asarray([float(row["leakage"]) for row in selected])
        axis.semilogy(
            index,
            in_class,
            color=color,
            linestyle="--",
            linewidth=1.1,
            marker=marker,
            markersize=2.8,
            label=f"{label} in-class $B$",
        )
        axis.semilogy(
            index,
            leakage,
            color=color,
            linestyle=":",
            linewidth=1.35,
            marker=marker,
            markersize=2.8,
            label=f"{label} leakage $L$",
        )
        values.extend(in_class)
        values.extend(leakage)
    for selected, label, color, marker in method_rows:
        total = np.asarray([float(row["population_error"]) for row in selected])
        axis.semilogy(
            index,
            total,
            color=color,
            linewidth=1.75,
            marker=marker,
            markersize=3.1,
            zorder=4,
            label=f"{label} total $E$",
        )
        values.extend(total)
    axis.set(
        title="Full-circle common-LB endpoints",
        xlabel="endpoint",
        ylabel=r"test $L^2$ error norm",
        xticks=index,
        xticklabels=[str(value) for value in index.astype(int)],
        ylim=(float(np.min(values)) * 0.65, float(np.max(values)) * 1.55),
    )
    axis.grid(alpha=0.25)
    axis.set_axisbelow(True)
    axis.legend(fontsize="x-small", loc="best", ncol=2)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_report_figures(
    *,
    output_dir: Path,
    semi: Any,
    full: Any,
    representative_family: int,
    representative_s: float,
    representative_values: np.ndarray,
    semi_mode_plot: dict[str, Any],
    full_mode_plot: dict[str, Any],
    semi_endpoint_rows: list[dict[str, Any]],
    semi_family_rows: list[dict[str, Any]],
    family_diagnostics: list[dict[str, Any]],
    full_decompositions: list[dict[str, Any]],
) -> None:
    """Write exactly the five figures embedded in the PDF note."""

    _plot_target_ensembles(
        output_dir / "target_ensembles.png",
        semi,
        full,
        representative_family=representative_family,
        representative_s=representative_s,
        representative_values=representative_values,
    )
    _plot_kernel_mode_comparison(
        output_dir / "kernel_mode_comparison.png", semi_mode_plot, full_mode_plot
    )
    _plot_semicircle_endpoints(output_dir / "semicircle_endpoints.png", semi_endpoint_rows)
    _plot_semicircle_family_focus(
        output_dir / "semicircle_family_focus_and_summary.png",
        semi_family_rows,
        family_diagnostics,
    )
    _plot_fullcircle_endpoints(
        output_dir / "fullcircle_lb_endpoints.png", full_decompositions
    )


__all__ = ["write_report_figures"]
