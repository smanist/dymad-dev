from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from common import (
    METHOD_LABELS,
    METHODS,
    Case,
    Group,
    SemiTorusGeometry,
    add_study_args,
    convergence_y_limits,
    draw_missing,
    make_semi_torus_ambient_fourier_target,
    make_semi_torus_fourier_target,
    make_semi_torus_sample,
    output_root,
    plot_error_curves,
    read_error_curves,
    report_root,
    run_group_cases,
    selected_groups,
    semi_torus_target_grid,
    wrap_scaled_target,
)

BASE_DIR = Path(__file__).resolve().parent
GEOMETRY = SemiTorusGeometry(major_radius=2.0)
COORDINATE_SCALE = 3.0
LAPLACE_MODES = ((1, 0), (3, 1), (6, 3))
SURFACE_LIMITS = ((-3.2, 3.2), (-1.2, 4.2), (-3.0, 3.0))


def laplace_case(
    boundary: str,
    m: int,
    j: int,
    ambient_dim: int = 3,
    *,
    show_ambient_dim: bool = False,
) -> Case:
    return Case(
        name=f"{boundary}_m{m}_j{j}_d{ambient_dim}",
        title=f"d={ambient_dim}" if show_ambient_dim or ambient_dim != 3 else f"m={m}, j={j}",
        ambient_dim=ambient_dim,
        target=wrap_scaled_target(
            make_semi_torus_fourier_target(
                GEOMETRY,
                boundary=boundary,
                m=m,
                j=j,
                fourier_order=16,
                quadrature_size=4096,
            ),
            coordinate_scale=COORDINATE_SCALE,
        ),
    )


GROUPS = (
    Group(
        "Semi-torus Neumann LB eigenfunctions",
        "neumann",
        tuple(laplace_case("neumann", m, j) for m, j in LAPLACE_MODES),
    ),
    Group(
        "Semi-torus Dirichlet LB eigenfunctions",
        "dirichlet",
        tuple(laplace_case("dirichlet", m, j) for m, j in LAPLACE_MODES),
    ),
    Group(
        "Semi-torus ambient Fourier modes",
        "semi_torus_ambient_fourier",
        tuple(
            Case(
                name=f"ambient_fourier_k{k}",
                title=f"k={k}",
                ambient_dim=3,
                target=wrap_scaled_target(
                    make_semi_torus_ambient_fourier_target(GEOMETRY, frequency=k),
                    coordinate_scale=COORDINATE_SCALE,
                ),
            )
            for k in (2, 6, 10)
        ),
    ),
    Group(
        "Semi-torus mid-frequency Dirichlet LB eigenfunction (m=3, j=1) by ambient dimension",
        "dirichlet_ambient",
        tuple(
            laplace_case("dirichlet", 3, 1, ambient_dim, show_ambient_dim=True)
            for ambient_dim in (3, 7, 11, 15)
        ),
        show_targets=False,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run semi-torus DM/RBF KRR convergence cases.")
    add_study_args(parser, GROUPS)
    return parser.parse_args()


def _grid_target(case: Case):
    return lambda points: case.target(points / COORDINATE_SCALE)


def _plot_target(ax: plt.Axes, case: Case, color_limit: float) -> None:
    x, y, z, values, _ = semi_torus_target_grid(
        GEOMETRY,
        _grid_target(case),
        ambient_dim=case.ambient_dim,
        n_theta=256,
        n_phi=128,
    )
    ax.plot_surface(
        x,
        y,
        z,
        facecolors=plt.cm.viridis(plt.Normalize(-color_limit, color_limit)(values)),
        linewidth=0,
        antialiased=True,
        shade=False,
    )
    ax.set_title(case.title, fontsize=14)
    ax.set_box_aspect(tuple(high - low for low, high in SURFACE_LIMITS))
    ax.set_xlim(*SURFACE_LIMITS[0])
    ax.set_ylim(*SURFACE_LIMITS[1])
    ax.set_zlim(*SURFACE_LIMITS[2])
    ax.view_init(elev=24, azim=-54)
    ax.tick_params(labelsize=14, pad=0)


def write_group_plot(group: Group, root: Path, reports: Path) -> Path:
    case_dirs = [root / group.slug / case.name for case in group.cases]
    curves = [read_error_curves(path) for path in case_dirs]
    y_limits = convergence_y_limits(curves)
    n_rows = 2 if group.show_targets else 1
    fig = plt.figure(
        figsize=(4.2 * len(group.cases) + 1.5, 6.4 if group.show_targets else 3.2),
    )
    grid = fig.add_gridspec(
        n_rows,
        len(group.cases),
        left=0.06,
        right=0.85,
        bottom=0.16,
        top=0.88 if group.show_targets else 0.80,
        hspace=0.12,
        wspace=0.12,
    )
    color_limit = 1.0
    if group.show_targets:
        values = [
            semi_torus_target_grid(GEOMETRY, _grid_target(case), ambient_dim=case.ambient_dim)[3]
            for case in group.cases
        ]
        color_limit = max(float(np.nanmax(np.abs(values))), 1.0e-12)

    axes: list[plt.Axes] = []
    for col, (case, case_dir, case_curves) in enumerate(
        zip(group.cases, case_dirs, curves, strict=True), start=1
    ):
        if group.show_targets:
            _plot_target(fig.add_subplot(grid[0, col - 1], projection="3d"), case, color_limit)
        curve_row = 1 if group.show_targets else 0
        ax = fig.add_subplot(
            grid[curve_row, col - 1],
            sharey=axes[0] if axes else None,
        )
        if axes:
            ax.tick_params(axis="y", which="both", left=False, labelleft=False)
        axes.append(ax)
        if case_dir.exists() and case_curves:
            plot_error_curves(ax, case_curves, y_limits)
        else:
            draw_missing(ax, "no curves")
        ax.set_title(case.title if not group.show_targets else "", fontsize=14)
        ax.set_xticks(
            (2**9, 2**10, 2**11, 2**12),
            labels=(r"$2^9$", r"$2^{10}$", r"$2^{11}$", r"$2^{12}$"),
        )
        ax.set_xlabel("N", fontsize=14)
        ax.tick_params(labelsize=14, length=2)

    axes[0].set_ylabel("RMSE", fontsize=14)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.87, 0.5), frameon=False)
    fig.suptitle(group.name, fontsize=14, y=0.97)
    reports.mkdir(parents=True, exist_ok=True)
    report_slug = group.slug.removeprefix("semi_torus_")
    path = reports / f"semi_torus_{report_slug}.png"
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    return path


def main() -> int:
    args = parse_args()
    groups = selected_groups(GROUPS, args.groups)
    root = output_root(args.workdir, BASE_DIR)
    run_group_cases(
        groups,
        root=root,
        args=args,
        sample_for_case=lambda case: make_semi_torus_sample(
            GEOMETRY,
            ambient_dim=case.ambient_dim,
            coordinate_scale=COORDINATE_SCALE,
        ),
    )
    if not args.no_plot:
        reports = report_root(args.workdir, BASE_DIR)
        paths = [write_group_plot(group, root, reports) for group in groups]
        (reports / "summary.md").write_text(
            "# Semi-torus DM/RBF KRR convergence\n\n"
            + "\n".join(f"- [{path.stem}]({path.name})" for path in paths)
            + f"\n\nMethods: {METHOD_LABELS[METHODS[1]]} and {METHOD_LABELS[METHODS[0]]}.\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
