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
    add_study_args,
    convergence_y_limits,
    draw_missing,
    make_ambient_periodic_disk_target,
    make_dirichlet_disk_target,
    make_neumann_disk_target,
    make_rbf_integral_disk_target,
    output_root,
    plot_error_curves,
    read_error_curves,
    report_root,
    run_group_cases,
    selected_groups,
    target_grid_on_disk,
    unit_disk_sample,
)

BASE_DIR = Path(__file__).resolve().parent


def radial_cases(prefix: str, label: str, target_factory) -> tuple[Case, ...]:
    return tuple(
        Case(
            name=f"{prefix}_m3_r{radial}",
            title=f"{label + ' ' if label else ''}m=3, r={radial}",
            target=target_factory(3, radial),
        )
        for radial in range(1, 5)
    )


GROUPS = (
    Group(
        "Disk Neumann LB eigenfunctions",
        "neumann",
        radial_cases("neumann", "", make_neumann_disk_target),
    ),
    Group(
        "Disk Dirichlet LB eigenfunctions",
        "dirichlet",
        radial_cases("dirichlet", "", make_dirichlet_disk_target),
    ),
    Group(
        "RBF Integral",
        "rbf_integral",
        radial_cases("rbf_integral", "", make_rbf_integral_disk_target),
    ),
    Group(
        "Disk ambient Fourier modes",
        "ambient_fourier",
        tuple(
            Case(
                name=f"ambient_fourier_k{k}",
                title=f"k=({k},{k})",
                target=make_ambient_periodic_disk_target(k, k),
            )
            for k in range(2, 6)
        ),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run disk DM/RBF KRR convergence cases.")
    add_study_args(parser, GROUPS)
    return parser.parse_args()


def write_group_plot(group: Group, root: Path, reports: Path) -> Path:
    case_dirs = [root / group.slug / case.name for case in group.cases]
    curves = [read_error_curves(path) for path in case_dirs]
    y_limits = convergence_y_limits(curves)
    fig, axes = plt.subplots(
        2,
        len(group.cases),
        figsize=(3.25 * len(group.cases), 5.9),
        constrained_layout=True,
        squeeze=False,
    )
    for ax in axes[1, 1:]:
        ax.sharey(axes[1, 0])
        ax.tick_params(axis="y", which="both", left=False, labelleft=False)
    targets = [target_grid_on_disk(case.target) for case in group.cases]
    color_limit = max(float(np.nanmax(np.abs(values))) for _, values in targets)
    target_mappable = None
    for col, (case, case_dir, case_curves, (points, values)) in enumerate(
        zip(group.cases, case_dirs, curves, targets, strict=True)
    ):
        grid_size = int(round(np.sqrt(values.size)))
        x = points[:, 0].reshape(grid_size, grid_size)
        y = points[:, 1].reshape(grid_size, grid_size)
        target = values.reshape(grid_size, grid_size)
        target_mappable = axes[0, col].contourf(
            x,
            y,
            target,
            levels=np.linspace(-color_limit, color_limit, 21),
            cmap="coolwarm",
            extend="both",
        )
        axes[0, col].set_aspect("equal", adjustable="box")
        axes[0, col].set_title(case.title, fontsize=14)
        axes[0, col].set_xticks([])
        axes[0, col].set_yticks([])

        ax = axes[1, col]
        if case_dir.exists() and case_curves:
            plot_error_curves(ax, case_curves, y_limits)
        else:
            draw_missing(ax, "no curves")
        ax.set_xticks(
            (2**9, 2**10, 2**11, 2**12),
            labels=(r"$2^9$", r"$2^{10}$", r"$2^{11}$", r"$2^{12}$"),
        )
        ax.set_xlabel("N", fontsize=14)
        ax.tick_params(labelsize=14, length=2)

    axes[1, 0].set_ylabel("RMSE", fontsize=14)
    if target_mappable is not None:
        fig.colorbar(target_mappable, ax=axes[0, :], fraction=0.025, pad=0.018)
    handles, labels = axes[1, 0].get_legend_handles_labels()
    if handles:
        if group.slug == "ambient_fourier":
            fig.legend(handles, labels, loc="center", bbox_to_anchor=(0.99, 0.36), frameon=False)
        else:
            fig.legend(
                handles, labels, loc="upper center", bbox_to_anchor=(0.955, 0.46), frameon=False
            )
    fig.suptitle(group.name, fontsize=14)
    reports.mkdir(parents=True, exist_ok=True)
    path = reports / f"disk_{group.slug}.png"
    fig.savefig(
        path,
        dpi=180,
        facecolor="white",
        bbox_inches="tight" if group.slug == "ambient_fourier" else None,
    )
    plt.close(fig)
    return path


def main() -> int:
    args = parse_args()
    groups = selected_groups(GROUPS, args.groups)
    root = output_root(args.workdir, BASE_DIR)
    run_group_cases(groups, root=root, args=args, sample_for_case=lambda _: unit_disk_sample)
    if not args.no_plot:
        reports = report_root(args.workdir, BASE_DIR)
        paths = [write_group_plot(group, root, reports) for group in groups]
        (reports / "summary.md").write_text(
            "# Disk DM/RBF KRR convergence\n\n"
            + "\n".join(f"- [{path.stem}]({path.name})" for path in paths)
            + f"\n\nMethods: {METHOD_LABELS[METHODS[1]]} and {METHOD_LABELS[METHODS[0]]}.\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
