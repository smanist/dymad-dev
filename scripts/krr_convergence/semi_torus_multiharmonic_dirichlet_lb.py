"""Semi-torus Dirichlet LB KRR cases with fixed and own harmonic-metric targets."""

from __future__ import annotations

import argparse
from pathlib import Path

from krr_common import (
    Case,
    Group,
    add_study_args,
    output_root,
    report_root,
    run_group_cases,
    selected_groups,
    wrap_scaled_target,
)
from semi_torus import write_group_plot
from semi_torus_multiharmonic_lb import (
    AMBIENT_DIMS,
    GEOMETRY,
    INPUT_SCALE,
    aligned_multiharmonic_theta_mode,
    make_multiharmonic_area_sample,
    make_multiharmonic_dirichlet_lb_target,
)

BASE_DIR = Path(__file__).resolve().parent
AZIMUTHAL_MODE = 3
THETA_MODE = 1
REFERENCE_AMBIENT_DIM = 3


def dirichlet_case(ambient_dim: int, *, own_target: bool) -> Case:
    source_ambient_dim = ambient_dim if own_target else REFERENCE_AMBIENT_DIM
    mode = aligned_multiharmonic_theta_mode(
        GEOMETRY.major_radius,
        AZIMUTHAL_MODE,
        THETA_MODE,
        source_ambient_dim,
    )
    source_name = "own" if own_target else "d3"
    source_label = "own" if own_target else "d=3"
    return Case(
        name=(f"dirichlet_m{AZIMUTHAL_MODE}_j{THETA_MODE}_target_{source_name}_d{ambient_dim}"),
        title=f"d={ambient_dim}, target={source_label}, λ={mode.eigenvalue:.4f}",
        ambient_dim=ambient_dim,
        embedding="harmonic",
        target=wrap_scaled_target(
            make_multiharmonic_dirichlet_lb_target(
                m=AZIMUTHAL_MODE,
                j=THETA_MODE,
                source_ambient_dim=source_ambient_dim,
            ),
            coordinate_scale=INPUT_SCALE,
        ),
    )


GROUPS = (
    Group(
        "Semi-torus Dirichlet m=3, j=1: common d=3 target by ambient dimension",
        "semi_torus_dirichlet_d3_target_ambient",
        tuple(dirichlet_case(ambient_dim, own_target=False) for ambient_dim in AMBIENT_DIMS),
    ),
    Group(
        "Semi-torus Dirichlet m=3, j=1: own aligned target by ambient dimension",
        "semi_torus_dirichlet_own_target_ambient",
        tuple(dirichlet_case(ambient_dim, own_target=True) for ambient_dim in AMBIENT_DIMS),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fixed- and own-target semi-torus harmonic Dirichlet LB KRR cases."
    )
    add_study_args(parser, GROUPS)
    parser.add_argument(
        "--ambient-dims",
        type=int,
        nargs="+",
        choices=AMBIENT_DIMS,
        help="run only the requested harmonic embedding dimensions",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    groups = selected_groups(GROUPS, args.groups)
    if args.ambient_dims is not None:
        selected_dims = set(args.ambient_dims)
        groups = tuple(
            Group(
                group.name,
                group.slug,
                tuple(case for case in group.cases if case.ambient_dim in selected_dims),
                show_targets=group.show_targets,
            )
            for group in groups
        )
    root = output_root(args.workdir, BASE_DIR)
    run_group_cases(
        groups,
        root=root,
        args=args,
        sample_for_case=lambda case: make_multiharmonic_area_sample(case.ambient_dim),
        x_transform=None,
        y_transform=None,
    )
    if not args.no_plot:
        reports = report_root(args.workdir, BASE_DIR)
        for group in groups:
            path = write_group_plot(group, root, reports)
            print(f"Wrote report to {path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
