"""CLI for the ambient-Euclidean semicircle/full-circle KRR study."""

from __future__ import annotations

import argparse
from pathlib import Path

from ambient_circle_study import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_REPORT_PATH,
    AmbientCircleStudyConfig,
    run_ambient_circle_study,
)


def _float_pair(value: str) -> tuple[float, float]:
    values = tuple(float(item) for item in value.split(",") if item.strip())
    if len(values) != 2 or values[0] <= 0.0 or values[0] >= values[1]:
        raise argparse.ArgumentTypeError("expected two increasing positive comma-separated floats")
    return values[0], values[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the ambient-2D semicircle and full-circle diffusion-map/RBF KRR study."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument(
        "--n-train",
        type=int,
        default=None,
        help="Compatibility shorthand: set both geometry-specific training counts.",
    )
    parser.add_argument("--semi-n-train", type=int, default=1024)
    parser.add_argument("--full-n-train", type=int, default=13)
    parser.add_argument("--semi-n-valid", type=int, default=1023)
    parser.add_argument("--full-n-valid", type=int, default=13)
    parser.add_argument("--n-test", type=int, default=65_536)
    parser.add_argument("--quadrature-order", type=int, default=512)
    parser.add_argument("--endpoint-count", type=int, default=12)
    parser.add_argument("--rbf-target-lengthscale", type=float, default=0.2)
    parser.add_argument("--bandwidth-bounds", type=_float_pair, default=(1.0e-4, 1.0e2))
    parser.add_argument("--ridge-bounds", type=_float_pair, default=(1.0e-16, 1.0e1))
    parser.add_argument(
        "--fixed-rbf-ridge-bounds", type=_float_pair, default=(1.0e-16, 1.0e-8)
    )
    parser.add_argument("--initial-grid-size", type=int, default=9)
    parser.add_argument("--refinement-budget", type=int, default=64)
    parser.add_argument("--fixed-rbf-ridge-count", type=int, default=65)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--no-report", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    semi_n_train = args.n_train if args.n_train is not None else args.semi_n_train
    full_n_train = args.n_train if args.n_train is not None else args.full_n_train
    summary = run_ambient_circle_study(
        AmbientCircleStudyConfig(
            output_dir=args.output_dir,
            report_path=args.report_path,
            semi_n_train=semi_n_train,
            full_n_train=full_n_train,
            semi_n_valid=args.semi_n_valid,
            full_n_valid=args.full_n_valid,
            test_count=args.n_test,
            quadrature_order=args.quadrature_order,
            endpoint_count=args.endpoint_count,
            rbf_target_lengthscale=args.rbf_target_lengthscale,
            bandwidth_bounds=args.bandwidth_bounds,
            ridge_bounds=args.ridge_bounds,
            fixed_rbf_ridge_bounds=args.fixed_rbf_ridge_bounds,
            initial_grid_size=args.initial_grid_size,
            refinement_budget=args.refinement_budget,
            fixed_rbf_ridge_count=args.fixed_rbf_ridge_count,
            seed=args.seed,
            max_workers=args.max_workers,
            plot=not args.no_plot,
            write_report=not args.no_report,
        )
    )
    print(
        "Wrote ambient-circle KRR evidence: "
        f"semicircle family crossings={summary['semicircle']['families']['crossing_count']}; "
        f"full-circle RBF/LB gap={summary['full_circle']['rbf_to_lb_subspace_gap']:.3e}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
