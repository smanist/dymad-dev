from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cartesian_high_freq_krr_problem import make_convergence_plot, problem
from scripts.cli_helpers import set_seed

from dymad.studies.convergence import ArrayRegressionStudyConfig, run_array_regression_study

BASE_DIR = Path(__file__).resolve().parent


def parse_int_list(text: str) -> tuple[int, ...]:
    return tuple(int(item) for item in text.split(",") if item.strip())


def parse_int_or_tuple(text: str) -> int | tuple[int, ...]:
    values = parse_int_list(text)
    if not values:
        raise ValueError("value must be a positive integer or comma list")
    return values[0] if len(values) == 1 else values


def _parse_int_or_tuple_arg(text: str) -> int | tuple[int, ...]:
    try:
        return parse_int_or_tuple(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a standalone tuning + convergence-study example for the "
            "cartesian_high_freq unit-disk KRR case."
        )
    )
    parser.add_argument("--workdir", type=Path, default=BASE_DIR / "cartesian_high_freq_study")
    parser.add_argument("--levels", default="32,64", help="Comma-separated training sizes.")
    parser.add_argument(
        "--trials",
        type=_parse_int_or_tuple_arg,
        default=1,
        help=(
            "Trial count. Use N for N trials at every level, or n1,n2,... for "
            "level-specific trial counts."
        ),
    )
    parser.add_argument("--n-val", type=int, default=32)
    parser.add_argument("--n-test", type=int, default=128)
    parser.add_argument(
        "--resampling-mode",
        choices=("legacy", "nested-fixed-test"),
        default="legacy",
    )
    parser.add_argument(
        "--validation-mode",
        choices=("holdout", "kfold", "train-valid-count"),
        default="holdout",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--validation-size", type=int, default=None)
    parser.add_argument("--k-folds", type=int, default=4)
    parser.add_argument("--pool-multiplier", type=int, default=1)
    parser.add_argument(
        "--confidence-band",
        choices=("std", "stderr", "iqr", "q05_q95"),
        default=None,
    )
    parser.add_argument(
        "--initial-budget",
        type=_parse_int_or_tuple_arg,
        default=5,
        help=(
            "Initial search budget. Use N for total candidates, or n1,n2,... for "
            "per-parameter grid counts."
        ),
    )
    parser.add_argument("--refinement-budget", type=int, default=0)
    parser.add_argument(
        "--refinement-strategy",
        choices=("nelder_mead_like", "batch_pattern_search"),
        default=None,
    )
    parser.add_argument("--tuning-policy", choices=("per_trial", "per_level"), default="per_trial")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--no-prediction-plots", action="store_true")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> ArrayRegressionStudyConfig:
    return ArrayRegressionStudyConfig(
        output_dir=args.workdir,
        levels=parse_int_list(args.levels),
        trials=args.trials,
        n_val=args.n_val,
        n_test=args.n_test,
        initial_budget=args.initial_budget,
        refinement_budget=args.refinement_budget,
        refinement_strategy=args.refinement_strategy,
        tuning_policy=args.tuning_policy,
        seed=args.seed,
        max_workers=args.max_workers,
        resampling_mode=args.resampling_mode,
        validation_mode=args.validation_mode,
        validation_fraction=args.validation_fraction,
        validation_size=args.validation_size,
        k_folds=args.k_folds,
        pool_multiplier=args.pool_multiplier,
        confidence_band=args.confidence_band,
        plot=not args.no_plot,
        prediction_plots=not args.no_prediction_plots,
    )


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    config = config_from_args(args)

    result = run_array_regression_study(problem, config, make_plot=make_convergence_plot)
    print(f"Wrote convergence artifacts to {Path(config.output_dir).resolve()}")
    if result.diagnostics:
        print(f"Diagnostics: {len(result.diagnostics)} advisory item(s); see diagnostics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
