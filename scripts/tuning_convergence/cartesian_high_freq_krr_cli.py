from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.cli_helpers import set_seed

from dymad.io import Split
from dymad.modules import make_krr
from dymad.studies.convergence import (
    ConvergenceEvaluationContext,
    ConvergenceStudySpec,
    HoldoutValidationPolicy,
    KFoldValidationPolicy,
    LevelSamplePlan,
    MedianPlotContext,
    NestedResamplingPolicy,
    TrainValidCountPolicy,
    TuningEvaluationContext,
    TuningPolicy,
    run_convergence_study,
)
from dymad.tuning import ParameterSpec, TuningSpec

BASE_DIR = Path(__file__).resolve().parent
CASE_NAME = "cartesian_high_freq"
METHODS = ("rbf_krr", "dm_krr")


def unit_disk_sample(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    radius = np.sqrt(rng.random(n_samples))
    theta = 2.0 * math.pi * rng.random(n_samples)
    return np.column_stack((radius * np.cos(theta), radius * np.sin(theta)))


def label_values(points: np.ndarray) -> np.ndarray:
    x = points[:, 0]
    y = points[:, 1]
    values = np.sin(18.0 * x + 11.0 * y) + 0.5 * np.cos(22.0 * x - 5.0 * y)
    return values.reshape(-1, 1)


def make_split(n_train: int, n_val: int, n_test: int, seed: int) -> Split:
    rng = np.random.default_rng(100_000 * seed + 97 * n_train + 19_889)
    x_train = unit_disk_sample(n_train, rng)
    x_val = unit_disk_sample(n_val, rng)
    x_test = unit_disk_sample(n_test, rng)
    y_train = label_values(x_train)
    y_val = label_values(x_val)
    y_test = label_values(x_test)
    return Split.from_arrays(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
    )


def make_split_from_arrays(
    *,
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
) -> Split:
    return Split.from_arrays(
        x_train=x_train,
        y_train=label_values(x_train),
        x_val=x_val,
        y_val=label_values(x_val),
        x_test=x_test,
        y_test=label_values(x_test),
    )


class NestedCartesianSamples:
    def __init__(self, *, max_train: int, n_test: int, seed: int, trials: tuple[int | str, ...]):
        test_rng = np.random.default_rng(1_000_000_007 + seed)
        self.x_test = unit_disk_sample(n_test, test_rng)
        self.x_dev_by_trial = {
            trial: unit_disk_sample(max_train, np.random.default_rng(2_000_000_011 + seed + index))
            for index, trial in enumerate(trials)
        }

    def split_for_fold(self, trial: int | str, fold) -> Split:
        x_dev = self.x_dev_by_trial[trial]
        return make_split_from_arrays(
            x_train=x_dev[list(fold.train_indices)],
            x_val=x_dev[list(fold.validation_indices)],
            x_test=self.x_test,
        )

    def split_for_refit(self, trial: int | str, plan: LevelSamplePlan) -> Split:
        x_dev = self.x_dev_by_trial[trial]
        x_train = x_dev[list(plan.refit_indices)]
        x_val = x_train[:1]
        return make_split_from_arrays(x_train=x_train, x_val=x_val, x_test=self.x_test)


def kernel_config(method: str, bandwidth_init: float) -> dict[str, Any]:
    if method == "rbf_krr":
        return {"type": "sc_rbf", "input_dim": 2, "lengthscale_init": float(bandwidth_init)}
    if method == "dm_krr":
        return {"type": "sc_dm", "input_dim": 2, "eps_init": float(bandwidth_init)}
    raise ValueError(f"unknown method {method}")


def realized_kernel_value(model: Any, method: str) -> float:
    with torch.no_grad():
        if method == "rbf_krr":
            return float(model.kernel.ell.detach().cpu())
        return float(model.kernel.eps.detach().cpu())


def rmse(truth: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((truth - pred) ** 2)))


def fit_model(method: str, split: Split, bandwidth_init: float, ridge_init: float) -> Any:
    model = make_krr(
        type="share",
        kernel=kernel_config(method, bandwidth_init),
        dtype=torch.float64,
        ridge_init=float(ridge_init),
        jitter=0.0,
    )
    model.set_train_data(split.x_train, split.y_train)
    model.fit()
    return model


def fit_and_score(
    method: str,
    split: Split,
    bandwidth_init: float,
    ridge_init: float,
    *,
    include_test: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    model = fit_model(method, split, bandwidth_init, ridge_init)
    with torch.no_grad():
        y_val_pred = model(torch.as_tensor(split.x_val, dtype=torch.float64)).cpu().numpy()
        y_test_pred = (
            model(torch.as_tensor(split.x_test, dtype=torch.float64)).cpu().numpy()
            if include_test
            else None
        )
        train_residual = (
            float(model._residual.detach().cpu()) if model._residual is not None else math.nan
        )
    row: dict[str, Any] = {
        "validation_normalized_rmse": rmse(split.y_val, y_val_pred),
        "fit_seconds": time.perf_counter() - started,
        "realized_bandwidth": realized_kernel_value(model, method),
        "realized_ridge": float(model.ridge.detach().cpu()),
        "train_residual": train_residual,
    }
    if y_test_pred is not None:
        y_test_physical_pred = split.inverse_y(y_test_pred)
        row.update(
            {
                "error": rmse(split.y_test, y_test_pred),
                "test_physical_rmse": rmse(split.y_test_raw, y_test_physical_pred),
                "test_normalized_max_abs": float(np.max(np.abs(split.y_test - y_test_pred))),
            }
        )
    return row


def fit_and_score_folds(
    method: str,
    samples: NestedCartesianSamples,
    plan: LevelSamplePlan,
    trial: int | str,
    bandwidth_init: float,
    ridge_init: float,
) -> dict[str, Any]:
    fold_rows = [
        fit_and_score(
            method,
            samples.split_for_fold(trial, fold),
            bandwidth_init,
            ridge_init,
            include_test=False,
        )
        for fold in plan.validation_folds
    ]
    values = np.asarray(
        [float(row["validation_normalized_rmse"]) for row in fold_rows],
        dtype=float,
    )
    return {
        "validation_normalized_rmse": float(np.mean(values)),
        "std_metric": float(np.std(values)),
        "fold_metrics": values.tolist(),
        "fit_seconds": float(sum(float(row["fit_seconds"]) for row in fold_rows)),
    }


def tuning_spec(
    metric_name: str, initial_budget: int | tuple[int, ...], refinement_budget: int
) -> TuningSpec:
    return TuningSpec(
        parameters=(
            ParameterSpec("bandwidth_init", bounds=(1e-4, 1e2), scale="log"),
            ParameterSpec("ridge_init", bounds=(1e-16, 1e1), scale="log"),
        ),
        metric_name=metric_name,
        initial_budget=initial_budget,
        initial_strategy="grid",
        refinement_strategy="nelder_mead_like" if refinement_budget > 0 else None,
        refinement_budget=refinement_budget,
        metadata={"case": CASE_NAME},
    )


def parse_int_list(text: str) -> tuple[int, ...]:
    return tuple(int(item) for item in text.split(",") if item.strip())


def parse_int_or_tuple(text: str) -> int | tuple[int, ...]:
    values = parse_int_list(text)
    if not values:
        raise argparse.ArgumentTypeError("value must be a positive integer or comma list")
    return values[0] if len(values) == 1 else values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a standalone tuning + convergence-study example for the "
            "dm_conv_v4 cartesian_high_freq unit-disk KRR case."
        )
    )
    parser.add_argument("--workdir", type=Path, default=BASE_DIR / "cartesian_high_freq_study")
    parser.add_argument("--levels", default="32,64", help="Comma-separated training sizes.")
    parser.add_argument(
        "--trials",
        type=parse_int_or_tuple,
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
        type=parse_int_or_tuple,
        default=5,
        help=(
            "Initial search budget. Use N for total candidates, or n1,n2,... for "
            "per-parameter grid counts."
        ),
    )
    parser.add_argument("--refinement-budget", type=int, default=0)
    parser.add_argument("--tuning-policy", choices=("per_trial", "per_level"), default="per_trial")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--no-prediction-plots", action="store_true")
    return parser.parse_args()


def make_plot(
    result,
    output_dir: Path,
    *,
    center: str = "mean",
    band: str = "std",
) -> None:
    by_method: dict[str, list[dict[str, Any]]] = {method: [] for method in METHODS}
    for row in result.convergence_summary:
        by_method[str(row["method"])].append(row)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for method, rows in by_method.items():
        rows = sorted(rows, key=lambda item: float(item["refinement"]))
        if not rows:
            continue
        x = np.array([float(row["refinement"]) for row in rows])
        y = np.array([float(row[center]) for row in rows])
        if band == "iqr":
            lower = np.array([float(row["q25"]) for row in rows])
            upper = np.array([float(row["q75"]) for row in rows])
            ax.plot(x, y, marker="o", label=method)
            ax.fill_between(x, lower, upper, alpha=0.18)
        elif band == "q05_q95":
            lower = np.array([float(row["q05"]) for row in rows])
            upper = np.array([float(row["q95"]) for row in rows])
            ax.plot(x, y, marker="o", label=method)
            ax.fill_between(x, lower, upper, alpha=0.18)
        else:
            yerr = np.array([float(row[band]) for row in rows])
            ax.errorbar(x, y, yerr=yerr, marker="o", capsize=3, label=method)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("n_train")
    ax.set_ylabel("test normalized RMSE")
    ax.set_title("cartesian_high_freq KRR convergence")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "convergence.png", dpi=160)
    plt.close(fig)


def plot_truth_vs_prediction(context: MedianPlotContext, split: Split) -> None:
    model = fit_model(
        context.method,
        split,
        float(context.params["bandwidth_init"]),
        float(context.params["ridge_init"]),
    )
    with torch.no_grad():
        y_pred_norm = model(torch.as_tensor(split.x_test, dtype=torch.float64)).cpu().numpy()
    truth = split.y_test_raw.reshape(-1)
    pred = split.inverse_y(y_pred_norm).reshape(-1)
    abs_error = np.abs(truth - pred)
    color_max = max(float(np.max(np.abs(truth))), float(np.max(np.abs(pred))), 1e-12)

    context.output_path.parent.mkdir(parents=True, exist_ok=True)
    x_coord = split.x_test_raw[:, 0]
    y_coord = split.x_test_raw[:, 1]
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.2), constrained_layout=True)
    panels = (
        ("truth", truth, "coolwarm", np.linspace(-color_max, color_max, 24)),
        ("prediction", pred, "coolwarm", np.linspace(-color_max, color_max, 24)),
        (
            "absolute error",
            abs_error,
            "magma",
            np.linspace(0.0, max(float(abs_error.max()), 1e-12), 24),
        ),
    )
    for ax, (title, values, cmap, levels) in zip(axes, panels, strict=True):
        contour = ax.tricontourf(x_coord, y_coord, values, levels=levels, cmap=cmap)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(
        f"{context.method}, n_train={context.refinement}, "
        f"trial={context.trial}, {context.metric_name}={context.metric_value:.3g}",
        fontsize=10,
    )
    fig.savefig(context.output_path, dpi=160)
    plt.close(fig)


def _trial_ids_for_all_levels(
    trials: int | tuple[int | str, ...], n_levels: int
) -> tuple[int | str, ...]:
    if isinstance(trials, int):
        return tuple(range(trials))
    if trials and all(isinstance(item, int) and item > 0 for item in trials):
        return tuple(range(max(int(item) for item in trials[:n_levels])))
    return trials


def _nested_resampling_policy(
    args: argparse.Namespace, *, max_level: int
) -> NestedResamplingPolicy:
    if args.validation_mode == "holdout":
        validation = HoldoutValidationPolicy(validation_fraction=args.validation_fraction)
    elif args.validation_mode == "kfold":
        validation = KFoldValidationPolicy(k=args.k_folds)
    else:
        validation = TrainValidCountPolicy(
            validation_fraction=None
            if args.validation_size is not None
            else args.validation_fraction,
            validation_size=args.validation_size,
        )
    return NestedResamplingPolicy(
        test_size=args.n_test,
        validation=validation,
        seed=args.seed,
        dev_pool_size=max_level * args.pool_multiplier,
    )


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    output_dir = args.workdir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    levels = parse_int_list(args.levels)
    trials = args.trials
    trial_ids = _trial_ids_for_all_levels(trials, len(levels))
    split_cache: dict[tuple[int, int], Split] = {}
    max_level = max(levels)
    nested_policy = (
        _nested_resampling_policy(args, max_level=max_level)
        if args.resampling_mode == "nested-fixed-test"
        else None
    )
    nested_samples = (
        NestedCartesianSamples(
            max_train=nested_policy.dev_pool_size or max_level,
            n_test=args.n_test,
            seed=args.seed,
            trials=trial_ids,
        )
        if nested_policy is not None
        else None
    )

    def split_for(refinement: int | float | str, trial: int | str) -> Split:
        key = (int(refinement), int(trial))
        if key not in split_cache:
            split_cache[key] = make_split(int(refinement), args.n_val, args.n_test, int(trial))
        return split_cache[key]

    def split_for_context(context) -> Split:
        if nested_samples is None or context.sample_plan is None:
            return split_for(context.refinement, context.trial)
        return nested_samples.split_for_refit(context.trial, context.sample_plan)

    def tune_eval(
        method: str, refinement: int | float | str, trial: int | str, params: dict[str, Any]
    ):
        split = split_for(refinement, trial)
        return fit_and_score(
            method,
            split,
            float(params["bandwidth_init"]),
            float(params["ridge_init"]),
            include_test=False,
        )

    def tune_context_eval(context: TuningEvaluationContext):
        if nested_samples is None or context.sample_plan is None:
            return tune_eval(context.method, context.refinement, context.trial, context.params)
        return fit_and_score_folds(
            context.method,
            nested_samples,
            context.sample_plan,
            context.trial,
            float(context.params["bandwidth_init"]),
            float(context.params["ridge_init"]),
        )

    def study_eval(context: ConvergenceEvaluationContext) -> dict[str, Any]:
        split = split_for_context(context)
        return fit_and_score(
            context.method,
            split,
            float(context.params["bandwidth_init"]),
            float(context.params["ridge_init"]),
            include_test=True,
        )

    def median_plotter(context: MedianPlotContext) -> None:
        plot_truth_vs_prediction(context, split_for_context(context))

    specs = {
        method: tuning_spec(
            "validation_normalized_rmse", args.initial_budget, args.refinement_budget
        )
        for method in METHODS
    }
    study_spec = ConvergenceStudySpec(
        methods=METHODS,
        refinement_levels=levels,
        trials=trials,
        metrics=("error", "test_physical_rmse", "test_normalized_max_abs", "fit_seconds"),
        tuning_policy=TuningPolicy(mode=args.tuning_policy, specs=specs),
        fit_window=levels,
        artifact_dir=output_dir,
        primary_metric="error",
        resampling=nested_policy,
    )
    result = run_convergence_study(
        study_spec,
        study_eval,
        tuning_evaluator=tune_eval if nested_policy is None else None,
        tuning_context_evaluator=tune_context_eval if nested_policy is not None else None,
        median_plotter=None if args.no_prediction_plots else median_plotter,
        max_workers=args.max_workers,
        tuning_max_workers=args.max_workers,
    )
    if not args.no_plot:
        center = "median" if nested_policy is not None else "mean"
        band = args.confidence_band or ("iqr" if nested_policy is not None else "std")
        make_plot(result, output_dir, center=center, band=band)
    print(f"Wrote convergence artifacts to {output_dir}")
    if result.diagnostics:
        print(f"Diagnostics: {len(result.diagnostics)} advisory item(s); see diagnostics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
