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
    MedianPlotContext,
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


def tuning_spec(metric_name: str, initial_budget: int, refinement_budget: int) -> TuningSpec:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a standalone tuning + convergence-study example for the "
            "dm_conv_v4 cartesian_high_freq unit-disk KRR case."
        )
    )
    parser.add_argument("--workdir", type=Path, default=BASE_DIR / "cartesian_high_freq_study")
    parser.add_argument("--levels", default="32,64", help="Comma-separated training sizes.")
    parser.add_argument("--trials", default="0", help="Comma-separated trial seeds.")
    parser.add_argument("--n-val", type=int, default=32)
    parser.add_argument("--n-test", type=int, default=128)
    parser.add_argument("--initial-budget", type=int, default=5)
    parser.add_argument("--refinement-budget", type=int, default=0)
    parser.add_argument("--tuning-policy", choices=("per_trial", "per_level"), default="per_trial")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--no-prediction-plots", action="store_true")
    return parser.parse_args()


def make_plot(result, output_dir: Path) -> None:
    by_method: dict[str, list[dict[str, Any]]] = {method: [] for method in METHODS}
    for row in result.convergence_summary:
        by_method[str(row["method"])].append(row)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for method, rows in by_method.items():
        rows = sorted(rows, key=lambda item: float(item["refinement"]))
        if not rows:
            continue
        x = np.array([float(row["refinement"]) for row in rows])
        y = np.array([float(row["mean"]) for row in rows])
        yerr = np.array([float(row["std"]) for row in rows])
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


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    output_dir = args.workdir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    levels = parse_int_list(args.levels)
    trials = parse_int_list(args.trials)
    split_cache: dict[tuple[int, int], Split] = {}

    def split_for(refinement: int | float | str, trial: int | str) -> Split:
        key = (int(refinement), int(trial))
        if key not in split_cache:
            split_cache[key] = make_split(int(refinement), args.n_val, args.n_test, int(trial))
        return split_cache[key]

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

    def study_eval(context: ConvergenceEvaluationContext) -> dict[str, Any]:
        split = split_for(context.refinement, context.trial)
        return fit_and_score(
            context.method,
            split,
            float(context.params["bandwidth_init"]),
            float(context.params["ridge_init"]),
            include_test=True,
        )

    def median_plotter(context: MedianPlotContext) -> None:
        plot_truth_vs_prediction(context, split_for(context.refinement, context.trial))

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
    )
    result = run_convergence_study(
        study_spec,
        study_eval,
        tuning_evaluator=tune_eval,
        median_plotter=None if args.no_prediction_plots else median_plotter,
    )
    if not args.no_plot:
        make_plot(result, output_dir)
    print(f"Wrote convergence artifacts to {output_dir}")
    if result.diagnostics:
        print(f"Diagnostics: {len(result.diagnostics)} advisory item(s); see diagnostics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
