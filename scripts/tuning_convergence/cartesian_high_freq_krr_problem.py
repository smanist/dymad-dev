from __future__ import annotations

import math
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dymad.io import Split
from dymad.modules import make_krr
from dymad.studies.convergence import (
    ArrayRegressionProblem,
    CurveStyle,
    LevelSamplePlan,
    MedianPlotContext,
    NestedArraySamples,
    plot_convergence_summary,
)
from dymad.tuning import ParameterSpec, TuningSpec

CASE_NAME = "cartesian_high_freq"
METHODS = ("rbf_krr", "dm_krr")


def unit_disk_sample(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    radius = np.sqrt(rng.random(n_samples))
    theta = 2.0 * math.pi * rng.random(n_samples)
    return np.column_stack((radius * np.cos(theta), radius * np.sin(theta)))


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


def fit_model(method: str, split: Split, params: dict[str, Any]) -> Any:
    model = make_krr(
        type="share",
        kernel=kernel_config(method, float(params["bandwidth_init"])),
        dtype=torch.float64,
        ridge_init=float(params["ridge_init"]),
        jitter=0.0,
    )
    model.set_train_data(split.x_train, split.y_train)
    model.fit()
    return model


def fit_and_score(
    method: str,
    split: Split,
    params: dict[str, Any],
    include_test: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    model = fit_model(method, split, params)
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
    samples: NestedArraySamples,
    plan: LevelSamplePlan,
    trial: int | str,
    params: dict[str, Any],
) -> dict[str, Any]:
    fold_rows = [
        fit_and_score(method, samples.split_for_fold(trial, fold), params, include_test=False)
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
    metric_name: str,
    initial_budget: int | tuple[int, ...],
    refinement_budget: int,
    refinement_strategy: str | None,
) -> TuningSpec:
    strategy = (refinement_strategy or "nelder_mead_like") if refinement_budget > 0 else None
    return TuningSpec(
        parameters=(
            ParameterSpec("bandwidth_init", bounds=(1e-4, 1e2), scale="log"),
            ParameterSpec("ridge_init", bounds=(1e-16, 1e1), scale="log"),
        ),
        metric_name=metric_name,
        initial_budget=initial_budget,
        initial_strategy="grid",
        refinement_strategy=strategy,
        refinement_budget=refinement_budget,
        metadata={"case": CASE_NAME},
    )


def make_convergence_plot(result: Any, output_dir: Path, center: str, band: str) -> None:
    plot_convergence_summary(
        result,
        output_dir / "convergence.png",
        methods=METHODS,
        center=center,
        band=band,
        title="cartesian_high_freq KRR convergence",
        xlabel="n_train",
        ylabel="test normalized RMSE",
        styles={
            "rbf_krr": CurveStyle(label="RBF KRR"),
            "dm_krr": CurveStyle(label="DM KRR"),
        },
    )


def plot_truth_vs_prediction(context: MedianPlotContext, split: Split) -> None:
    model = fit_model(context.method, split, context.params)
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


def make_problem(
    target_name: str, target: Callable[[np.ndarray], np.ndarray]
) -> ArrayRegressionProblem:
    return ArrayRegressionProblem(
        name=f"{CASE_NAME}_{target_name}",
        methods=METHODS,
        sample=unit_disk_sample,
        target=target,
        fit_and_score=fit_and_score,
        fit_and_score_folds=fit_and_score_folds,
        tuning_spec=tuning_spec,
        metrics=("error", "test_physical_rmse", "test_normalized_max_abs", "fit_seconds"),
        primary_metric="error",
        prediction_plotter=plot_truth_vs_prediction,
    )
