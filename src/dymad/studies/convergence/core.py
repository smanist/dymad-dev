from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from dymad.tuning import TuningResult, TuningSpec, tune, write_tuning_artifacts


@dataclass(frozen=True)
class TuningPolicy:
    mode: str = "none"
    specs: Mapping[str, TuningSpec] = field(default_factory=dict)
    fixed_params: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    external_params: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    reference_level: float | int | str | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"none", "per_trial", "per_level", "reference_level", "external"}:
            raise ValueError(
                "TuningPolicy.mode must be none, per_trial, per_level, reference_level, or external"
            )


@dataclass(frozen=True)
class ConvergenceStudySpec:
    methods: tuple[str, ...]
    refinement_levels: tuple[float | int | str, ...]
    trials: int | tuple[int | str, ...]
    metrics: tuple[str, ...]
    tuning_policy: TuningPolicy = field(default_factory=TuningPolicy)
    fit_window: tuple[float | int | str, ...] | None = None
    group_columns: tuple[str, ...] = ("method", "metric")
    artifact_dir: str | Path | None = None
    primary_metric: str | None = None
    high_variance_cv_threshold: float = 0.5
    low_r2_threshold: float = 0.9

    def __post_init__(self) -> None:
        if not self.methods:
            raise ValueError("ConvergenceStudySpec.methods must be non-empty")
        if not self.refinement_levels:
            raise ValueError("ConvergenceStudySpec.refinement_levels must be non-empty")
        if isinstance(self.trials, int):
            if self.trials <= 0:
                raise ValueError("ConvergenceStudySpec.trials must be positive")
        elif not self.trials:
            raise ValueError("ConvergenceStudySpec.trials must be non-empty")
        elif _looks_like_trial_count_tuple(self.trials):
            if len(self.trials) != len(self.refinement_levels):
                raise ValueError(
                    "ConvergenceStudySpec.trials count tuple length must match refinement_levels"
                )
        if not self.metrics:
            raise ValueError("ConvergenceStudySpec.metrics must be non-empty")
        if self.primary_metric is not None and self.primary_metric not in self.metrics:
            raise ValueError("ConvergenceStudySpec.primary_metric must be listed in metrics")


@dataclass
class ConvergenceEvaluationContext:
    method: str
    refinement: float | int | str
    trial: int | str
    params: dict[str, Any]
    tuning_result: TuningResult | None = None


@dataclass
class MedianPlotContext:
    method: str
    refinement: float | int | str
    trial: int | str
    params: dict[str, Any]
    metric_name: str
    metric_value: float
    raw_row: dict[str, Any]
    output_path: Path
    tuning_result: TuningResult | None = None


@dataclass
class Diagnostic:
    kind: str
    severity: str
    message: str
    recommendation: str
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConvergenceStudyResult:
    raw_rows: list[dict[str, Any]]
    metric_values: list[dict[str, Any]]
    trial_statistics: list[dict[str, Any]]
    convergence_summary: list[dict[str, Any]]
    convergence_rates: list[dict[str, Any]]
    diagnostics: list[Diagnostic]
    tuning_results: dict[str, TuningResult]
    median_plot_paths: dict[str, str] = field(default_factory=dict)


StudyEvaluator = Callable[[ConvergenceEvaluationContext], Mapping[str, Any]]
TuningEvaluator = Callable[
    [str, float | int | str, int | str, dict[str, Any]], float | Mapping[str, Any]
]
MedianPlotter = Callable[[MedianPlotContext], None]


def run_convergence_study(
    spec: ConvergenceStudySpec,
    evaluator: StudyEvaluator,
    *,
    tuning_evaluator: TuningEvaluator | None = None,
    median_plotter: MedianPlotter | None = None,
    max_workers: int = 1,
    tuning_max_workers: int | None = None,
) -> ConvergenceStudyResult:
    if max_workers <= 0:
        raise ValueError("max_workers must be positive")
    if tuning_max_workers is None:
        tuning_max_workers = max_workers
    if tuning_max_workers <= 0:
        raise ValueError("tuning_max_workers must be positive")
    tuning_cache: dict[tuple[Any, ...], TuningResult] = {}
    tuning_results: dict[str, TuningResult] = {}
    contexts: list[ConvergenceEvaluationContext] = []

    artifact_dir = Path(spec.artifact_dir) if spec.artifact_dir is not None else None
    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=True)

    for method in spec.methods:
        for level_index, refinement in enumerate(spec.refinement_levels):
            for trial in _trials_for_level(spec, level_index):
                tuning_result = _resolve_tuning(
                    spec,
                    method,
                    refinement,
                    trial,
                    tuning_evaluator=tuning_evaluator,
                    cache=tuning_cache,
                    max_workers=tuning_max_workers,
                )
                tuning_key = _tuning_artifact_key(
                    spec.tuning_policy.mode, method, refinement, trial
                )
                if tuning_result is not None:
                    tuning_results.setdefault(tuning_key, tuning_result)
                    if artifact_dir is not None:
                        write_tuning_artifacts(
                            tuning_result,
                            artifact_dir / "tuning" / tuning_key,
                        )
                params = (
                    dict(tuning_result.selected_params)
                    if tuning_result is not None
                    else dict(spec.tuning_policy.fixed_params.get(method, {}))
                )
                context = ConvergenceEvaluationContext(
                    method, refinement, trial, params, tuning_result
                )
                contexts.append(context)

    raw_rows = _evaluate_study_contexts(
        contexts,
        metrics=spec.metrics,
        evaluator=evaluator,
        max_workers=max_workers,
    )

    metric_values = _metric_values(raw_rows, spec.metrics)
    trial_statistics = aggregate_trials(
        metric_values,
        group_columns=[*spec.group_columns, "refinement"],
        value_column="value",
    )
    primary_metric = spec.primary_metric or spec.metrics[0]
    convergence_summary = [row for row in trial_statistics if row.get("metric") == primary_metric]
    convergence_rates = fit_convergence_rates(
        convergence_summary,
        group_columns=[column for column in spec.group_columns if column != "metric"],
        refinement_column="refinement",
        value_column="mean",
        window=list(spec.fit_window) if spec.fit_window is not None else None,
        loglog=True,
    )
    median_plot_paths: dict[str, str] = {}
    plot_diagnostics: list[Diagnostic] = []
    if artifact_dir is not None and median_plotter is not None:
        median_plot_paths, plot_diagnostics = _write_median_prediction_plots(
            artifact_dir,
            spec,
            raw_rows=raw_rows,
            tuning_results=tuning_results,
            median_plotter=median_plotter,
        )
    diagnostics = diagnose_convergence(
        spec,
        raw_rows=raw_rows,
        trial_statistics=trial_statistics,
        convergence_summary=convergence_summary,
        convergence_rates=convergence_rates,
        tuning_results=tuning_results,
    )
    diagnostics.extend(plot_diagnostics)
    result = ConvergenceStudyResult(
        raw_rows=raw_rows,
        metric_values=metric_values,
        trial_statistics=trial_statistics,
        convergence_summary=convergence_summary,
        convergence_rates=convergence_rates,
        diagnostics=diagnostics,
        tuning_results=tuning_results,
        median_plot_paths=median_plot_paths,
    )
    if artifact_dir is not None:
        _write_study_artifacts(artifact_dir, result)
    return result


def _resolve_tuning(
    spec: ConvergenceStudySpec,
    method: str,
    refinement: float | int | str,
    trial: int | str,
    *,
    tuning_evaluator: TuningEvaluator | None,
    cache: dict[tuple[Any, ...], TuningResult],
    max_workers: int,
) -> TuningResult | None:
    policy = spec.tuning_policy
    if policy.mode == "none":
        params = dict(policy.fixed_params.get(method, {}))
        return _recorded_params_result(params, {"mode": "none", "method": method})
    if policy.mode == "external":
        params = dict(policy.external_params.get(method, policy.fixed_params.get(method, {})))
        return _recorded_params_result(params, {"mode": "external", "method": method})
    if tuning_evaluator is None:
        raise ValueError(f"tuning_evaluator is required for tuning policy {policy.mode!r}")
    if method not in policy.specs:
        raise ValueError(f"missing tuning spec for method {method!r}")
    if policy.mode == "per_trial":
        key = ("per_trial", method, refinement, trial)
    elif policy.mode == "per_level":
        key = ("per_level", method, refinement)
    elif policy.mode == "reference_level":
        reference_level = (
            policy.reference_level
            if policy.reference_level is not None
            else spec.refinement_levels[-1]
        )
        key = ("reference_level", method, reference_level)
    else:
        raise AssertionError(f"unsupported policy {policy.mode}")
    if key not in cache:
        tuning_refinement = key[2]
        tuning_trial = trial

        def objective(params: dict[str, Any]) -> float | Mapping[str, Any]:
            return tuning_evaluator(method, tuning_refinement, tuning_trial, params)

        cache[key] = tune(policy.specs[method], objective, max_workers=max_workers)
    return cache[key]


def _recorded_params_result(params: dict[str, Any], policy: dict[str, Any]) -> TuningResult:
    return TuningResult(
        selected_params=dict(params),
        selected_metric=math.nan,
        evaluations=[],
        failures=[],
        candidate_plan={"strategy": "none", "candidates": []},
        policy=policy,
    )


def _evaluate_study_contexts(
    contexts: Sequence[ConvergenceEvaluationContext],
    *,
    metrics: Sequence[str],
    evaluator: StudyEvaluator,
    max_workers: int,
) -> list[dict[str, Any]]:
    if max_workers > 1 and len(contexts) > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(
                executor.map(
                    lambda context: _evaluate_study_context(context, metrics, evaluator),
                    contexts,
                )
            )
    return [_evaluate_study_context(context, metrics, evaluator) for context in contexts]


def _evaluate_study_context(
    context: ConvergenceEvaluationContext,
    metrics: Sequence[str],
    evaluator: StudyEvaluator,
) -> dict[str, Any]:
    try:
        metric_values = dict(evaluator(context))
        status = str(metric_values.pop("status", "ok"))
        failure_reason = str(metric_values.pop("failure_reason", "")) if status != "ok" else ""
    except Exception as exc:  # noqa: BLE001 - failed trials are study artifacts.
        metric_values = {}
        status = "failed"
        failure_reason = f"{type(exc).__name__}: {exc}"
    row = {
        "method": context.method,
        "refinement": context.refinement,
        "trial": context.trial,
        "status": status,
        "failure_reason": failure_reason,
        "params": dict(context.params),
    }
    for metric in metrics:
        row[metric] = metric_values.get(metric, math.nan)
    return row


def _trials_for_level(spec: ConvergenceStudySpec, level_index: int) -> tuple[int | str, ...]:
    trials = spec.trials
    if isinstance(trials, int):
        return tuple(range(trials))
    if _looks_like_trial_count_tuple(trials):
        return tuple(range(int(trials[level_index])))
    return trials


def _looks_like_trial_count_tuple(trials: tuple[int | str, ...]) -> bool:
    return all(isinstance(item, int) and item > 0 for item in trials)


def aggregate_trials(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_columns: Sequence[str],
    value_column: str,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for row in rows:
        try:
            value = float(row[value_column])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        key = tuple(str(row.get(column, "")) for column in group_columns)
        groups[key].append(value)
    output = []
    for key in sorted(groups):
        values = groups[key]
        mean = sum(values) / len(values)
        std = (
            math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
            if len(values) > 1
            else 0.0
        )
        summary_row: dict[str, Any] = {
            column: key[index] for index, column in enumerate(group_columns)
        }
        summary_row.update(
            {
                "count": len(values),
                "mean": mean,
                "std": std,
                "min": min(values),
                "max": max(values),
            }
        )
        output.append(summary_row)
    return output


def fit_convergence_rates(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_columns: Sequence[str],
    refinement_column: str,
    value_column: str,
    window: Sequence[float | int | str] | None = None,
    loglog: bool = True,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[tuple[float, float]]] = defaultdict(list)
    window_values = {float(value) for value in window} if window is not None else None
    for row in rows:
        try:
            refinement = float(row[refinement_column])
            value = float(row[value_column])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(refinement) or not math.isfinite(value):
            continue
        if refinement <= 0 or (loglog and value <= 0):
            continue
        if window_values is not None and refinement not in window_values:
            continue
        groups[tuple(str(row.get(column, "")) for column in group_columns)].append(
            (refinement, value)
        )
    output = []
    for key in sorted(groups):
        points = sorted(groups[key])
        if len(points) < 2:
            output.append(
                {
                    **{column: key[index] for index, column in enumerate(group_columns)},
                    "status": "insufficient_fit_window",
                    "n_points": len(points),
                    "fit_window": [point[0] for point in points],
                }
            )
            continue
        x_values = [math.log(x) if loglog else x for x, _ in points]
        y_values = [math.log(y) if loglog else y for _, y in points]
        slope, intercept = _least_squares_line(x_values, y_values)
        fitted = [intercept + slope * x for x in x_values]
        output.append(
            {
                **{column: key[index] for index, column in enumerate(group_columns)},
                "status": "ok",
                "n_points": len(points),
                "fit_window": [point[0] for point in points],
                "slope": slope,
                "intercept": intercept,
                "order": -slope if loglog else slope,
                "r2": _r2(y_values, fitted),
                "monotone_decreasing": all(
                    points[index + 1][1] <= points[index][1] for index in range(len(points) - 1)
                ),
            }
        )
    return output


def diagnose_convergence(
    spec: ConvergenceStudySpec,
    *,
    raw_rows: Sequence[Mapping[str, Any]],
    trial_statistics: Sequence[Mapping[str, Any]],
    convergence_summary: Sequence[Mapping[str, Any]],
    convergence_rates: Sequence[Mapping[str, Any]],
    tuning_results: Mapping[str, TuningResult],
) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    for row in convergence_rates:
        if row.get("status") != "ok":
            diagnostics.append(
                Diagnostic(
                    "missing_or_invalid_fit_window",
                    "warning",
                    "Convergence rate could not be fit for one group.",
                    "Choose a fit window with at least two positive finite points.",
                    dict(row),
                )
            )
            continue
        if not bool(row.get("monotone_decreasing", False)):
            diagnostics.append(
                Diagnostic(
                    "non_monotone_fit_window",
                    "warning",
                    "Metric is not monotone decreasing over the selected fit window.",
                    "Inspect raw levels, consider reporting this window as pre-asymptotic, or add finer levels.",
                    dict(row),
                )
            )
        r2 = float(row.get("r2", math.nan))
        if not math.isfinite(r2) or r2 < spec.low_r2_threshold:
            diagnostics.append(
                Diagnostic(
                    "low_loglog_fit_quality",
                    "warning",
                    "Log-log convergence fit has low or undefined R2.",
                    "Inspect alternate windows and raw curves before making rate claims.",
                    dict(row),
                )
            )
    for item in tuning_results.values():
        if item.failures:
            diagnostics.append(
                Diagnostic(
                    "tuning_failures_or_boundary",
                    "warning",
                    "At least one tuning invocation had failed or boundary-hit evaluations.",
                    "Inspect tuning_failures.csv before interpreting convergence claims.",
                    {"selected_params": item.selected_params, "failure_count": len(item.failures)},
                )
            )
    failed_trials = [row for row in raw_rows if row.get("status") != "ok"]
    if failed_trials:
        diagnostics.append(
            Diagnostic(
                "failed_trials",
                "warning",
                "One or more study trials failed.",
                "Inspect raw_results.csv and rerun or explain failed trials explicitly.",
                {"failed_count": len(failed_trials)},
            )
        )
    for row in trial_statistics:
        mean = abs(float(row.get("mean", 0.0)))
        std = float(row.get("std", 0.0))
        if mean > 0.0 and std / mean > spec.high_variance_cv_threshold:
            diagnostics.append(
                Diagnostic(
                    "high_trial_variance",
                    "warning",
                    "Trial standard deviation is large relative to the mean.",
                    "Increase trials or audit seeds/splits before making fine-grained comparisons.",
                    dict(row),
                )
            )
    if spec.fit_window is not None:
        available = {str(row.get("refinement")) for row in convergence_summary}
        missing = [
            value
            for value in spec.fit_window
            if str(float(value)) not in available and str(value) not in available
        ]
        if missing:
            diagnostics.append(
                Diagnostic(
                    "missing_fit_window_levels",
                    "warning",
                    "Some requested fit-window levels are missing from the convergence summary.",
                    "Use available levels or rerun the missing refinements.",
                    {"missing": list(missing)},
                )
            )
    return diagnostics


def _metric_values(
    raw_rows: Sequence[Mapping[str, Any]], metrics: Sequence[str]
) -> list[dict[str, Any]]:
    rows = []
    for row in raw_rows:
        if row.get("status") != "ok":
            continue
        for metric in metrics:
            rows.append(
                {
                    "method": row["method"],
                    "refinement": row["refinement"],
                    "trial": row["trial"],
                    "metric": metric,
                    "value": row.get(metric, math.nan),
                }
            )
    return rows


def _write_median_prediction_plots(
    root: Path,
    spec: ConvergenceStudySpec,
    *,
    raw_rows: Sequence[Mapping[str, Any]],
    tuning_results: Mapping[str, TuningResult],
    median_plotter: MedianPlotter,
) -> tuple[dict[str, str], list[Diagnostic]]:
    primary_metric = spec.primary_metric or spec.metrics[0]
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        if row.get("status") != "ok":
            continue
        try:
            metric_value = float(row[primary_metric])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(metric_value):
            continue
        grouped[(str(row["method"]), str(row["refinement"]))].append(row)

    paths: dict[str, str] = {}
    diagnostics: list[Diagnostic] = []
    output_root = root / "median_predictions"
    output_root.mkdir(parents=True, exist_ok=True)
    for _, rows in sorted(grouped.items()):
        values = [float(row[primary_metric]) for row in rows]
        median_value = float(np.median(np.asarray(values, dtype=float)))
        selected = min(
            rows,
            key=lambda row: (
                abs(float(row[primary_metric]) - median_value),
                str(row.get("trial", "")),
            ),
        )
        method = str(selected["method"])
        refinement = selected["refinement"]
        trial = selected["trial"]
        plot_key = f"{_slug(method)}__level_{_slug(refinement)}"
        output_path = output_root / f"{plot_key}.png"
        params = dict(selected.get("params", {}))
        tuning_key = _tuning_artifact_key(spec.tuning_policy.mode, method, refinement, trial)
        context = MedianPlotContext(
            method=method,
            refinement=refinement,
            trial=trial,
            params=params,
            metric_name=primary_metric,
            metric_value=float(selected[primary_metric]),
            raw_row=dict(selected),
            output_path=output_path,
            tuning_result=tuning_results.get(tuning_key),
        )
        try:
            median_plotter(context)
        except Exception as exc:  # noqa: BLE001 - plot failures are advisory artifacts.
            diagnostics.append(
                Diagnostic(
                    "median_prediction_plot_failed",
                    "warning",
                    "Median truth-vs-prediction plot generation failed for one group.",
                    "Inspect the supplied median_plotter and rerun plotting if needed.",
                    {
                        "method": method,
                        "refinement": refinement,
                        "trial": trial,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
            )
            continue
        if output_path.exists():
            paths[plot_key] = str(output_path.relative_to(root))
    return paths, diagnostics


def _write_study_artifacts(root: Path, result: ConvergenceStudyResult) -> None:
    _write_csv(root / "raw_results.csv", result.raw_rows)
    _write_csv(root / "metric_values.csv", result.metric_values)
    _write_csv(root / "trial_statistics.csv", result.trial_statistics)
    _write_csv(root / "convergence_summary.csv", result.convergence_summary)
    _write_csv(root / "convergence_rates.csv", result.convergence_rates)
    (root / "convergence_rates.json").write_text(
        json.dumps(_jsonable(result.convergence_rates), indent=2) + "\n",
        encoding="utf-8",
    )
    (root / "diagnostics.json").write_text(
        json.dumps([_jsonable(asdict(item)) for item in result.diagnostics], indent=2) + "\n",
        encoding="utf-8",
    )
    if result.median_plot_paths:
        (root / "median_prediction_plots.json").write_text(
            json.dumps(_jsonable(result.median_plot_paths), indent=2) + "\n",
            encoding="utf-8",
        )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format_csv_value(row.get(key, "")) for key in fieldnames})


def _tuning_artifact_key(
    mode: str,
    method: str,
    refinement: float | int | str,
    trial: int | str,
) -> str:
    if mode == "per_trial":
        return f"{_slug(method)}__level_{_slug(refinement)}__trial_{_slug(trial)}"
    if mode == "per_level":
        return f"{_slug(method)}__level_{_slug(refinement)}"
    if mode == "reference_level":
        return f"{_slug(method)}__reference_level"
    return f"{_slug(method)}__{mode}"


def _least_squares_line(
    x_values: Sequence[float], y_values: Sequence[float]
) -> tuple[float, float]:
    if len(x_values) < 2:
        raise ValueError("at least two points are required to fit convergence")
    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(y_values) / len(y_values)
    denom = sum((x - x_mean) ** 2 for x in x_values)
    if denom == 0:
        raise ValueError("refinement values must not all be equal")
    slope = (
        sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values, strict=True)) / denom
    )
    return slope, y_mean - slope * x_mean


def _r2(y_values: Sequence[float], fitted: Sequence[float]) -> float:
    mean = sum(y_values) / len(y_values)
    total = sum((y - mean) ** 2 for y in y_values)
    if total == 0.0:
        return 1.0
    residual = sum((y - fit) ** 2 for y, fit in zip(y_values, fitted, strict=True))
    return 1.0 - residual / total


def _slug(value: Any) -> str:
    text = str(value).replace("/", "_").replace(" ", "_")
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)


def _format_csv_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return json.dumps(_jsonable(value), sort_keys=True)
    if isinstance(value, (list, tuple)):
        return json.dumps(_jsonable(value))
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return f"{value:.12g}"
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
