from __future__ import annotations

import csv
import json
import math
import random
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

MetricEvaluator = Callable[[dict[str, Any]], float | Mapping[str, Any]]


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    bounds: tuple[float, float] | None = None
    values: tuple[Any, ...] | None = None
    scale: str = "linear"
    value_kind: str = "float"
    step: float | None = None
    parity: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("parameter name must be non-empty")
        if self.values is None and self.bounds is None:
            raise ValueError(f"{self.name}: provide either values or bounds")
        if self.values is not None and len(self.values) == 0:
            raise ValueError(f"{self.name}: values must not be empty")
        if self.bounds is not None:
            if len(self.bounds) != 2:
                raise ValueError(f"{self.name}: bounds must contain two values")
            lower, upper = float(self.bounds[0]), float(self.bounds[1])
            if lower > upper:
                raise ValueError(f"{self.name}: lower bound must be <= upper bound")
            object.__setattr__(self, "bounds", (lower, upper))
        if self.scale not in {"linear", "log"}:
            raise ValueError(f"{self.name}: scale must be linear or log")
        if (
            self.scale == "log"
            and self.bounds is not None
            and (self.bounds[0] <= 0.0 or self.bounds[1] <= 0.0)
        ):
            raise ValueError(f"{self.name}: log bounds must be positive")
        value_kind = "int" if self.value_kind == "integer" else self.value_kind
        if value_kind not in {"float", "int"}:
            raise ValueError(f"{self.name}: value_kind must be float or int")
        object.__setattr__(self, "value_kind", value_kind)
        if self.step is not None and self.step <= 0.0:
            raise ValueError(f"{self.name}: step must be positive")
        if value_kind == "int" and self.step is not None and not float(self.step).is_integer():
            raise ValueError(f"{self.name}: integer step must be an integer")
        if self.parity is not None:
            if self.parity not in {"even", "odd"}:
                raise ValueError(f"{self.name}: parity must be even or odd")
            if value_kind != "int":
                raise ValueError(f"{self.name}: parity requires integer value_kind")
        if self.values is None and value_kind == "int" and _integer_value_count(self) <= 0:
            raise ValueError(f"{self.name}: integer constraints admit no values")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> ParameterSpec:
        if "name" not in raw:
            raise ValueError("parameter missing required field: name")
        bounds = tuple(raw["bounds"]) if "bounds" in raw else None
        values = tuple(raw["values"]) if "values" in raw else None
        return cls(
            name=str(raw["name"]),
            bounds=bounds,  # type: ignore[arg-type]
            values=values,
            scale=str(raw.get("scale", "linear")),
            value_kind=str(raw.get("value_kind", raw.get("type", "float"))),
            step=float(raw["step"]) if "step" in raw else None,
            parity=str(raw["parity"]) if raw.get("parity") is not None else None,
        )

    @property
    def is_discrete(self) -> bool:
        return self.values is not None or self.value_kind == "int" or self.step is not None

    def domain_summary(self) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "name": self.name,
            "scale": self.scale,
            "value_kind": self.value_kind,
        }
        if self.bounds is not None:
            summary["bounds"] = list(self.bounds)
        if self.values is not None:
            summary["values"] = list(self.values)
        if self.step is not None:
            summary["step"] = self.step
        if self.parity is not None:
            summary["parity"] = self.parity
        return summary

    def project(self, value: Any) -> Any:
        if self.values is not None:
            try:
                numeric = float(value)
                return min(self.values, key=lambda candidate: abs(float(candidate) - numeric))
            except (TypeError, ValueError):
                return value if value in self.values else self.values[0]
        assert self.bounds is not None
        clipped = min(max(float(value), self.bounds[0]), self.bounds[1])
        if self.value_kind == "int":
            return _project_integer(self, clipped)
        if self.step is not None:
            index = round((clipped - self.bounds[0]) / self.step)
            return _clean_float(_stepped_float_value_at_index(self, int(index)))
        return clipped


@dataclass(frozen=True)
class TuningSpec:
    parameters: tuple[ParameterSpec, ...]
    metric_name: str = "metric"
    goal: str = "minimize"
    initial_budget: int = 1
    initial_strategy: str = "auto"
    refinement_strategy: str | None = None
    refinement_budget: int = 0
    selection_tie_breakers: tuple[str, ...] = ("metric", "param_l1", "candidate_index")
    seed: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.parameters:
            raise ValueError("TuningSpec.parameters must be non-empty")
        if self.goal not in {"minimize", "maximize"}:
            raise ValueError("TuningSpec.goal must be minimize or maximize")
        if self.initial_budget <= 0:
            raise ValueError("TuningSpec.initial_budget must be positive")
        if self.initial_strategy not in {"auto", "grid", "random"}:
            raise ValueError("TuningSpec.initial_strategy must be auto, grid, or random")
        if self.refinement_strategy not in {None, "nelder_mead_like"}:
            raise ValueError("TuningSpec.refinement_strategy must be None or nelder_mead_like")
        if self.refinement_budget < 0:
            raise ValueError("TuningSpec.refinement_budget must be non-negative")


@dataclass
class TuningEvaluation:
    params: dict[str, Any]
    phase: str
    index: int
    metric_value: float
    status: str
    elapsed_seconds: float
    cache_hit: bool = False
    boundary_hit: bool = False
    failure_reason: str | None = None
    extra_metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class TuningResult:
    selected_params: dict[str, Any]
    selected_metric: float
    evaluations: list[TuningEvaluation]
    failures: list[TuningEvaluation]
    candidate_plan: dict[str, Any]
    policy: dict[str, Any] = field(default_factory=dict)


def iter_param_grid(param_grid: Mapping[str, Iterable[Any]]) -> Iterable[dict[str, Any]]:
    keys = list(param_grid.keys())
    values_lists: list[list[Any]] = []
    for key in keys:
        value = param_grid[key]
        if isinstance(value, list):
            values_lists.append(value)
        elif isinstance(value, tuple):
            if value[0] == "linspace":
                values_lists.append(np.linspace(*value[1]).tolist())
            elif value[0] == "logspace":
                values_lists.append(np.logspace(*value[1]).tolist())
            else:
                raise ValueError(f"Unknown param grid specifier: {value}")
        else:
            raise ValueError(f"Param grid values must be lists or tuples, got {type(value)}")
    for values in product(*values_lists):
        yield dict(zip(keys, values, strict=False))


def initial_search_plan(spec: TuningSpec) -> dict[str, Any]:
    strategy = spec.initial_strategy
    if strategy == "auto":
        strategy = "grid" if len(spec.parameters) <= 3 else "random"
    candidates = (
        _grid_candidates(spec.parameters, spec.initial_budget)
        if strategy == "grid"
        else _random_candidates(spec.parameters, spec.initial_budget, spec.seed)
    )
    return {
        "strategy": strategy,
        "initial_budget": spec.initial_budget,
        "candidate_count": len(candidates),
        "parameter_domains": [parameter.domain_summary() for parameter in spec.parameters],
        "candidates": candidates,
        "refinement": {
            "strategy": spec.refinement_strategy,
            "budget": spec.refinement_budget,
        },
    }


def tune(spec: TuningSpec, evaluator: MetricEvaluator) -> TuningResult:
    plan = initial_search_plan(spec)
    evaluations: list[TuningEvaluation] = []
    failures: list[TuningEvaluation] = []
    cache: dict[tuple[tuple[str, Any], ...], TuningEvaluation] = {}

    def evaluate(params: dict[str, Any], phase: str, index: int) -> TuningEvaluation:
        projected = {
            parameter.name: parameter.project(params[parameter.name])
            for parameter in spec.parameters
        }
        key = _params_key(projected)
        if key in cache:
            cached = cache[key]
            item = TuningEvaluation(
                params=dict(cached.params),
                phase=phase,
                index=index,
                metric_value=cached.metric_value,
                status=cached.status,
                elapsed_seconds=0.0,
                cache_hit=True,
                boundary_hit=cached.boundary_hit,
                failure_reason=cached.failure_reason,
                extra_metrics=dict(cached.extra_metrics),
            )
            evaluations.append(item)
            return item
        started = time.perf_counter()
        try:
            raw = evaluator(dict(projected))
            if isinstance(raw, Mapping):
                metric = float(
                    raw.get(spec.metric_name, raw.get("metric_value", raw.get("metric")))
                )
                extra = {str(k): v for k, v in raw.items() if k != spec.metric_name}
            else:
                metric = float(raw)
                extra = {}
            item = TuningEvaluation(
                params=dict(projected),
                phase=phase,
                index=index,
                metric_value=metric,
                status="ok",
                elapsed_seconds=time.perf_counter() - started,
                boundary_hit=_any_boundary_hit(spec.parameters, projected),
                extra_metrics=extra,
            )
        except Exception as exc:  # noqa: BLE001 - failed candidates are tuning artifacts.
            item = TuningEvaluation(
                params=dict(projected),
                phase=phase,
                index=index,
                metric_value=math.inf if spec.goal == "minimize" else -math.inf,
                status="failed",
                elapsed_seconds=time.perf_counter() - started,
                boundary_hit=_any_boundary_hit(spec.parameters, projected),
                failure_reason=f"{type(exc).__name__}: {exc}",
            )
            failures.append(item)
        evaluations.append(item)
        cache[key] = item
        return item

    for index, candidate in enumerate(plan["candidates"]):
        evaluate(candidate, "initial", index)

    ok = [item for item in evaluations if item.status == "ok" and math.isfinite(item.metric_value)]
    if not ok:
        return TuningResult({}, math.inf, evaluations, failures, plan, _policy_from_spec(spec))

    best = ok[select_best_evaluation(ok, goal=spec.goal, tie_breakers=spec.selection_tie_breakers)]
    if spec.refinement_strategy == "nelder_mead_like" and spec.refinement_budget > 0:
        numeric_params = [
            parameter for parameter in spec.parameters if parameter.bounds is not None
        ]
        if len(numeric_params) == len(spec.parameters):
            lower = [
                float(parameter.bounds[0])
                for parameter in spec.parameters
                if parameter.bounds is not None
            ]
            upper = [
                float(parameter.bounds[1])
                for parameter in spec.parameters
                if parameter.bounds is not None
            ]

            def _evaluate_point(point: np.ndarray) -> float:
                params = {
                    parameter.name: parameter.project(value)
                    for parameter, value in zip(spec.parameters, point, strict=True)
                }
                return evaluate(params, "refinement", len(evaluations)).metric_value

            bounded_nelder_mead_search_points(
                lower_bounds=lower,
                upper_bounds=upper,
                evaluate_point=_evaluate_point,
                goal=spec.goal,
                max_iterations=spec.refinement_budget,
            )
            ok = [
                item
                for item in evaluations
                if item.status == "ok" and math.isfinite(item.metric_value)
            ]
            best = ok[
                select_best_evaluation(ok, goal=spec.goal, tie_breakers=spec.selection_tie_breakers)
            ]

    return TuningResult(
        selected_params=dict(best.params),
        selected_metric=float(best.metric_value),
        evaluations=evaluations,
        failures=[item for item in evaluations if item.status != "ok" or item.boundary_hit],
        candidate_plan=plan,
        policy=_policy_from_spec(spec),
    )


def select_best_evaluation(
    evaluations: Sequence[TuningEvaluation],
    *,
    goal: str = "minimize",
    tie_breakers: Sequence[str] = ("metric", "param_l1", "candidate_index"),
) -> int:
    if not evaluations:
        raise ValueError("evaluations must be non-empty")
    if goal not in {"minimize", "maximize"}:
        raise ValueError("goal must be minimize or maximize")

    def key(index: int, item: TuningEvaluation) -> tuple[float, ...]:
        primary = item.metric_value if goal == "minimize" else -item.metric_value
        parts = [primary]
        for tie_breaker in tie_breakers:
            if tie_breaker in {"metric", "mean_metric"}:
                continue
            if tie_breaker == "std_metric":
                parts.append(float(item.extra_metrics.get("std_metric", 0.0)))
            elif tie_breaker == "param_l1":
                parts.append(_param_l1_score(item.params))
            elif tie_breaker in {"candidate_index", "combo_index"}:
                parts.append(float(item.index))
            else:
                raise ValueError(f"unsupported tie breaker: {tie_breaker}")
        return tuple(parts)

    return min(range(len(evaluations)), key=lambda index: key(index, evaluations[index]))


def nelder_mead_like_search_indices(
    combos: Sequence[dict[str, Any]],
    *,
    evaluate_index: Callable[[int], float],
    goal: str = "minimize",
    max_iterations: int | None = None,
    reflection: float = 1.0,
    expansion: float = 2.0,
    contraction: float = 0.5,
    shrink: float = 0.5,
) -> list[int]:
    if goal not in {"minimize", "maximize"}:
        raise ValueError("goal must be either 'minimize' or 'maximize'")
    if max_iterations is not None and max_iterations <= 0:
        raise ValueError("max_iterations must be a positive integer when provided")
    if not combos:
        return []

    evaluated_order: list[int] = []
    score_cache: dict[int, float] = {}

    def objective(metric: float) -> float:
        return metric if goal == "minimize" else -metric

    def ensure_score(index: int) -> float:
        if index not in score_cache:
            score_cache[index] = objective(float(evaluate_index(index)))
            evaluated_order.append(index)
        return score_cache[index]

    vectors = _combo_numeric_matrix(combos)
    if vectors is None:
        for index in range(len(combos)):
            ensure_score(index)
        return evaluated_order
    vectors = _normalize_vectors(vectors)

    simplex_size = min(vectors.shape[1] + 1, len(combos))
    simplex = _initial_simplex_indices(vectors, simplex_size=simplex_size)
    for index in simplex:
        ensure_score(index)
    if len(score_cache) == len(combos):
        return evaluated_order

    iteration_budget = max_iterations if max_iterations is not None else len(combos)
    for _ in range(iteration_budget):
        simplex = sorted(simplex, key=ensure_score)
        best = simplex[0]
        worst = simplex[-1]
        second_worst = simplex[-2] if len(simplex) > 1 else worst
        centroid = (
            np.mean(vectors[simplex[:-1]], axis=0) if len(simplex) > 1 else vectors[best].copy()
        )
        best_score = ensure_score(best)
        worst_score = ensure_score(worst)
        second_worst_score = ensure_score(second_worst)
        reflected = centroid + reflection * (centroid - vectors[worst])
        reflected_idx = _nearest_candidate_index(reflected, vectors, excluded=set(score_cache))
        if reflected_idx is None:
            break
        reflected_score = ensure_score(reflected_idx)
        if reflected_score < best_score:
            expanded = centroid + expansion * (vectors[reflected_idx] - centroid)
            expanded_idx = _nearest_candidate_index(expanded, vectors, excluded=set(score_cache))
            if expanded_idx is not None:
                expanded_score = ensure_score(expanded_idx)
                simplex[-1] = expanded_idx if expanded_score < reflected_score else reflected_idx
            else:
                simplex[-1] = reflected_idx
            if len(score_cache) == len(combos):
                break
            continue
        if reflected_score < second_worst_score:
            simplex[-1] = reflected_idx
            if len(score_cache) == len(combos):
                break
            continue
        contracted = (
            centroid + contraction * (vectors[reflected_idx] - centroid)
            if reflected_score < worst_score
            else centroid + contraction * (vectors[worst] - centroid)
        )
        contracted_idx = _nearest_candidate_index(contracted, vectors, excluded=set(score_cache))
        if contracted_idx is not None:
            contracted_score = ensure_score(contracted_idx)
            if contracted_score < min(worst_score, reflected_score):
                simplex[-1] = contracted_idx
                if len(score_cache) == len(combos):
                    break
                continue
        shrunk_simplex = [best]
        for vertex in simplex[1:]:
            shrink_target = vectors[best] + shrink * (vectors[vertex] - vectors[best])
            shrink_idx = _nearest_candidate_index(shrink_target, vectors, excluded=set(score_cache))
            if shrink_idx is not None:
                ensure_score(shrink_idx)
                shrunk_simplex.append(shrink_idx)
        if len(shrunk_simplex) <= 1:
            break
        simplex = shrunk_simplex[:simplex_size]
        if len(score_cache) == len(combos):
            break
    return evaluated_order


def bounded_nelder_mead_search_points(
    *,
    lower_bounds: Sequence[float],
    upper_bounds: Sequence[float],
    evaluate_point: Callable[[np.ndarray], float],
    goal: str = "minimize",
    max_iterations: int | None = None,
    reflection: float = 1.0,
    expansion: float = 2.0,
    contraction: float = 0.5,
    shrink: float = 0.5,
) -> list[np.ndarray]:
    if goal not in {"minimize", "maximize"}:
        raise ValueError("goal must be either 'minimize' or 'maximize'")
    if max_iterations is not None and max_iterations <= 0:
        raise ValueError("max_iterations must be a positive integer when provided")
    lower = np.asarray(lower_bounds, dtype=float)
    upper = np.asarray(upper_bounds, dtype=float)
    if lower.ndim != 1 or upper.ndim != 1 or lower.shape != upper.shape:
        raise ValueError("lower_bounds and upper_bounds must be 1D arrays with matching shapes")
    if lower.size == 0:
        return []
    if np.any(lower >= upper):
        raise ValueError(
            "each lower bound must be strictly less than the corresponding upper bound"
        )
    span = upper - lower
    evaluated_points: list[np.ndarray] = []
    score_cache: dict[tuple[float, ...], float] = {}

    def objective(metric: float) -> float:
        return metric if goal == "minimize" else -metric

    def clip_unit(point: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(point, dtype=float), 0.0, 1.0)

    def denormalize(unit_point: np.ndarray) -> np.ndarray:
        return lower + clip_unit(unit_point) * span

    def ensure_score(unit_point: np.ndarray) -> float:
        key = tuple(float(value) for value in np.round(clip_unit(unit_point), decimals=12))
        if key not in score_cache:
            point = denormalize(np.asarray(key, dtype=float))
            score_cache[key] = objective(float(evaluate_point(point.copy())))
            evaluated_points.append(point)
        return score_cache[key]

    dim = lower.size
    simplex = _initial_bounded_simplex(dim)
    for vertex in simplex:
        ensure_score(vertex)
    iteration_budget = max_iterations if max_iterations is not None else max(20, 8 * dim)
    for _ in range(iteration_budget):
        simplex = sorted(simplex, key=ensure_score)
        best = simplex[0]
        worst = simplex[-1]
        second_worst = simplex[-2] if len(simplex) > 1 else worst
        centroid = np.mean(simplex[:-1], axis=0) if len(simplex) > 1 else best.copy()
        best_score = ensure_score(best)
        worst_score = ensure_score(worst)
        second_worst_score = ensure_score(second_worst)
        reflected = clip_unit(centroid + reflection * (centroid - worst))
        reflected_score = ensure_score(reflected)
        if reflected_score < best_score:
            expanded = clip_unit(centroid + expansion * (reflected - centroid))
            expanded_score = ensure_score(expanded)
            simplex[-1] = expanded if expanded_score < reflected_score else reflected
            continue
        if reflected_score < second_worst_score:
            simplex[-1] = reflected
            continue
        contracted = (
            clip_unit(centroid + contraction * (reflected - centroid))
            if reflected_score < worst_score
            else clip_unit(centroid + contraction * (worst - centroid))
        )
        contracted_score = ensure_score(contracted)
        if contracted_score < min(worst_score, reflected_score):
            simplex[-1] = contracted
            continue
        simplex = [best] + [clip_unit(best + shrink * (vertex - best)) for vertex in simplex[1:]]
    return evaluated_points


def write_tuning_artifacts(
    result: TuningResult, output_dir: str | Path, *, plot: bool = True
) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    result_payload = {
        "selected_params": result.selected_params,
        "selected_metric": result.selected_metric,
        "candidate_plan": result.candidate_plan,
        "policy": result.policy,
    }
    (root / "tuning_result.json").write_text(
        json.dumps(_jsonable(result_payload), indent=2) + "\n",
        encoding="utf-8",
    )
    _write_evaluations_csv(root / "tuning_evaluations.csv", result.evaluations)
    if result.failures:
        _write_evaluations_csv(root / "tuning_failures.csv", result.failures)
    if plot:
        try:
            plot_tuning_search(result, root / "tuning_search.png")
        except Exception as exc:  # noqa: BLE001 - plotting must not corrupt tuning artifacts.
            (root / "tuning_plot_error.txt").write_text(
                f"{type(exc).__name__}: {exc}\n",
                encoding="utf-8",
            )


def plot_tuning_search(result: TuningResult, output_path: str | Path) -> Path | None:
    """Plot standalone search history using the same visual conventions as CV plots."""
    ok = [
        item
        for item in result.evaluations
        if item.status == "ok" and math.isfinite(item.metric_value)
    ]
    if not ok:
        return None
    names = _plot_parameter_names(result, ok)
    if not names:
        return None

    from dymad.utils.plot import plot_search_results

    metric_values = np.asarray([item.metric_value for item in ok], dtype=float)
    selected_index = _selected_evaluation_index(result, ok)
    path = Path(output_path)
    value_scale = "log" if np.all(metric_values > 0.0) else "linear"
    if len(names) == 1:
        params, _ = _plot_axis_values(ok, names[0])
        mode = "1d"
    elif len(names) == 2:
        x_values, _ = _plot_axis_values(ok, names[0])
        y_values, _ = _plot_axis_values(ok, names[1])
        params = np.column_stack((x_values, y_values))
        mode = "2d"
    else:
        params = np.arange(len(ok), dtype=float)
        mode = "history"
    plot_search_results(
        params,
        metric_values,
        key_labels=names if mode != "history" else ["Evaluation Index"],
        metric_name="evaluation metric",
        best_idx=selected_index,
        mode=mode,
        title="Hyperparameter Search",
        output_path=path,
        ifclose=True,
        value_scale=value_scale,
    )
    return path


def _write_evaluations_csv(path: Path, evaluations: Sequence[TuningEvaluation]) -> None:
    fieldnames = [
        "phase",
        "index",
        "status",
        "metric_value",
        "elapsed_seconds",
        "cache_hit",
        "boundary_hit",
        "failure_reason",
        "params",
        "extra_metrics",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in evaluations:
            writer.writerow(
                {
                    "phase": item.phase,
                    "index": item.index,
                    "status": item.status,
                    "metric_value": _format_float(item.metric_value),
                    "elapsed_seconds": _format_float(item.elapsed_seconds),
                    "cache_hit": str(item.cache_hit).lower(),
                    "boundary_hit": str(item.boundary_hit).lower(),
                    "failure_reason": item.failure_reason or "",
                    "params": json.dumps(_jsonable(item.params), sort_keys=True),
                    "extra_metrics": json.dumps(_jsonable(item.extra_metrics), sort_keys=True),
                }
            )


def _plot_parameter_names(
    result: TuningResult, evaluations: Sequence[TuningEvaluation]
) -> list[str]:
    names: list[str] = []
    for domain in result.candidate_plan.get("parameter_domains", []):
        if isinstance(domain, Mapping) and isinstance(domain.get("name"), str):
            names.append(str(domain["name"]))
    for item in evaluations:
        for name in item.params:
            if name not in names:
                names.append(name)
    return names


def _plot_axis_values(
    evaluations: Sequence[TuningEvaluation], parameter_name: str
) -> tuple[np.ndarray, list[str] | None]:
    raw_values = [item.params.get(parameter_name) for item in evaluations]
    try:
        numeric_values = []
        for value in raw_values:
            if value is None:
                raise TypeError("missing parameter value")
            numeric_values.append(float(value))
        return np.asarray(numeric_values, dtype=float), None
    except (TypeError, ValueError):
        labels = sorted({str(value) for value in raw_values})
        positions = {label: float(index) for index, label in enumerate(labels)}
        return np.asarray([positions[str(value)] for value in raw_values], dtype=float), labels


def _selected_evaluation_index(
    result: TuningResult, evaluations: Sequence[TuningEvaluation]
) -> int:
    selected_key = _params_key(result.selected_params)
    for index, item in enumerate(evaluations):
        if _params_key(item.params) == selected_key:
            return index
    return min(
        range(len(evaluations)),
        key=lambda index: abs(evaluations[index].metric_value - result.selected_metric),
    )


def _grid_candidates(parameters: Sequence[ParameterSpec], budget: int) -> list[dict[str, Any]]:
    counts = _grid_counts(parameters, budget)
    axes = [
        _values_for_parameter(parameter, count)
        for parameter, count in zip(parameters, counts, strict=True)
    ]
    candidates = [
        {parameter.name: value for parameter, value in zip(parameters, values, strict=True)}
        for values in product(*axes)
    ]
    return candidates[:budget]


def _grid_counts(parameters: Sequence[ParameterSpec], budget: int) -> list[int]:
    counts = [1] * len(parameters)
    limits = [_axis_limit(parameter, budget) for parameter in parameters]
    while math.prod(counts) < budget:
        expandable = [index for index, count in enumerate(counts) if count < limits[index]]
        if not expandable:
            break
        axis = min(expandable, key=lambda index: counts[index])
        counts[axis] += 1
    return counts


def _values_for_parameter(parameter: ParameterSpec, count: int) -> list[Any]:
    if parameter.values is not None:
        return list(parameter.values[:count])
    assert parameter.bounds is not None
    if parameter.value_kind == "int":
        return _integer_values_for_parameter(parameter, count)
    if parameter.step is not None:
        return _stepped_float_values_for_parameter(parameter, count)
    lower, upper = parameter.bounds
    if count == 1:
        midpoint = (
            (lower + upper) / 2.0 if parameter.scale == "linear" else math.sqrt(lower * upper)
        )
        return [parameter.project(midpoint)]
    if parameter.scale == "log":
        log_lower = math.log(lower)
        log_upper = math.log(upper)
        return [
            parameter.project(math.exp(log_lower + (log_upper - log_lower) * index / (count - 1)))
            for index in range(count)
        ]
    return [
        parameter.project(lower + (upper - lower) * index / (count - 1)) for index in range(count)
    ]


def _random_candidates(
    parameters: Sequence[ParameterSpec], budget: int, seed: int
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    candidates = []
    for _ in range(budget):
        item = {}
        for parameter in parameters:
            if parameter.values is not None:
                item[parameter.name] = rng.choice(parameter.values)
                continue
            assert parameter.bounds is not None
            lower, upper = parameter.bounds
            value = (
                math.exp(rng.uniform(math.log(lower), math.log(upper)))
                if parameter.scale == "log"
                else rng.uniform(lower, upper)
            )
            item[parameter.name] = parameter.project(value)
        candidates.append(item)
    return candidates


def _axis_limit(parameter: ParameterSpec, budget: int) -> int:
    if parameter.values is not None:
        return len(parameter.values)
    if parameter.value_kind == "int":
        return min(_integer_value_count(parameter), budget)
    if parameter.step is not None:
        return min(_stepped_float_value_count(parameter), budget)
    return budget


def _integer_values_for_parameter(parameter: ParameterSpec, count: int) -> list[int]:
    total = _integer_value_count(parameter)
    indexes = _selected_indexes(total, min(count, total), scale=parameter.scale)
    return [_integer_value_at_index(parameter, index) for index in indexes]


def _stepped_float_values_for_parameter(parameter: ParameterSpec, count: int) -> list[float]:
    total = _stepped_float_value_count(parameter)
    indexes = _selected_indexes(total, min(count, total), scale=parameter.scale)
    return [_clean_float(_stepped_float_value_at_index(parameter, index)) for index in indexes]


def _selected_indexes(total: int, count: int, *, scale: str) -> list[int]:
    if count <= 1:
        return [total // 2]
    if scale == "log" and total > 1:
        raw = [round((total - 1) ** (index / (count - 1))) for index in range(count)]
    else:
        raw = [round((total - 1) * index / (count - 1)) for index in range(count)]
    indexes: list[int] = []
    for index in raw:
        clipped = min(max(int(index), 0), total - 1)
        if clipped not in indexes:
            indexes.append(clipped)
    candidate = 0
    while len(indexes) < count and candidate < total:
        if candidate not in indexes:
            indexes.append(candidate)
        candidate += 1
    return sorted(indexes)


def _integer_value_count(parameter: ParameterSpec) -> int:
    assert parameter.bounds is not None
    lower, upper = parameter.bounds
    anchor = math.ceil(lower)
    upper_int = math.floor(upper)
    if anchor > upper_int:
        return 0
    step = int(parameter.step) if parameter.step is not None else 1
    count = ((upper_int - anchor) // step) + 1
    if parameter.parity is None:
        return count
    if step % 2 == 0:
        return count if _matches_parity(anchor, parameter.parity) else 0
    return (count + 1) // 2 if _matches_parity(anchor, parameter.parity) else count // 2


def _integer_value_at_index(parameter: ParameterSpec, index: int) -> int:
    assert parameter.bounds is not None
    lower, upper = parameter.bounds
    anchor = math.ceil(lower)
    upper_int = math.floor(upper)
    step = int(parameter.step) if parameter.step is not None else 1
    if parameter.parity is None or step % 2 == 0:
        value = anchor + index * step
    elif _matches_parity(anchor, parameter.parity):
        value = anchor + 2 * index * step
    else:
        value = anchor + (2 * index + 1) * step
    if value > upper_int or (
        parameter.parity is not None and not _matches_parity(value, parameter.parity)
    ):
        raise ValueError(f"{parameter.name}: integer constraints admit no value at index {index}")
    return value


def _stepped_float_value_count(parameter: ParameterSpec) -> int:
    assert parameter.bounds is not None
    assert parameter.step is not None
    return max(0, math.floor((parameter.bounds[1] - parameter.bounds[0]) / parameter.step) + 1)


def _stepped_float_value_at_index(parameter: ParameterSpec, index: int) -> float:
    assert parameter.bounds is not None
    assert parameter.step is not None
    return min(parameter.bounds[0] + index * parameter.step, parameter.bounds[1])


def _project_integer(parameter: ParameterSpec, value: float) -> int:
    total = _integer_value_count(parameter)
    if total <= 0:
        raise ValueError(f"{parameter.name}: integer constraints admit no values")
    assert parameter.bounds is not None
    anchor = math.ceil(parameter.bounds[0])
    upper = math.floor(parameter.bounds[1])
    step = int(parameter.step) if parameter.step is not None else 1
    nearest = anchor + round((round(value) - anchor) / step) * step
    candidates = []
    for offset in range(-8, 9):
        candidate = nearest + offset * step
        if anchor <= candidate <= upper and (
            parameter.parity is None or _matches_parity(candidate, parameter.parity)
        ):
            candidates.append(candidate)
    if not candidates:
        candidates = [
            _integer_value_at_index(parameter, 0),
            _integer_value_at_index(parameter, total - 1),
        ]
    return min(candidates, key=lambda candidate: (abs(candidate - value), candidate))


def _matches_parity(value: int, parity: str) -> bool:
    return value % 2 == (0 if parity == "even" else 1)


def _clean_float(value: float) -> float:
    return round(float(value), 12)


def _params_key(params: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
    return tuple((key, params[key]) for key in sorted(params))


def _any_boundary_hit(parameters: Sequence[ParameterSpec], params: Mapping[str, Any]) -> bool:
    for parameter in parameters:
        if parameter.bounds is None:
            continue
        value = float(params[parameter.name])
        if math.isclose(value, parameter.bounds[0], rel_tol=1e-10, abs_tol=1e-12):
            return True
        if math.isclose(value, parameter.bounds[1], rel_tol=1e-10, abs_tol=1e-12):
            return True
    return False


def _policy_from_spec(spec: TuningSpec) -> dict[str, Any]:
    return {
        "metric_name": spec.metric_name,
        "goal": spec.goal,
        "initial_budget": spec.initial_budget,
        "initial_strategy": spec.initial_strategy,
        "refinement_strategy": spec.refinement_strategy,
        "refinement_budget": spec.refinement_budget,
        "selection_tie_breakers": list(spec.selection_tie_breakers),
        **spec.metadata,
    }


def _param_l1_score(params: Mapping[str, Any]) -> float:
    score = 0.0
    for key in sorted(params):
        value = params[key]
        if isinstance(value, bool):
            score += float(value)
        elif isinstance(value, (int, float, np.integer, np.floating)):
            score += abs(float(value))
    return score


def _combo_numeric_matrix(combos: Sequence[dict[str, Any]]) -> np.ndarray | None:
    if not combos:
        return np.zeros((0, 0), dtype=float)
    key_order = tuple(sorted(combos[0]))
    rows: list[list[float]] = []
    for combo in combos:
        if tuple(sorted(combo)) != key_order:
            return None
        row = []
        for key in key_order:
            value = combo[key]
            if isinstance(value, bool):
                row.append(float(value))
            elif isinstance(value, (int, float, np.integer, np.floating)):
                row.append(float(value))
            else:
                return None
        rows.append(row)
    return np.asarray(rows, dtype=float)


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    if vectors.size == 0:
        return vectors.copy()
    mins = np.min(vectors, axis=0)
    spans = np.ptp(vectors, axis=0)
    spans = np.where(spans == 0.0, 1.0, spans)
    return (vectors - mins) / spans


def _initial_simplex_indices(vectors: np.ndarray, *, simplex_size: int) -> list[int]:
    n_candidates = vectors.shape[0]
    if simplex_size >= n_candidates:
        return list(range(n_candidates))
    selected = [0]
    while len(selected) < simplex_size:
        best_idx = None
        best_distance = -1.0
        for index in range(n_candidates):
            if index in selected:
                continue
            distance = min(
                float(np.linalg.norm(vectors[index] - vectors[current])) for current in selected
            )
            if distance > best_distance:
                best_distance = distance
                best_idx = index
        if best_idx is None:
            break
        selected.append(best_idx)
    return selected


def _nearest_candidate_index(
    target: np.ndarray, vectors: np.ndarray, *, excluded: set[int]
) -> int | None:
    best_idx = None
    best_distance = float("inf")
    for index in range(vectors.shape[0]):
        if index in excluded:
            continue
        distance = float(np.linalg.norm(vectors[index] - target))
        if distance < best_distance:
            best_distance = distance
            best_idx = index
    return best_idx


def _initial_bounded_simplex(dim: int) -> list[np.ndarray]:
    base = np.full(dim, 0.5, dtype=float)
    simplex = [base]
    for axis in range(dim):
        vertex = base.copy()
        vertex[axis] = 1.0
        simplex.append(vertex)
    return simplex


def _format_float(value: float) -> str:
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    if math.isnan(value):
        return "nan"
    return f"{value:.12g}"


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, TuningEvaluation):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
