from __future__ import annotations

import csv
import json
import math
import random
import time
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from itertools import product
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
from scipy.stats import qmc

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
    initial_budget: int | tuple[int, ...] = 1
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
        _validate_initial_budget(self.initial_budget, len(self.parameters))
        if self.initial_strategy not in {"auto", "grid", "random"}:
            raise ValueError("TuningSpec.initial_strategy must be auto, grid, or random")
        if self.refinement_strategy not in {
            None,
            "nelder_mead_like",
            "batch_pattern_search",
            "multi_start_nelder_mead",
        }:
            raise ValueError(
                "TuningSpec.refinement_strategy must be None, nelder_mead_like, "
                "batch_pattern_search, or multi_start_nelder_mead"
            )
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
    budget_total = _initial_budget_total(spec.initial_budget)
    candidates = (
        _grid_candidates(spec.parameters, spec.initial_budget)
        if strategy == "grid"
        else _random_candidates(spec.parameters, budget_total, spec.seed)
    )
    return {
        "strategy": strategy,
        "initial_budget": spec.initial_budget,
        "initial_budget_mode": "per_parameter"
        if isinstance(spec.initial_budget, tuple)
        else "total",
        "candidate_count": len(candidates),
        "parameter_domains": [parameter.domain_summary() for parameter in spec.parameters],
        "candidates": candidates,
        "refinement": {
            "strategy": spec.refinement_strategy,
            "budget": spec.refinement_budget,
        },
    }


def tune(spec: TuningSpec, evaluator: MetricEvaluator, *, max_workers: int = 1) -> TuningResult:
    if max_workers <= 0:
        raise ValueError("max_workers must be positive")
    plan = initial_search_plan(spec)
    evaluations: list[TuningEvaluation] = []
    cache: dict[tuple[tuple[str, Any], ...], TuningEvaluation] = {}

    def project(params: dict[str, Any]) -> dict[str, Any]:
        return {
            parameter.name: parameter.project(params[parameter.name])
            for parameter in spec.parameters
        }

    def run_projected(projected: dict[str, Any], phase: str, index: int) -> TuningEvaluation:
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
            return TuningEvaluation(
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
            return TuningEvaluation(
                params=dict(projected),
                phase=phase,
                index=index,
                metric_value=math.inf if spec.goal == "minimize" else -math.inf,
                status="failed",
                elapsed_seconds=time.perf_counter() - started,
                boundary_hit=_any_boundary_hit(spec.parameters, projected),
                failure_reason=f"{type(exc).__name__}: {exc}",
            )

    def evaluate(params: dict[str, Any], phase: str, index: int) -> TuningEvaluation:
        projected = project(params)
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
        item = run_projected(projected, phase, index)
        evaluations.append(item)
        cache[key] = item
        return item

    if max_workers > 1 and len(plan["candidates"]) > 1:
        _evaluate_initial_candidates_parallel(
            candidates=plan["candidates"],
            phase="initial",
            project=project,
            run_projected=run_projected,
            evaluations=evaluations,
            cache=cache,
            max_workers=max_workers,
        )
    else:
        for index, candidate in enumerate(plan["candidates"]):
            evaluate(candidate, "initial", index)

    ok = [item for item in evaluations if item.status == "ok" and math.isfinite(item.metric_value)]
    if not ok:
        return TuningResult(
            {},
            math.inf,
            evaluations,
            [item for item in evaluations if item.status != "ok" or item.boundary_hit],
            plan,
            _policy_from_spec(spec),
        )

    best = ok[select_best_evaluation(ok, goal=spec.goal, tie_breakers=spec.selection_tie_breakers)]
    if spec.refinement_strategy == "nelder_mead_like" and spec.refinement_budget > 0:
        if max_workers > 1:
            warnings.warn(
                "refinement_strategy='nelder_mead_like' is sequential; use "
                "refinement_strategy='batch_pattern_search' to use parallel workers.",
                RuntimeWarning,
                stacklevel=2,
            )
        numeric_params = [
            parameter for parameter in spec.parameters if parameter.bounds is not None
        ]
        if len(numeric_params) == len(spec.parameters):
            lower = [
                _parameter_search_lower_bound(parameter)
                for parameter in spec.parameters
                if parameter.bounds is not None
            ]
            upper = [
                _parameter_search_upper_bound(parameter)
                for parameter in spec.parameters
                if parameter.bounds is not None
            ]

            def _evaluate_point(point: np.ndarray) -> float:
                params = {
                    parameter.name: parameter.project(
                        _parameter_value_from_search_coordinate(parameter, value)
                    )
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
    elif spec.refinement_strategy == "batch_pattern_search" and spec.refinement_budget > 0:
        if max_workers == 1:
            warnings.warn(
                "refinement_strategy='batch_pattern_search' is intended for max_workers > 1; "
                "with max_workers=1 it runs as sequential batched pattern search.",
                RuntimeWarning,
                stacklevel=2,
            )
        numeric_params = [
            parameter for parameter in spec.parameters if parameter.bounds is not None
        ]
        if len(numeric_params) == len(spec.parameters):
            lower = [
                _parameter_search_lower_bound(parameter)
                for parameter in spec.parameters
                if parameter.bounds is not None
            ]
            upper = [
                _parameter_search_upper_bound(parameter)
                for parameter in spec.parameters
                if parameter.bounds is not None
            ]
            ranked = sorted(
                ok,
                key=lambda item: (
                    _objective_score(item.metric_value, goal=spec.goal),
                    item.index,
                ),
            )
            initial_points = [
                [
                    _parameter_search_coordinate_from_value(parameter, item.params[parameter.name])
                    for parameter in spec.parameters
                ]
                for item in ranked[:1]
            ]
            initial_step = _batch_pattern_initial_step(spec, plan)

            def _evaluate_points(points: Sequence[np.ndarray]) -> list[float]:
                params_by_point = [
                    {
                        parameter.name: parameter.project(
                            _parameter_value_from_search_coordinate(parameter, value)
                        )
                        for parameter, value in zip(spec.parameters, point, strict=True)
                    }
                    for point in points
                ]
                if max_workers > 1 and len(params_by_point) > 1:
                    items = _evaluate_projected_candidates_parallel(
                        projected_candidates=params_by_point,
                        phase="refinement",
                        start_index=len(evaluations),
                        run_projected=run_projected,
                        evaluations=evaluations,
                        cache=cache,
                        max_workers=max_workers,
                    )
                    return [item.metric_value for item in items]
                return [
                    evaluate(params, "refinement", len(evaluations)).metric_value
                    for params in params_by_point
                ]

            batch_pattern_search_points(
                lower_bounds=lower,
                upper_bounds=upper,
                evaluate_points=_evaluate_points,
                goal=spec.goal,
                max_evaluations=spec.refinement_budget,
                batch_size=max_workers,
                initial_points=initial_points,
                initial_step=initial_step,
            )
            ok = [
                item
                for item in evaluations
                if item.status == "ok" and math.isfinite(item.metric_value)
            ]
            best = ok[
                select_best_evaluation(ok, goal=spec.goal, tie_breakers=spec.selection_tie_breakers)
            ]
    elif spec.refinement_strategy == "multi_start_nelder_mead" and spec.refinement_budget > 0:
        numeric_params = [
            parameter for parameter in spec.parameters if parameter.bounds is not None
        ]
        if len(numeric_params) == len(spec.parameters):
            lower = [
                _parameter_search_lower_bound(parameter)
                for parameter in spec.parameters
                if parameter.bounds is not None
            ]
            upper = [
                _parameter_search_upper_bound(parameter)
                for parameter in spec.parameters
                if parameter.bounds is not None
            ]
            evaluation_lock = Lock()
            next_refinement_index = len(evaluations)

            def _evaluate_point(point: np.ndarray) -> float:
                nonlocal next_refinement_index
                params = {
                    parameter.name: parameter.project(
                        _parameter_value_from_search_coordinate(parameter, value)
                    )
                    for parameter, value in zip(spec.parameters, point, strict=True)
                }
                projected = project(params)
                key = _params_key(projected)
                with evaluation_lock:
                    index = next_refinement_index
                    next_refinement_index += 1
                    cached = cache.get(key)
                    if cached is not None:
                        item = TuningEvaluation(
                            params=dict(cached.params),
                            phase="refinement",
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
                        return item.metric_value
                item = run_projected(projected, "refinement", index)
                with evaluation_lock:
                    evaluations.append(item)
                    cache.setdefault(key, item)
                return item.metric_value

            multi_start_bounded_nelder_mead_search_points(
                lower_bounds=lower,
                upper_bounds=upper,
                evaluate_point=_evaluate_point,
                goal=spec.goal,
                max_iterations=spec.refinement_budget,
                num_simplices=max_workers,
                max_workers=max_workers,
                seed=spec.seed,
            )
            evaluations.sort(key=lambda item: item.index)
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


def _evaluate_initial_candidates_parallel(
    *,
    candidates: Sequence[dict[str, Any]],
    phase: str,
    project: Callable[[dict[str, Any]], dict[str, Any]],
    run_projected: Callable[[dict[str, Any], str, int], TuningEvaluation],
    evaluations: list[TuningEvaluation],
    cache: dict[tuple[tuple[str, Any], ...], TuningEvaluation],
    max_workers: int,
) -> None:
    projected_by_index = [project(candidate) for candidate in candidates]
    first_index_by_key: dict[tuple[tuple[str, Any], ...], int] = {}
    for index, projected in enumerate(projected_by_index):
        first_index_by_key.setdefault(_params_key(projected), index)
    first_indexes = sorted(set(first_index_by_key.values()))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_by_index = {
            index: executor.submit(run_projected, projected_by_index[index], phase, index)
            for index in first_indexes
        }
        first_results = {index: future_by_index[index].result() for index in first_indexes}

    result_by_key = {
        _params_key(first_results[index].params): first_results[index] for index in first_indexes
    }
    for index, projected in enumerate(projected_by_index):
        key = _params_key(projected)
        first_result = result_by_key[key]
        if first_index_by_key[key] == index:
            item = first_result
        else:
            item = TuningEvaluation(
                params=dict(first_result.params),
                phase=phase,
                index=index,
                metric_value=first_result.metric_value,
                status=first_result.status,
                elapsed_seconds=0.0,
                cache_hit=True,
                boundary_hit=first_result.boundary_hit,
                failure_reason=first_result.failure_reason,
                extra_metrics=dict(first_result.extra_metrics),
            )
        evaluations.append(item)
        cache[key] = first_result


def _evaluate_projected_candidates_parallel(
    *,
    projected_candidates: Sequence[dict[str, Any]],
    phase: str,
    start_index: int,
    run_projected: Callable[[dict[str, Any], str, int], TuningEvaluation],
    evaluations: list[TuningEvaluation],
    cache: dict[tuple[tuple[str, Any], ...], TuningEvaluation],
    max_workers: int,
) -> list[TuningEvaluation]:
    first_index_by_key: dict[tuple[tuple[str, Any], ...], int] = {}
    for offset, projected in enumerate(projected_candidates):
        key = _params_key(projected)
        if key not in cache:
            first_index_by_key.setdefault(key, offset)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_by_offset = {
            offset: executor.submit(
                run_projected,
                projected_candidates[offset],
                phase,
                start_index + offset,
            )
            for offset in sorted(first_index_by_key.values())
        }
        first_results = {offset: future_by_offset[offset].result() for offset in future_by_offset}

    result_by_key = {
        _params_key(projected_candidates[offset]): first_results[offset] for offset in first_results
    }
    batch_items: list[TuningEvaluation] = []
    for offset, projected in enumerate(projected_candidates):
        key = _params_key(projected)
        if key in cache:
            source = cache[key]
            item = TuningEvaluation(
                params=dict(source.params),
                phase=phase,
                index=start_index + offset,
                metric_value=source.metric_value,
                status=source.status,
                elapsed_seconds=0.0,
                cache_hit=True,
                boundary_hit=source.boundary_hit,
                failure_reason=source.failure_reason,
                extra_metrics=dict(source.extra_metrics),
            )
        else:
            first_offset = first_index_by_key[key]
            source = result_by_key[key]
            if first_offset == offset:
                item = source
            else:
                item = TuningEvaluation(
                    params=dict(source.params),
                    phase=phase,
                    index=start_index + offset,
                    metric_value=source.metric_value,
                    status=source.status,
                    elapsed_seconds=0.0,
                    cache_hit=True,
                    boundary_hit=source.boundary_hit,
                    failure_reason=source.failure_reason,
                    extra_metrics=dict(source.extra_metrics),
                )
            cache[key] = source
        evaluations.append(item)
        batch_items.append(item)
    return batch_items


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


def batch_pattern_search_indices(
    combos: Sequence[dict[str, Any]],
    *,
    evaluate_indices: Callable[[Sequence[int]], Sequence[float]],
    goal: str = "minimize",
    max_evaluations: int | None = None,
    batch_size: int = 1,
    initial_step: float = 0.25,
    step_shrink: float = 0.5,
) -> list[int]:
    if goal not in {"minimize", "maximize"}:
        raise ValueError("goal must be either 'minimize' or 'maximize'")
    if max_evaluations is not None and max_evaluations <= 0:
        raise ValueError("max_evaluations must be a positive integer when provided")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if initial_step <= 0.0:
        raise ValueError("initial_step must be positive")
    if not 0.0 < step_shrink < 1.0:
        raise ValueError("step_shrink must be in (0, 1)")
    if not combos:
        return []

    vectors = _combo_numeric_matrix(combos)
    if vectors is None:
        budget = len(combos) if max_evaluations is None else min(max_evaluations, len(combos))
        return _evaluate_index_batch(
            list(range(budget)),
            evaluate_indices=evaluate_indices,
            score_cache={},
            evaluated_order=[],
            goal=goal,
        )

    vectors = _normalize_vectors(vectors)
    center = np.full(vectors.shape[1], 0.5, dtype=float)
    first_index = _nearest_candidate_index(center, vectors, excluded=set())
    if first_index is None:
        return []

    budget = max_evaluations if max_evaluations is not None else len(combos)
    budget = min(budget, len(combos))
    evaluated_order: list[int] = []
    score_cache: dict[int, float] = {}
    current_best = first_index
    step = initial_step

    while len(score_cache) < budget:
        pending: list[int] = []
        if current_best not in score_cache:
            pending.append(current_best)
        excluded = set(score_cache) | set(pending)
        for target in _pattern_search_targets(vectors[current_best], step=step):
            if len(score_cache) + len(pending) >= budget or len(pending) >= batch_size:
                break
            index = _nearest_candidate_index(target, vectors, excluded=excluded)
            if index is None:
                continue
            pending.append(index)
            excluded.add(index)
        if not pending:
            step *= step_shrink
            if step <= 1e-12:
                break
            continue
        previous_best = _best_index(score_cache, goal=goal) if score_cache else None
        _evaluate_index_batch(
            pending,
            evaluate_indices=evaluate_indices,
            score_cache=score_cache,
            evaluated_order=evaluated_order,
            goal=goal,
        )
        current_best = _best_index(score_cache, goal=goal)
        if (
            previous_best is not None
            and current_best == previous_best
            and not _has_new_pattern_index(
                vectors[current_best],
                vectors=vectors,
                step=step,
                score_cache=score_cache,
            )
        ):
            step *= step_shrink
    return evaluated_order


def batch_pattern_search_points(
    *,
    lower_bounds: Sequence[float],
    upper_bounds: Sequence[float],
    evaluate_points: Callable[[Sequence[np.ndarray]], Sequence[float]],
    goal: str = "minimize",
    max_evaluations: int | None = None,
    batch_size: int = 1,
    initial_points: Sequence[Sequence[float]] | None = None,
    initial_step: float = 0.25,
    step_shrink: float = 0.5,
) -> list[np.ndarray]:
    if goal not in {"minimize", "maximize"}:
        raise ValueError("goal must be either 'minimize' or 'maximize'")
    if max_evaluations is not None and max_evaluations <= 0:
        raise ValueError("max_evaluations must be a positive integer when provided")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if initial_step <= 0.0:
        raise ValueError("initial_step must be positive")
    if not 0.0 < step_shrink < 1.0:
        raise ValueError("step_shrink must be in (0, 1)")
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
    budget = max_evaluations if max_evaluations is not None else max(20, 8 * lower.size)
    unit_centers = _initial_pattern_centers(initial_points, lower=lower, span=span)
    evaluated_points: list[np.ndarray] = []
    score_cache: dict[tuple[float, ...], float] = {}
    current_centers = unit_centers[: max(1, batch_size)]
    step = initial_step

    def denormalize(unit_point: np.ndarray) -> np.ndarray:
        return lower + _clip_unit(unit_point) * span

    while len(score_cache) < budget:
        pending_keys: list[tuple[float, ...]] = []
        for center in current_centers:
            if len(score_cache) + len(pending_keys) >= budget or len(pending_keys) >= batch_size:
                break
            _append_pattern_candidate(
                pending_keys,
                _unit_key(center),
                score_cache=score_cache,
            )
        for center in current_centers:
            if len(score_cache) + len(pending_keys) >= budget or len(pending_keys) >= batch_size:
                break
            for target in _pattern_search_targets(center, step=step):
                if (
                    len(score_cache) + len(pending_keys) >= budget
                    or len(pending_keys) >= batch_size
                ):
                    break
                _append_pattern_candidate(
                    pending_keys,
                    _unit_key(target),
                    score_cache=score_cache,
                )
        if not pending_keys:
            step *= step_shrink
            if step <= 1e-12:
                break
            continue
        previous_best = _best_point_key(score_cache, goal=goal) if score_cache else None
        points = [denormalize(np.asarray(key, dtype=float)) for key in pending_keys]
        scores = evaluate_points([point.copy() for point in points])
        if len(scores) != len(points):
            raise ValueError("evaluate_points must return one score per input point")
        for key, point, metric in zip(pending_keys, points, scores, strict=True):
            score_cache[key] = _objective_score(float(metric), goal=goal)
            evaluated_points.append(point)
        best_keys = sorted(score_cache, key=lambda key: score_cache[key])
        current_centers = [np.asarray(best_keys[0], dtype=float)]
        current_best = best_keys[0]
        if (
            previous_best is not None
            and current_best == previous_best
            and not _has_new_pattern_point(
                np.asarray(current_best, dtype=float),
                step=step,
                score_cache=score_cache,
            )
        ):
            step *= step_shrink
    return evaluated_points


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
    iteration_budget = max_iterations if max_iterations is not None else max(20, 8 * lower.size)
    return _run_bounded_nelder_mead_unit_simplex(
        simplex=_initial_bounded_simplex(lower.size),
        lower=lower,
        span=span,
        evaluate_point=evaluate_point,
        goal=goal,
        max_iterations=iteration_budget,
        reflection=reflection,
        expansion=expansion,
        contraction=contraction,
        shrink=shrink,
    )


def multi_start_bounded_nelder_mead_search_points(
    *,
    lower_bounds: Sequence[float],
    upper_bounds: Sequence[float],
    evaluate_point: Callable[[np.ndarray], float],
    goal: str = "minimize",
    max_iterations: int | None = None,
    num_simplices: int = 1,
    max_workers: int = 1,
    seed: int = 0,
    simplex_scale: float = 0.2,
    reflection: float = 1.0,
    expansion: float = 2.0,
    contraction: float = 0.5,
    shrink: float = 0.5,
) -> list[np.ndarray]:
    if goal not in {"minimize", "maximize"}:
        raise ValueError("goal must be either 'minimize' or 'maximize'")
    if max_iterations is not None and max_iterations <= 0:
        raise ValueError("max_iterations must be a positive integer when provided")
    if num_simplices <= 0:
        raise ValueError("num_simplices must be positive")
    if max_workers <= 0:
        raise ValueError("max_workers must be positive")
    if simplex_scale <= 0.0:
        raise ValueError("simplex_scale must be positive")
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
    dim = lower.size
    total_iterations = (
        max(20, 8 * dim) * num_simplices if max_iterations is None else max_iterations
    )
    simplex_count = min(num_simplices, total_iterations)
    iteration_budgets = _split_iteration_budget(total_iterations, simplex_count)
    centers = _sobol_unit_centers(dim, simplex_count, seed=seed)
    simplices = [
        _initial_bounded_simplex_around_center(center, scale=simplex_scale) for center in centers
    ]

    def run_simplex(index: int) -> list[np.ndarray]:
        return _run_bounded_nelder_mead_unit_simplex(
            simplex=simplices[index],
            lower=lower,
            span=span,
            evaluate_point=evaluate_point,
            goal=goal,
            max_iterations=iteration_budgets[index],
            reflection=reflection,
            expansion=expansion,
            contraction=contraction,
            shrink=shrink,
        )

    if max_workers > 1 and len(simplices) > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            traces = list(executor.map(run_simplex, range(len(simplices))))
    else:
        traces = [run_simplex(index) for index in range(len(simplices))]
    return [point for trace in traces for point in trace]


def _run_bounded_nelder_mead_unit_simplex(
    *,
    simplex: Sequence[np.ndarray],
    lower: np.ndarray,
    span: np.ndarray,
    evaluate_point: Callable[[np.ndarray], float],
    goal: str,
    max_iterations: int,
    reflection: float,
    expansion: float,
    contraction: float,
    shrink: float,
) -> list[np.ndarray]:
    evaluated_points: list[np.ndarray] = []
    score_cache: dict[tuple[float, ...], float] = {}

    def denormalize(unit_point: np.ndarray) -> np.ndarray:
        return lower + _clip_unit(unit_point) * span

    def ensure_score(unit_point: np.ndarray) -> float:
        key = _unit_key(unit_point)
        if key not in score_cache:
            point = denormalize(np.asarray(key, dtype=float))
            score_cache[key] = _objective_score(float(evaluate_point(point.copy())), goal=goal)
            evaluated_points.append(point)
        return score_cache[key]

    current_simplex = [_clip_unit(vertex) for vertex in simplex]
    for vertex in current_simplex:
        ensure_score(vertex)
    for _ in range(max_iterations):
        current_simplex = sorted(current_simplex, key=ensure_score)
        best = current_simplex[0]
        worst = current_simplex[-1]
        second_worst = current_simplex[-2] if len(current_simplex) > 1 else worst
        centroid = (
            np.mean(current_simplex[:-1], axis=0) if len(current_simplex) > 1 else best.copy()
        )
        best_score = ensure_score(best)
        worst_score = ensure_score(worst)
        second_worst_score = ensure_score(second_worst)
        reflected = _clip_unit(centroid + reflection * (centroid - worst))
        reflected_score = ensure_score(reflected)
        if reflected_score < best_score:
            expanded = _clip_unit(centroid + expansion * (reflected - centroid))
            expanded_score = ensure_score(expanded)
            current_simplex[-1] = expanded if expanded_score < reflected_score else reflected
            continue
        if reflected_score < second_worst_score:
            current_simplex[-1] = reflected
            continue
        contracted = (
            _clip_unit(centroid + contraction * (reflected - centroid))
            if reflected_score < worst_score
            else _clip_unit(centroid + contraction * (worst - centroid))
        )
        contracted_score = ensure_score(contracted)
        if contracted_score < min(worst_score, reflected_score):
            current_simplex[-1] = contracted
            continue
        current_simplex = [
            best,
            *[_clip_unit(best + shrink * (vertex - best)) for vertex in current_simplex[1:]],
        ]
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


def read_tuning_artifacts(output_dir: str | Path) -> TuningResult:
    root = Path(output_dir)
    result_path = root / "tuning_result.json"
    evaluations_path = root / "tuning_evaluations.csv"
    if not result_path.is_file() or not evaluations_path.is_file():
        raise FileNotFoundError(f"incomplete tuning artifacts under {root}")

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    evaluations = _read_evaluations_csv(evaluations_path)
    failures_path = root / "tuning_failures.csv"
    failures = (
        _read_evaluations_csv(failures_path)
        if failures_path.is_file()
        else [item for item in evaluations if item.status != "ok" or item.boundary_hit]
    )
    return TuningResult(
        selected_params=dict(payload.get("selected_params", {})),
        selected_metric=float(payload.get("selected_metric", math.nan)),
        evaluations=evaluations,
        failures=failures,
        candidate_plan=dict(payload.get("candidate_plan", {})),
        policy=dict(payload.get("policy", {})),
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
        axis_scales=_plot_axis_scales(result, names if mode != "history" else []),
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


def _read_evaluations_csv(path: Path) -> list[TuningEvaluation]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    evaluations = []
    for row in rows:
        evaluations.append(
            TuningEvaluation(
                params=json.loads(row.get("params") or "{}"),
                phase=str(row.get("phase", "")),
                index=int(row.get("index") or 0),
                metric_value=_parse_float(row.get("metric_value", "nan")),
                status=str(row.get("status", "")),
                elapsed_seconds=_parse_float(row.get("elapsed_seconds", "0")),
                cache_hit=_parse_bool(row.get("cache_hit", "false")),
                boundary_hit=_parse_bool(row.get("boundary_hit", "false")),
                failure_reason=row.get("failure_reason") or None,
                extra_metrics=json.loads(row.get("extra_metrics") or "{}"),
            )
        )
    return evaluations


def _parse_bool(value: str | None) -> bool:
    return str(value).strip().lower() == "true"


def _parse_float(value: str | None) -> float:
    text = str(value).strip().lower()
    if text == "inf":
        return math.inf
    if text == "-inf":
        return -math.inf
    if text == "nan":
        return math.nan
    return float(text)


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


def _plot_axis_scales(result: TuningResult, parameter_names: Sequence[str]) -> list[str]:
    scale_by_name = {}
    for domain in result.candidate_plan.get("parameter_domains", []):
        if isinstance(domain, Mapping) and isinstance(domain.get("name"), str):
            scale_by_name[str(domain["name"])] = str(domain.get("scale", "linear"))
    return [scale_by_name.get(name, "linear") for name in parameter_names]


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


def _validate_initial_budget(budget: int | tuple[int, ...], parameter_count: int) -> None:
    if isinstance(budget, tuple):
        if len(budget) != parameter_count:
            raise ValueError(
                "TuningSpec.initial_budget tuple length must match number of parameters"
            )
        if any(int(value) <= 0 for value in budget):
            raise ValueError("TuningSpec.initial_budget entries must be positive")
        return
    if int(budget) <= 0:
        raise ValueError("TuningSpec.initial_budget must be positive")


def _initial_budget_total(budget: int | tuple[int, ...]) -> int:
    return math.prod(int(value) for value in budget) if isinstance(budget, tuple) else int(budget)


def _batch_pattern_initial_step(spec: TuningSpec, plan: Mapping[str, Any]) -> float:
    if plan.get("strategy") != "grid":
        return 0.25
    counts = _grid_counts(spec.parameters, spec.initial_budget)
    spacings = [1.0 / float(count - 1) for count in counts if count > 1]
    return min(spacings) if spacings else 0.25


def _grid_candidates(
    parameters: Sequence[ParameterSpec], budget: int | tuple[int, ...]
) -> list[dict[str, Any]]:
    counts = _grid_counts(parameters, budget)
    axes = [
        _values_for_parameter(parameter, count)
        for parameter, count in zip(parameters, counts, strict=True)
    ]
    candidates = [
        {parameter.name: value for parameter, value in zip(parameters, values, strict=True)}
        for values in product(*axes)
    ]
    return candidates if isinstance(budget, tuple) else candidates[: int(budget)]


def _grid_counts(parameters: Sequence[ParameterSpec], budget: int | tuple[int, ...]) -> list[int]:
    if isinstance(budget, tuple):
        return [
            min(int(count), _axis_limit(parameter, int(count)))
            for parameter, count in zip(parameters, budget, strict=True)
        ]
    budget = int(budget)
    counts = [1] * len(parameters)
    limits = [_axis_limit(parameter, budget) for parameter in parameters]
    while math.prod(counts) < budget:
        expandable = [index for index, count in enumerate(counts) if count < limits[index]]
        if not expandable:
            break
        axis = min(expandable, key=lambda index: counts[index])
        counts[axis] += 1
    return counts


def _parameter_search_lower_bound(parameter: ParameterSpec) -> float:
    assert parameter.bounds is not None
    lower = float(parameter.bounds[0])
    return math.log(lower) if parameter.scale == "log" else lower


def _parameter_search_upper_bound(parameter: ParameterSpec) -> float:
    assert parameter.bounds is not None
    upper = float(parameter.bounds[1])
    return math.log(upper) if parameter.scale == "log" else upper


def _parameter_value_from_search_coordinate(parameter: ParameterSpec, coordinate: float) -> float:
    return math.exp(float(coordinate)) if parameter.scale == "log" else float(coordinate)


def _parameter_search_coordinate_from_value(parameter: ParameterSpec, value: Any) -> float:
    numeric = float(value)
    return math.log(numeric) if parameter.scale == "log" else numeric


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


def _objective_score(metric: float, *, goal: str) -> float:
    return metric if goal == "minimize" else -metric


def _clip_unit(point: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(point, dtype=float), 0.0, 1.0)


def _unit_key(point: np.ndarray) -> tuple[float, ...]:
    return tuple(float(value) for value in np.round(_clip_unit(point), decimals=12))


def _append_pattern_candidate(
    pending_keys: list[tuple[float, ...]],
    key: tuple[float, ...],
    *,
    score_cache: Mapping[tuple[float, ...], float],
) -> None:
    if key not in score_cache and key not in pending_keys:
        pending_keys.append(key)


def _initial_pattern_centers(
    initial_points: Sequence[Sequence[float]] | None,
    *,
    lower: np.ndarray,
    span: np.ndarray,
) -> list[np.ndarray]:
    if initial_points:
        centers = []
        for point in initial_points:
            raw = np.asarray(point, dtype=float)
            if raw.shape != lower.shape:
                raise ValueError("initial_points must match lower_bounds and upper_bounds shape")
            centers.append(_clip_unit((raw - lower) / span))
        if centers:
            return centers
    return [np.full(lower.size, 0.5, dtype=float)]


def _pattern_search_targets(center: np.ndarray, *, step: float) -> list[np.ndarray]:
    dim = center.size
    targets: list[np.ndarray] = []
    for axis in range(dim):
        plus = center.copy()
        plus[axis] += step
        targets.append(_clip_unit(plus))
        minus = center.copy()
        minus[axis] -= step
        targets.append(_clip_unit(minus))
    if dim > 1:
        diagonal = np.full(dim, step / math.sqrt(dim), dtype=float)
        targets.append(_clip_unit(center + diagonal))
        targets.append(_clip_unit(center - diagonal))
    return targets


def _evaluate_index_batch(
    indices: Sequence[int],
    *,
    evaluate_indices: Callable[[Sequence[int]], Sequence[float]],
    score_cache: dict[int, float],
    evaluated_order: list[int],
    goal: str,
) -> list[int]:
    fresh = [index for index in indices if index not in score_cache]
    if not fresh:
        return []
    metrics = evaluate_indices(fresh)
    if len(metrics) != len(fresh):
        raise ValueError("evaluate_indices must return one score per input index")
    for index, metric in zip(fresh, metrics, strict=True):
        score_cache[index] = _objective_score(float(metric), goal=goal)
        evaluated_order.append(index)
    return fresh


def _has_new_pattern_index(
    center: np.ndarray,
    *,
    vectors: np.ndarray,
    step: float,
    score_cache: Mapping[int, float],
) -> bool:
    excluded = set(score_cache)
    for target in _pattern_search_targets(center, step=step):
        index = _nearest_candidate_index(target, vectors, excluded=excluded)
        if index is not None:
            return True
    return False


def _has_new_pattern_point(
    center: np.ndarray,
    *,
    step: float,
    score_cache: Mapping[tuple[float, ...], float],
) -> bool:
    for target in _pattern_search_targets(center, step=step):
        if _unit_key(target) not in score_cache:
            return True
    return False


def _best_index(score_cache: Mapping[int, float], *, goal: str) -> int:
    del goal
    return min(score_cache, key=lambda index: score_cache[index])


def _best_point_key(
    score_cache: Mapping[tuple[float, ...], float], *, goal: str
) -> tuple[float, ...]:
    del goal
    return min(score_cache, key=lambda key: score_cache[key])


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


def _initial_bounded_simplex_around_center(center: np.ndarray, *, scale: float) -> list[np.ndarray]:
    center = _clip_unit(center)
    simplex = [center]
    for axis in range(center.size):
        vertex = center.copy()
        direction = 1.0 if center[axis] <= 0.5 else -1.0
        vertex[axis] += direction * scale
        if not 0.0 <= vertex[axis] <= 1.0:
            vertex[axis] = center[axis] - direction * scale
        simplex.append(_clip_unit(vertex))
    return simplex


def _sobol_unit_centers(dim: int, count: int, *, seed: int) -> list[np.ndarray]:
    if count <= 0:
        return []
    exponent = math.ceil(math.log2(count)) if count > 1 else 0
    try:
        sampler = qmc.Sobol(d=dim, scramble=True, rng=np.random.default_rng(seed))
    except TypeError:
        # SciPy 1.14, the oldest supported release, still names this
        # argument ``seed``. Newer releases renamed it to ``rng``.
        sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
    samples = sampler.random_base2(m=exponent)[:count]
    return [np.asarray(sample, dtype=float) for sample in samples]


def _split_iteration_budget(total_iterations: int, num_simplices: int) -> list[int]:
    if total_iterations <= 0:
        raise ValueError("total_iterations must be positive")
    if num_simplices <= 0:
        raise ValueError("num_simplices must be positive")
    base = total_iterations // num_simplices
    remainder = total_iterations % num_simplices
    return [base + (1 if index < remainder else 0) for index in range(num_simplices)]


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
