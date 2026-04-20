from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from itertools import product
from typing import Any

import numpy as np


@dataclass
class CVResult:
    params: dict[str, Any]
    fold_metrics: list[float]
    mean_metric: float = 0.0
    std_metric: float = 0.0
    checkpoint_paths: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.mean_metric = float(np.mean(self.fold_metrics))
        self.std_metric = float(np.std(self.fold_metrics))


def aggregate_cv_results(results: list[dict[str, Any]]):
    """
    The results are potentially from concurrent runs, and each is in the format of

    {'combo_idx', 'fold_idx', 'combo', 'metric_value', 'model_prefix'}

    This function aggregates them into CVResult objects by collecting fold results for each combo_idx.
    """
    tmp = [res["combo_idx"] for res in results]
    max_combo_idx, min_combo_idx = max(tmp), min(tmp)
    tmp = [[[], [], []] for _ in range(max_combo_idx - min_combo_idx + 1)]
    for res in results:
        c_idx = res["combo_idx"] - min_combo_idx
        tmp[c_idx][0].append(res["combo"])
        tmp[c_idx][1].append(res["metric_value"])
        tmp[c_idx][2].append(res["model_prefix"])

    cv_results = []
    for combos, metrics, paths in tmp:
        assert len(set(tuple(sorted(c.items())) for c in combos)) == 1, (
            "Inconsistent combos for same combo_idx"
        )
        cv_results.append(CVResult(params=combos[0], fold_metrics=metrics, checkpoint_paths=paths))

    return cv_results


def _param_l1_score(params: dict[str, Any]) -> float:
    score = 0.0
    for key in sorted(params):
        value = params[key]
        if isinstance(value, bool):
            score += float(value)
            continue
        if isinstance(value, (int, float, np.integer, np.floating)):
            score += abs(float(value))
    return score


def select_best_cv_result(
    cv_results: list[CVResult],
    *,
    goal: str = "minimize",
    tie_breakers: tuple[str, ...] | list[str] = ("std_metric", "combo_index"),
    combo_indices: Sequence[int] | None = None,
) -> int:
    """
    Return the selected best CV-result index using explicit selection rules.

    Supported goals:
      - minimize: lower mean metric is better
      - maximize: higher mean metric is better

    Supported tie breakers (applied in order):
      - std_metric: lower std is better
      - param_l1: lower numeric L1 score of tuned params is better
      - combo_index: lower candidate index is better. Uses `combo_indices` when provided;
        otherwise uses the cv_results list position.
    """
    if not cv_results:
        raise ValueError("cv_results must be non-empty")
    if goal not in {"minimize", "maximize"}:
        raise ValueError("goal must be either 'minimize' or 'maximize'")
    if combo_indices is not None and len(combo_indices) != len(cv_results):
        raise ValueError("combo_indices must have the same length as cv_results")

    allowed_tie_breakers = {"std_metric", "param_l1", "combo_index"}
    normalized_tie_breakers = tuple(tie_breakers)
    unknown_tie_breakers = sorted(set(normalized_tie_breakers) - allowed_tie_breakers)
    if unknown_tie_breakers:
        raise ValueError(f"unsupported tie breaker(s): {unknown_tie_breakers}")

    def _selection_key(index: int, result: CVResult) -> tuple[float, ...]:
        primary = result.mean_metric if goal == "minimize" else -result.mean_metric
        key_parts: list[float] = [primary]
        for tie_breaker in normalized_tie_breakers:
            if tie_breaker == "std_metric":
                key_parts.append(result.std_metric)
            elif tie_breaker == "param_l1":
                key_parts.append(_param_l1_score(result.params))
            elif tie_breaker == "combo_index":
                combo_index = combo_indices[index] if combo_indices is not None else index
                key_parts.append(float(combo_index))
        return tuple(key_parts)

    return min(range(len(cv_results)), key=lambda idx: _selection_key(idx, cv_results[idx]))


def iter_param_grid(param_grid: dict[str, Iterable[Any]]):
    """
    param_grid: dict mapping dotted keys to iterables.
    Yields dicts mapping dotted keys -> single value.
    """
    keys = list(param_grid.keys())
    values_lists = []
    for k in keys:
        val = param_grid[k]
        if isinstance(val, list):
            values_lists.append(val)
        elif isinstance(val, tuple):
            if val[0] == "linspace":
                values_lists.append(np.linspace(*val[1]).tolist())
            elif val[0] == "logspace":
                values_lists.append(np.logspace(*val[1]).tolist())
            else:
                raise ValueError(f"Unknown param grid specifier: {val}")
        else:
            raise ValueError(f"Param grid values must be lists or tuples, got {type(val)}")
    for values in product(*values_lists):
        yield dict(zip(keys, values, strict=False))


def _combo_numeric_matrix(combos: Sequence[dict[str, Any]]) -> np.ndarray | None:
    if not combos:
        return np.zeros((0, 0), dtype=float)

    key_order = tuple(sorted(combos[0]))
    vectors: list[list[float]] = []
    for combo in combos:
        if tuple(sorted(combo)) != key_order:
            return None
        row: list[float] = []
        for key in key_order:
            value = combo[key]
            if isinstance(value, bool):
                row.append(float(value))
                continue
            if isinstance(value, (int, float, np.integer, np.floating)):
                row.append(float(value))
                continue
            return None
        vectors.append(row)
    return np.asarray(vectors, dtype=float)


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
            min_distance = min(
                float(np.linalg.norm(vectors[index] - vectors[current])) for current in selected
            )
            if min_distance > best_distance:
                best_distance = min_distance
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
    """
    Evaluate combo candidates through a discrete Nelder-Mead-like search path.

    Returns the ordered list of evaluated combo indices. If candidate values are non-numeric
    (or key structure is inconsistent), the function deterministically falls back to evaluating
    the full grid order.
    """
    if goal not in {"minimize", "maximize"}:
        raise ValueError("goal must be either 'minimize' or 'maximize'")
    if max_iterations is not None and max_iterations <= 0:
        raise ValueError("max_iterations must be a positive integer when provided")
    if not combos:
        return []

    evaluated_order: list[int] = []
    score_cache: dict[int, float] = {}

    def _objective(metric: float) -> float:
        return metric if goal == "minimize" else -metric

    def _ensure_score(index: int) -> float:
        if index not in score_cache:
            metric = float(evaluate_index(index))
            score_cache[index] = _objective(metric)
            evaluated_order.append(index)
        return score_cache[index]

    vectors = _combo_numeric_matrix(combos)
    if vectors is None:
        for index in range(len(combos)):
            _ensure_score(index)
        return evaluated_order
    vectors = _normalize_vectors(vectors)

    simplex_size = min(vectors.shape[1] + 1, len(combos))
    simplex = _initial_simplex_indices(vectors, simplex_size=simplex_size)
    for index in simplex:
        _ensure_score(index)

    if len(score_cache) == len(combos):
        return evaluated_order

    iteration_budget = max_iterations if max_iterations is not None else len(combos)
    for _ in range(iteration_budget):
        simplex = sorted(simplex, key=_ensure_score)
        best = simplex[0]
        worst = simplex[-1]
        second_worst = simplex[-2] if len(simplex) > 1 else worst
        centroid = (
            np.mean(vectors[simplex[:-1]], axis=0) if len(simplex) > 1 else vectors[best].copy()
        )

        best_score = _ensure_score(best)
        worst_score = _ensure_score(worst)
        second_worst_score = _ensure_score(second_worst)

        reflected = centroid + reflection * (centroid - vectors[worst])
        reflected_idx = _nearest_candidate_index(reflected, vectors, excluded=set(score_cache))
        if reflected_idx is None:
            break
        reflected_score = _ensure_score(reflected_idx)

        if reflected_score < best_score:
            expanded = centroid + expansion * (vectors[reflected_idx] - centroid)
            expanded_idx = _nearest_candidate_index(expanded, vectors, excluded=set(score_cache))
            if expanded_idx is not None:
                expanded_score = _ensure_score(expanded_idx)
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

        if reflected_score < worst_score:
            contracted = centroid + contraction * (vectors[reflected_idx] - centroid)
        else:
            contracted = centroid + contraction * (vectors[worst] - centroid)
        contracted_idx = _nearest_candidate_index(contracted, vectors, excluded=set(score_cache))
        if contracted_idx is not None:
            contracted_score = _ensure_score(contracted_idx)
            if contracted_score < min(worst_score, reflected_score):
                simplex[-1] = contracted_idx
                if len(score_cache) == len(combos):
                    break
                continue

        shrunk_simplex = [best]
        for vertex in simplex[1:]:
            shrink_target = vectors[best] + shrink * (vectors[vertex] - vectors[best])
            shrink_idx = _nearest_candidate_index(shrink_target, vectors, excluded=set(score_cache))
            if shrink_idx is None:
                continue
            _ensure_score(shrink_idx)
            shrunk_simplex.append(shrink_idx)

        if len(shrunk_simplex) <= 1:
            break

        for index in simplex:
            if len(shrunk_simplex) >= simplex_size:
                break
            if index not in shrunk_simplex:
                shrunk_simplex.append(index)
        simplex = shrunk_simplex[:simplex_size]

        if len(score_cache) == len(combos):
            break

    return evaluated_order


def set_by_dotted_key(d: dict[str, Any], dotted_key: str, value: Any):
    """
    Set nested dict/list paths for dotted keys such as 'a.b.c' or 'phases.0.n_epochs'.
    Creates intermediate containers as needed.
    """
    parts = dotted_key.split(".")
    curr: dict[str, Any] | list[Any] = d
    for index, part in enumerate(parts[:-1]):
        next_part = parts[index + 1]
        next_is_index = next_part.isdigit()

        if isinstance(curr, list):
            list_index = int(part)
            while len(curr) <= list_index:
                curr.append([] if next_is_index else {})
            curr = curr[list_index]
            continue

        assert isinstance(curr, dict)
        if part not in curr or not isinstance(curr[part], (dict, list)):
            curr[part] = [] if next_is_index else {}
        curr = curr[part]

    last = parts[-1]
    if isinstance(curr, list):
        list_index = int(last)
        while len(curr) <= list_index:
            curr.append(None)
        curr[list_index] = value
        return
    assert isinstance(curr, dict)
    curr[last] = value
