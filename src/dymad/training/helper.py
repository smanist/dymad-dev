from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

import dymad.tuning as _tuning


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
    Aggregate concrete fold results into CVResult objects by combo index.

    The input rows are produced by ``run_cv_single`` and contain ``combo_idx``,
    ``fold_idx``, ``combo``, ``metric_value``, and ``model_prefix``.
    """
    tmp = [res["combo_idx"] for res in results]
    max_combo_idx, min_combo_idx = max(tmp), min(tmp)
    grouped = [[[], [], []] for _ in range(max_combo_idx - min_combo_idx + 1)]
    for res in results:
        c_idx = res["combo_idx"] - min_combo_idx
        grouped[c_idx][0].append(res["combo"])
        grouped[c_idx][1].append(res["metric_value"])
        grouped[c_idx][2].append(res["model_prefix"])

    cv_results = []
    for combos, metrics, paths in grouped:
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
    return _tuning.iter_param_grid(param_grid)


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
    return _tuning.nelder_mead_like_search_indices(
        combos,
        evaluate_index=evaluate_index,
        goal=goal,
        max_iterations=max_iterations,
        reflection=reflection,
        expansion=expansion,
        contraction=contraction,
        shrink=shrink,
    )


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
    return _tuning.batch_pattern_search_indices(
        combos,
        evaluate_indices=evaluate_indices,
        goal=goal,
        max_evaluations=max_evaluations,
        batch_size=batch_size,
        initial_step=initial_step,
        step_shrink=step_shrink,
    )


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
    return _tuning.batch_pattern_search_points(
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        evaluate_points=evaluate_points,
        goal=goal,
        max_evaluations=max_evaluations,
        batch_size=batch_size,
        initial_points=initial_points,
        initial_step=initial_step,
        step_shrink=step_shrink,
    )


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
    return _tuning.bounded_nelder_mead_search_points(
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        evaluate_point=evaluate_point,
        goal=goal,
        max_iterations=max_iterations,
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
    return _tuning.multi_start_bounded_nelder_mead_search_points(
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        evaluate_point=evaluate_point,
        goal=goal,
        max_iterations=max_iterations,
        num_simplices=num_simplices,
        max_workers=max_workers,
        seed=seed,
        simplex_scale=simplex_scale,
        reflection=reflection,
        expansion=expansion,
        contraction=contraction,
        shrink=shrink,
    )


def get_by_dotted_key(d: dict[str, Any], dotted_key: str) -> Any:
    """
    Read nested dict/list paths for dotted keys such as 'a.b.c' or 'phases.0.n_epochs'.
    """
    curr: Any = d
    for part in dotted_key.split("."):
        if isinstance(curr, list):
            curr = curr[int(part)]
        else:
            curr = curr[part]
    return curr


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
