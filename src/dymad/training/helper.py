from collections.abc import Iterable
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
) -> int:
    """
    Return the selected best CV-result index using explicit selection rules.

    Supported goals:
      - minimize: lower mean metric is better
      - maximize: higher mean metric is better

    Supported tie breakers (applied in order):
      - std_metric: lower std is better
      - param_l1: lower numeric L1 score of tuned params is better
      - combo_index: lower candidate index is better
    """
    if not cv_results:
        raise ValueError("cv_results must be non-empty")
    if goal not in {"minimize", "maximize"}:
        raise ValueError("goal must be either 'minimize' or 'maximize'")

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
                key_parts.append(float(index))
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
