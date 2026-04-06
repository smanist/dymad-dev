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
    Set d['a']['b']['c'] when dotted_key is 'a.b.c'.
    Creates intermediate dicts as needed.
    """
    parts = dotted_key.split(".")
    curr = d
    for p in parts[:-1]:
        if p not in curr or not isinstance(curr[p], dict):
            curr[p] = {}
        curr = curr[p]
    curr[parts[-1]] = value
