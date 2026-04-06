from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np


SAFETY_FACTOR = 1.5
ABS_TOLERANCES = {
    "best_valid_total": 1.0e-12,
    "final_valid_loss": 1.0e-12,
    "crit_train_last": 1.0e-12,
    "crit_valid_last": 1.0e-12,
    "rmse": 1.0e-9,
}


def load_baselines(path: Path) -> dict:
    with open(path, "r") as fh:
        return json.load(fh)


def scaled_limit(metric_name: str, baseline_value: float, factor: float | None = None) -> float:
    use_factor = SAFETY_FACTOR if factor is None else factor
    abs_tol = ABS_TOLERANCES.get(metric_name, 1.0e-9)
    return max(baseline_value * use_factor, baseline_value + abs_tol)


def build_mpl_env(workdir: Path) -> dict[str, str]:
    mpl_dir = workdir / ".mpl"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = str(mpl_dir)
    return env


def load_baseline_store(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r") as fh:
        return json.load(fh)


def write_baseline_store(path: Path, store: dict) -> None:
    with open(path, "w") as fh:
        json.dump(store, fh, indent=2, sort_keys=True)
        fh.write("\n")


def load_summary(summary_path: Path) -> dict:
    with np.load(summary_path, allow_pickle=True) as npz:
        return {key: npz[key] for key in npz.files}


def summary_signature(summary: dict) -> dict:
    model_name = summary["model_name"].item() if hasattr(summary["model_name"], "item") else summary["model_name"]
    best_valid = summary["best_valid_loss"].item()
    hist = summary["hist"]
    hist0 = hist[0] if len(hist) > 0 else {}
    crit_name = summary["crit_name"].item() if hasattr(summary["crit_name"], "item") else summary["crit_name"]
    crit_epoch = summary["crit_epoch"]
    crits = summary["crits"]
    return {
        "top_level_keys": sorted(summary.keys()),
        "model_name_prefix": str(model_name).split("_c")[0],
        "best_valid_loss_keys": sorted(best_valid.keys()),
        "hist_count": int(len(hist)),
        "hist_entry_keys": sorted(hist0.keys()),
        "crit_name": crit_name,
        "crit_epoch_count": int(len(crit_epoch)),
        "crit_series_count": int(crits.shape[0]) if getattr(crits, "ndim", 0) == 2 else 0,
    }


def assert_summary_against_baseline(summary: dict, baseline: dict) -> None:
    assert summary_signature(summary) == baseline["summary_signature"]

    total_training_time = float(summary["total_training_time"])
    avg_epoch_time = float(summary["avg_epoch_time"])
    final_train_loss = float(summary["final_train_loss"])
    final_valid_loss = float(summary["final_valid_loss"])
    assert math.isfinite(total_training_time)
    assert math.isfinite(avg_epoch_time)
    assert math.isfinite(final_train_loss)
    assert math.isfinite(final_valid_loss)
    assert total_training_time >= 0.0
    assert avg_epoch_time >= 0.0

    best_valid = summary["best_valid_loss"].item()
    assert int(best_valid["epoch"]) >= 0
    assert math.isfinite(float(best_valid["train_total"]))
    assert math.isfinite(float(best_valid["valid_total"]))

    hist = summary["hist"]
    assert len(hist) == baseline["summary_signature"]["hist_count"]
    if len(hist) > 0:
        hist0 = hist[0]
        assert len(hist0["epoch"]) > 0


def extract_record(summary: dict, rmse: float) -> dict:
    best_valid = summary["best_valid_loss"].item()
    crits = summary["crits"]
    metrics = {
        "best_valid_total": float(best_valid["valid_total"]),
        "final_valid_loss": float(summary["final_valid_loss"]),
        "rmse": float(rmse),
    }
    if getattr(crits, "ndim", 0) == 2 and crits.shape[0] >= 2 and crits.shape[1] > 0:
        metrics["crit_train_last"] = float(crits[0, -1])
        metrics["crit_valid_last"] = float(crits[1, -1])
    return {
        "summary_signature": summary_signature(summary),
        "metrics": metrics,
    }


def compare_record_metrics(record: dict, baseline: dict, metric_factors: dict[str, float] | None = None) -> None:
    metric_factors = metric_factors or {}
    for metric_name, baseline_value in baseline["metrics"].items():
        factor = metric_factors.get(metric_name)
        assert record["metrics"][metric_name] <= scaled_limit(metric_name, baseline_value, factor)
