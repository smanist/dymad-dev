from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
import torch

from dymad.io import load_model
from dymad.models import DGKBF, DGKM, DGKMSK, DGLDM
from dymad.utils import TrajectorySampler, adj_to_edge

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = REPO_ROOT / "scripts" / "ltg_dt"
BASELINE_PATH = Path(__file__).with_name("slow_ltg_dt_cli_baselines.json")
SAFETY_FACTOR = 1.5
TEST_SEED = 12345
ABS_TOLERANCES = {
    "best_valid_total": 1.0e-12,
    "final_valid_loss": 1.0e-12,
    "crit_train_last": 1.0e-12,
    "crit_valid_last": 1.0e-12,
    "rmse": 1.0e-9,
}

A = np.array(
    [
        [0.0, 1.0],
        [-1.0, -0.1],
    ]
)


def f(t, x, u):
    return (x @ A.T) + u


def g(t, x, u):
    return x


ADJ = np.array(
    [
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
    ]
)
EDGE_INDEX = adj_to_edge(ADJ)[0]

CONFIG_GAU = {
    "control": {
        "kind": "gaussian",
        "params": {
            "mean": 0.5,
            "std": 1.0,
            "t1": 4.0,
            "dt": 0.2,
            "mode": "zoh",
        },
    }
}


@dataclass(frozen=True)
class SlowLTGDTCase:
    idx: int
    model_name: str
    model_class: type
    metric_factors: dict[str, float] = field(default_factory=dict)

    @property
    def script_path(self) -> Path:
        return SCRIPT_ROOT / "ltg_dt_cli.py"

    @property
    def run_dir_name(self) -> str:
        return f"ltg_{self.model_name}"

    @property
    def checkpoint_path(self) -> Path:
        return Path(self.run_dir_name) / f"{self.run_dir_name}.pt"

    @property
    def summary_path(self) -> Path:
        return Path(self.run_dir_name) / f"{self.run_dir_name}_summary.npz"


CASES = [
    SlowLTGDTCase(idx=0, model_name="dldm", model_class=DGLDM),
    SlowLTGDTCase(idx=1, model_name="dkbf", model_class=DGKBF),
    SlowLTGDTCase(idx=2, model_name="dkbl", model_class=DGKBF),
    SlowLTGDTCase(idx=3, model_name="ltil", model_class=DGKBF),
    SlowLTGDTCase(idx=4, model_name="dkm", model_class=DGKM),
    SlowLTGDTCase(idx=5, model_name="dkmsk", model_class=DGKMSK),
]


def _gaussian_eval_sample():
    np.random.seed(TEST_SEED)
    torch.manual_seed(TEST_SEED)
    sampler = TrajectorySampler(
        f,
        g,
        config=SCRIPT_ROOT / "ltg_data.yaml",
        config_mod=CONFIG_GAU,
    )
    t_grid = np.linspace(0, 5, 501)
    ts, xs, us, _ = sampler.sample(t_grid, batch=1)
    x_data = np.concatenate([xs[0], xs[0], xs[0]], axis=-1)
    t_data = ts[0]
    u_data = np.concatenate([us[0], us[0], us[0]], axis=-1)
    return x_data, t_data, u_data


def _rollout_rmse(case: SlowLTGDTCase, checkpoint_path: Path) -> float:
    x_data, t_data, u_data = _gaussian_eval_sample()
    _, predict_fn = load_model(case.model_class, checkpoint_path)
    with torch.no_grad():
        pred = predict_fn(x_data, t_data, u=u_data, ei=EDGE_INDEX)
    return float(np.sqrt(np.mean((pred - x_data) ** 2)))


def _run_case(case: SlowLTGDTCase, workdir: Path) -> None:
    mpl_dir = workdir / ".mpl"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = str(mpl_dir)

    subprocess.run(
        [
            sys.executable,
            str(case.script_path),
            "--case",
            str(case.idx),
            "--workdir",
            str(workdir),
            "--seed",
            str(TEST_SEED),
            "--no-plot",
            "--no-predict",
            "--no-show",
        ],
        check=True,
        cwd=REPO_ROOT,
        env=env,
    )


def _load_baselines() -> dict:
    with open(BASELINE_PATH) as fh:
        return json.load(fh)


def _scaled_limit(case: SlowLTGDTCase, metric_name: str, baseline_value: float) -> float:
    factor = case.metric_factors.get(metric_name, SAFETY_FACTOR)
    return max(baseline_value * factor, baseline_value + ABS_TOLERANCES[metric_name])


def _load_summary(summary_path: Path) -> dict:
    with np.load(summary_path, allow_pickle=True) as npz:
        return {key: npz[key] for key in npz.files}


def _summary_signature(summary: dict) -> dict:
    model_name = (
        summary["model_name"].item()
        if hasattr(summary["model_name"], "item")
        else summary["model_name"]
    )
    best_valid = summary["best_valid_loss"].item()
    hist = summary["hist"]
    hist0 = hist[0] if len(hist) > 0 else {}
    crit_name = (
        summary["crit_name"].item()
        if hasattr(summary["crit_name"], "item")
        else summary["crit_name"]
    )
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


def _assert_summary_against_baseline(summary: dict, baseline: dict) -> None:
    signature = _summary_signature(summary)
    assert signature == baseline["summary_signature"]

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
    assert len(hist) == signature["hist_count"]
    if len(hist) > 0:
        hist0 = hist[0]
        assert len(hist0["epoch"]) > 0


def _extract_record(summary: dict, rmse: float) -> dict:
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
        "summary_signature": _summary_signature(summary),
        "metrics": metrics,
    }


@pytest.fixture(scope="session")
def baseline_store(request):
    if not request.config.getoption("--record-baselines"):
        yield None
        return

    store = {}
    if BASELINE_PATH.exists():
        with open(BASELINE_PATH) as fh:
            store = json.load(fh)

    yield store

    with open(BASELINE_PATH, "w") as fh:
        json.dump(store, fh, indent=2, sort_keys=True)
        fh.write("\n")


@pytest.mark.slow
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.model_name)
def test_ltg_dt_cli_training_regression(
    case: SlowLTGDTCase, tmp_path: Path, request, baseline_store
):
    _run_case(case, tmp_path)

    checkpoint_path = tmp_path / case.checkpoint_path
    summary_path = tmp_path / case.summary_path

    assert checkpoint_path.exists()
    assert summary_path.exists()

    summary = _load_summary(summary_path)
    rmse = _rollout_rmse(case, checkpoint_path)
    record = _extract_record(summary, rmse)

    if request.config.getoption("--record-baselines"):
        baseline_store[case.model_name] = record
        return

    baseline = _load_baselines()[case.model_name]
    _assert_summary_against_baseline(summary, baseline)
    for metric_name, baseline_value in baseline["metrics"].items():
        assert record["metrics"][metric_name] <= _scaled_limit(case, metric_name, baseline_value)
