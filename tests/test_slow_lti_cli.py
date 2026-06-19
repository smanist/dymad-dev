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
from dymad.models import KBF, LDM, LTI
from dymad.utils import TrajectorySampler

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = REPO_ROOT / "scripts" / "linear_time_invariant"
BASELINE_PATH = Path(__file__).with_name("slow_lti_cli_baselines.json")
SAFETY_FACTOR = 1.5
DIAGNOSTIC_SAFETY_FACTORS = {
    "crit_train_last": 10.0,
    "crit_valid_last": 10.0,
}
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
class SlowLTICase:
    script_name: str
    idx: int
    model_name: str
    model_class: type
    seed: int = TEST_SEED
    metric_factors: dict[str, float] = field(default_factory=dict)

    @property
    def script_path(self) -> Path:
        return SCRIPT_ROOT / self.script_name

    @property
    def run_dir_name(self) -> str:
        return f"lti_{self.model_name}"

    @property
    def checkpoint_path(self) -> Path:
        return Path(self.run_dir_name) / f"{self.run_dir_name}.pt"

    @property
    def summary_path(self) -> Path:
        return Path(self.run_dir_name) / f"{self.run_dir_name}_summary.npz"


SLOW_CASES = [
    SlowLTICase(
        script_name="lti_train_cli.py",
        idx=0,
        model_name="ldm_wf",
        model_class=LDM,
    ),
    SlowLTICase(
        script_name="lti_train_cli.py",
        idx=1,
        model_name="ldm_node",
        model_class=LDM,
        seed=0,
    ),
    SlowLTICase(
        script_name="lti_train_cli.py",
        idx=2,
        model_name="kbf_wf",
        model_class=KBF,
    ),
    SlowLTICase(
        script_name="lti_train_cli.py",
        idx=4,
        model_name="kbf_ln",
        model_class=KBF,
    ),
    SlowLTICase(
        script_name="lti_train_cli.py",
        idx=5,
        model_name="lti_wf",
        model_class=LTI,
        seed=3,
    ),
    SlowLTICase(
        script_name="lti_train_cli.py",
        idx=6,
        model_name="lti_ln",
        model_class=LTI,
        seed=1,
    ),
    SlowLTICase(
        script_name="lti_multi_cli.py",
        idx=0,
        model_name="kbf_two",
        model_class=KBF,
        seed=14,
    ),
    SlowLTICase(
        script_name="lti_multi_cli.py",
        idx=1,
        model_name="kbf_mcri",
        model_class=KBF,
        seed=1,
    ),
]

EXTRA_SLOW_CASES = [
    SlowLTICase(
        script_name="lti_mp_cli.py",
        idx=0,
        model_name="kbf_cv",
        model_class=KBF,
        seed=14,
    ),
]


def _gaussian_eval_sample(seed: int):
    sampler = TrajectorySampler(
        f,
        g,
        config=SCRIPT_ROOT / "lti_data.yaml",
        rng=seed,
        config_mod=CONFIG_GAU,
    )
    t_grid = np.linspace(0, 5, 501)
    ts, xs, us, _ = sampler.sample(t_grid, batch=1)
    return xs[0], ts[0], us[0]


def _best_valid_total(summary_path: Path) -> float:
    npz = np.load(summary_path, allow_pickle=True)
    best_valid = npz["best_valid_loss"].item()
    return float(best_valid["valid_total"])


def _rollout_rmse(case: SlowLTICase, checkpoint_path: Path) -> float:
    x_data, t_data, u_data = _gaussian_eval_sample(case.seed)
    _, predict_fn = load_model(case.model_class, checkpoint_path)
    with torch.no_grad():
        pred = predict_fn(x_data, t_data, u=u_data)
    return float(np.sqrt(np.mean((pred - x_data) ** 2)))


def _run_case(case: SlowLTICase, workdir: Path) -> None:
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
            str(case.seed),
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


def _scaled_limit(case: SlowLTICase, metric_name: str, baseline_value: float) -> float:
    factor = case.metric_factors.get(
        metric_name, DIAGNOSTIC_SAFETY_FACTORS.get(metric_name, SAFETY_FACTOR)
    )
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


def _assert_summary_against_baseline(summary: dict, case: SlowLTICase, baseline: dict) -> None:
    signature = _summary_signature(summary)
    baseline_signature = baseline["summary_signature"]
    assert set(baseline_signature["top_level_keys"]) <= set(signature["top_level_keys"])
    stable_signature = {
        key: value
        for key, value in signature.items()
        if key not in {"top_level_keys", "crit_epoch_count"}
    }
    stable_baseline_signature = {
        key: value
        for key, value in baseline_signature.items()
        if key not in {"top_level_keys", "crit_epoch_count"}
    }
    assert stable_signature == stable_baseline_signature
    assert signature["crit_epoch_count"] > 0

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


def _extract_record(summary: dict, case: SlowLTICase, rmse: float) -> dict:
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


def _run_and_check(case: SlowLTICase, tmp_path: Path, request, baseline_store):
    _run_case(case, tmp_path)

    checkpoint_path = tmp_path / case.checkpoint_path
    summary_path = tmp_path / case.summary_path

    assert checkpoint_path.exists()
    assert summary_path.exists()

    summary = _load_summary(summary_path)
    rmse = _rollout_rmse(case, checkpoint_path)
    record = _extract_record(summary, case, rmse)

    if request.config.getoption("--record-baselines"):
        baseline_store[case.model_name] = record
        return

    baseline = _load_baselines()[case.model_name]
    _assert_summary_against_baseline(summary, case, baseline)
    for metric_name, baseline_value in baseline["metrics"].items():
        assert record["metrics"][metric_name] <= _scaled_limit(case, metric_name, baseline_value)


@pytest.mark.slow
@pytest.mark.parametrize("case", SLOW_CASES, ids=lambda case: case.model_name)
def test_lti_cli_training_regression(case: SlowLTICase, tmp_path: Path, request, baseline_store):
    _run_and_check(case, tmp_path, request, baseline_store)


@pytest.mark.extra_slow
@pytest.mark.parametrize("case", EXTRA_SLOW_CASES, ids=lambda case: case.model_name)
def test_lti_cli_training_regression_extra_slow(
    case: SlowLTICase, tmp_path: Path, request, baseline_store
):
    _run_and_check(case, tmp_path, request, baseline_store)
