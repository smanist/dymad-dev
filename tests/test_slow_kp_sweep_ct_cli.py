from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
import torch

from dymad.io import load_model
from dymad.models import KBF
from dymad.utils import TrajectorySampler
from tests.slow_regression_utils import (
    assert_summary_against_baseline,
    extract_record,
    load_baselines,
    load_summary,
    scaled_limit,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = REPO_ROOT / "scripts" / "2d_koopman"
BASELINE_PATH = Path(__file__).with_name("slow_kp_sweep_ct_cli_baselines.json")
TEST_SEED = 12345

N = 301
t_grid = np.linspace(0, 6, N)
mu = -0.5
lm = -3.0


def f(t, x):
    return np.array([mu * x[0], lm * (x[1] - x[0] ** 2)])


@dataclass(frozen=True)
class Case:
    idx: int
    model_name: str
    model_class: type
    metric_factors: dict[str, float] = field(default_factory=dict)

    @property
    def run_dir_name(self) -> str:
        return f"kp_{self.model_name}"


CASES = [
    Case(0, "nd1", KBF),
    Case(1, "nd2", KBF),
    Case(2, "nd3", KBF),
    Case(3, "nd4", KBF),
]


def _eval_rmse(case: Case, checkpoint_path: Path) -> float:
    np.random.seed(TEST_SEED)
    torch.manual_seed(TEST_SEED)
    sampler = TrajectorySampler(f, config=SCRIPT_ROOT / "kp_data.yaml")
    ts, xs, _ = sampler.sample(t_grid, batch=1)
    x_data = xs[0]
    t_data = ts[0]
    _, predict_fn = load_model(case.model_class, checkpoint_path)
    with torch.no_grad():
        pred = predict_fn(x_data, t_data)
    return float(np.sqrt(np.mean((pred - x_data) ** 2)))


def _run_case(case: Case, workdir: Path) -> None:
    mpl_dir = workdir / ".mpl"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = str(mpl_dir)
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT_ROOT / "kp_sweep_ct_cli.py"),
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
@pytest.mark.parametrize("case", CASES, ids=lambda c: c.model_name)
def test_kp_sweep_ct_cli(case: Case, tmp_path: Path, request, baseline_store):
    _run_case(case, tmp_path)
    checkpoint = tmp_path / case.run_dir_name / f"{case.run_dir_name}.pt"
    summary_path = tmp_path / case.run_dir_name / f"{case.run_dir_name}_summary.npz"
    assert checkpoint.exists()
    assert summary_path.exists()
    summary = load_summary(summary_path)
    rmse = _eval_rmse(case, checkpoint)
    record = extract_record(summary, rmse)
    if request.config.getoption("--record-baselines"):
        baseline_store[case.model_name] = record
        return
    baseline = load_baselines(BASELINE_PATH)[case.model_name]
    assert_summary_against_baseline(summary, baseline)
    for metric_name, baseline_value in baseline["metrics"].items():
        factor = case.metric_factors.get(metric_name)
        assert record["metrics"][metric_name] <= scaled_limit(metric_name, baseline_value, factor)
