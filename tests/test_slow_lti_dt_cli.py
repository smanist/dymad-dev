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
from dymad.models import DKBF, DLDM, DLTI
from dymad.utils import TrajectorySampler
from tests.slow_regression_utils import (
    assert_summary_against_baseline,
    extract_record,
    load_baselines,
    load_summary,
    scaled_limit,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = REPO_ROOT / "scripts" / "lti_dt"
BASELINE_PATH = Path(__file__).with_name("slow_lti_dt_cli_baselines.json")
TEST_SEED = 12345

B = 128
N = 501
t_grid = np.linspace(0, 5, N)
A = np.array([[0.0, 1.0], [-1.0, -0.1]])


def f(t, x, u):
    return (x @ A.T) + u


def g(t, x, u):
    return x


CONFIG_GAU = {
    "control": {
        "kind": "gaussian",
        "params": {"mean": 0.5, "std": 1.0, "t1": 4.0, "dt": 0.2, "mode": "zoh"},
    }
}


@dataclass(frozen=True)
class Case:
    idx: int
    model_name: str
    model_class: type
    metric_factors: dict[str, float] = field(default_factory=dict)

    @property
    def run_dir_name(self) -> str:
        return f"lti_{self.model_name}"


CASES = [
    Case(0, "dldm", DLDM),
    Case(1, "dkbf", DKBF),
    Case(2, "dkbl", DKBF, metric_factors={"crit_train_last": 5.0, "crit_valid_last": 5.0}),
    Case(3, "ltil", DLTI),
]


def _eval_rmse(case: Case, checkpoint_path: Path) -> float:
    np.random.seed(TEST_SEED)
    torch.manual_seed(TEST_SEED)
    sampler = TrajectorySampler(f, g, config=SCRIPT_ROOT / "lti_data.yaml", config_mod=CONFIG_GAU)
    ts, xs, us, ys = sampler.sample(t_grid, batch=1)
    x_data = xs[0]
    t_data = ts[0]
    u_data = us[0]
    _, predict_fn = load_model(case.model_class, checkpoint_path)
    with torch.no_grad():
        pred = predict_fn(x_data, t_data, u=u_data)
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
            str(SCRIPT_ROOT / "lti_dt_cli.py"),
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
def test_lti_dt_cli(case: Case, tmp_path: Path, request, baseline_store):
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
