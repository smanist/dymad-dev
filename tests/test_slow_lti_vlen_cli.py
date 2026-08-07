from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch

from dymad.io import load_model
from dymad.models import LDM
from dymad.utils import TrajectorySampler
from tests.slow_regression_utils import (
    assert_summary_against_baseline,
    build_mpl_env,
    compare_record_metrics,
    extract_record,
    load_baseline_store,
    load_summary,
    write_baseline_store,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = REPO_ROOT / "scripts" / "lti_vlen"
BASELINE_PATH = Path(__file__).with_name("slow_lti_vlen_cli_baselines.json")
TEST_SEED = 12345
DT = 0.05
PRED_STEPS = 101

A = np.array([[0.0, 1.0], [-1.0, -0.1]])


def f(t, x, u):
    return (x @ A.T) + u


def g(t, x, u):
    return x


@dataclass(frozen=True)
class Case:
    idx: int
    model_name: str
    model_class: type
    seed: int = TEST_SEED

    @property
    def run_dir_name(self) -> str:
        return f"lti_vlen_{self.model_name}"


CASES = [
    Case(0, "ldm_node", LDM),
    Case(1, "ldm_wf", LDM),
]


def _run_case(case: Case, workdir: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT_ROOT / "lti_vlen_cli.py"),
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
        env=build_mpl_env(workdir),
    )


def _eval_rmse(case: Case, checkpoint_path: Path) -> float:
    sampler = TrajectorySampler(
        f,
        g,
        config=SCRIPT_ROOT / "lti_vlen_data.yaml",
        rng=case.seed,
    )
    t_grid = DT * np.arange(PRED_STEPS)
    ts, xs, us, _ = sampler.sample(t_grid, batch=1)
    x_data = xs[0]
    t_data = ts[0]
    u_data = us[0]
    _, predict_fn = load_model(case.model_class, checkpoint_path)
    with torch.no_grad():
        pred = predict_fn(x_data, t_data, u=u_data)
    return float(np.sqrt(np.mean((pred - x_data) ** 2)))


@pytest.fixture(scope="session")
def baseline_store(request):
    if not request.config.getoption("--record-baselines"):
        yield None
        return
    store = load_baseline_store(BASELINE_PATH)
    yield store
    write_baseline_store(BASELINE_PATH, store)


@pytest.mark.slow
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.model_name)
def test_lti_vlen_cli(case: Case, tmp_path: Path, request, baseline_store):
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
    baseline = load_baseline_store(BASELINE_PATH)[case.model_name]
    assert_summary_against_baseline(summary, baseline)
    compare_record_metrics(record, baseline)


@pytest.mark.slow
def test_lti_vlen_cli_prediction_smoke(tmp_path: Path):
    case = CASES[0]
    _run_case(case, tmp_path)
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT_ROOT / "lti_vlen_cli.py"),
            "--case",
            str(case.idx),
            "--workdir",
            str(tmp_path),
            "--seed",
            str(case.seed),
            "--no-train",
            "--no-plot",
            "--no-show",
        ],
        check=True,
        cwd=REPO_ROOT,
        env=build_mpl_env(tmp_path),
    )
    assert (tmp_path / "lti_vlen_prediction.png").exists()
