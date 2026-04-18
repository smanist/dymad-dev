from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
import torch

from dymad.io import load_model
from dymad.models import GKBF
from dymad.utils import TrajectorySampler, adj_to_edge
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
SCRIPT_ROOT = REPO_ROOT / "scripts" / "linear_graph_auto"
BASELINE_PATH = Path(__file__).with_name("slow_linear_graph_auto_cli_baselines.json")
TEST_SEED = 12345
EVAL_SEED = 12346

A = np.array([[0.0, 1.0], [-1.0, -0.1]])
ADJ = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
EDGE_INDEX = adj_to_edge(ADJ)[0]


def f(t, x):
    return x @ A.T


@dataclass(frozen=True)
class Case:
    idx: int
    model_name: str
    model_class: type
    seed: int = TEST_SEED
    metric_factors: dict[str, float] = field(default_factory=dict)

    @property
    def run_dir_name(self) -> str:
        return f"ltga_{self.model_name}"


CASES = [
    Case(
        2,
        "kbf_wf",
        GKBF,
        metric_factors={
            "best_valid_total": 15.0,
            "final_valid_loss": 15.0,
            "crit_train_last": 6.0,
            "crit_valid_last": 6.0,
            "rmse": 2.0,
        },
    ),
    Case(
        3,
        "kbf_node",
        GKBF,
        metric_factors={
            "best_valid_total": 10.0,
            "final_valid_loss": 10.0,
            "crit_train_last": 6.0,
            "crit_valid_last": 6.0,
            "rmse": 2.0,
        },
    ),
    Case(
        4,
        "kbf_ln",
        GKBF,
        metric_factors={
            "best_valid_total": 5.0,
            "final_valid_loss": 5.0,
            "crit_train_last": 5.0,
            "crit_valid_last": 5.0,
            "rmse": 2.0,
        },
    ),
]


def _run_case(case: Case, workdir: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT_ROOT / "ltga_train_cli.py"),
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
    sampler = TrajectorySampler(f, config=SCRIPT_ROOT / "ltga_data.yaml", rng=EVAL_SEED)
    ts, xs, ys = sampler.sample(np.linspace(0, 5, 501), batch=1)
    x_data = np.concatenate([xs[0], xs[0], xs[0]], axis=-1)
    t_data = ts[0]
    _, predict_fn = load_model(case.model_class, checkpoint_path)
    with torch.no_grad():
        pred = predict_fn(x_data, t_data, ei=EDGE_INDEX)
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
def test_linear_graph_auto_cli(case: Case, tmp_path: Path, request, baseline_store):
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
    compare_record_metrics(record, baseline, case.metric_factors)
