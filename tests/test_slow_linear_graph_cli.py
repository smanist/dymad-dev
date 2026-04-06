from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
import torch

from dymad.io import load_model
from dymad.models import GKBF, GLDM, GLTI
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
SCRIPT_ROOT = REPO_ROOT / "scripts" / "linear_graph"
BASELINE_PATH = Path(__file__).with_name("slow_linear_graph_cli_baselines.json")
TEST_SEED = 12345

A = np.array([[0.0, 1.0], [-1.0, -0.1]])
ADJ = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
EDGE_INDEX = adj_to_edge(ADJ)[0]


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
class Case:
    idx: int
    model_name: str
    model_class: type
    metric_factors: dict[str, float] = field(default_factory=dict)
    ode_method: str | None = None

    @property
    def run_dir_name(self) -> str:
        return f"ltg_{self.model_name}"


CASES = [
    Case(0, "ldm_wf", GLDM),
    Case(1, "ldm_node", GLDM),
    Case(2, "kbf_wf", GKBF),
    Case(3, "kbf_node", GKBF),
    Case(5, "lti_wf", GLTI),
    Case(6, "lti_ln", GLTI, ode_method="rk4"),
]


def _run_case(case: Case, workdir: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT_ROOT / "ltg_train_cli.py"),
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
        env=build_mpl_env(workdir),
    )


def _eval_rmse(case: Case, checkpoint_path: Path) -> float:
    np.random.seed(TEST_SEED)
    torch.manual_seed(TEST_SEED)
    sampler = TrajectorySampler(f, g, config=SCRIPT_ROOT / "ltg_data.yaml", config_mod=CONFIG_GAU)
    ts, xs, us, ys = sampler.sample(np.linspace(0, 5, 501), batch=1)
    x_data = np.concatenate([ys[0], ys[0], ys[0]], axis=-1)
    t_data = ts[0]
    u_data = np.concatenate([us[0], us[0], us[0]], axis=-1)
    _, predict_fn = load_model(case.model_class, checkpoint_path)
    kwargs = {"u": u_data, "ei": EDGE_INDEX}
    if case.ode_method is not None:
        kwargs["method"] = case.ode_method
    with torch.no_grad():
        pred = predict_fn(x_data, t_data, **kwargs)
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
def test_linear_graph_cli(case: Case, tmp_path: Path, request, baseline_store):
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


@pytest.mark.slow
def test_linear_graph_linear_prediction_regression(tmp_path: Path):
    case = next(case for case in CASES if case.model_name == "lti_ln")
    _run_case(case, tmp_path)
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT_ROOT / "ltg_train_cli.py"),
            "--case",
            str(case.idx),
            "--workdir",
            str(tmp_path),
            "--seed",
            str(TEST_SEED),
            "--no-train",
            "--no-plot",
            "--no-show",
        ],
        check=True,
        cwd=REPO_ROOT,
        env=build_mpl_env(tmp_path),
    )
