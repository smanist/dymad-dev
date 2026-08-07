from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch

from dymad.io import load_model
from dymad.models import DKMSK
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
SCRIPT_ROOT = REPO_ROOT / "scripts" / "lorenz63"
BASELINE_PATH = Path(__file__).with_name("slow_lorenz63_cli_baselines.json")
TEST_SEED = 12345


@dataclass(frozen=True)
class Case:
    idx: int
    model_name: str
    model_class: type
    setup_idx: int
    expected_cv_results: int
    seed: int = TEST_SEED

    @property
    def run_dir_name(self) -> str:
        return f"lor_{self.model_name}"

    @property
    def baseline_key(self) -> str:
        return f"{self.model_name}_setup{self.setup_idx}"


CASES = [
    Case(1, "ddm_dm", DKMSK, setup_idx=0, expected_cv_results=10),
    Case(1, "ddm_dm", DKMSK, setup_idx=1, expected_cv_results=20),
]


def _run_case(case: Case, workdir: Path) -> None:
    for setup_idx in range(case.setup_idx + 1):
        subprocess.run(
            [
                sys.executable,
                str(SCRIPT_ROOT / "lor_train_cli.py"),
                "--case",
                str(case.idx),
                "--setup",
                str(setup_idx),
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


def _eval_rmse(case: Case, workdir: Path, checkpoint_path: Path) -> float:
    data = np.load(workdir / "data" / "l63_test.npz")
    x_data = data["x"][:5]
    t_data = data["t"][:5]
    _, predict_fn = load_model(case.model_class, checkpoint_path)
    with torch.no_grad():
        pred = predict_fn(x_data, t_data)
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
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.baseline_key)
def test_lorenz63_cli(case: Case, tmp_path: Path, request, baseline_store):
    _run_case(case, tmp_path)
    checkpoint = tmp_path / case.run_dir_name / f"{case.run_dir_name}.pt"
    summary_path = tmp_path / case.run_dir_name / f"{case.run_dir_name}_summary.npz"
    cv_results = np.load(
        tmp_path / case.run_dir_name / f"{case.run_dir_name}_cv.npz", allow_pickle=True
    )
    assert checkpoint.exists()
    assert summary_path.exists()
    assert len(cv_results["all_results"]) == case.expected_cv_results
    summary = load_summary(summary_path)
    rmse = _eval_rmse(case, tmp_path, checkpoint)
    record = extract_record(summary, rmse)
    if request.config.getoption("--record-baselines"):
        baseline_store[case.baseline_key] = record
        return
    baseline = load_baseline_store(BASELINE_PATH)[case.baseline_key]
    assert_summary_against_baseline(summary, baseline)
    compare_record_metrics(record, baseline)
