from __future__ import annotations

from dataclasses import dataclass, field
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from dymad.io import load_model
from dymad.utils import TrajectorySampler

from scripts.pirom_res.res_train import DPJ, DPT, f, t_grid
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
SCRIPT_ROOT = REPO_ROOT / "scripts" / "pirom_res"
BASELINE_PATH = Path(__file__).with_name("slow_pirom_res_cli_baselines.json")
TEST_SEED = 12345


@dataclass(frozen=True)
class Case:
    idx: int
    model_name: str
    model_class: type
    metric_factors: dict[str, float] = field(default_factory=dict)

    @property
    def run_dir_name(self) -> str:
        return f"res_{self.model_name}"


CASES = [
    Case(0, "dp_nd", DPT),
    Case(1, "dp_wf", DPT),
    Case(2, "dj_nd", DPJ),
    Case(3, "dj_wf", DPJ),
]


def _run_case(case: Case, workdir: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT_ROOT / "res_train_cli.py"),
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
    sampler = TrajectorySampler(f, config=SCRIPT_ROOT / "res_test.yaml")
    ts, xs, _, ps = sampler.sample(t_grid, batch=5)
    _, predict_fn = load_model(case.model_class, checkpoint_path)
    with torch.no_grad():
        pred = np.stack(
            [predict_fn(xs[j], ts[0], p=ps[j]) for j in range(len(xs))],
            axis=0,
        )
    return float(np.sqrt(np.mean((pred - xs) ** 2)))


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
def test_pirom_res_cli(case: Case, tmp_path: Path, request, baseline_store):
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
