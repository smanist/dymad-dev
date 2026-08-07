from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch
from scripts.kuramoto.train import DSDMSKG

from dymad.io import load_model
from dymad.utils import adj_to_edge
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
SCRIPT_ROOT = REPO_ROOT / "scripts" / "kuramoto"
BASELINE_PATH = Path(__file__).with_name("slow_kuramoto_cli_baselines.json")
TEST_SEED = 12345
DATA_STEM = "data/data_n4_s5_k4_s5"


@dataclass(frozen=True)
class Case:
    idx: int
    model_name: str
    model_class: type
    seed: int = TEST_SEED

    @property
    def run_dir_name(self) -> str:
        return self.model_name


CASES = [
    Case(0, "sdm_skip", DSDMSKG),
]


def _run_case(case: Case, workdir: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT_ROOT / "kuramoto_cli.py"),
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


def _eval_rmse(case: Case, workdir: Path, checkpoint_path: Path) -> float:
    dat = np.load(workdir / f"{DATA_STEM}_test.npz", allow_pickle=True)
    x_data = dat["x"]
    t_data = np.arange(x_data.shape[1]) * 0.01
    u_data = dat["u"]
    ei_data, ew_data = adj_to_edge(dat["adj"])
    _, predict_fn = load_model(case.model_class, checkpoint_path)
    with torch.no_grad():
        pred = np.stack(
            [
                predict_fn(
                    x_data[j],
                    t_data,
                    u=u_data[j],
                    ei=ei_data[j],
                    ew=ew_data[j],
                )
                for j in range(len(x_data))
            ],
            axis=0,
        )
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
def test_kuramoto_cli(case: Case, tmp_path: Path, request, baseline_store):
    _run_case(case, tmp_path)
    checkpoint = tmp_path / case.run_dir_name / f"{case.run_dir_name}.pt"
    summary_path = tmp_path / case.run_dir_name / f"{case.run_dir_name}_summary.npz"
    assert checkpoint.exists()
    assert summary_path.exists()
    summary = load_summary(summary_path)
    rmse = _eval_rmse(case, tmp_path, checkpoint)
    record = extract_record(summary, rmse)
    if request.config.getoption("--record-baselines"):
        baseline_store[case.model_name] = record
        return
    baseline = load_baseline_store(BASELINE_PATH)[case.model_name]
    assert_summary_against_baseline(summary, baseline)
    compare_record_metrics(record, baseline)
