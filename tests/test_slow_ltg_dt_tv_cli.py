from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch

from dymad.io import load_model
from dymad.models import DLDMG
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
SCRIPT_ROOT = REPO_ROOT / "scripts" / "ltg_dt_tv"
BASELINE_PATH = Path(__file__).with_name("slow_ltg_dt_tv_cli_baselines.json")
TEST_SEED = 12345


@dataclass(frozen=True)
class Case:
    idx: int
    model_name: str
    model_class: type
    run_dir_name: str
    seed: int = TEST_SEED


CASES = [
    Case(0, "dldmg", DLDMG, "kura_model", seed=8),
]


def _run_case(case: Case, workdir: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT_ROOT / "ltg_dt_tv_cli.py"),
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


def _load_case_data(path: Path):
    data = np.load(path, allow_pickle=True)
    return data.item() if isinstance(data, np.ndarray) and data.shape == () else data


def _eval_rmse(case: Case, workdir: Path, checkpoint_path: Path) -> float:
    data = _load_case_data(workdir / "data" / "data_n2_s3_k4_s20.pkl")
    x_data = np.asarray(data["x"][10])
    t_data = np.arange(x_data.shape[0])
    ei_data = data["ei"][10]
    ew_data = data["ew"][10]
    _, predict_fn = load_model(case.model_class, checkpoint_path)
    with torch.no_grad():
        pred = predict_fn(x_data, t_data, ei=ei_data, ew=ew_data)
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
def test_ltg_dt_tv_cli(case: Case, tmp_path: Path, request, baseline_store):
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
