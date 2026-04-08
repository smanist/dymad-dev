from __future__ import annotations

import importlib.util
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from dymad.io import load_model
from dymad.models import DKBF
from dymad.utils import TrajectorySampler
from tests.slow_regression_utils import build_mpl_env, load_summary

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "sa_2dk" / "kp_sa_cli.py"
SCRIPT_ROOT = SCRIPT_PATH.parent
TEST_SEED = 12345

N = 21
t_grid = np.linspace(0, 10, N)

_t = 0.2
_T = np.array([[1, _t], [0, 1]])
_S = np.array([[1, -_t], [0, 1]])

mu = -0.5
lm = -3.0


def f(t, x):
    _y = _T.dot(x)
    _d = np.array([mu * _y[0], lm * (_y[1] - _y[0] ** 2)])
    return _S.dot(_d)


@dataclass(frozen=True)
class Case:
    idx: int
    model_name: str

    @property
    def run_dir_name(self) -> str:
        return f"kp_{self.model_name}"

    @property
    def checkpoint_path(self) -> Path:
        return Path(self.run_dir_name) / f"{self.run_dir_name}.pt"

    @property
    def summary_path(self) -> Path:
        return Path(self.run_dir_name) / f"{self.run_dir_name}_summary.npz"


CASES = [
    Case(0, "dkbf_ln"),
    Case(2, "dkbf_sa"),
]


def _load_kp_sa_cli_module():
    spec = importlib.util.spec_from_file_location("kp_sa_cli_test_module", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert_finite_array(values) -> None:
    arr = np.asarray(values)
    assert np.isfinite(arr.real).all()
    assert np.isfinite(arr.imag).all()


def _run_case(case: Case, workdir: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--case",
            str(case.idx),
            "--workdir",
            str(workdir),
            "--seed",
            str(TEST_SEED),
            "--train",
            "--no-show",
        ],
        check=True,
        cwd=REPO_ROOT,
        env=build_mpl_env(workdir),
    )


def _eval_rmse(checkpoint_path: Path) -> float:
    np.random.seed(TEST_SEED)
    torch.manual_seed(TEST_SEED)
    sampler = TrajectorySampler(f, config=SCRIPT_ROOT / "kp_data.yaml")
    ts, xs, _ = sampler.sample(t_grid, batch=1)
    x_data = xs[0]
    t_data = ts[0]
    _, predict_fn = load_model(DKBF, checkpoint_path)
    with torch.no_grad():
        pred = predict_fn(x_data, t_data)
    return float(np.sqrt(np.mean((pred - x_data) ** 2)))


@pytest.mark.slow
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.model_name)
def test_kp_sa_cli(case: Case, tmp_path: Path):
    _run_case(case, tmp_path)

    data_path = tmp_path / "data" / "kp.npz"
    checkpoint = tmp_path / case.checkpoint_path
    summary_path = tmp_path / case.summary_path

    assert data_path.exists()
    assert checkpoint.exists()
    assert summary_path.exists()

    summary = load_summary(summary_path)
    assert math.isfinite(float(summary["total_training_time"]))
    assert math.isfinite(float(summary["avg_epoch_time"]))
    assert math.isfinite(float(summary["final_train_loss"]))
    assert math.isfinite(float(summary["final_valid_loss"]))
    best_valid = summary["best_valid_loss"].item()
    assert int(best_valid["epoch"]) >= 0
    assert math.isfinite(float(best_valid["train_total"]))
    assert math.isfinite(float(best_valid["valid_total"]))

    rmse = _eval_rmse(checkpoint)
    assert math.isfinite(rmse)
    assert rmse >= 0.0


@pytest.mark.slow
def test_kp_sa_cli_analyze_returns_spectral_and_conjugacy_diagnostics(tmp_path: Path):
    module = _load_kp_sa_cli_module()
    module.set_seed(TEST_SEED)
    module.prepare_workdir(tmp_path)
    module.generate_data(tmp_path)
    module.train([0, 1, 2], tmp_path)

    diagnostics = module.analyze(
        [0, 1, 2],
        tmp_path,
        pred_batch=4,
        map_batch=4,
        ps_points=9,
        measure_thetas=31,
    )

    assert len(diagnostics["cases"]) == 3
    for case_diag in diagnostics["cases"]:
        _assert_finite_array(case_diag["discrete"]["grid"])
        _assert_finite_array(case_diag["discrete"]["standard"])
        _assert_finite_array(case_diag["discrete"]["sako"])
        _assert_finite_array(case_diag["continuous"]["grid"])
        _assert_finite_array(case_diag["continuous"]["standard"])
        _assert_finite_array(case_diag["continuous"]["sako"])
        _assert_finite_array(case_diag["measure"]["theta"])
        _assert_finite_array(case_diag["measure"]["values"])
        _assert_finite_array(case_diag["conjugacy"]["trajectory_cnj"])
        _assert_finite_array(case_diag["conjugacy"]["trajectory_nrm"])
        _assert_finite_array(case_diag["conjugacy"]["slow_cnj"])
        _assert_finite_array(case_diag["conjugacy"]["slow_nrm"])
    plt.close("all")
