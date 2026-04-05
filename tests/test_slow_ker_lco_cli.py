from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import scipy.integrate as spi
import torch

from dymad.io import load_model
from dymad.models import DKM, DKMSK, KM
from dymad.utils import TrajectorySampler

from tests.slow_regression_utils import assert_summary_against_baseline, extract_record, load_baselines, load_summary, scaled_limit


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = REPO_ROOT / "scripts" / "ker_lco"
BASELINE_PATH = Path(__file__).with_name("slow_ker_lco_cli_baselines.json")
TEST_SEED = 12345

B = 50
N = 81
t_grid = np.linspace(0, 8, N)
mu = 1.0


def f(t, x):
    _x, _y = x
    return np.array([_y, mu * (1 - _x**2) * _y - _x])


g = lambda t, x: x
_Nt = 161
_ts = np.linspace(0, 40.0, 8 * _Nt)
_res = spi.solve_ivp(f, [0, _ts[-1]], [2, 2], t_eval=_ts)
_ref = _res.y[:, -220:].T
db = 0.4
SMPL = {"x0": {"kind": "perturb", "params": {"bounds": [-db, db], "ref": _ref}}}


@dataclass(frozen=True)
class Case:
    idx: int
    model_name: str
    model_class: type
    metric_factors: dict[str, float] = field(default_factory=dict)

    @property
    def run_dir_name(self) -> str:
        return f"ker_{self.model_name}"


CASES = [
    Case(0, "km_ln", KM),
    Case(2, "dkm_ln", DKM, metric_factors={"rmse": 10.0}),
    Case(4, "dks_ln", DKMSK),
]


def _eval_rmse(case: Case, checkpoint_path: Path) -> float:
    np.random.seed(TEST_SEED)
    torch.manual_seed(TEST_SEED)
    sampler = TrajectorySampler(f, g, config=SCRIPT_ROOT / "ker_data.yaml", config_mod=SMPL)
    ts, xs, ys = sampler.sample(t_grid, batch=32)
    x_data = xs
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
            str(SCRIPT_ROOT / "ker_lco_cli.py"),
            "--case", str(case.idx),
            "--workdir", str(workdir),
            "--seed", str(TEST_SEED),
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
        with open(BASELINE_PATH, "r") as fh:
            store = json.load(fh)
    yield store
    with open(BASELINE_PATH, "w") as fh:
        json.dump(store, fh, indent=2, sort_keys=True)
        fh.write("\n")


@pytest.mark.slow
@pytest.mark.parametrize("case", CASES, ids=lambda c: c.model_name)
def test_ker_lco_cli(case: Case, tmp_path: Path, request, baseline_store):
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
