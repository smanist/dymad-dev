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
from dymad.models import DKMSK, KM, KMM
from dymad.utils import TrajectorySampler
from tests.slow_regression_utils import (
    assert_summary_against_baseline,
    extract_record,
    load_baselines,
    load_summary,
    scaled_limit,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = REPO_ROOT / "scripts" / "ker_s1u"
BASELINE_PATH = Path(__file__).with_name("slow_ker_s1u_cli_baselines.json")
TEST_SEED = 12345

B = 20
N = 101
t_grid = np.linspace(0, 8, N)
t_pred = np.linspace(0, 16, N * 2)
s5 = np.sqrt(5)
K0, D0 = 3, 0.1


def f(t, x, u):
    _x = np.atleast_2d(x)
    _t = np.arctan2(_x[:, 1], _x[:, 0])
    _v = 1.5 - np.cos(_t) + u
    _r = 1 + D0 * np.cos(K0 * _t)
    _d = -K0 * D0 * np.sin(K0 * _t)
    _c, _s = np.cos(_t) * _v, np.sin(_t) * _v
    return np.vstack([-_r * _s + _d * _c, _r * _c + _d * _s]).T.squeeze()


def g(t, x, u):
    return x


def dyn(tt, K=K0, D=D0):
    vv = 2 * np.arctan(np.tan(s5 * tt / 4) / s5)
    rr = 1 + D * np.cos(K * vv)
    return vv, np.array([rr * np.cos(vv), rr * np.sin(vv)]).T


t_ref = np.linspace(0, 6, 51)
_ref = dyn(t_ref)[1]
CONFIG_CHR = {
    "control": {
        "kind": "chirp",
        "params": {
            "t1": 8.0,
            "freq_range": (0.25, 0.5),
            "amp_range": (0.5, 1.0),
            "phase_range": (0.0, 360.0),
        },
    },
    "x0": {"kind": "perturb", "params": {"bounds": [0, 0], "ref": _ref}},
}


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
    Case(0, "km_ln", KM, metric_factors={"rmse": 10.0}),
    Case(1, "kmm_ln", KMM),
    Case(2, "dks_ln", DKMSK),
]


def _eval_rmse(case: Case, checkpoint_path: Path) -> float:
    np.random.seed(TEST_SEED)
    torch.manual_seed(TEST_SEED)
    sampler = TrajectorySampler(f, g, config=SCRIPT_ROOT / "ker_data.yaml", config_mod=CONFIG_CHR)
    ts, xs, us, ys = sampler.sample(t_pred, batch=1, save=None)
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
            str(SCRIPT_ROOT / "ker_s1u_cli.py"),
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
def test_ker_s1u_cli(case: Case, tmp_path: Path, request, baseline_store):
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
