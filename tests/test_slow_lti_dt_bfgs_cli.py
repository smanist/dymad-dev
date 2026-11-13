"""Full-budget CLI regressions: refinement MSE <= 1e-12 and unseen rollout error <= 1%."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.linalg import expm

from dymad.io import load_model
from dymad.models import DKBF, DLDM
from tests.slow_regression_utils import build_mpl_env

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/lti_dt/lti_dt_bfgs_cli.py"


def _heldout_trajectories():
    # Independent Gaussian ZOH inputs, active for four seconds then zero for one.
    # Build the exact teacher independently of the CLI's sampler/cont2discrete path.
    rng = np.random.default_rng(32345)
    ts = np.linspace(0, 5, 501)
    xs = np.zeros((16, len(ts), 2))
    xs[:, 0] = rng.uniform(-1, 1, (16, 2))
    us = np.zeros((16, len(ts), 1))
    us[:, :400] = np.repeat(rng.normal(0.5, 1, (16, 20, 1)), 20, axis=1)
    generator = np.array([[0.0, 1.0, 1.0], [-1.0, -0.1, 1.0], [0.0, 0.0, 0.0]])
    transition = expm((ts[1] - ts[0]) * generator)
    for k in range(len(ts) - 1):
        xs[:, k + 1] = xs[:, k] @ transition[:2, :2].T + us[:, k] @ transition[:2, 2:].T
    return ts, xs, us


@pytest.mark.slow
@pytest.mark.parametrize("case_index,case_name,model_class", [(0, "dldm", DLDM), (1, "dkbf", DKBF)])
def test_lti_dt_bfgs_cli_learns_unseen_inputs(tmp_path, case_index, case_name, model_class):
    env = build_mpl_env(tmp_path)
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--case",
            str(case_index),
            "--workdir",
            str(tmp_path),
            "--seed",
            "12345",
            "--trajectories",
            "128",
            "--steps",
            "501",
            "--duration",
            "5",
            "--warmup",
            "200",
            "--refine",
            "800",
            "--no-plot",
            "--no-predict",
            "--no-show",
        ],
        cwd=tmp_path,
        env=env,
        check=True,
        timeout=600,
    )
    ts, truth, controls = _heldout_trajectories()
    initial = np.zeros_like(truth)
    initial[:, 0] = truth[:, 0]  # Supply no future states to public checkpoint prediction.
    np.savez_compressed(tmp_path / "heldout.npz", t=ts, x=truth, u=controls)
    warmups, metrics = [], {}
    for optimizer in ("adam", "bfgs", "lbfgs"):
        name = f"lti_{case_name}_{optimizer}"
        checkpoint = tmp_path / name / f"{name}.pt"
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert len(payload["hist"]) == 2
        warmup, refinement = payload["hist"]
        warmups.append(warmup)
        assert len(warmup["valid_total"]) == 200
        assert 0 < len(refinement["valid_total"]) <= 800
        assert np.isfinite(refinement["valid_total"]).all()
        _, predict = load_model(model_class, checkpoint)
        with torch.no_grad():
            predictions = np.stack(
                [predict(x0, ts, u=u) for x0, u in zip(initial, controls, strict=True)]
            )
        assert predictions.shape == truth.shape
        assert np.isfinite(predictions).all()
        relative_errors = np.sqrt(
            np.mean((predictions - truth) ** 2, axis=(1, 2)) / np.mean(truth**2, axis=(1, 2))
        )
        metrics[optimizer] = {
            "refinement_valid_mse": float(min(refinement["valid_total"])),
            "worst_heldout_relative_rmse": float(relative_errors.max()),
        }
    (tmp_path / "regression_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    assert warmups[0] == warmups[1] == warmups[2]
    for optimizer, result in metrics.items():
        # The validation gate requires real refinement beyond the shared Adam warmup.
        # The rollout gate preserves the previously established 1% learning criterion.
        assert result["refinement_valid_mse"] <= 1e-12, (optimizer, result)
        assert result["worst_heldout_relative_rmse"] <= 0.01, (optimizer, result)
