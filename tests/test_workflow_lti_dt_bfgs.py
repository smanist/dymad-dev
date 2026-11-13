"""Short CLI execution checks; learning accuracy belongs in the slow regression."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from scipy.signal import cont2discrete

from tests.slow_regression_utils import build_mpl_env

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/lti_dt/lti_dt_bfgs_cli.py"


@pytest.mark.parametrize("case_index,case_name", [(0, "dldm"), (1, "dkbf")])
def test_lti_dt_adam_bfgs_example(tmp_path, case_index, case_name):
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
            "6",
            "--steps",
            "9",
            "--duration",
            "0.4",
            "--warmup",
            "2",
            "--refine",
            "2",
            "--no-show",
        ],
        cwd=tmp_path,
        env=env,
        check=True,
        timeout=120,
    )

    warmups = []
    for optimizer in ("adam", "bfgs", "lbfgs"):
        name = f"lti_{case_name}_{optimizer}"
        payload = torch.load(tmp_path / name / f"{name}.pt", map_location="cpu", weights_only=False)
        warmups.append(payload["hist"][0])
        assert len(payload["hist"]) == 2
        assert all(tensor.dtype == torch.float64 for tensor in payload["model_state_dict"].values())
        losses = [value for phase in payload["hist"] for value in phase["valid_total"]]
        assert 2 < len(losses) <= 4
        assert np.isfinite(losses).all()
        assert len(payload["epoch_times"]) == len(losses)
        assert all(seconds > 0 for seconds in payload["epoch_times"])
        config = yaml.safe_load((tmp_path / f"{name}.yaml").read_text())
        assert config["seed"] == config["data"]["split_seed"] == 12345
        assert config["dataloader"]["batch_size"] == 6
        assert config["phases"][0]["optimizer"] == "adam"
        assert config["phases"][1]["optimizer"] == optimizer
        if optimizer == "lbfgs":
            assert config["phases"][1]["history_size"] == 100
        assert all(phase["sweep_lengths"] == [None] for phase in config["phases"])
    assert warmups[0] == warmups[1] == warmups[2]
    ad, bd, *_ = cont2discrete(
        (np.array([[0.0, 1.0], [-1.0, -0.1]]), np.ones((2, 1)), np.eye(2), np.zeros((2, 1))),
        0.05,
    )
    with np.load(tmp_path / "lti.npz") as data:
        np.testing.assert_allclose(
            data["x"][:, 1:],
            data["x"][:, :-1] @ ad.T + data["u"][:, :-1] @ bd.T,
            rtol=1e-13,
            atol=1e-14,
        )
    with np.load(tmp_path / f"{case_name}_predictions.npz") as predictions:
        assert predictions["x"].shape == (4, 9, 2)  # Truth plus three optimizers.
        assert np.isfinite(predictions["x"]).all()
        truth = predictions["x"][0]
        np.testing.assert_allclose(
            truth[1:],
            truth[:-1] @ ad.T + predictions["u"][:-1] @ bd.T,
            rtol=1e-13,
            atol=1e-14,
        )
    assert (tmp_path / f"{case_name}_loss.png").exists()
    assert (tmp_path / f"{case_name}_prediction.png").exists()
