from __future__ import annotations

import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.slow_regression_utils import build_mpl_env, load_summary

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "vortex" / "vor_train_cli.py"


@dataclass(frozen=True)
class Case:
    idx: int
    model_name: str

    @property
    def run_dir_name(self) -> str:
        return f"kp_{self.model_name}"


CASES = [Case(1, "dkbf_ln")]


def _run_case(case: Case, workdir: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--case",
            str(case.idx),
            "--workdir",
            str(workdir),
            "--no-plot",
            "--no-predict",
            "--no-show",
        ],
        check=True,
        cwd=REPO_ROOT,
        env=build_mpl_env(workdir),
    )


@pytest.mark.slow
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.model_name)
def test_vortex_train_cli(case: Case, tmp_path: Path):
    _run_case(case, tmp_path)
    checkpoint = tmp_path / case.run_dir_name / f"{case.run_dir_name}.pt"
    summary_path = tmp_path / case.run_dir_name / f"{case.run_dir_name}_summary.npz"
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
