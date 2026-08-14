from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "tuning_convergence" / "cartesian_high_freq_krr_cli.py"
BASELINE_PATH = Path(__file__).with_name("slow_tuning_convergence_baseline.json")
TARGET = "laplace_neumann_m2_k2"
LEVELS = (512, 1024, 2048, 4096)
METHODS = ("rbf_krr", "dm_krr")


def _run_full_neumann(workdir: Path) -> Path:
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = str(workdir / ".mpl")
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--workdir",
            str(workdir),
            "--target",
            TARGET,
            "--levels",
            ",".join(str(level) for level in LEVELS),
            "--trials",
            "5",
            "--n-val",
            "1024",
            "--n-test",
            "4096",
            "--resampling-mode",
            "nested-fixed-test",
            "--validation-mode",
            "train-valid-count",
            "--validation-size",
            "1024",
            "--pool-multiplier",
            "2",
            "--initial-budget",
            "9,9",
            "--refinement-strategy",
            "nelder_mead_like",
            "--refinement-budget",
            "64",
            "--tuning-policy",
            "per_trial",
            "--seed",
            "0",
            "--max-workers",
            "1",
            "--no-prediction-plots",
        ],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        timeout=7200,
    )
    return workdir / TARGET


def _record(result_dir: Path) -> dict[str, object]:
    with (result_dir / "convergence_summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with (result_dir / "convergence_rates.json").open(encoding="utf-8") as handle:
        rates = json.load(handle)

    medians = {
        method: {
            str(level): float(
                next(
                    row["median"]
                    for row in rows
                    if row["method"] == method
                    and row["metric"] == "error"
                    and int(row["refinement"]) == level
                )
            )
            for level in LEVELS
        }
        for method in METHODS
    }
    slopes = {
        method: float(next(row["slope"] for row in rates if row["method"] == method))
        for method in METHODS
    }
    return {"error_medians": medians, "error_slopes": slopes}


def _write_baseline(record: dict[str, object]) -> None:
    BASELINE_PATH.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@pytest.mark.slow
def test_neumann_full_nelder_mead_convergence_baseline(
    tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    result_dir = _run_full_neumann(tmp_path)
    assert (result_dir / "raw_results.csv").is_file()
    assert (result_dir / "convergence.png").is_file()
    assert len(list((result_dir / "tuning").glob("*/tuning_result.json"))) == 40

    record = _record(result_dir)
    if request.config.getoption("--record-baselines"):
        _write_baseline(record)
        return

    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    for method in METHODS:
        for level in LEVELS:
            observed_error = record["error_medians"][method][str(level)]
            baseline_error = baseline["error_medians"][method][str(level)]
            assert observed_error <= baseline_error * 1.25 + 1.0e-12
        assert np.isclose(
            record["error_slopes"][method],
            baseline["error_slopes"][method],
            rtol=0.25,
            atol=0.1,
        )
