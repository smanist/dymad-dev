from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "ker_heat"))

import circle  # noqa: E402
from common import epsilon_curve_at_largest_n, fit_loglog_rate, study_artifact_paths  # noqa: E402

BASELINE_PATH = Path(__file__).with_name("slow_heat_kernel_circle_no_mass_baseline.json")


def _record(raw_csv: Path) -> dict[str, object]:
    with raw_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    epsilons, errors, n_values = epsilon_curve_at_largest_n(rows, metric="max_abs_error")
    rate = fit_loglog_rate(epsilons, errors)
    if rate is None:
        raise AssertionError("Expected a convergence rate from the circle no-mass sweep.")
    return {
        "largest_n": int(n_values[0]),
        "mae_at_largest_n": {
            f"{epsilon:g}": float(error) for epsilon, error in zip(epsilons, errors, strict=True)
        },
        "epsilon_rate": rate[0],
    }


def _write_baseline(record: dict[str, object]) -> None:
    BASELINE_PATH.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@pytest.mark.slow
def test_circle_no_mass_full_heat_study(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / ".mpl"))
    monkeypatch.setattr(circle, "BASE_DIR", tmp_path)

    case = circle.CASES["no_mass"]
    circle.run_case("no_mass")

    raw_csv, _conv_en, _conv, _section = study_artifact_paths(circle.BASE_DIR, case)
    record = _record(raw_csv)
    if request.config.getoption("--record-baselines"):
        _write_baseline(record)
        return

    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    assert record["largest_n"] == baseline["largest_n"]
    for epsilon, error in record["mae_at_largest_n"].items():
        assert np.isclose(error, baseline["mae_at_largest_n"][epsilon], rtol=0.25, atol=1e-12)
    assert np.isclose(record["epsilon_rate"], baseline["epsilon_rate"], rtol=0.25, atol=0.1)
