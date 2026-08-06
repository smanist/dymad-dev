"""Workflow coverage for the figure-only ambient-circle KRR study."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

FIGURES = {
    "circle_labels.png",
    "circle_semi_lb.png",
    "circle_semi_mode.png",
    "circle_semi_rbf.png",
    "circle_full_lb.png",
    "circle_angles.png",
    "semicircle_endpoints.png",
    "circle_krr.png",
    "fullcircle_lb_endpoints.png",
}
SCRIPT = Path("scripts/circle_krr_evidence/circle_krr.py")


def _run(output: Path, mpl: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--quick", "--workers", "1", "--output-dir", str(output)],
        cwd=Path.cwd(),
        env={
            **os.environ,
            "MPLCONFIGDIR": str(mpl),
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        },
        text=True,
        capture_output=True,
        check=True,
        timeout=120,
    )


def test_circle_krr_study_writes_only_reproducible_report_figures(tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    result = _run(first, tmp_path / "mpl")
    _run(second, tmp_path / "mpl")

    assert "Wrote 9 figures" in result.stdout
    assert {path.name for path in first.iterdir()} == FIGURES
    assert {path.name for path in second.iterdir()} == FIGURES
    assert (first / "circle_labels.png").read_bytes() == (second / "circle_labels.png").read_bytes()

    coefficients = json.loads(
        Path("scripts/circle_krr_evidence/label_coefficients.json").read_text(encoding="utf-8")
    )
    assert set(coefficients) == {"semicircle_lb", "semicircle_rbf", "full_circle_lb"}
    assert all(len(values) == 12 for values in coefficients.values())
    source = SCRIPT.read_text(encoding="utf-8")
    assert len(source.splitlines()) < 500
    assert "target" not in source.lower()
