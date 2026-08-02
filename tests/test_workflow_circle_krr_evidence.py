"""Workflow coverage for the ambient-circle diffusion-map/RBF KRR study."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path


def test_circle_krr_evidence_cli_writes_paired_tuning_and_decomposition_artifacts(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "circle_oracle"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/circle_krr_evidence/circle_krr_evidence_cli.py",
            "--output-dir",
            str(output_dir),
            "--semi-n-train",
            "16",
            "--full-n-train",
            "16",
            "--semi-n-valid",
            "15",
            "--full-n-valid",
            "16",
            "--n-test",
            "128",
            "--quadrature-order",
            "32",
            "--endpoint-count",
            "2",
            "--initial-grid-size",
            "2",
            "--refinement-budget",
            "0",
            "--fixed-rbf-ridge-count",
            "3",
            "--max-workers",
            "1",
            "--no-report",
        ],
        cwd=Path.cwd(),
        env={
            **os.environ,
            "MPLCONFIGDIR": str(tmp_path / "mpl"),
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        },
        text=True,
        capture_output=True,
        check=True,
        timeout=120,
    )
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert "Wrote ambient-circle KRR evidence" in result.stdout
    assert summary["protocol"]["kernel_distance"] == "ambient Euclidean chord distance in R^2"
    assert summary["protocol"]["semi_circle"] == {
        "n_train": 16,
        "n_valid": 15,
        "test_sampling": "seeded uniform random theta",
    }
    assert summary["protocol"]["full_circle"]["n_train"] == 16
    assert summary["protocol"]["tuning"]["rbf_fixed_sweep"] == {
        "ambient_lengthscale": 0.2,
        "ridge_count": 3,
        "ridge_bounds": [1.0e-16, 1.0e-8],
    }
    # The theoretical gap is zero; a principal-angle calculation near one
    # incurs a square-root amplification of floating-point roundoff.
    assert summary["full_circle"]["rbf_to_lb_subspace_gap"] < 1.0e-6
    semi_angles = summary["semicircle"]["krr_mode_angles"]["comparisons"]
    assert len(semi_angles) == 5
    assert all(
        all(0.0 <= angle <= 90.0 + 1.0e-8 for angle in row["principal_angles_degrees"])
        for row in semi_angles.values()
    )
    full_angles = summary["full_circle"]["krr_mode_angles"]["comparisons"]
    assert len(full_angles) == 2
    assert all(
        all(0.0 <= angle <= 90.0 + 1.0e-8 for angle in row["principal_angles_degrees"])
        for row in full_angles.values()
    )
    assert summary["audit"]["maximum_decomposition_defect"] < 1.0e-8
    assert summary["audit"]["leakage_exceeds_total_count"] == 0
    assert summary["audit"]["maximum_leakage_to_total_ratio"] <= 1.0
    assert summary["semicircle"]["families"]["count"] == 2
    with (output_dir / "decompositions.csv").open(newline="", encoding="utf-8") as file:
        decomposition_rows = list(csv.DictReader(file))
    assert all(
        float(row["leakage"]) <= float(row["population_error"])
        for row in decomposition_rows
    )
    for name in (
        "selected_models.csv",
        "tuning_evaluations.csv",
        "decompositions.csv",
        "semicircle_family_diagnostics.csv",
        "krr_mode_angles.csv",
        "summary.json",
    ):
        assert (output_dir / name).is_file()
    assert {path.name for path in output_dir.glob("*.png")} == {
        "target_ensembles.png",
        "kernel_mode_comparison.png",
        "semicircle_endpoints.png",
        "semicircle_family_focus_and_summary.png",
        "fullcircle_lb_endpoints.png",
    }
