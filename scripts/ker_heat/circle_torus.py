"""Compact circle/torus heat-kernel convergence comparison."""

from __future__ import annotations

# ruff: noqa: E402, I001

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from runtime_env import configure_script_runtime  # noqa: E402

configure_script_runtime(__file__, matplotlib=True)

from common import plot_circle_torus_convergence, read_rows, study_artifact_paths  # noqa: E402
import circle  # noqa: E402
import torus  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent
CIRCLE_CASE = "mass"
TORUS_CASE = "mass"
CIRCLE_EPSILON = 0.000625
TORUS_EPSILON = 0.01
OUTPUT_PATH = BASE_DIR / "runs" / "heat_circle_torus_convergence.png"

ifplt = 1


def plot() -> Path | None:
    """Write the four-panel, LaTeX-width circle/torus convergence figure."""

    circle_config = circle.CASES[CIRCLE_CASE]
    torus_config = torus.CASES[TORUS_CASE]
    circle_raw_csv, *_circle_paths = study_artifact_paths(circle.BASE_DIR, circle_config)
    torus_raw_csv, *_torus_paths = study_artifact_paths(torus.BASE_DIR, torus_config)
    missing = [path for path in (circle_raw_csv, torus_raw_csv) if not path.exists()]
    if missing:
        print("Missing raw results: " + ", ".join(str(path) for path in missing))
        return None
    plot_circle_torus_convergence(
        circle_rows=read_rows(circle_raw_csv),
        torus_rows=read_rows(torus_raw_csv),
        circle_case=circle_config,
        torus_case=torus_config,
        circle_epsilon=CIRCLE_EPSILON,
        torus_epsilon=TORUS_EPSILON,
        path=OUTPUT_PATH,
    )
    return OUTPUT_PATH


if __name__ == "__main__" and ifplt:
    output = plot()
    if output is not None:
        print(f"Wrote {output}")
