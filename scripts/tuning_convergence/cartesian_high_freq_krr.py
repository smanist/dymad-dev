from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cartesian_high_freq_krr_problem import make_convergence_plot, make_problem  # noqa: E402
from cartesian_high_freq_krr_targets import TARGETS  # noqa: E402
from scripts.cli_helpers import set_seed  # noqa: E402

from dymad.studies.convergence import (  # noqa: E402
    ArrayRegressionStudyConfig,
    run_array_regression_study,
)

# fmt: off
TARGET_NAME = "smooth_radial"
if TARGET_NAME == "oscillatory":
    LEVELS = (512, 1024, 2048, 4096, 8192)
else:
    LEVELS = (512, 1024, 2048, 4096)
TRIALS = 5
N_VAL = 1024
N_TEST = 4096
INITIAL_BUDGET = (9, 9)
REFINEMENT_STRATEGY = "batch_pattern_search"
REFINEMENT_BUDGET = 64 if REFINEMENT_STRATEGY == "batch_pattern_search" else 20
TUNING_POLICY = "per_trial"
SEED = 0
MAX_WORKERS = 4
RESAMPLING_MODE = "nested-fixed-test"
VALIDATION_MODE = "train-valid-count"
VALIDATION_FRACTION = 0.25
VALIDATION_SIZE = 1024
K_FOLDS = 4
POOL_MULTIPLIER = 2
CONFIDENCE_BAND = None

RESTART = True

ifrun = 1
ifplt = 1
ifprd = 1
# fmt: on


OUTPUT_DIR = Path("./runs") / TARGET_NAME
problem = make_problem(TARGET_NAME, TARGETS[TARGET_NAME])

config = ArrayRegressionStudyConfig(
    output_dir=OUTPUT_DIR,
    levels=LEVELS,
    trials=TRIALS,
    n_val=N_VAL,
    n_test=N_TEST,
    initial_budget=INITIAL_BUDGET,
    refinement_budget=REFINEMENT_BUDGET,
    refinement_strategy=REFINEMENT_STRATEGY,
    tuning_policy=TUNING_POLICY,
    seed=SEED,
    max_workers=MAX_WORKERS,
    resampling_mode=RESAMPLING_MODE,
    validation_mode=VALIDATION_MODE,
    validation_fraction=VALIDATION_FRACTION,
    validation_size=VALIDATION_SIZE,
    k_folds=K_FOLDS,
    pool_multiplier=POOL_MULTIPLIER,
    confidence_band=CONFIDENCE_BAND,
    restart=RESTART,
    plot=bool(ifplt),
    prediction_plots=bool(ifprd),
)

set_seed(config.seed)
result: Any | None = None

if ifrun:
    result = run_array_regression_study(problem, config, make_plot=make_convergence_plot)
    print(f"Wrote convergence artifacts to {Path(config.output_dir).resolve()}")
    if result.diagnostics:
        print(f"Diagnostics: {len(result.diagnostics)} advisory item(s); see diagnostics.json")
elif ifplt:
    raise RuntimeError("Set ifrun=1 before plotting so the study result is available.")
