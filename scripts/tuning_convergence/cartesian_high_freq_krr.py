from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from cartesian_high_freq_krr_cli import METHODS, fit_and_score, make_plot, make_split, tuning_spec

from dymad.studies.convergence import (
    ConvergenceEvaluationContext,
    ConvergenceStudySpec,
    TuningPolicy,
    run_convergence_study,
)

OUTPUT_DIR = Path('./runs')
LEVELS = (32, 64)
TRIALS = (0,)
N_VAL = 32
N_TEST = 128
INITIAL_BUDGET = 5
REFINEMENT_BUDGET = 0
TUNING_POLICY = "per_trial"
SEED = int(os.environ.get("DYMAD_CARTESIAN_SEED", "0"))

ifrun = 1
ifplt = 1


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)
split_cache = {}
result = None


def split_for(refinement: int | float | str, trial: int | str):
    key = (int(refinement), int(trial))
    if key not in split_cache:
        split_cache[key] = make_split(int(refinement), N_VAL, N_TEST, int(trial))
    return split_cache[key]


def tune_eval(method: str, refinement: int | float | str, trial: int | str, params: dict[str, Any]):
    split = split_for(refinement, trial)
    return fit_and_score(
        method,
        split,
        float(params["bandwidth_init"]),
        float(params["ridge_init"]),
        include_test=False,
    )


def study_eval(context: ConvergenceEvaluationContext) -> dict[str, Any]:
    split = split_for(context.refinement, context.trial)
    return fit_and_score(
        context.method,
        split,
        float(context.params["bandwidth_init"]),
        float(context.params["ridge_init"]),
        include_test=True,
    )


if ifrun:
    specs = {
        method: tuning_spec("validation_normalized_rmse", INITIAL_BUDGET, REFINEMENT_BUDGET)
        for method in METHODS
    }
    study_spec = ConvergenceStudySpec(
        methods=METHODS,
        refinement_levels=LEVELS,
        trials=TRIALS,
        metrics=("error", "test_physical_rmse", "test_normalized_max_abs", "fit_seconds"),
        tuning_policy=TuningPolicy(mode=TUNING_POLICY, specs=specs),
        fit_window=LEVELS,
        artifact_dir=OUTPUT_DIR,
        primary_metric="error",
    )
    result = run_convergence_study(study_spec, study_eval, tuning_evaluator=tune_eval)
    print(f"Wrote convergence artifacts to {OUTPUT_DIR}")
    if result.diagnostics:
        print(f"Diagnostics: {len(result.diagnostics)} advisory item(s); see diagnostics.json")

if ifplt:
    if result is None:
        raise RuntimeError("Set ifrun=1 before plotting so the study result is available.")
    make_plot(result, OUTPUT_DIR)
    print(f"Wrote plot to {OUTPUT_DIR / 'convergence.png'}")
