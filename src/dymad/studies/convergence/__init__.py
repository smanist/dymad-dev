"""Standalone convergence-study utilities."""

from dymad.studies.convergence.array_regression import (
    ArrayRegressionProblem,
    ArrayRegressionStudyConfig,
    NestedArraySamples,
    run_array_regression_study,
)
from dymad.studies.convergence.core import (
    ConvergenceEvaluationContext,
    ConvergenceStudyResult,
    ConvergenceStudySpec,
    Diagnostic,
    MedianPlotContext,
    TuningEvaluationContext,
    TuningPolicy,
    fit_convergence_rates,
    run_convergence_study,
)
from dymad.studies.convergence.plotting import CurveStyle, plot_convergence_summary
from dymad.studies.convergence.resampling import (
    HoldoutValidationPolicy,
    KFoldValidationPolicy,
    LevelSamplePlan,
    NestedResamplingPolicy,
    TrainValidCountPolicy,
    TrialSamplePlan,
    ValidationFold,
    build_nested_trial_sample_plan,
)

__all__ = [
    "ArrayRegressionProblem",
    "ArrayRegressionStudyConfig",
    "ConvergenceEvaluationContext",
    "ConvergenceStudyResult",
    "ConvergenceStudySpec",
    "CurveStyle",
    "Diagnostic",
    "fit_convergence_rates",
    "build_nested_trial_sample_plan",
    "HoldoutValidationPolicy",
    "KFoldValidationPolicy",
    "LevelSamplePlan",
    "MedianPlotContext",
    "NestedArraySamples",
    "NestedResamplingPolicy",
    "plot_convergence_summary",
    "run_array_regression_study",
    "run_convergence_study",
    "TrialSamplePlan",
    "TrainValidCountPolicy",
    "TuningEvaluationContext",
    "TuningPolicy",
    "ValidationFold",
]
