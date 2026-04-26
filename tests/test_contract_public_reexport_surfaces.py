from __future__ import annotations

import dymad.core as core
import dymad.models as models
import dymad.training as training

EXPECTED_CORE_SURFACE = {
    "AddOneTransform",
    "AutoencoderTransform",
    "CallableExternalTransform",
    "ComposeTransform",
    "DenoisingTransform",
    "DelayEmbeddingTransform",
    "DiffMapTransform",
    "DiffMapVBTransform",
    "ExternalTransformModule",
    "FieldTransformModule",
    "FixedGraphSeries",
    "GraphModelContext",
    "GraphSeries",
    "GraphSeriesBatch",
    "GraphTrainerBatch",
    "IdentityTransform",
    "IsomapTransform",
    "LiftTransform",
    "RaggedGraphSeriesBatch",
    "RaggedRegularRuntime",
    "RaggedRegularSeriesBatch",
    "RegularModelContext",
    "RegularSeries",
    "RegularSeriesBatch",
    "RegularTrainerBatch",
    "ScalerTransform",
    "SeriesTransformPipeline",
    "SVDTransform",
    "UniformGraphRuntime",
    "UniformLengthGraphSeriesBatch",
    "UniformLengthRegularSeriesBatch",
    "UniformRegularRuntime",
    "VariableEdgeGraphSeries",
    "build_model_context",
    "build_transform_module",
}

EXPECTED_MODELS_SURFACE = {
    "DGKBF",
    "DGKM",
    "DGKMSK",
    "DGLDM",
    "DGLTI",
    "DKBF",
    "DKM",
    "DKMSK",
    "DLDM",
    "DLDMG",
    "DLTI",
    "DSDM",
    "DSDMG",
    "DecoderSpec",
    "DynamicsSpec",
    "EncoderSpec",
    "FeatureSpec",
    "GKBF",
    "GKM",
    "GLDM",
    "GLTI",
    "KBF",
    "KM",
    "KMM",
    "LDM",
    "LDMG",
    "LTI",
    "MemorySpec",
    "ModelSpec",
    "ModelSpecValidationError",
    "PredefinedModel",
    "RecipeSpec",
    "ResolvedModelSpec",
    "RolloutSpec",
    "build_model",
    "build_model_from_spec",
    "get_dims",
}

EXPECTED_TRAINING_SURFACE = {
    "aggregate_cv_results",
    "AnalysisPhaseSpec",
    "ArtifactRegistry",
    "CVResult",
    "DataPhaseSpec",
    "DriverBase",
    "EvaluationArtifact",
    "ExecutionServices",
    "ExportArtifact",
    "ExportPhaseSpec",
    "iter_param_grid",
    "LinearSolvePhaseSpec",
    "LinearSolveReportArtifact",
    "LinearTrainer",
    "ModelArtifact",
    "NODETrainer",
    "normalize_phase_specs",
    "OneStepTrainer",
    "OptimizerPhaseSpec",
    "OptimizerStateArtifact",
    "PhaseContext",
    "PhasePipeline",
    "PhaseRecord",
    "PhaseResult",
    "PhaseSpecValidationError",
    "select_best_cv_result",
    "set_by_dotted_key",
    "SingleSplitDriver",
    "StackedTrainer",
    "TrainerRun",
    "TrainerState",
    "TrainingCheckpointError",
    "TrainingHistoryArtifact",
    "WeakFormTrainer",
}


def test_core_reexport_surface_is_explicit_and_bounded() -> None:
    assert set(core.__all__) == EXPECTED_CORE_SURFACE
    # Internal-only helpers should stay off the package barrel.
    assert not hasattr(core, "build_legacy_transform")
    assert not hasattr(core, "to_padded_regular_runtime")


def test_models_reexport_surface_is_explicit_and_bounded() -> None:
    assert set(models.__all__) == EXPECTED_MODELS_SURFACE
    # Model internals remain importable from concrete modules only.
    assert not hasattr(models, "predict_discrete")
    assert not hasattr(models, "DEC_MAP")


def test_training_reexport_surface_is_explicit_and_bounded() -> None:
    assert set(training.__all__) == EXPECTED_TRAINING_SURFACE
    assert not hasattr(training, "RunState")
    assert not hasattr(training, "OptBase")
    assert not hasattr(training, "OptNODE")
    assert not hasattr(training, "OptWeakForm")
    assert not hasattr(training, "OptLinear")
    assert not hasattr(training, "StackedOpt")
