from __future__ import annotations

import dymad.core as core
import dymad.models as models


EXPECTED_CORE_SURFACE = {
    "AddOneTransform",
    "ComposeTransform",
    "DelayEmbeddingTransform",
    "FieldTransformModule",
    "FixedGraphSeries",
    "GraphModelContext",
    "GraphSeries",
    "GraphSeriesBatch",
    "GraphTrainerBatch",
    "IdentityTransform",
    "LiftTransform",
    "NDRTransformModuleAdapter",
    "RaggedGraphSeriesBatch",
    "RaggedRegularRuntime",
    "RaggedRegularSeriesBatch",
    "RegularModelContext",
    "RegularSeries",
    "RegularSeriesBatch",
    "RegularTrainerBatch",
    "ScalerTransform",
    "SeriesTransformPipeline",
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
