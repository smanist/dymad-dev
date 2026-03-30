"""Core data abstractions for the DyMAD migration."""

from dymad.core.graph_series import FixedGraphSeries, GraphSeries, GraphSeriesBatch, VariableEdgeGraphSeries
from dymad.core.model_context import GraphModelContext, RegularModelContext, build_model_context
from dymad.core.series import RegularSeries, RegularSeriesBatch
from dymad.core.torch_transforms import (
    AddOneTransform,
    ComposeTransform,
    DelayEmbeddingTransform,
    IdentityTransform,
    LiftTransform,
    ScalerTransform,
)
from dymad.core.transform_builder import build_legacy_transform, build_transform_module, export_transform_state
from dymad.core.transform_module import (
    FieldTransformModule,
    LegacyTransformModuleAdapter,
    NDRTransformModuleAdapter,
    SeriesTransformPipeline,
    TransformMetadata,
    TransformModule,
)
from dymad.core.transform_pipeline import FieldTransform, RegularSeriesTransformPipeline

__all__ = [
    "AddOneTransform",
    "ComposeTransform",
    "FieldTransform",
    "FieldTransformModule",
    "FixedGraphSeries",
    "GraphSeries",
    "GraphSeriesBatch",
    "GraphModelContext",
    "IdentityTransform",
    "build_legacy_transform",
    "build_model_context",
    "build_transform_module",
    "export_transform_state",
    "LegacyTransformModuleAdapter",
    "LiftTransform",
    "NDRTransformModuleAdapter",
    "RegularSeries",
    "RegularSeriesBatch",
    "RegularModelContext",
    "RegularSeriesTransformPipeline",
    "ScalerTransform",
    "SeriesTransformPipeline",
    "TransformMetadata",
    "TransformModule",
    "DelayEmbeddingTransform",
    "VariableEdgeGraphSeries",
]
