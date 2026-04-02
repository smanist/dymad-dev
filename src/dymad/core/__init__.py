"""Core data abstractions for the DyMAD migration."""

from dymad.core.graph_series import (
    FixedGraphSeries,
    GraphSeries,
    GraphSeriesBatch,
    RaggedGraphSeriesBatch,
    UniformLengthGraphSeriesBatch,
    VariableEdgeGraphSeries,
)
from dymad.core.model_context import GraphModelContext, RegularModelContext, build_model_context
from dymad.core.runtime import (
    EmptyRegularRuntime,
    GraphRuntimeStep,
    RaggedGraphRuntime,
    RaggedRegularRuntime,
    RegularRuntimeStep,
    UniformGraphRuntime,
    UniformRegularRuntime,
    to_padded_graph_runtime,
    to_padded_regular_runtime,
)
from dymad.core.series import (
    RaggedRegularSeriesBatch,
    RegularSeries,
    RegularSeriesBatch,
    UniformLengthRegularSeriesBatch,
)
from dymad.core.trainer_batch import GraphTrainerBatch, RegularTrainerBatch
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
    "EmptyRegularRuntime",
    "GraphSeries",
    "GraphSeriesBatch",
    "GraphRuntimeStep",
    "GraphTrainerBatch",
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
    "RegularRuntimeStep",
    "RegularTrainerBatch",
    "RegularModelContext",
    "RegularSeriesTransformPipeline",
    "RaggedGraphRuntime",
    "RaggedGraphSeriesBatch",
    "RaggedRegularRuntime",
    "RaggedRegularSeriesBatch",
    "ScalerTransform",
    "SeriesTransformPipeline",
    "TransformMetadata",
    "TransformModule",
    "UniformLengthGraphSeriesBatch",
    "UniformLengthRegularSeriesBatch",
    "UniformGraphRuntime",
    "UniformRegularRuntime",
    "DelayEmbeddingTransform",
    "VariableEdgeGraphSeries",
    "to_padded_graph_runtime",
    "to_padded_regular_runtime",
]
