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
from dymad.core.runtime import RaggedRegularRuntime, UniformGraphRuntime, UniformRegularRuntime
from dymad.core.series import (
    RaggedRegularSeriesBatch,
    RegularSeries,
    RegularSeriesBatch,
    UniformLengthRegularSeriesBatch,
)
from dymad.core.torch_transforms import (
    AddOneTransform,
    ComposeTransform,
    DelayEmbeddingTransform,
    IdentityTransform,
    LiftTransform,
    ScalerTransform,
)
from dymad.core.trainer_batch import GraphTrainerBatch, RegularTrainerBatch
from dymad.core.transform_builder import build_transform_module
from dymad.core.transform_module import (
    FieldTransformModule,
    NDRTransformModuleAdapter,
    SeriesTransformPipeline,
)

__all__ = [
    "AddOneTransform",
    "ComposeTransform",
    "FieldTransformModule",
    "FixedGraphSeries",
    "GraphSeries",
    "GraphSeriesBatch",
    "GraphTrainerBatch",
    "GraphModelContext",
    "IdentityTransform",
    "build_model_context",
    "build_transform_module",
    "LiftTransform",
    "NDRTransformModuleAdapter",
    "RegularSeries",
    "RegularSeriesBatch",
    "RegularTrainerBatch",
    "RaggedGraphSeriesBatch",
    "RaggedRegularRuntime",
    "RaggedRegularSeriesBatch",
    "RegularModelContext",
    "ScalerTransform",
    "SeriesTransformPipeline",
    "UniformLengthGraphSeriesBatch",
    "UniformLengthRegularSeriesBatch",
    "UniformGraphRuntime",
    "UniformRegularRuntime",
    "DelayEmbeddingTransform",
    "VariableEdgeGraphSeries",
]
