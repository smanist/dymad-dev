"""Agent-facing capability registry accessors."""

from dymad.agent.registry.analyses import list_analysis_capabilities, resolve_analysis_capability
from dymad.agent.registry.evaluations import (
    SUPPORTED_EVALUATION_METRICS,
    list_evaluation_capabilities,
)
from dymad.agent.registry.models import list_model_capabilities, resolve_model_capability
from dymad.agent.registry.training_schema import (
    describe_training_capability,
    list_training_phase_entry_schemas,
)
from dymad.agent.registry.types import (
    AnalysisCapability,
    AnalysisImplementation,
    AnalysisSupportLevel,
    DatasetKind,
    EvaluationCapability,
    ModelCapability,
    ModelVariantCapability,
    ProfileCapability,
    TrainingCapabilityDetail,
    TrainingPhaseEntrySchema,
    TrainingWorkflowCapability,
    WorkflowKind,
)
from dymad.agent.registry.workflows import (
    available_profiles,
    list_profile_capabilities,
    list_training_capabilities,
    profile_alias_mapping,
    profile_config,
    profile_registry_payload,
    resolve_profile_name,
)

__all__ = [
    "AnalysisCapability",
    "AnalysisImplementation",
    "AnalysisSupportLevel",
    "DatasetKind",
    "EvaluationCapability",
    "ModelCapability",
    "ModelVariantCapability",
    "TrainingCapabilityDetail",
    "TrainingPhaseEntrySchema",
    "ProfileCapability",
    "TrainingWorkflowCapability",
    "WorkflowKind",
    "available_profiles",
    "describe_training_capability",
    "list_analysis_capabilities",
    "list_evaluation_capabilities",
    "list_model_capabilities",
    "list_profile_capabilities",
    "list_training_phase_entry_schemas",
    "list_training_capabilities",
    "profile_alias_mapping",
    "profile_config",
    "profile_registry_payload",
    "resolve_analysis_capability",
    "resolve_model_capability",
    "resolve_profile_name",
    "SUPPORTED_EVALUATION_METRICS",
]
