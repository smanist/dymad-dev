"""Training override and phase-schema registry."""

from __future__ import annotations

from functools import lru_cache

from dymad.agent.registry.models import resolve_model_capability
from dymad.agent.registry.types import (
    DatasetKind,
    TrainingCapabilityDetail,
    TrainingPhaseEntrySchema,
)
from dymad.agent.registry.workflows import list_training_capabilities

ALLOWED_TOP_LEVEL_OVERRIDE_KEYS: tuple[str, ...] = (
    "criterion",
    "dataloader",
    "model",
    "phases",
    "plotting",
    "split",
    "transform_u",
    "transform_x",
)
RUNTIME_OWNED_OVERRIDE_PATHS: tuple[str, ...] = (
    "path",
    "data_valid",
    "data.path",
    "data_valid.path",
    "model.name",
)
ALLOWED_DATA_OVERRIDE_KEYS: tuple[str, ...] = ("double_precision",)
RUNTIME_OWNED_MODEL_KEYS: tuple[str, ...] = ("name",)
AUTO_APPENDED_PHASES: tuple[str, ...] = (
    "analysis",
    "export_best_model",
    "export_run_checkpoint",
    "export_summary",
)


@lru_cache(maxsize=1)
def _phase_entry_schemas() -> tuple[TrainingPhaseEntrySchema, ...]:
    return (
        TrainingPhaseEntrySchema(
            key="legacy_optimizer",
            summary="Legacy shorthand for one optimizer phase.",
            accepted_shape='{"trainer": "NODE" | "Weak" | "Linear", ...}',
            required_fields=("trainer",),
            optional_fields=("name",),
            enum_fields={"trainer": ("NODE", "Weak", "Linear")},
            allows_additional_keys=True,
            notes=(
                "Any extra keys are treated as optimizer-phase config.",
                "Legacy shorthand normalizes to an explicit optimizer phase.",
            ),
            example={"trainer": "NODE", "name": "node", "n_epochs": 25},
        ),
        TrainingPhaseEntrySchema(
            key="optimizer",
            summary="Explicit optimizer phase.",
            accepted_shape='{"type": "optimizer", "trainer": "NODE" | "Weak" | "Linear", ...}',
            required_fields=("type", "trainer"),
            optional_fields=("name",),
            enum_fields={"trainer": ("NODE", "Weak", "Linear")},
            allows_additional_keys=True,
            notes=("Any extra keys are treated as optimizer-phase config.",),
            example={"type": "optimizer", "name": "Refine", "trainer": "NODE", "n_epochs": 25},
        ),
        TrainingPhaseEntrySchema(
            key="linear_solve",
            summary="Explicit linear-solve phase.",
            accepted_shape='{"type": "linear_solve", ...}',
            required_fields=("type",),
            optional_fields=("name", "method", "params", "kwargs", "reset_optimizer"),
            allows_additional_keys=True,
            notes=("Any extra keys are stored in the phase config payload.",),
            example={"type": "linear_solve", "name": "ls", "method": "truncated", "params": 2},
        ),
        TrainingPhaseEntrySchema(
            key="data",
            summary="Explicit data/context phase.",
            accepted_shape='{"type": "data", ...}',
            required_fields=("type",),
            optional_fields=("name", "operation"),
            allows_additional_keys=True,
            notes=("Any extra keys are stored in the phase config payload.",),
            example={"type": "data", "name": "refresh_context", "operation": "context"},
        ),
        TrainingPhaseEntrySchema(
            key="analysis",
            summary="Explicit analysis phase.",
            accepted_shape='{"type": "analysis", ...}',
            required_fields=("type",),
            optional_fields=("name", "split", "evaluate_all"),
            allows_additional_keys=True,
            notes=("Any extra keys are stored in the phase config payload.",),
            example={"type": "analysis", "name": "analysis", "split": "valid"},
        ),
        TrainingPhaseEntrySchema(
            key="export",
            summary="Explicit export phase.",
            accepted_shape='{"type": "export", ...}',
            required_fields=("type",),
            optional_fields=("name", "export_kind"),
            allows_additional_keys=True,
            notes=("Any extra keys are stored in the phase config payload.",),
            example={"type": "export", "name": "export_summary", "export_kind": "summary"},
        ),
        TrainingPhaseEntrySchema(
            key="repeat",
            summary="Repeat block that expands nested phases.",
            accepted_shape='{"repeat": {"times": int, "phases": [...]}}',
            required_fields=("repeat.times", "repeat.phases"),
            optional_fields=("repeat.name",),
            allows_additional_keys=False,
            notes=(
                "A repeat entry may only contain the top-level key 'repeat'.",
                "Nested phases are normalized exactly like top-level phases.",
                "Analysis/export phases inside repeat are allowed but warned about.",
            ),
            example={
                "repeat": {
                    "name": "cycle",
                    "times": 2,
                    "phases": [
                        {"type": "linear_solve", "name": "ls", "method": "truncated", "params": 2},
                        {"type": "optimizer", "name": "node", "trainer": "NODE", "n_epochs": 5},
                    ],
                }
            },
        ),
    )


@lru_cache(maxsize=1)
def _examples() -> tuple[dict[str, object], ...]:
    return (
        {
            "name": "linear_then_node",
            "overrides": {
                "phases": [
                    {"type": "optimizer", "name": "LinearInit", "trainer": "Linear", "n_epochs": 1},
                    {
                        "type": "optimizer",
                        "name": "NODERefine",
                        "trainer": "NODE",
                        "n_epochs": 25,
                        "learning_rate": 5e-3,
                        "decay_rate": 0.999,
                    },
                ]
            },
        },
        {
            "name": "repeat_linear_solve_and_node",
            "overrides": {
                "phases": [
                    {
                        "repeat": {
                            "name": "cycle",
                            "times": 2,
                            "phases": [
                                {
                                    "type": "linear_solve",
                                    "name": "ls",
                                    "method": "truncated",
                                    "params": 2,
                                },
                                {
                                    "type": "optimizer",
                                    "name": "node",
                                    "trainer": "NODE",
                                    "n_epochs": 5,
                                },
                            ],
                        }
                    }
                ]
            },
        },
    )


def list_training_phase_entry_schemas() -> tuple[TrainingPhaseEntrySchema, ...]:
    return _phase_entry_schemas()


def describe_training_capability(
    *,
    model_key: str,
    dataset_kind: DatasetKind,
) -> TrainingCapabilityDetail:
    resolved_model = resolve_model_capability(model_key)
    for capability in list_training_capabilities(dataset_kind=dataset_kind):
        if capability.model_key == resolved_model.key:
            return TrainingCapabilityDetail(
                capability=capability,
                allowed_override_top_level_keys=ALLOWED_TOP_LEVEL_OVERRIDE_KEYS,
                runtime_owned_override_paths=RUNTIME_OWNED_OVERRIDE_PATHS,
                allowed_data_override_keys=ALLOWED_DATA_OVERRIDE_KEYS,
                runtime_owned_model_keys=RUNTIME_OWNED_MODEL_KEYS,
                phase_entry_schemas=_phase_entry_schemas(),
                auto_appended_phases=AUTO_APPENDED_PHASES,
                examples=_examples(),
            )
    raise ValueError(f"model '{resolved_model.key}' does not support dataset kind '{dataset_kind}'")
