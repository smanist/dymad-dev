"""Training override and phase-schema registry."""

from __future__ import annotations

from functools import lru_cache

from dymad.agent.registry.models import resolve_model_capability
from dymad.agent.registry.types import (
    DatasetKind,
    TrainingCapabilityDetail,
    TrainingCapabilityExample,
    TrainingCVSchema,
    TrainingPhaseEntrySchema,
)
from dymad.agent.registry.workflows import list_training_capabilities

ALLOWED_TOP_LEVEL_OVERRIDE_KEYS: tuple[str, ...] = (
    "criterion",
    "cv",
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
CV_ALLOWED_KEYS: tuple[str, ...] = ("param_grid", "metric")
CV_DEFAULT_METRIC = "total"
CV_PARAM_GRID_VALUE_FORMS: tuple[str, ...] = ("list", "linspace_tuple", "logspace_tuple")
CV_NOTES: tuple[str, ...] = (
    "This v1 user-mode CV surface runs the existing single-split parameter sweep; it is not "
    "true k-fold cross-validation.",
    "The best parameter combination is selected by the lowest aggregated metric value.",
    "Param-grid dotted keys may target either explicit phases.* paths or legacy training.* "
    "shorthand, which is normalized onto the first optimizer phase.",
)
AUTO_APPENDED_PHASES: tuple[str, ...] = (
    "analysis",
    "export_best_model",
    "export_run_checkpoint",
    "export_summary",
)
TRANSLATION_GUIDANCE: tuple[str, ...] = (
    "For any ordered trainer names mentioned by the user, emit one overrides.phases "
    "entry per trainer in the same order.",
    "Encode hyperparameter sweep requests as overrides.cv.param_grid, with optional "
    "overrides.cv.metric to choose the optimization metric.",
    "Supported optimizer trainer names are Linear, Weak, and NODE.",
    "Prefer minimal legacy optimizer entries such as {'trainer': 'Linear'} or "
    "{'trainer': 'Weak'} unless the user asks for explicit phase-level hyperparameters.",
    "Add phase names only when they improve readability or reflect user-provided "
    "labels such as initialization or refinement.",
)
CONSTRAINT_NOTES: tuple[str, ...] = (
    "Setting encoder_layers=0 or decoder_layers=0 only yields a true identity map "
    "when the latent dimension matches the dataset state dimension.",
    "When the user requests identity encoder/decoder behavior without naming the "
    "latent dimension, inspect the dataset and set the latent dimension to the "
    "dataset state dimension.",
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
def _examples() -> tuple[TrainingCapabilityExample, ...]:
    return (
        TrainingCapabilityExample(
            name="linear_then_node_from_plain_english",
            user_request=(
                "Use staged training: first a Linear phase for initialization, then a "
                "NODE phase for refinement."
            ),
            overrides={
                "phases": [
                    {"trainer": "Linear", "name": "initialization"},
                    {"trainer": "NODE", "name": "refinement"},
                ]
            },
            notes=(
                "This uses the minimal legacy optimizer shorthand because the user did "
                "not specify per-phase hyperparameters.",
            ),
        ),
        TrainingCapabilityExample(
            name="weak_then_node_from_plain_english",
            user_request="Use weak form training first, then refine with NODE.",
            overrides={
                "phases": [
                    {"trainer": "Weak"},
                    {"trainer": "NODE"},
                ]
            },
            notes=(
                "The same ordered-trainer translation rule applies to any supported "
                "mix of Linear, Weak, and NODE phases.",
            ),
        ),
        TrainingCapabilityExample(
            name="hyperparameter_sweep_from_plain_english",
            user_request=(
                "Sweep Koopman dimensions 4 and 6, and choose the model with the lowest total "
                "validation metric."
            ),
            overrides={
                "cv": {
                    "param_grid": {"model.koopman_dimension": [4, 6]},
                    "metric": "total",
                }
            },
            notes=(
                "This uses the existing single-split CV sweep runtime rather than true k-fold "
                "cross-validation.",
            ),
        ),
        TrainingCapabilityExample(
            name="identity_encoder_decoder_for_two_state_lti",
            user_request=(
                "Use trivial encoder and decoder, i.e. identity maps, for a 2-state LTI model."
            ),
            overrides={
                "model": {
                    "encoder_layers": 0,
                    "decoder_layers": 0,
                    "koopman_dimension": 2,
                }
            },
            notes=(
                "In general, replace 2 with the inspected dataset state dimension "
                "before compiling the request.",
            ),
        ),
        TrainingCapabilityExample(
            name="repeat_linear_solve_and_node",
            user_request="Repeat a linear solve followed by NODE refinement twice.",
            overrides={
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
            notes=(
                "Repeat blocks are useful when the user explicitly asks for a cyclic "
                "schedule rather than a simple ordered list of trainers.",
            ),
        ),
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
                cv_schema=TrainingCVSchema(
                    supported=True,
                    workflow_kind="single_split_param_sweep",
                    allowed_keys=CV_ALLOWED_KEYS,
                    default_metric=CV_DEFAULT_METRIC,
                    param_grid_value_forms=CV_PARAM_GRID_VALUE_FORMS,
                    notes=CV_NOTES,
                ),
                translation_guidance=TRANSLATION_GUIDANCE,
                constraint_notes=CONSTRAINT_NOTES,
                auto_appended_phases=AUTO_APPENDED_PHASES,
                examples=_examples(),
            )
    raise ValueError(f"model '{resolved_model.key}' does not support dataset kind '{dataset_kind}'")
