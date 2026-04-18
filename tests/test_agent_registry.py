from __future__ import annotations

from dymad.agent.exec.training_profiles import (
    PROFILE_ALIASES,
    available_profiles,
    profile_config,
    resolve_profile_name,
)
from dymad.agent.registry import (
    describe_training_capability,
    list_model_capabilities,
    list_profile_capabilities,
    list_training_capabilities,
    resolve_model_capability,
)


def test_registry_lists_current_model_families_and_variants() -> None:
    capabilities = {capability.key: capability for capability in list_model_capabilities()}

    assert {"kbf", "km", "kmsk", "kmm", "ldm", "lti", "sdm"} <= set(capabilities)
    assert capabilities["ldm"].dataset_kinds == ("regular", "graph")
    assert capabilities["ldm"].default_model_ref_by_dataset_kind["regular"] == (
        "dymad.models.collections:LDM"
    )
    assert capabilities["ldm"].default_model_ref_by_dataset_kind["graph"] == (
        "dymad.models.collections:GLDM"
    )
    assert {variant.name for variant in capabilities["ldm"].variants} == {
        "LDM",
        "DLDM",
        "GLDM",
        "DGLDM",
        "LDMG",
        "DLDMG",
    }
    assert capabilities["kbf"].default_model_ref_by_dataset_kind["graph"] == (
        "dymad.models.collections:GKBF"
    )
    assert capabilities["lti"].default_model_ref_by_dataset_kind["graph"] == (
        "dymad.models.collections:GLTI"
    )


def test_registry_resolves_model_capabilities_from_aliases() -> None:
    assert resolve_model_capability("ldm").key == "ldm"
    assert resolve_model_capability("LDM").key == "ldm"
    assert resolve_model_capability("dymad.models.collections:GKBF").key == "kbf"
    assert resolve_model_capability("GLTI").key == "lti"


def test_registry_lists_profile_capabilities_and_compatibility_metadata() -> None:
    profiles = {capability.key: capability for capability in list_profile_capabilities()}

    assert set(available_profiles()) == set(profiles)
    assert profiles["kbf-regular-default"].dataset_kind == "regular"
    assert profiles["kbf-regular-default"].model_keys == ("kbf",)
    assert profiles["kbf-graph-default"].dataset_kind == "graph"
    assert profiles["kbf-graph-default"].implementation_model_refs == (
        "dymad.models.collections:DGKBF",
        "dymad.models.collections:GKBF",
    )
    assert profiles["ldm-graph-default"].implementation_model_refs == (
        "dymad.models.collections:DGLDM",
        "dymad.models.collections:GLDM",
    )
    assert profile_config("lti-regular-default")["model"]["koopman_dimension"] == 4


def test_profile_inference_works_through_registry_backed_training_profiles() -> None:
    assert (
        resolve_profile_name(
            model_ref="dymad.models.collections:KBF",
            dataset_kind="regular",
            reference_profile=None,
        )
        == "kbf-regular-default"
    )
    assert (
        resolve_profile_name(
            model_ref="dymad.models.collections:DGLTI",
            dataset_kind="graph",
            reference_profile=None,
        )
        == "lti-graph-default"
    )
    assert PROFILE_ALIASES[("dymad.models.collections:GLDM", "graph")] == "ldm-graph-default"


def test_training_capabilities_cover_current_profiled_families_and_dataset_kinds() -> None:
    capabilities = {(cap.model_key, cap.dataset_kind): cap for cap in list_training_capabilities()}

    assert capabilities[("kbf", "regular")].default_profile == "kbf-regular-default"
    assert capabilities[("kbf", "graph")].default_model_ref == "dymad.models.collections:GKBF"
    assert capabilities[("ldm", "graph")].default_profile == "ldm-graph-default"
    assert capabilities[("lti", "regular")].default_profile == "lti-regular-default"
    assert capabilities[("sdm", "regular")].default_profile is None


def test_describe_training_capability_exposes_phase_schema_and_override_contract() -> None:
    detail = describe_training_capability(model_key="lti", dataset_kind="regular")

    assert detail.capability.model_key == "lti"
    assert detail.capability.dataset_kind == "regular"
    assert detail.allowed_override_top_level_keys == (
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
    assert "data.path" in detail.runtime_owned_override_paths
    assert "model.name" in detail.runtime_owned_override_paths
    assert {entry.key for entry in detail.phase_entry_schemas} == {
        "legacy_optimizer",
        "optimizer",
        "linear_solve",
        "data",
        "analysis",
        "export",
        "repeat",
    }
    assert detail.cv_schema.supported is True
    assert detail.cv_schema.workflow_kind == "single_split_param_sweep"
    assert detail.cv_schema.allowed_keys == ("param_grid", "metric")
    assert detail.cv_schema.default_metric == "total"
    assert detail.cv_schema.param_grid_value_forms == ("list", "linspace_tuple", "logspace_tuple")
    assert detail.cv_schema.notes == (
        "This v1 user-mode CV surface runs the existing single-split parameter sweep; it is not "
        "true k-fold cross-validation.",
        "The best parameter combination is selected by the lowest aggregated metric value.",
        "Param-grid dotted keys may target either explicit phases.* paths or legacy training.* "
        "shorthand, which is normalized onto the first optimizer phase.",
    )
    assert detail.translation_guidance == (
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
    assert detail.constraint_notes == (
        "Setting encoder_layers=0 or decoder_layers=0 only yields a true identity map "
        "when the latent dimension matches the dataset state dimension.",
        "When the user requests identity encoder/decoder behavior without naming the "
        "latent dimension, inspect the dataset and set the latent dimension to the "
        "dataset state dimension.",
    )
    assert "export_summary" in detail.auto_appended_phases
    assert detail.examples[0].name == "linear_then_node_from_plain_english"
    assert (
        detail.examples[0].user_request
        == "Use staged training: first a Linear phase for initialization, then a "
        "NODE phase for refinement."
    )
    assert detail.examples[0].overrides == {
        "phases": [
            {"trainer": "Linear", "name": "initialization"},
            {"trainer": "NODE", "name": "refinement"},
        ]
    }
    assert detail.examples[1].overrides == {
        "phases": [
            {"trainer": "Weak"},
            {"trainer": "NODE"},
        ]
    }
    assert detail.examples[2].overrides == {
        "cv": {
            "param_grid": {"model.koopman_dimension": [4, 6]},
            "metric": "total",
        }
    }
    assert detail.examples[3].overrides["model"]["koopman_dimension"] == 2
