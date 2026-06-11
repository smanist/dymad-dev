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
    assert detail.cv_schema.allowed_keys == ("param_grid", "metric", "search", "selection")
    assert detail.cv_schema.default_metric == "total"
    assert detail.cv_schema.param_grid_value_forms == ("list", "linspace_tuple", "logspace_tuple")
    assert detail.cv_schema.search_schema == {
        "allowed_keys": (
            "mode",
            "bounds",
            "max_iterations",
            "reflection",
            "expansion",
            "contraction",
            "shrink",
        ),
        "mode_options": (
            "grid",
            "nelder_mead_like",
            "batch_pattern_search",
            "multi_start_nelder_mead",
        ),
        "default_mode": "grid",
    }
    assert detail.cv_schema.selection_schema == {
        "allowed_keys": ("goal", "tie_breakers"),
        "goal_options": ("minimize", "maximize"),
        "default_goal": "minimize",
        "tie_breaker_options": ("std_metric", "param_l1", "combo_index"),
        "default_tie_breakers": ("std_metric", "combo_index"),
    }
    phase_schemas = {entry.key: entry for entry in detail.phase_entry_schemas}
    assert "reset_optimizer" in phase_schemas["legacy_optimizer"].optional_fields
    assert "reset_optimizer" in phase_schemas["optimizer"].optional_fields
    assert detail.cv_schema.notes == (
        "This v1 user-mode CV surface runs the existing single-split parameter sweep; it is not "
        "true k-fold cross-validation.",
        "cv.search.mode selects the CV optimizer. Grid search operates on cv.param_grid; "
        "Nelder-Mead-like, batch pattern search, and multi-start Nelder-Mead operate on "
        "cv.search.bounds when provided.",
        "The best parameter combination is selected by cv.selection (default: minimize mean "
        "metric, then std_metric, then combo_index).",
        "Param-grid dotted keys may target either explicit phases.* paths or legacy training.* "
        "shorthand, which is normalized onto the first optimizer phase.",
        "cv.search.mode='nelder_mead_like' with cv.search.bounds runs a bounded Nelder-Mead "
        "search over lower/upper parameter ranges in single-split mode; integer-valued bounds "
        "may also specify parity='odd' or 'even'. Without bounds, it falls back to the legacy "
        "adaptive path over numeric param_grid values; non-numeric values fall back to grid "
        "order.",
        "cv.search.mode='batch_pattern_search' runs a bounded batched pattern search over "
        "cv.search.bounds or a batched adaptive walk over numeric param_grid values; use it with "
        "run.max_workers > 1 to keep parallel workers busy during refinement.",
        "cv.search.mode='multi_start_nelder_mead' runs Sobol-started bounded Nelder-Mead "
        "simplices over cv.search.bounds; run.max_workers controls the number of simplices, and "
        "cv.search.max_iterations is divided across those simplices as the total iteration budget.",
    )
    assert detail.translation_guidance == (
        "For any ordered trainer names mentioned by the user, emit one overrides.phases "
        "entry per trainer in the same order.",
        "Encode grid-search sweep requests as overrides.cv.param_grid, with optional "
        "overrides.cv.metric to choose the optimization metric and optional "
        "overrides.cv.search.mode='grid' for explicitness.",
        "For bounded Nelder-Mead requests, set overrides.cv.search.mode='nelder_mead_like' and "
        "provide lower/upper bounds in overrides.cv.search.bounds, optionally adding "
        "parity='odd' or 'even' for integer-valued fields. If bounds are omitted, the runtime "
        "uses the legacy adaptive walk over param_grid candidates.",
        "For parallel refinement requests, prefer overrides.cv.search.mode='batch_pattern_search' "
        "with run.max_workers greater than 1; it uses cv.search.max_iterations as an evaluation "
        "budget.",
        "For parallel multi-start Nelder-Mead requests, set "
        "overrides.cv.search.mode='multi_start_nelder_mead', provide overrides.cv.search.bounds, "
        "and set run.max_workers to the requested number of Sobol-started simplices.",
        "Use overrides.cv.selection to control model choice policy (goal and tie_breakers).",
        "Supported optimizer trainer names are Linear, OneStep, Weak, and NODE.",
        "Prefer minimal legacy optimizer entries such as {'trainer': 'Linear'} or "
        "{'trainer': 'Weak'} unless the user asks for explicit phase-level hyperparameters; "
        "matching trainer defaults from the selected profile are preserved unless overridden.",
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
    assert detail.examples[3].overrides == {
        "cv": {
            "search": {
                "mode": "nelder_mead_like",
                "bounds": {
                    "model.koopman_dimension": [4, 8],
                    "training.weak_form_params.N": {
                        "lower": 9,
                        "upper": 17,
                        "parity": "odd",
                    },
                },
                "max_iterations": 12,
                "reflection": 1.0,
                "expansion": 2.0,
                "contraction": 0.5,
                "shrink": 0.5,
            },
            "selection": {
                "goal": "minimize",
                "tie_breakers": ["std_metric", "combo_index"],
            },
        }
    }
    assert detail.examples[3].notes == (
        "This executes a bounded Nelder-Mead search over the provided lower/upper parameter "
        "ranges.",
    )
    assert detail.examples[4].overrides == {
        "cv": {
            "search": {
                "mode": "multi_start_nelder_mead",
                "bounds": {"model.koopman_dimension": [4, 8]},
                "max_iterations": 80,
            },
            "selection": {
                "goal": "minimize",
                "tie_breakers": ["std_metric", "combo_index"],
            },
        }
    }
    assert detail.examples[5].overrides["model"]["koopman_dimension"] == 2
