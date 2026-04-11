from __future__ import annotations

from dymad.agent.exec.training_profiles import (
    PROFILE_ALIASES,
    available_profiles,
    profile_config,
    resolve_profile_name,
)
from dymad.agent.registry import (
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
