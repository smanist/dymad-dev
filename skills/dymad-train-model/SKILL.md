---
name: dymad-train-model
description: Use DyMAD MCP tools to discover compatible model families and reference profiles, validate and materialize a structured training config, and train one DyMAD model from registered datasets.
---

# DyMAD Train Model

Use this skill when the user provides one training dataset, optionally a validation dataset, and wants a DyMAD model trained.

Required MCP tools:
- `register_dataset_file`
- `inspect_dataset`
- `list_model_families`
- `describe_model_family`
- `list_reference_profiles`
- `describe_reference_profile`
- `resolve_training_intent`
- `validate_dataset_compatibility`
- `validate_training_config`
- `materialize_training_config`
- `train_model`

Workflow:
1. Register each provided dataset file with `register_dataset_file`.
2. Inspect the registered datasets with `inspect_dataset`.
3. Resolve the user's free text request into sparse structured intent with `resolve_training_intent`.
4. Use `intent.accepted_inputs` and `intent.trace` as the authoritative translation surface for dataset kinds, compatible model refs, reference profiles, supported phase trainers, and config-construction strategy.
5. Only if `resolve_training_intent` leaves unresolved fields or the user explicitly asks for options, use `list_model_families` and `describe_model_family` to inspect candidates.
6. Only if needed after that, use `list_reference_profiles` and `describe_reference_profile` to inspect compatible defaults.
7. Validate dataset/model compatibility with `validate_dataset_compatibility`.
8. Validate the full structured training request with `validate_training_config`.
9. Materialize the normalized config with `materialize_training_config`.
10. Call `train_model`.
11. Report the run handle, checkpoint handle, config path, summary path, and key metrics.

Rules:
- Prefer `resolve_training_intent` to translate free text into sparse structured overrides before validation.
- Treat a valid `resolve_training_intent` result as sufficient guidance for model family, dataset kind, transforms, and phase translation; do not probe the repo or installed package for trainer names or config enums.
- For config construction, use `intent.accepted_inputs.suggested_validate_request` or `intent.structured_config()` directly as the validator payload. Do not expand profile defaults by hand and do not mine local YAML files, traces, or artifacts for a template config.
- Treat `validate_training_config(...).normalized_config` as the source of truth for the full config shape. The intent result should stay sparse.
- Discovery comes first, validation comes second, execution comes last.
- Require dataset handles for validation and training; never pass raw dataset paths directly into `train_model`.
- Keep runtime-owned fields out of user config:
  - `data.path`
  - `data_valid.path`
  - `path.*`
- Translate architecture requests into `config.model`.
- Translate optimizer and schedule requests into `config.phases`.
- If the dataset schema is incompatible with the requested model family, stop and explain the mismatch.
- If the validation result is invalid, surface the rejection reason directly instead of guessing a fallback.
- If the user does not specify a `reference_profile`, let validation infer it.
- If `register_dataset_file` rejects `kind`, correct it using the MCP error message or `intent.accepted_inputs.dataset_kinds`; do not inspect repo code to infer supported dataset kinds.
- Do not inspect or modify repo code unless an MCP tool reports that the requested workflow is unsupported.

Expected final report:
- selected `model_ref`
- selected or inferred `reference_profile`
- training run handle
- checkpoint handle and checkpoint path
- materialized config path
- training summary path
- history plot path, if present
- prediction plot path, if present
- aggregate training metrics
