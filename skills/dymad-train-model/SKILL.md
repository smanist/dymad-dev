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
- `validate_dataset_compatibility`
- `validate_training_config`
- `materialize_training_config`
- `train_model`

Workflow:
1. Register each provided dataset file with `register_dataset_file`.
2. Inspect the registered datasets with `inspect_dataset`.
3. Discover candidate model families with `list_model_families` and `describe_model_family`.
4. Discover compatible reference profiles with `list_reference_profiles` and `describe_reference_profile`.
5. Translate the user's free text request into structured `model_ref`, optional `reference_profile`, and `config`.
6. Validate dataset/model compatibility with `validate_dataset_compatibility`.
7. Validate the full structured training request with `validate_training_config`.
8. Materialize the normalized config with `materialize_training_config`.
9. Call `train_model`.
10. Report the run handle, checkpoint handle, config path, summary path, and key metrics.

Rules:
- Do not pass free text into MCP tools. Natural-language interpretation happens in the skill, not the tool layer.
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
