---
name: dymad-evaluate-model
description: Use DyMAD MCP tools to register or reuse a checkpoint and test dataset, validate compatibility, and evaluate one DyMAD model with the Phase 1 rollout evaluation surface.
---

# DyMAD Evaluate Model

Use this skill when the user wants to evaluate an existing DyMAD checkpoint on a registered or newly provided dataset.

Required MCP tools:
- `register_dataset_file`
- `inspect_dataset`
- `register_checkpoint`
- `validate_dataset_compatibility`
- `evaluate_model`

Workflow:
1. Register the test dataset with `register_dataset_file` if it is not already registered.
2. Inspect the dataset with `inspect_dataset`.
3. Register the checkpoint with `register_checkpoint` if it is not already registered.
4. Validate checkpoint and dataset compatibility with `validate_dataset_compatibility` using the checkpoint's `model_ref`.
5. Call `evaluate_model`.
6. Report the evaluation handle, metrics path, aggregate metrics, and plot path or skip reason.

Rules:
- Do not pass free text into MCP tools. The skill translates the user request into structured tool inputs.
- Always validate compatibility before calling `evaluate_model`.
- Phase 1 evaluation is limited to the current rollout evaluation behavior and current metric coverage.
- If the user asks for unsupported metrics, say so directly instead of inventing a fallback.
- Report graph-plot skip reasons explicitly when plots are unavailable.

Expected final report:
- checkpoint handle
- test dataset handle
- evaluation handle
- metrics path
- aggregate metrics
- representative plot path, or the skip reason
- note that Phase 1 evaluation currently supports `rollout_rmse`
