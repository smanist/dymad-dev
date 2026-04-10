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
- `prepare_prediction_request`
- `predict_checkpoint`
- `compute_rollout_metrics`
- `plot_rollouts`

Workflow:
1. Register the test dataset with `register_dataset_file` if it is not already registered.
2. Inspect the dataset with `inspect_dataset`.
3. Register the checkpoint with `register_checkpoint` if it is not already registered.
4. Validate checkpoint and dataset compatibility with `validate_dataset_compatibility` using the checkpoint's `model_ref`.
5. Create a prediction request with `prepare_prediction_request`.
6. Call `predict_checkpoint`.
7. Call `compute_rollout_metrics`.
8. Call `plot_rollouts`.
9. Report the prediction handle, evaluation handle, metrics path, aggregate metrics, and plot path or skip reason.

Rules:
- Do not pass free text into MCP tools. The skill translates the user request into structured tool inputs.
- Always validate compatibility before calling `evaluate_model`.
- Phase 1 evaluation is limited to the current rollout evaluation behavior and current metric coverage.
- If the user asks for unsupported metrics, say so directly instead of inventing a fallback.
- Report graph-plot skip reasons explicitly when plots are unavailable.
- Use structured metric specs. Current v1 metrics are `rollout_rmse`, `rollout_mae`, `horizon_rmse`, and `horizon_mae`.

Expected final report:
- checkpoint handle
- test dataset handle
- prediction handle
- evaluation handle
- metrics path
- aggregate metrics
- representative plot path, or the skip reason
- note that graph plotting is still unsupported in v1
