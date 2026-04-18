---
name: dymad-train-eval-workflow
description: Use DyMAD MCP tools to register train/test datasets, inspect schemas, translate natural-language model/training requests into structured config, train a model, and evaluate rollout error on a test dataset.
---

# DyMAD Train/Eval Workflow

Use this skill when the user provides train/test dataset files and wants a DyMAD model trained and evaluated.

Required MCP tools:
- `register_dataset_file`
- `inspect_dataset`
- `list_training_capabilities`
- `list_evaluation_capabilities`
- `compile_training_request`
- `train_compiled_request`
- `evaluate_checkpoint`

Workflow:
1. Register each provided dataset file with `register_dataset_file`.
2. Inspect train and test datasets with `inspect_dataset`.
3. Use `list_training_capabilities` if you need to confirm which model families support the dataset kind.
4. Translate the user's natural-language modeling request into a structured `compile_training_request.overrides` dict.
5. Pick the stable `model_key` from the requested DyMAD model family.
6. Call `compile_training_request`.
7. Call `train_compiled_request`.
8. Call `list_evaluation_capabilities` for the evaluation dataset handle before selecting an evaluation metric.
9. Use the advertised `supported_metrics` and `parameter_schema.metric.default` as the authoritative source of evaluation metric names.
10. Call `evaluate_checkpoint` on the returned checkpoint handle and the registered test dataset.
11. Report the checkpoint path, training summary path, evaluation metrics, and representative rollout plot path.

Rules:
- Do not pass free text into MCP tools. Natural-language interpretation happens in the skill, not the tool layer.
- Require dataset handles for training and evaluation; never pass raw dataset paths directly into compile/train/evaluate tools.
- Prefer `overrides.phases` for staged training, such as weak form followed by NODE.
- Treat `list_evaluation_capabilities` as the authoritative contract for supported evaluation metric names. Do not guess metric keys or inspect code to infer them.
- If the user does not name an evaluation metric, use `parameter_schema.metric.default` when present, otherwise use the only advertised supported metric.
- If the dataset schema is incompatible with the requested model family, stop and explain the mismatch.
- If the user asks for unsupported metrics, formats, or model families, say so directly instead of inventing a fallback workflow.

Translation guidance:
- Map architecture requests into the nested `model` config fields.
- Map optimizer/training requests into `phases`.
- Keep runtime-owned fields out of user overrides:
  - `data.path`
  - `data_valid.path`
  - `path.*`
- If the user does not specify a `reference_profile`, let `compile_training_request` infer it.

Expected final report:
- checkpoint handle and checkpoint path
- training run handle
- training summary path
- evaluation handle
- evaluation metrics path
- aggregate rollout metrics
- representative plot path, or the graph-plot skip reason
