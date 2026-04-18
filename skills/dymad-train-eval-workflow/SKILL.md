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
- `describe_training_capability`
- `list_evaluation_capabilities`
- `compile_training_request`
- `train_compiled_request`
- `evaluate_checkpoint`

Workflow:
1. Register each provided dataset file with `register_dataset_file`.
2. Inspect train and test datasets with `inspect_dataset`.
3. Use `list_training_capabilities` if you need to confirm which model families support the dataset kind.
4. Pick the stable `model_key` from the requested DyMAD model family.
5. Call `describe_training_capability` for that `model_key` and dataset kind before translating any nontrivial training request.
6. Translate the user's natural-language modeling request into a structured `compile_training_request.overrides` dict using `translation_guidance`, `constraint_notes`, `phase_entry_schemas`, `cv_schema`, and `examples` from `describe_training_capability`.
7. Call `compile_training_request`.
8. Call `train_compiled_request`.
9. Call `list_evaluation_capabilities` for the evaluation dataset handle before selecting an evaluation metric.
10. Use the advertised `supported_metrics` and `parameter_schema.metric.default` as the authoritative source of evaluation metric names.
11. Call `evaluate_checkpoint` on the returned checkpoint handle and the registered test dataset.
12. Report the checkpoint path, training summary path, CV artifact paths when present, evaluation metrics, and representative rollout plot path.

Rules:
- Do not pass free text into MCP tools. Natural-language interpretation happens in the skill, not the tool layer.
- Require dataset handles for training and evaluation; never pass raw dataset paths directly into compile/train/evaluate tools.
- For staged training, prefer `overrides.phases`, such as weak form followed by NODE.
- For hyperparameter sweeps, encode the request as `overrides.cv.param_grid` and only use `overrides.cv.metric` when the user names a specific selection metric.
- Treat `list_evaluation_capabilities` as the authoritative contract for supported evaluation metric names. Do not guess metric keys or inspect code to infer them.
- If the user does not name an evaluation metric, use `parameter_schema.metric.default` when present, otherwise use the only advertised supported metric.
- If the dataset schema is incompatible with the requested model family, stop and explain the mismatch.
- If the user asks for unsupported metrics, formats, or model families, say so directly instead of inventing a fallback workflow.
- Treat `describe_training_capability` as the authoritative contract for allowed overrides, phase schema, CV sweep support, translation guidance, and surfaced training constraints.

Translation guidance:
- Map architecture requests into the nested `model` config fields.
- Map optimizer/training requests into `phases`.
- Map hyperparameter sweep requests into `cv.param_grid`.
- Keep runtime-owned fields out of user overrides:
  - `data.path`
  - `data_valid.path`
  - `path.*`
- If the user does not specify a `reference_profile`, let `compile_training_request` infer it.

Expected final report:
- checkpoint handle and checkpoint path
- training run handle
- training summary path
- cv_results_path when present
- cv_plot_path when present
- evaluation handle
- evaluation metrics path
- aggregate rollout metrics
- representative plot path, or the graph-plot skip reason
