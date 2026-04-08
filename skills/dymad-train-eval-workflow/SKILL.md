---
name: dymad-train-eval-workflow
description: Use DyMAD MCP tools to register train/test datasets, inspect schemas, translate natural-language model/training requests into structured config, train a model, and evaluate rollout error on a test dataset.
---

# DyMAD Train/Eval Workflow

Use this skill when the user provides train/test dataset files and wants a DyMAD model trained and evaluated.

Required MCP tools:
- `register_dataset_file`
- `inspect_dataset`
- `train_model`
- `evaluate_model`

Workflow:
1. Register each provided dataset file with `register_dataset_file`.
2. Inspect train and test datasets with `inspect_dataset`.
3. Translate the user's natural-language modeling request into a structured `train_model.config` dict.
4. Pick `model_ref` and optional `reference_profile` from the requested DyMAD model family.
5. Call `train_model`.
6. Call `evaluate_model` on the returned checkpoint handle and the registered test dataset.
7. Report the checkpoint path, training summary path, evaluation metrics, and representative rollout plot path.

Rules:
- Do not pass free text into MCP tools. Natural-language interpretation happens in the skill, not the tool layer.
- Require dataset handles for training and evaluation; never pass raw dataset paths directly into `train_model` or `evaluate_model`.
- Prefer `config.phases` for staged training, such as weak form followed by NODE.
- If the dataset schema is incompatible with the requested model family, stop and explain the mismatch.
- If the user asks for unsupported metrics, formats, or model families, say so directly instead of inventing a fallback workflow.

Translation guidance:
- Map architecture requests into the nested `model` config fields.
- Map optimizer/training requests into `phases`.
- Keep runtime-owned fields out of user config:
  - `data.path`
  - `data_valid.path`
  - `path.*`
- If the user does not specify a `reference_profile`, let `train_model` infer it.

Expected final report:
- checkpoint handle and checkpoint path
- training run handle
- training summary path
- evaluation handle
- evaluation metrics path
- aggregate rollout metrics
- representative plot path, or the graph-plot skip reason
