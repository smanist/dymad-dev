---
name: dymad-train-eval-workflow
description: Use DyMAD MCP tools to register datasets, inspect schemas, translate natural-language model/training requests into structured config, train and evaluate models, then export a reproducible CLI config/run manifest so users can audit, edit, and rerun with the package-level dymad CLI.
---

# DyMAD Train/Eval Workflow

Use this skill when the user provides train/test dataset files and wants a DyMAD model trained,
evaluated, and reproducibly recorded.

Default to MCP as the agent control plane. Use the CLI for exported reproducibility and for user
reruns, not as the primary skill control plane unless the user explicitly asks to operate only
through shell commands.

Prefer an MCP server mode that exposes both developer dataset-registration tools and user workflow
tools. If the server is developer-only, `list_evaluation_capabilities` is still expected to be
available for metric discovery.

## Required Surfaces

MCP tools:
- `register_dataset_file`
- `inspect_dataset`
- `list_training_capabilities`
- `describe_training_capability`
- `list_evaluation_capabilities`
- `compile_training_request`
- `start_training_run`
- `describe_training_run`
- `read_training_run_log`
- `evaluate_checkpoint`

CLI commands for exported reruns:
- `dymad config validate CONFIG --out RUN_DIR`
- `dymad train --config CONFIG --out RUN_DIR`
- `dymad status --run RUN_DIR`
- `dymad log --run RUN_DIR`
- `dymad eval --run RUN_DIR`
- `dymad report --run RUN_DIR`

## MCP Workflow

1. Register each provided dataset file with `register_dataset_file`.
2. Inspect train and test datasets with `inspect_dataset`.
3. Use `list_training_capabilities` if you need to confirm which model families support the dataset kind.
4. Pick the stable `model_key` from the requested DyMAD model family.
5. Call `describe_training_capability` for that `model_key` and dataset kind before translating any nontrivial training request.
6. Translate the user's natural-language modeling request into `compile_training_request.overrides` using `translation_guidance`, `constraint_notes`, `phase_entry_schemas`, `cv_schema`, and `examples`. Before compiling, compare overrides against `runtime_owned_override_paths`, `allowed_override_top_level_keys`, `allowed_data_override_keys`, and `runtime_owned_model_keys`.
7. Call `compile_training_request`.
8. If validation fails because overrides contain runtime-owned fields, remove all runtime-owned fields in the same repair pass before retrying once. If validation fails for any other issue directly addressed by surfaced constraints or the error message, repair once and retry. Do not inspect package code to infer phase/config shape.
9. Call `start_training_run`.
10. Poll with `describe_training_run` until the run reaches `SUCCEEDED` or `FAILED`; use `read_training_run_log` when you need incremental worker logs.
11. If the run fails, report the structured error metadata and relevant log excerpt instead of continuing.
12. Call `list_evaluation_capabilities` for the evaluation dataset handle before selecting a metric.
13. Use the advertised `supported_metrics` and `parameter_schema.metric.default` as the authoritative source of evaluation metric names.
14. Call `evaluate_checkpoint` on the returned checkpoint handle and registered test dataset once `describe_training_run` reports `SUCCEEDED`.
15. Report checkpoint path, training summary path, CV artifact paths when present, evaluation metrics, and representative rollout plot path.

## Export Reproducible CLI Run

After a successful MCP compile/train/eval workflow, export a CLI rerun package when the user wants
auditability or reproducibility, or when dataset paths were provided as files:

1. Write a CLI config YAML with:
   - `version: 1`
   - `model_key`
   - optional `reference_profile`
   - `data.train.path`, optional `data.valid.path`, optional `data.test.path`
   - dataset `kind` and `format` when known
   - translated `overrides`
   - `run.name`, `run.seed`, `run.device`, `run.max_workers` when used
   - `evaluation.metric`, `plot_selection`, `max_plots`, `predict_kwargs` when used
2. Choose an output run directory matching `run.name`, for example `runs/foo`.
3. Validate the exported config with `dymad config validate CONFIG --out RUN_DIR`.
4. Tell the user the rerun commands:
   - `dymad train --config CONFIG --out RUN_DIR`
   - `dymad eval --run RUN_DIR`
   - `dymad report --run RUN_DIR`
5. Explain that the rerun will create new handles/checkpoints but should reproduce the same user-mode request.

If the MCP workflow used only handles, recover source dataset paths, kinds, and formats from facade
dataset records before exporting. Do not export developer-mode-only `model_ref` or raw config flows
as user-mode CLI configs unless they can be represented as `model_key` plus allowed `overrides`.

## Rules

- MCP uses handles. CLI configs use dataset paths that the CLI registers internally.
- Do not pass free text into MCP tools. Natural-language interpretation happens in the skill, not the tool layer.
- Require dataset handles for MCP training and evaluation; never pass raw dataset paths directly into compile/train/evaluate tools.
- Use `model_key` to choose the model family. Do not set implementation selectors such as `overrides.model.name`; the runtime selects the concrete `model_ref` from the model key, dataset kind, and reference profile.
- For staged training, translate the requested trainer order into `overrides.phases` in the same order.
- For ordered optimizer sequences, support any mix of `Linear`, `Weak`, and `NODE` named by the user.
- Prefer minimal optimizer entries such as `{"trainer": "Linear"}`, `{"trainer": "Weak"}`, or `{"trainer": "NODE"}` unless the user asks for per-phase hyperparameters.
- For hyperparameter sweeps, encode the request as `overrides.cv.param_grid` and only use `overrides.cv.metric` when the user names a specific selection metric.
- Treat `describe_training_capability` as the authoritative contract for allowed overrides, phase schema, CV sweep support, translation guidance, and surfaced constraints.
- Treat `list_evaluation_capabilities` as the authoritative contract for supported evaluation metric names. Do not guess metric keys or inspect code to infer them.
- If no metric is named, use `parameter_schema.metric.default` when present, otherwise use the only advertised supported metric.
- Keep runtime-owned fields out of user overrides:
  - `data.path`
  - `data_valid.path`
  - `path`
  - `model.name`
- If the user does not specify a `reference_profile`, let `compile_training_request` infer it.
- If the dataset schema is incompatible with the requested model family, stop and explain the mismatch.
- If the user asks for unsupported metrics, formats, or model families, say so instead of inventing a fallback.

## Expected Final Report

- checkpoint handle and checkpoint path
- training run handle
- training summary path
- `cv_results_path` when present
- `cv_plot_path` when present
- evaluation handle
- evaluation metrics path
- aggregate rollout metrics
- representative plot path, or graph-plot skip reason
- exported CLI config path, if exported
- rerun run directory and `dymad-run.json` manifest path, if exported
