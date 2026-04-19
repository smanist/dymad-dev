# Repository Architecture

## Purpose

This document is the current source of truth for:

- repo package boundaries
- MCP user/developer surfaces
- registry/compiler/execution layering
- checkpoint compatibility flow
- where new features should plug in

Use it instead of the old `docs/mcp_*.md` and `docs/checkpoint-e2e-layering.md` design notes.

Maintenance rule:

- If a change alters MCP surfaces, boundary ownership, registry/compiler responsibilities,
  checkpoint materialization flow, or persisted handle types, update this document in the same
  change.

## Package Map

| Area | Owns |
| --- | --- |
| `src/dymad/agent/mcp` | MCP-facing tool adapters and server assembly |
| `src/dymad/agent/registry` | user-facing capability metadata, profiles, schemas, supported analyses/evaluations |
| `src/dymad/agent/compiler` | typed request validation and compilation into persisted requests |
| `src/dymad/agent/exec` | workflow orchestration and compatibility execution |
| `src/dymad/agent/facade` | stable typed boundary over persisted objects |
| `src/dymad/agent/store` | in-memory/filesystem-backed artifact records and handle persistence |
| `src/dymad/models` | model families, collections, typed model specs, rollout helpers |
| `src/dymad/training` | training runtime, phases, trainers, execution helpers |
| `src/dymad/io` | checkpoint loading, trajectory/data managers, legacy public runtime entrypoints |
| `src/dymad/core` | typed runtime/series/transform building blocks |
| `src/dymad/numerics` | math and linear-algebra utilities |
| `src/dymad/sako` | spectral analysis runtime and adapters |

## Layer Stack

Current user-facing stack:

```text
MCP server
  -> user_tools / developer_tools
  -> registry + compiler
  -> CompatibilityExecutor
  -> FacadeOperations
  -> ObjectStore / FilesystemArtifactStore
  -> legacy runtime/training/checkpoint/analysis code
```

Important distinction:

- `server.py` only registers tools and mode splits.
- `user_tools.py` is the high-level surface.
- `demo_tools.py` plus `developer_tools.py` expose the raw/developer surface.
- `CompatibilityExecutor` still owns orchestration, but some compatibility flows intentionally
  materialize through legacy `io/*` code instead of fully executor-native implementations.

## MCP Surfaces

`build_server(mode=...)` supports three registrations:

- `mode="user"`: high-level workflows
- `mode="developer"`: raw/debug/compatibility tools
- `mode="both"`: both surfaces on one server

### User Mode

User mode is registry/compiler-backed. It currently exposes:

- `list_training_capabilities`
- `list_analysis_capabilities`
- `list_evaluation_capabilities`
- `describe_training_capability`
- `compile_training_request`
- `train_compiled_request`
- `evaluate_checkpoint`
- `compile_analysis_request`
- `run_analysis_request`

Notes:

- user mode does not require raw `model_ref`
- user mode compiles `model_key` plus validated overrides into persisted compiled requests
- `describe_training_capability` is the authoritative contract for allowed overrides, phase-entry
  schemas, CV sweep support metadata, natural-language-to-override translation guidance, and
  surfaced training constraints
- user mode currently assumes dataset handles already exist

### Developer Mode

Developer mode keeps the raw and compatibility-oriented path available:

- `register_dataset_file`
- `inspect_dataset`
- `register_checkpoint`
- `prepare_prediction_request`
- `plan_checkpoint_prediction`
- `train_model`
- `evaluate_model`
- `list_model_capabilities`
- `resolve_model_capability`
- `list_profile_capabilities`
- `describe_training_capability`
- `describe_object`
- `list_objects`

Use developer mode when debugging boundary behavior, raw config/profile selection, or compatibility
flows.

## Current Workflow Paths

### Training and Evaluation

High-level path:

```text
register_dataset_file
  -> describe_training_capability / list_training_capabilities
  -> compile_training_request
  -> train_compiled_request
  -> evaluate_checkpoint
```

Compilation resolves:

- `model_key` -> model capability -> default `model_ref`
- dataset kind compatibility
- default or explicit profile
- allowed user overrides
- optional single-split CV sweep settings under `overrides.cv`
- phase overrides normalized against matching profile defaults so trainer-specific phase config is
  preserved unless explicitly overridden
- translation guidance and surfaced constraint notes for clients that map natural-language requests
  into structured overrides, including CV sweep requests
- effective config
- trainer kind

Execution still routes through the current training runtime in `src/dymad/training/*`.

### Analysis

Current analysis path:

```text
compile_analysis_request
  -> persisted compiled analysis request
  -> run_analysis_request
  -> analysis-specific execution in CompatibilityExecutor
```

Currently supported workflow keys:

- `spectral_koopman`
- `vortex_transform_modes`

### Checkpoint Compatibility

Current checkpoint load path:

```text
dymad.io.load_model(...)
  -> CompatibilityExecutor.plan_checkpoint_prediction(...)
  -> FacadeOperations.register_checkpoint(...)
  -> FacadeOperations.prepare_prediction_request(...)
  -> legacy checkpoint materialization in dymad.io.checkpoint
```

This is an important current-state detail:

- `CompatibilityExecutor.plan_checkpoint_prediction(...)` is active.
- `CompatibilityExecutor.materialize_checkpoint_prediction(...)` is not the active materialization
  path today; it is a placeholder that raises `NotImplementedError`.
- the persisted checkpoint and prediction-request handles still record the boundary state used by
  `load_model(...)`.

So the boundary plan is real, but final checkpoint materialization still goes through
`dymad.io.checkpoint`.

## Persisted Artifacts and Handles

The object store persists the main boundary objects used by MCP and compatibility workflows:

- datasets: `ds_*`
- checkpoints: `chk_*`
- training runs: `run_*`
- compiled training requests: `trainreq_*`
- compiled analysis requests: `analysisreq_*`
- evaluations: `eval_*`
- prediction requests: `pred_*`
- spectral snapshots: `specsnap_*`

If a new workflow needs durable planning or inspection across calls, it usually needs a new record
type in `agent/store` plus matching facade helpers.

## Design Rules

- Keep policy and validation out of `server.py`.
- Prefer stable user-facing keys in `registry/*` over raw import strings in user-mode flows.
- Put request-shape validation in `compiler/*`, not in MCP adapters.
- Put orchestration in `exec/*`, not in registry or MCP modules.
- Put persistence logic in `store/*` and `facade/*`, not in executor methods.
- Keep model/math/runtime behavior in the implementation packages unless the public boundary
  changes.

## Tests That Define the Boundary

Use these as the fastest ground truth for the current architecture:

- `tests/test_mcp_server_modes.py`: user/developer mode split
- `tests/test_mcp_user_tools.py`: user-mode compile/train/evaluate path
- `tests/test_training_compiler.py`: typed training compiler behavior
- `tests/test_analysis_workflows.py`: compiled analysis workflows
- `tests/test_checkpoint_e2e_layering.py`: checkpoint planning through exec/facade/store
- `tests/test_public_load_model_boundary.py`: `load_model(...)` still materializes through
  `dymad.io.checkpoint`

## When Adding Features

If you are deciding where a change belongs, use [feature-placement.md](/Users/daninghuang/Repos/dymad-dev/docs/feature-placement.md).

If your change moves the answer, update that file too.
