# MCP Current State Audit

## Scope and Evidence

This audit is based on the current repo checkout, with direct inspection of:

- `src/dymad/agent/mcp/server.py`
- `src/dymad/agent/mcp/demo_tools.py`
- `src/dymad/agent/exec/context.py`
- `src/dymad/agent/exec/workflow.py`
- `src/dymad/agent/facade/operations.py`
- `src/dymad/agent/facade/handles.py`
- `src/dymad/agent/store/object_store.py`
- `src/dymad/agent/store/filesystem_artifact_store.py`
- `src/dymad/agent/exec/training_profiles.py`
- `src/dymad/models/model_spec.py`
- `src/dymad/models/collections.py`
- `src/dymad/models/recipes.py`
- `src/dymad/models/helpers.py`
- `src/dymad/training/phases.py`
- `src/dymad/training/phase_pipeline.py`
- `src/dymad/training/trainer_run.py`
- `src/dymad/io/checkpoint.py`
- `skills/dymad-train-eval-workflow/SKILL.md`
- representative scripts under `scripts/`
- representative tests under `tests/`
- docs including `docs/checkpoint-e2e-layering.md`

Important uncertainty:

- I did **not** find an explicit MCP-side prompt parser or keyword-resolution module in this checkout under `src/dymad/agent` or `src/dymad/agent/mcp`.
- The closest repo-local resolution logic is:
  - model/profile aliasing in `src/dymad/agent/exec/training_profiles.py`
  - typed predefined-model resolution in `src/dymad/models/*`
  - skill guidance in `skills/dymad-train-eval-workflow/SKILL.md`
- If a prompt parser exists, it is either out of tree, not committed here, or lives outside the inspected MCP packages.

## Executive Summary

The repo already has a meaningful layered skeleton:

```text
FastMCP server
  -> DemoTools JSON wrapper
  -> CompatibilityExecutor workflows
  -> FacadeOperations
  -> ObjectStore / FilesystemArtifactStore
  -> legacy training / checkpoint / analysis code
```

This is real progress, not just aspiration. The strongest pieces already in place are:

- typed artifact handles and persisted records
- a narrow MCP tool surface instead of exposing every internal function
- a training workflow that can infer reference profiles and write an effective YAML config
- a typed internal training phase system
- typed predefined model specs and recipe resolution

The main weakness is not that there is no structure; it is that the structure stops short of a true capability registry and typed user-facing compiler. The current tool surface still depends on raw strings and raw nested config dicts for important decisions.

## 1. Current MCP Entrypoints and Tool Inventory

## Server entrypoint

`src/dymad/agent/mcp/server.py` builds one FastMCP server around `DemoTools`.

Registered tools:

1. `register_dataset_file(path, format="npz", kind="regular")`
2. `inspect_dataset(dataset_handle)`
3. `register_checkpoint(model_ref, checkpoint_path, device="cpu")`
4. `prepare_prediction_request(checkpoint_handle, horizon, has_control=False, has_graph=False)`
5. `plan_checkpoint_prediction(model_ref, checkpoint_path, horizon, has_control=False, has_graph=False)`
6. `train_model(train_dataset_handle, artifact_root, model_ref, valid_dataset_handle=None, reference_profile=None, config=None, run_name=None, seed=None, device="auto", max_workers=1)`
7. `evaluate_model(checkpoint_handle, test_dataset_handle, metric, artifact_root, plot_selection="median", max_plots=1, predict_kwargs=None)`
8. `describe_object(handle)`
9. `list_objects(kind=None)`

Observations:

- The current MCP surface is already narrower than the problem statement suggests.
- It does **not** expose every atomic training/postprocess/analysis step as a tool.
- It is mostly centered on:
  - artifact registration
  - one training workflow
  - one evaluation workflow
  - prediction planning
  - object inspection

## DemoTools wrapper

`src/dymad/agent/mcp/demo_tools.py` is a thin adapter. It does not do capability resolution or config compilation itself. It:

- forwards to `ExecutionContext.facade` or `ExecutionContext.executor`
- wraps outputs into `{ok: bool, data|error: ...}`
- serializes dataclasses with `asdict`

Implication:

- the MCP layer today is mostly transport shaping, not policy or workflow intelligence

## 2. Current Layering That Already Exists

`src/dymad/agent/exec/context.py` wires:

```text
FilesystemArtifactStore
  -> ObjectStore
  -> FacadeOperations
  -> CompatibilityExecutor
```

This is then exposed through:

```text
DemoTools
  -> FastMCP server
```

That matches the repo doc `docs/checkpoint-e2e-layering.md`, which explicitly documents:

```text
core -> agent.facade -> agent.exec -> agent.mcp.demo_tools -> agent.mcp.server
```

What already resembles the target layers:

- façade-like layer: `FacadeOperations`
- registry-like seeds: model collections, training profiles, phase specs
- execution graph fragments: training phase pipeline
- internal atomic functions: `CompatibilityExecutor`, legacy training classes, `load_model`, spectral classes

## 3. Current Workflow Patterns

## Training/evaluation workflow

Current happy path:

```text
register_dataset_file
  -> inspect_dataset
  -> train_model
  -> evaluate_model
```

Tool guidance for this flow lives in `skills/dymad-train-eval-workflow/SKILL.md`.

The skill explicitly says:

- translate natural language into a structured `train_model.config` dict outside the tool layer
- pick `model_ref` and optional `reference_profile`
- do not pass free text into MCP tools

Implication:

- the skill is compensating for weak MCP abstractions
- the agent, not the MCP layer, is expected to perform capability resolution

## Prediction compatibility workflow

Current checkpoint load flow:

```text
dymad.io.load_model(...)
  -> CompatibilityExecutor.plan_checkpoint_prediction(...)
  -> FacadeOperations.register_checkpoint(...)
  -> FacadeOperations.prepare_prediction_request(...)
  -> legacy checkpoint materialization
```

This is verified by tests such as:

- `tests/test_checkpoint_e2e_layering.py`
- `tests/test_boundary_skeleton.py`

## Spectral planning workflow

There is a partial spectral boundary:

- `CompatibilityExecutor.plan_spectral_analysis(...)`
- `FacadeOperations.register_spectral_snapshot(...)`
- `CompatibilityExecutor.materialize_spectral_adapter(...)`

But this is not exposed as MCP tools today.

## 4. Where Tools Are Too Granular

The current MCP tool surface is not extremely wide, but several tools are still too low-level for user mode:

- `register_checkpoint`
  - requires exact `model_ref` import path string
- `prepare_prediction_request`
  - internal planning primitive, not a user goal
- `plan_checkpoint_prediction`
  - compatibility seam for checkpoint loading, not a user-facing workflow
- `describe_object` / `list_objects`
  - useful for developer mode, low value for ordinary user mode

`train_model` and `evaluate_model` are closer to user workflows, but they still expose internal choices that make them developer-leaning:

- `model_ref`
- `reference_profile`
- raw `config: dict[str, Any]`
- `metric`
- `predict_kwargs`

## 5. Where Raw Strings Are Required

The interface is still strongly stringly typed.

## MCP / façade / exec strings

- `model_ref`
  - exact `"<module>:<name>"` import string
  - resolved by `_resolve_model_ref` in `src/dymad/agent/exec/workflow.py`
- `reference_profile`
  - exact profile key such as `kbf-regular-default`
- `metric`
  - currently only `"rollout_rmse"` is valid
- `plot_selection`
  - `"best"`, `"worst"`, `"median"`
- dataset `format`
  - currently only `"npz"`
- dataset `kind`
  - `"regular"` or `"graph"`
- `device`
  - strings such as `"auto"` or `"cpu"`

## Training config strings

The nested config dict can contain many raw string selectors, including:

- model-level:
  - `activation`
  - `weight_init`
  - `autoencoder_type`
  - `processor_type`
  - `gcl`
  - `predictor_type`
- phase-level:
  - `type`
  - `trainer`
  - `method`
  - `export_kind`
  - `operation`
- transform-level:
  - transform `type`
  - transform-specific mode strings
- kernel-level:
  - kernel `type`

The typed model spec work in `src/dymad/models/model_spec.py` reduces some internal string dispatch, but that typing does not yet reach the MCP request surface.

## 6. How Config Is Built Today

The effective training config path is:

```text
train_model(...)
  -> CompatibilityExecutor.train_model(...)
  -> _effective_config(...)
  -> profile_config(profile_name)
  -> deep merge user config
  -> inject runtime-owned paths and run name
  -> normalize legacy training aliases
  -> write YAML
  -> instantiate legacy trainer class
```

Key pieces:

- `src/dymad/agent/exec/training_profiles.py`
  - `PROFILE_REGISTRY`
  - `PROFILE_ALIASES`
  - `resolve_profile_name(...)`
  - `profile_config(...)`
- `src/dymad/agent/exec/workflow.py`
  - `_effective_config(...)`
  - `_validate_user_config(...)`
  - `_select_trainer(...)`
  - `_write_training_config(...)`
- `src/dymad/utils/misc.py`
  - `_normalize_legacy_training_config(...)`

What this means in practice:

- config assembly is currently a dict-template merge, not a typed compile step
- the main guardrails are:
  - reserved-field blocking
  - profile name validation
  - trainer selection sanity checks
- there is no schema-driven validation of which overrides are legal for a given model family

## 7. What Validation Already Exists

There is meaningful validation, but it is fragmented.

## Artifact and handle validation

`src/dymad/agent/facade/handles.py`:

- regex-validated typed handles:
  - `CheckpointHandle`
  - `DatasetHandle`
  - `TrainingRunHandle`
  - `EvaluationHandle`
  - `PredictionHandle`
  - `SpectralSnapshotHandle`

`src/dymad/agent/facade/operations.py`:

- dataset path exists
- supported dataset formats and kinds
- non-empty `model_ref`, `checkpoint_path`, `run_name`, `metric`
- positive prediction horizon
- listable object kind validation

## Workflow validation

`src/dymad/agent/exec/workflow.py`:

- `model_ref` must be importable
- model graph expectation must match dataset kind
- valid dataset kind must match training dataset kind
- `run_name` must not contain path separators
- user config cannot override runtime-owned paths
- evaluation metric currently restricted to `rollout_rmse`
- plot selection must be one of a small enum

## Training-phase validation

`src/dymad/training/phases.py`:

- explicit typed phase specs:
  - `OptimizerPhaseSpec`
  - `LinearSolvePhaseSpec`
  - `DataPhaseSpec`
  - `AnalysisPhaseSpec`
  - `ExportPhaseSpec`
- normalization and validation of repeat blocks
- rejection of deprecated `ls_update`
- validation of trainer names

## Model-spec validation

`src/dymad/models/*`:

- typed `ModelSpec`
- typed `RolloutSpec`
- typed `MemorySpec`
- recipe resolution checks
- rollout-engine checks for incompatible predictors

## Missing validation

Still missing at the user-facing boundary:

- canonical capability discovery
- legal override lists per model/workflow
- dataset-to-workflow recommendation
- type-safe config request objects
- structured rejection of unknown nested config fields

## 8. What Already Resembles Workflows, Recipes, Registries, Schemas, and Handles

## Handles and persisted records

Strong existing pieces:

- typed handles in `src/dymad/agent/facade/handles.py`
- persisted records in `src/dymad/agent/store/object_store.py`
- JSON persistence in `src/dymad/agent/store/filesystem_artifact_store.py`

These are already a good basis for layer 2 and layer 4 bookkeeping.

## Model recipes and typed specs

Strong existing pieces:

- `ModelSpec` and related typed spec objects in `src/dymad/models/model_spec.py`
- predefined model collection in `src/dymad/models/collections.py`
- typed recipe resolution in `src/dymad/models/recipes.py`

This is already registry-like metadata, even though it is not packaged as an agent-facing capability registry.

## Training profiles

Strong existing pieces:

- `PROFILE_REGISTRY`
- `PROFILE_ALIASES`

These are currently the closest thing to a training-capability registry.

## Training execution graph

Strong existing pieces:

- typed phase specs in `src/dymad/training/phases.py`
- `PhasePipeline` in `src/dymad/training/phase_pipeline.py`
- `TrainerRun` in `src/dymad/training/trainer_run.py`
- typed artifacts and state in `src/dymad/training/phase_runtime.py`

This is already a genuine internal execution graph for training.

## 9. Where Repo Inspection Is Currently Needed

In practice, the current MCP surface does not tell the agent enough to stay inside tools.

The agent often needs repo inspection to answer questions such as:

- Which model families exist?
  - discovered from `src/dymad/models/collections.py`
- Which `model_ref` strings are valid?
  - derived from module/class names
- Which profile names exist?
  - discovered from `PROFILE_REGISTRY`
- Which config keys and combinations are valid?
  - often inferred from:
    - `src/dymad/models/helpers.py`
    - `src/dymad/training/phases.py`
    - YAML files in `scripts/`
    - tests under `tests/test_workflow_*`
- Which training schedules are known-good?
  - inferred from scripts and tests
- Which analysis/postprocess flows exist?
  - inferred almost entirely from scripts and legacy APIs

Representative examples:

- `scripts/linear_time_invariant/lti_train_cli.py`
  - hard-codes case matrices of model + trainer + YAML
- `scripts/lorenz63/lor_train_cli.py`
  - encodes model/kernel/CV combinations in Python
- `scripts/vortex/vor_train_cli.py`
  - encodes alternating phase schedules and transform stacks
- `scripts/vortex/vor_proc_cli.py`
  - mode-analysis workflow via `DataInterface`
- `scripts/vortex/vor_post.py`
  - spectral/postprocessing workflow via `SpectralAnalysis`

Implication:

- repo inspection is currently compensating for missing capability metadata and missing typed request compilation

## 10. Current Post-Processing and Analysis State

The current MCP implementation is training/evaluation-centric.

Analysis and post-processing remain mostly library- or script-driven:

- `DataInterface` in `src/dymad/io/checkpoint.py`
  - transform inspection
  - encode/decode
  - forward/backward modes
- `SpectralAnalysis` in `src/dymad/sako/base.py`
  - checkpoint-backed spectral analysis
- `SpectralAnalysisAdapter` in `src/dymad/sako/adapter.py`
  - typed adapter seam

These are not exposed through MCP today.

This is a major reason the agent falls back to reading scripts and examples.

## 11. Tests and Docs That Confirm Current Intent

Useful evidence:

- `tests/test_demo_tools.py`
  - MCP tool registration and envelope behavior
- `tests/test_mcp_train_eval_tools.py`
  - train/evaluate contract
- `tests/test_checkpoint_e2e_layering.py`
  - load-model boundary routing
- `tests/test_boundary_skeleton.py`
  - handle and spectral boundary checks
- `tests/test_model_spec_resolver.py`
  - typed model/rollout resolution
- `tests/test_training_phase_runtime.py`
  - typed phase normalization/runtime
- `docs/checkpoint-e2e-layering.md`
  - explicit description of current layering

## 12. Audit Conclusions

What already exists:

- a real MCP entrypoint and server
- a typed handle/persistence boundary
- a useful execution layer for training/evaluation
- typed model specifications and typed training phases
- seed registries for profiles and predefined models

What is still weak:

- no unified capability registry
- no typed user-facing config compiler
- no explicit user mode vs developer mode split
- user-facing tools still require internal strings
- analysis/postprocess workflows are mostly outside MCP
- repo inspection is still required to infer legal combinations

Most important architectural reading:

- The repo does **not** need a fresh rewrite from scratch.
- It already contains good candidates for layers 2, 3, and 4.
- The next work should mostly be:
  - consolidating registries
  - lifting existing typing to the MCP boundary
  - wrapping script-era workflows behind stable user-mode tools
